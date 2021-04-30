# pragma pylint: disable=W0603
""" Edge positioning package """
import logging
from typing import Any, Dict, List, NamedTuple

import arrow
import numpy as np
import utils_find_1st as utf1st
from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.constants import DATETIME_PRINT_FORMAT, UNLIMITED_STAKE_AMOUNT
from freqtrade.data.history import get_timerange, load_data, refresh_data
from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.strategy.interface import SellType


logger = logging.getLogger(__name__)


class PairInfo(NamedTuple):
    stoploss: float
    winrate: float
    risk_reward_ratio: float
    required_risk_reward: float
    expectancy: float
    nb_trades: int
    avg_trade_duration: float


class Edge:
    """
    Calculates Win Rate, Risk Reward Ratio, Expectancy
    against historical data for a give set of markets and a strategy
    it then adjusts stoploss and position size accordingly
    and force it into the strategy
    Author: https://github.com/mishaker
    """

    config: Dict = {}
    _cached_pairs: Dict[str, Any] = {}  # Keeps a list of pairs

    def __init__(self, config: Dict[str, Any], exchange, strategy) -> None:

        self.config = config
        self.exchange = exchange
        self.strategy = strategy

        self.edge_config = self.config.get('edge', {})
        self._cached_pairs: Dict[str, Any] = {}  # Keeps a list of pairs
        self._final_pairs: list = []

        # checking max_open_trades. it should be -1 as with Edge
        # the number of trades is determined by position size
        if self.config['max_open_trades'] != float('inf'):
            logger.critical('max_open_trades should be -1 in config !')

        if self.config['stake_amount'] != UNLIMITED_STAKE_AMOUNT:
            raise OperationalException('Edge works only with unlimited stake amount')

        self._capital_ratio: float = self.config['tradable_balance_ratio']
        self._allowed_risk: float = self.edge_config.get('allowed_risk')
        self._since_number_of_days: int = self.edge_config.get('calculate_since_number_of_days', 14)
        self._last_updated: int = 0  # Timestamp of pairs last updated time
        self._refresh_pairs = True

        self._stoploss_range_min = float(self.edge_config.get('stoploss_range_min', -0.01))
        self._stoploss_range_max = float(self.edge_config.get('stoploss_range_max', -0.05))
        self._stoploss_range_step = float(self.edge_config.get('stoploss_range_step', -0.001))

        # calculating stoploss range
        self._stoploss_range = np.arange(
            self._stoploss_range_min,
            self._stoploss_range_max,
            self._stoploss_range_step
        )

        self._timerange: TimeRange = TimeRange.parse_timerange("%s-" % arrow.now().shift(
            days=-1 * self._since_number_of_days).format('YYYYMMDD'))
        if config.get('fee'):
            self.fee = config['fee']
        else:
            try:
                self.fee = self.exchange.get_fee(symbol=expand_pairlist(
                    self.config['exchange']['pair_whitelist'], list(self.exchange.markets))[0])
            except IndexError:
                self.fee = None

    def calculate(self, pairs: List[str]) -> bool:
        if self.fee is None and pairs:
            self.fee = self.exchange.get_fee(pairs[0])

        heartbeat = self.edge_config.get('process_throttle_secs')

        if (self._last_updated > 0) and (
                self._last_updated + heartbeat > arrow.utcnow().int_timestamp):
            return False

        data: Dict[str, Any] = {}
        logger.info('Using stake_currency: %s ...', self.config['stake_currency'])
        logger.info('Using local backtesting data (using whitelist in given config) ...')

        if self._refresh_pairs:
            refresh_data(
                datadir=self.config['datadir'],
                pairs=pairs,
                exchange=self.exchange,
                timeframe=self.strategy.timeframe,
                timerange=self._timerange,
                data_format=self.config.get('dataformat_ohlcv', 'json'),
            )

        data = load_data(
            datadir=self.config['datadir'],
            pairs=pairs,
            timeframe=self.strategy.timeframe,
            timerange=self._timerange,
            startup_candles=self.strategy.startup_candle_count,
            data_format=self.config.get('dataformat_ohlcv', 'json'),
        )

        if not data:
            # Reinitializing cached pairs
            self._cached_pairs = {}
            logger.critical("No data found. Edge is stopped ...")
            return False

        preprocessed = self.strategy.ohlcvdata_to_dataframe(data)

        # Print timeframe
        min_date, max_date = get_timerange(preprocessed)
        logger.info(f'Measuring data from {min_date.strftime(DATETIME_PRINT_FORMAT)} '
                    f'up to {max_date.strftime(DATETIME_PRINT_FORMAT)} '
                    f'({(max_date - min_date).days} days)..')
        headers = ['date', 'buy', 'open', 'close', 'sell', 'high', 'low']

        trades: list = []
        for pair, pair_data in preprocessed.items():
            # Sorting dataframe by date and reset index
            pair_data = pair_data.sort_values(by=['date'])
            pair_data = pair_data.reset_index(drop=True)

            df_analyzed = self.strategy.advise_sell(
                self.strategy.advise_buy(pair_data, {'pair': pair}), {'pair': pair})[headers].copy()

            trades += self._find_trades_for_stoploss_range(df_analyzed, pair, self._stoploss_range)

        # If no trade found then exit
        if len(trades) == 0:
            logger.info("No trades found.")
            return False

        # Fill missing, calculable columns, profit, duration , abs etc.
        trades_df = self._fill_calculable_fields(DataFrame(trades))
        self._cached_pairs = self._process_expectancy(trades_df)
        self._last_updated = arrow.utcnow().int_timestamp

        return True

    def stake_amount(self, pair: str, free_capital: float,
                     total_capital: float, capital_in_trade: float) -> float:
        stoploss = self.stoploss(pair)
        available_capital = (total_capital + capital_in_trade) * self._capital_ratio
        allowed_capital_at_risk = available_capital * self._allowed_risk
        max_position_size = abs(allowed_capital_at_risk / stoploss)
        # Position size must be below available capital.
        position_size = min(min(max_position_size, free_capital), available_capital)
        if pair in self._cached_pairs:
            logger.info(
                'winrate: %s, expectancy: %s, position size: %s, pair: %s,'
                ' capital in trade: %s, free capital: %s, total capital: %s,'
                ' stoploss: %s, available capital: %s.',
                self._cached_pairs[pair].winrate,
                self._cached_pairs[pair].expectancy,
                position_size, pair,
                capital_in_trade, free_capital, total_capital,
                stoploss, available_capital
            )
        return round(position_size, 15)

    def stoploss(self, pair: str) -> float:
        if pair in self._cached_pairs:
            return self._cached_pairs[pair].stoploss
        else:
            logger.warning('tried to access stoploss of a non-existing pair, '
                           'strategy stoploss is returned instead.')
            return self.strategy.stoploss

    def adjust(self, pairs: List[str]) -> list:
        """
        Filters out and sorts "pairs" according to Edge calculated pairs
        """
        final = []
        for pair, info in self._cached_pairs.items():
            if info.expectancy > float(self.edge_config.get('minimum_expectancy', 0.2)) and \
                info.winrate > float(self.edge_config.get('minimum_winrate', 0.60)) and \
                    pair in pairs:
                final.append(pair)

        if self._final_pairs != final:
            self._final_pairs = final
            if self._final_pairs:
                logger.info(
                    'Minimum expectancy and minimum winrate are met only for %s,'
                    ' so other pairs are filtered out.',
                    self._final_pairs
                    )
            else:
                logger.info(
                    'Edge removed all pairs as no pair with minimum expectancy '
                    'and minimum winrate was found !'
                    )

        return self._final_pairs

    def accepted_pairs(self) -> list:
        """
        return a list of accepted pairs along with their winrate, expectancy and stoploss
        """
        final = []
        for pair, info in self._cached_pairs.items():
            if info.expectancy > float(self.edge_config.get('minimum_expectancy', 0.2)) and \
                 info.winrate > float(self.edge_config.get('minimum_winrate', 0.60)):
                final.append({
                    'Pair': pair,
                    'Winrate': info.winrate,
                    'Expectancy': info.expectancy,
                    'Stoploss': info.stoploss,
                })
        return final

    def _fill_calculable_fields(self, result: DataFrame) -> DataFrame:
        """
        The result frame contains a number of columns that are calculable
        from other columns. These are left blank till all rows are added,
        to be populated in single vector calls.

        Columns to be populated are:
        - Profit
        - trade duration
        - profit abs
        :param result Dataframe
        :return: result Dataframe
        """
        # We set stake amount to an arbitrary amount, as it doesn't change the calculation.
        # All returned values are relative, they are defined as ratios.
        stake = 0.015

        result['trade_duration'] = result['close_date'] - result['open_date']

        result['trade_duration'] = result['trade_duration'].map(
            lambda x: int(x.total_seconds() / 60))

        # Spends, Takes, Profit, Absolute Profit

        # Buy Price
        result['buy_vol'] = stake / result['open_rate']  # How many target are we buying
        result['buy_fee'] = stake * self.fee
        result['buy_spend'] = stake + result['buy_fee']  # How much we're spending

        # Sell price
        result['sell_sum'] = result['buy_vol'] * result['close_rate']
        result['sell_fee'] = result['sell_sum'] * self.fee
        result['sell_take'] = result['sell_sum'] - result['sell_fee']

        # profit_ratio
        result['profit_ratio'] = (result['sell_take'] - result['buy_spend']) / result['buy_spend']

        # Absolute profit
        result['profit_abs'] = result['sell_take'] - result['buy_spend']

        return result

    def _process_expectancy(self, results: DataFrame) -> Dict[str, Any]:
        """
        This calculates WinRate, Required Risk Reward, Risk Reward and Expectancy of all pairs
        The calulation will be done per pair and per strategy.
        """
        # Removing pairs having less than min_trades_number
        min_trades_number = self.edge_config.get('min_trade_number', 10)
        results = results.groupby(['pair', 'stoploss']).filter(lambda x: len(x) > min_trades_number)
        ###################################

        # Removing outliers (Only Pumps) from the dataset
        # The method to detect outliers is to calculate standard deviation
        # Then every value more than (standard deviation + 2*average) is out (pump)
        #
        # Removing Pumps
        if self.edge_config.get('remove_pumps', False):
            results = results[results['profit_abs'] < 2 * results['profit_abs'].std()
                              + results['profit_abs'].mean()]
        ##########################################################################

        # Removing trades having a duration more than X minutes (set in config)
        max_trade_duration = self.edge_config.get('max_trade_duration_minute', 1440)
        results = results[results.trade_duration < max_trade_duration]
        #######################################################################

        if results.empty:
            return {}

        groupby_aggregator = {
            'profit_abs': [
                ('nb_trades', 'count'),  # number of all trades
                ('profit_sum', lambda x: x[x > 0].sum()),  # cumulative profit of all winning trades
                ('loss_sum', lambda x: abs(x[x < 0].sum())),  # cumulative loss of all losing trades
                ('nb_win_trades', lambda x: x[x > 0].count())  # number of winning trades
            ],
            'trade_duration': [('avg_trade_duration', 'mean')]
        }

        # Group by (pair and stoploss) by applying above aggregator
        df = results.groupby(['pair', 'stoploss'])[['profit_abs', 'trade_duration']].agg(
            groupby_aggregator).reset_index(col_level=1)

        # Dropping level 0 as we don't need it
        df.columns = df.columns.droplevel(0)

        # Calculating number of losing trades, average win and average loss
        df['nb_loss_trades'] = df['nb_trades'] - df['nb_win_trades']
        df['average_win'] = np.where(df['nb_win_trades'] == 0, 0.0,
                                     df['profit_sum'] / df['nb_win_trades'])
        df['average_loss'] = np.where(df['nb_loss_trades'] == 0, 0.0,
                                      df['loss_sum'] / df['nb_loss_trades'])

        # Win rate = number of profitable trades / number of trades
        df['winrate'] = df['nb_win_trades'] / df['nb_trades']

        # risk_reward_ratio = average win / average loss
        df['risk_reward_ratio'] = df['average_win'] / df['average_loss']

        # required_risk_reward = (1 / winrate) - 1
        df['required_risk_reward'] = (1 / df['winrate']) - 1

        # expectancy = (risk_reward_ratio * winrate) - (lossrate)
        df['expectancy'] = (df['risk_reward_ratio'] * df['winrate']) - (1 - df['winrate'])

        # sort by expectancy and stoploss
        df = df.sort_values(by=['expectancy', 'stoploss'], ascending=False).groupby(
            'pair').first().sort_values(by=['expectancy'], ascending=False).reset_index()

        final = {}
        for x in df.itertuples():
            final[x.pair] = PairInfo(
                x.stoploss,
                x.winrate,
                x.risk_reward_ratio,
                x.required_risk_reward,
                x.expectancy,
                x.nb_trades,
                x.avg_trade_duration
            )

        # Returning a list of pairs in order of "expectancy"
        return final

    def _find_trades_for_stoploss_range(self, df, pair, stoploss_range):
        buy_column = df['buy'].values
        sell_column = df['sell'].values
        date_column = df['date'].values
        ohlc_columns = df[['open', 'high', 'low', 'close']].values

        result: list = []
        for stoploss in stoploss_range:
            result += self._detect_next_stop_or_sell_point(
                buy_column, sell_column, date_column, ohlc_columns, round(stoploss, 6), pair
            )

        return result

    def _detect_next_stop_or_sell_point(self, buy_column, sell_column, date_column,
                                        ohlc_columns, stoploss, pair):
        """
        Iterate through ohlc_columns in order to find the next trade
        Next trade opens from the first buy signal noticed to
        The sell or stoploss signal after it.
        It then cuts OHLC, buy_column, sell_column and date_column.
        Cut from (the exit trade index) + 1.

        Author: https://github.com/mishaker
        """

        result: list = []
        start_point = 0

        while True:
            open_trade_index = utf1st.find_1st(buy_column, 1, utf1st.cmp_equal)

            # Return empty if we don't find trade entry (i.e. buy==1) or
            # we find a buy but at the end of array
            if open_trade_index == -1 or open_trade_index == len(buy_column) - 1:
                break
            else:
                # When a buy signal is seen,
                # trade opens in reality on the next candle
                open_trade_index += 1

            open_price = ohlc_columns[open_trade_index, 0]
            stop_price = (open_price * (stoploss + 1))

            # Searching for the index where stoploss is hit
            stop_index = utf1st.find_1st(
                ohlc_columns[open_trade_index:, 2], stop_price, utf1st.cmp_smaller)

            # If we don't find it then we assume stop_index will be far in future (infinite number)
            if stop_index == -1:
                stop_index = float('inf')

            # Searching for the index where sell is hit
            sell_index = utf1st.find_1st(sell_column[open_trade_index:], 1, utf1st.cmp_equal)

            # If we don't find it then we assume sell_index will be far in future (infinite number)
            if sell_index == -1:
                sell_index = float('inf')

            # Check if we don't find any stop or sell point (in that case trade remains open)
            # It is not interesting for Edge to consider it so we simply ignore the trade
            # And stop iterating there is no more entry
            if stop_index == sell_index == float('inf'):
                break

            if stop_index <= sell_index:
                exit_index = open_trade_index + stop_index
                exit_type = SellType.STOP_LOSS
                exit_price = stop_price
            elif stop_index > sell_index:
                # If exit is SELL then we exit at the next candle
                exit_index = open_trade_index + sell_index + 1

                # Check if we have the next candle
                if len(ohlc_columns) - 1 < exit_index:
                    break

                exit_type = SellType.SELL_SIGNAL
                exit_price = ohlc_columns[exit_index, 0]

            trade = {'pair': pair,
                     'stoploss': stoploss,
                     'profit_ratio': '',
                     'profit_abs': '',
                     'open_date': date_column[open_trade_index],
                     'close_date': date_column[exit_index],
                     'trade_duration': '',
                     'open_rate': round(open_price, 15),
                     'close_rate': round(exit_price, 15),
                     'exit_type': exit_type
                     }

            result.append(trade)

            # Giving a view of exit_index till the end of array
            buy_column = buy_column[exit_index:]
            sell_column = sell_column[exit_index:]
            date_column = date_column[exit_index:]
            ohlc_columns = ohlc_columns[exit_index:]
            start_point += exit_index

        return result
