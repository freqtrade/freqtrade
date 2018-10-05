# pragma pylint: disable=W0603
""" Edge positioning package """
import logging
from typing import Any, Dict
import arrow

import numpy as np
import utils_find_1st as utf1st
from pandas import DataFrame

import freqtrade.optimize as optimize
from freqtrade.arguments import Arguments
from freqtrade.arguments import TimeRange
from freqtrade.strategy.interface import SellType
from freqtrade.strategy.resolver import IStrategy, StrategyResolver
from freqtrade.optimize.backtesting import Backtesting


logger = logging.getLogger(__name__)


class Edge():

    config: Dict = {}
    _last_updated: int  # Timestamp of pairs last updated time
    _cached_pairs: list = []  # Keeps an array of
    # [pair, stoploss, winrate, risk reward ratio, required risk reward, expectancy]

    _total_capital: float
    _allowed_risk: float
    _since_number_of_days: int
    _timerange: TimeRange

    def __init__(self, config: Dict[str, Any], exchange=None) -> None:
        self.config = config
        self.exchange = exchange
        self.strategy: IStrategy = StrategyResolver(self.config).strategy
        self.ticker_interval = self.strategy.ticker_interval
        self.tickerdata_to_dataframe = self.strategy.tickerdata_to_dataframe
        self.get_timeframe = Backtesting.get_timeframe
        self.advise_sell = self.strategy.advise_sell
        self.advise_buy = self.strategy.advise_buy

        self.edge_config = self.config.get('edge', {})
        self._cached_pairs: list = []
        self._total_capital = self.edge_config.get('total_capital_in_stake_currency')
        self._allowed_risk = self.edge_config.get('allowed_risk')
        self._since_number_of_days = self.edge_config.get('calculate_since_number_of_days', 14)
        self._last_updated = 0

        self._timerange = Arguments.parse_timerange("%s-" % arrow.now().shift(
                 days=-1 * self._since_number_of_days).format('YYYYMMDD'))

        self.fee = self.exchange.get_fee()

    def calculate(self) -> bool:
        pairs = self.config['exchange']['pair_whitelist']
        heartbeat = self.edge_config.get('process_throttle_secs')

        if (self._last_updated > 0) and (
                self._last_updated + heartbeat > arrow.utcnow().timestamp):
            return False

        data: Dict[str, Any] = {}
        logger.info('Using stake_currency: %s ...', self.config['stake_currency'])
        logger.info('Using local backtesting data (using whitelist in given config) ...')

        data = optimize.load_data(
            self.config['datadir'],
            pairs=pairs,
            ticker_interval=self.ticker_interval,
            refresh_pairs=True,
            exchange=self.exchange,
            timerange=self._timerange
        )

        if not data:
            logger.critical("No data found. Edge is stopped ...")
            return False

        preprocessed = self.tickerdata_to_dataframe(data)

        # Print timeframe
        min_date, max_date = self.get_timeframe(preprocessed)
        logger.info(
            'Measuring data from %s up to %s (%s days) ...',
            min_date.isoformat(),
            max_date.isoformat(),
            (max_date - min_date).days
        )
        headers = ['date', 'buy', 'open', 'close', 'sell', 'high', 'low']

        stoploss_range_min = float(self.edge_config.get('stoploss_range_min', -0.01))
        stoploss_range_max = float(self.edge_config.get('stoploss_range_max', -0.05))
        stoploss_range_step = float(self.edge_config.get('stoploss_range_step', -0.001))
        stoploss_range = np.arange(stoploss_range_min, stoploss_range_max, stoploss_range_step)

        trades: list = []
        for pair, pair_data in preprocessed.items():
            # Sorting dataframe by date and reset index
            pair_data = pair_data.sort_values(by=['date'])
            pair_data = pair_data.reset_index(drop=True)

            ticker_data = self.advise_sell(
                self.advise_buy(pair_data, {'pair': pair}), {'pair': pair})[headers].copy()

            trades += self._find_trades_for_stoploss_range(ticker_data, pair, stoploss_range)

        # If no trade found then exit
        if len(trades) == 0:
            return False

        # Fill missing, calculable columns, profit, duration , abs etc.
        trades_df = self._fill_calculable_fields(DataFrame(trades))
        self._cached_pairs = self._process_expectancy(trades_df)
        self._last_updated = arrow.utcnow().timestamp

        # Not a nice hack but probably simplest solution:
        # When backtest load data it loads the delta between disk and exchange
        # The problem is that exchange consider that recent.
        # it is but it is incomplete (c.f. _async_get_candle_history)
        # So it causes get_signal to exit cause incomplete ticker_hist
        # A patch to that would be update _pairs_last_refresh_time of exchange
        # so it will download again all pairs
        # Another solution is to add new data to klines instead of reassigning it:
        # self.klines[pair].update(data) instead of self.klines[pair] = data in exchange package.
        # But that means indexing timestamp and having a verification so that
        # there is no empty range between two timestaps (recently added and last
        # one)
        self.exchange._pairs_last_refresh_time = {}

        return True

    def stake_amount(self, pair: str) -> float:
        info = [x for x in self._cached_pairs if x[0] == pair][0]
        stoploss = info[1]
        allowed_capital_at_risk = round(self._total_capital * self._allowed_risk, 5)
        position_size = abs(round((allowed_capital_at_risk / stoploss), 5))
        return position_size

    def stoploss(self, pair: str) -> float:
        info = [x for x in self._cached_pairs if x[0] == pair][0]
        return info[1]

    def filter(self, pairs) -> list:
        # Filtering pairs acccording to the expectancy
        filtered_expectancy: list = []
        filtered_expectancy = [
            x[0] for x in self._cached_pairs if x[5] > float(
                self.edge_config.get(
                    'minimum_expectancy', 0.2))]

        # Only return pairs which are included in "pairs" argument list
        final = [x for x in filtered_expectancy if x in pairs]
        if final:
            logger.info(
                'Edge validated only %s',
                final
            )
        else:
            logger.info('Edge removed all pairs as no pair with minimum expectancy was found !')

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

        # stake and fees
        # stake = 0.015
        # 0.05% is 0.0005
        # fee = 0.001

        stake = self.config.get('stake_amount')
        fee = self.fee

        open_fee = fee / 2
        close_fee = fee / 2

        result['trade_duration'] = result['close_time'] - result['open_time']

        result['trade_duration'] = result['trade_duration'].map(
            lambda x: int(x.total_seconds() / 60))

        # Spends, Takes, Profit, Absolute Profit

        # Buy Price
        result['buy_vol'] = stake / result['open_rate']  # How many target are we buying
        result['buy_fee'] = stake * open_fee
        result['buy_spend'] = stake + result['buy_fee']  # How much we're spending

        # Sell price
        result['sell_sum'] = result['buy_vol'] * result['close_rate']
        result['sell_fee'] = result['sell_sum'] * close_fee
        result['sell_take'] = result['sell_sum'] - result['sell_fee']

        # profit_percent
        result['profit_percent'] = (result['sell_take'] - result['buy_spend']) / result['buy_spend']

        # Absolute profit
        result['profit_abs'] = result['sell_take'] - result['buy_spend']

        return result

    def _process_expectancy(self, results: DataFrame) -> list:
        """
        This is a temporary version of edge positioning calculation.
        The function will be eventually moved to a plugin called Edge in order
        to calculate necessary WR, RRR and
        other indictaors related to money management periodically (each X minutes)
        and keep it in a storage.
        The calulation will be done per pair and per strategy.
        """

        # Removing pairs having less than min_trades_number
        min_trades_number = self.edge_config.get('min_trade_number', 15)
        results = results.groupby('pair').filter(lambda x: len(x) > min_trades_number)
        ###################################

        # Removing outliers (Only Pumps) from the dataset
        # The method to detect outliers is to calculate standard deviation
        # Then every value more than (standard deviation + 2*average) is out (pump)
        #
        # Calculating standard deviation of profits
        std = results[["profit_abs"]].std()
        #
        # Calculating average of profits
        avg = results[["profit_abs"]].mean()
        #
        # Removing Pumps
        if self.edge_config.get('remove_pumps', True):
            results = results[results.profit_abs <= float(avg + 2 * std)]
        ##########################################################################

        # Removing trades having a duration more than X minutes (set in config)
        max_trade_duration = self.edge_config.get('max_trade_duration_minute', 1440)
        results = results[results.trade_duration < max_trade_duration]
        #######################################################################

        # Win Rate is the number of profitable trades
        # Divided by number of trades
        def winrate(x):
            x = x[x > 0].count() / x.count()
            return x
        #############################

        # Risk Reward Ratio
        # 1 / ((loss money / losing trades) / (gained money / winning trades))
        def risk_reward_ratio(x):
            x = abs(1 / ((x[x < 0].sum() / x[x < 0].count()) / (x[x > 0].sum() / x[x > 0].count())))
            return x
        ##############################

        # Required Risk Reward
        # (1/(winrate - 1)
        def required_risk_reward(x):
            x = (1 / (x[x > 0].count() / x.count()) - 1)
            return x
        ##############################

        # Expectancy
        # Tells you the interest percentage you should hope
        # E.x. if expectancy is 0.35, on $1 trade you should expect a target of $1.35
        def expectancy(x):
            average_win = float(x[x > 0].sum() / x[x > 0].count())
            average_loss = float(abs(x[x < 0].sum() / x[x < 0].count()))
            winrate = float(x[x > 0].count() / x.count())
            x = ((1 + average_win / average_loss) * winrate) - 1
            return x
        ##############################

        final = results.groupby(['pair', 'stoploss'])['profit_abs'].\
            agg([winrate, risk_reward_ratio, required_risk_reward, expectancy]).\
            reset_index().sort_values(by=['expectancy', 'stoploss'], ascending=False)\
            .groupby('pair').first().sort_values(by=['expectancy'], ascending=False)

        # Returning an array of pairs in order of "expectancy"
        return final.reset_index().values

    def _find_trades_for_stoploss_range(self, ticker_data, pair, stoploss_range):
        buy_column = ticker_data['buy'].values
        sell_column = ticker_data['sell'].values
        date_column = ticker_data['date'].values
        ohlc_columns = ticker_data[['open', 'high', 'low', 'close']].values

        result: list = []
        for stoploss in stoploss_range:
            result += self._detect_next_stop_or_sell_point(
                buy_column, sell_column, date_column, ohlc_columns, round(stoploss, 6), pair
            )

        return result

    def _detect_next_stop_or_sell_point(
            self,
            buy_column,
            sell_column,
            date_column,
            ohlc_columns,
            stoploss,
            pair,
            start_point=0):

        result: list = []
        open_trade_index = utf1st.find_1st(buy_column, 1, utf1st.cmp_equal)

        # return empty if we don't find trade entry (i.e. buy==1) or
        # we find a buy but at the of array
        if open_trade_index == -1 or open_trade_index == len(buy_column) - 1:
            return []

        stop_price_percentage = stoploss + 1
        open_price = ohlc_columns[open_trade_index + 1, 0]
        stop_price = (open_price * stop_price_percentage)

        # Searching for the index where stoploss is hit
        stop_index = utf1st.find_1st(
            ohlc_columns[open_trade_index + 1:, 2], stop_price, utf1st.cmp_smaller)

        # If we don't find it then we assume stop_index will be far in future (infinite number)
        if stop_index == -1:
            stop_index = float('inf')

        # Searching for the index where sell is hit
        sell_index = utf1st.find_1st(sell_column[open_trade_index + 1:], 1, utf1st.cmp_equal)

        # If we don't find it then we assume sell_index will be far in future (infinite number)
        if sell_index == -1:
            sell_index = float('inf')

        # Check if we don't find any stop or sell point (in that case trade remains open)
        # It is not interesting for Edge to consider it so we simply ignore the trade
        # And stop iterating there is no more entry
        if stop_index == sell_index == float('inf'):
            return []

        if stop_index <= sell_index:
            exit_index = open_trade_index + stop_index + 1
            exit_type = SellType.STOP_LOSS
            exit_price = stop_price
        elif stop_index > sell_index:
            exit_index = open_trade_index + sell_index + 1
            exit_type = SellType.SELL_SIGNAL
            exit_price = ohlc_columns[exit_index, 0]

        trade = {'pair': pair,
                 'stoploss': stoploss,
                 'profit_percent': '',
                 'profit_abs': '',
                 'open_time': date_column[open_trade_index],
                 'close_time': date_column[exit_index],
                 'open_index': start_point + open_trade_index + 1,
                 'close_index': start_point + exit_index,
                 'trade_duration': '',
                 'open_rate': round(open_price, 15),
                 'close_rate': round(exit_price, 15),
                 'exit_type': exit_type
                 }

        result.append(trade)

        # Calling again the same function recursively but giving
        # it a view of exit_index till the end of array
        return result + self._detect_next_stop_or_sell_point(
            buy_column[exit_index:],
            sell_column[exit_index:],
            date_column[exit_index:],
            ohlc_columns[exit_index:],
            stoploss,
            pair,
            (start_point + exit_index)
        )
