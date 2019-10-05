# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logic
"""
import logging
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

from pandas import DataFrame

from freqtrade import OperationalException
from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.dataprovider import DataProvider
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.misc import file_dump_json
from freqtrade.persistence import Trade
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.state import RunMode
from freqtrade.strategy.interface import IStrategy, SellType
from tabulate import tabulate

logger = logging.getLogger(__name__)


class BacktestResult(NamedTuple):
    """
    NamedTuple Defining BacktestResults inputs.
    """
    pair: str
    profit_percent: float
    profit_abs: float
    open_time: datetime
    close_time: datetime
    open_index: int
    close_index: int
    trade_duration: float
    open_at_end: bool
    open_rate: float
    close_rate: float
    sell_reason: SellType


class Backtesting:
    """
    Backtesting class, this class contains all the logic to run a backtest

    To run a backtest:
    backtesting = Backtesting(config)
    backtesting.start()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        # Reset keys for backtesting
        self.config['exchange']['key'] = ''
        self.config['exchange']['secret'] = ''
        self.config['exchange']['password'] = ''
        self.config['exchange']['uid'] = ''
        self.config['dry_run'] = True
        self.strategylist: List[IStrategy] = []

        self.exchange = ExchangeResolver(self.config['exchange']['name'], self.config).exchange
        self.fee = self.exchange.get_fee()

        if self.config.get('runmode') != RunMode.HYPEROPT:
            self.dataprovider = DataProvider(self.config, self.exchange)
            IStrategy.dp = self.dataprovider

        if self.config.get('strategy_list', None):
            for strat in list(self.config['strategy_list']):
                stratconf = deepcopy(self.config)
                stratconf['strategy'] = strat
                self.strategylist.append(StrategyResolver(stratconf).strategy)

        else:
            # No strategy list specified, only one strategy
            self.strategylist.append(StrategyResolver(self.config).strategy)

        if "ticker_interval" not in self.config:
            raise OperationalException("Ticker-interval needs to be set in either configuration "
                                       "or as cli argument `--ticker-interval 5m`")
        self.ticker_interval = str(self.config.get('ticker_interval'))
        self.ticker_interval_mins = timeframe_to_minutes(self.ticker_interval)

        # Load one (first) strategy
        self._set_strategy(self.strategylist[0])

    def _set_strategy(self, strategy):
        """
        Load strategy into backtesting
        """
        self.strategy = strategy
        # Set stoploss_on_exchange to false for backtesting,
        # since a "perfect" stoploss-sell is assumed anyway
        # And the regular "stoploss" function would not apply to that case
        self.strategy.order_types['stoploss_on_exchange'] = False

    def _generate_text_table(self, data: Dict[str, Dict], results: DataFrame,
                             skip_nan: bool = False) -> str:
        """
        Generates and returns a text table for the given backtest data and the results dataframe
        :return: pretty printed table with tabulate as str
        """
        stake_currency = str(self.config.get('stake_currency'))
        max_open_trades = self.config.get('max_open_trades')

        floatfmt = ('s', 'd', '.2f', '.2f', '.8f', '.2f', 'd', '.1f', '.1f')
        tabular_data = []
        headers = ['pair', 'buy count', 'avg profit %', 'cum profit %',
                   'tot profit ' + stake_currency, 'tot profit %', 'avg duration',
                   'profit', 'loss']
        for pair in data:
            result = results[results.pair == pair]
            if skip_nan and result.profit_abs.isnull().all():
                continue

            tabular_data.append([
                pair,
                len(result.index),
                result.profit_percent.mean() * 100.0,
                result.profit_percent.sum() * 100.0,
                result.profit_abs.sum(),
                result.profit_percent.sum() * 100.0 / max_open_trades,
                str(timedelta(
                    minutes=round(result.trade_duration.mean()))) if not result.empty else '0:00',
                len(result[result.profit_abs > 0]),
                len(result[result.profit_abs < 0])
            ])

        # Append Total
        tabular_data.append([
            'TOTAL',
            len(results.index),
            results.profit_percent.mean() * 100.0,
            results.profit_percent.sum() * 100.0,
            results.profit_abs.sum(),
            results.profit_percent.sum() * 100.0 / max_open_trades,
            str(timedelta(
                minutes=round(results.trade_duration.mean()))) if not results.empty else '0:00',
            len(results[results.profit_abs > 0]),
            len(results[results.profit_abs < 0])
        ])
        # Ignore type as floatfmt does allow tuples but mypy does not know that
        return tabulate(tabular_data, headers=headers,
                        floatfmt=floatfmt, tablefmt="pipe")  # type: ignore

    def _generate_text_table_sell_reason(self, data: Dict[str, Dict], results: DataFrame) -> str:
        """
        Generate small table outlining Backtest results
        """
        tabular_data = []
        headers = ['Sell Reason', 'Count']
        for reason, count in results['sell_reason'].value_counts().iteritems():
            tabular_data.append([reason.value,  count])
        return tabulate(tabular_data, headers=headers, tablefmt="pipe")

    def _generate_text_table_strategy(self, all_results: dict) -> str:
        """
        Generate summary table per strategy
        """
        stake_currency = str(self.config.get('stake_currency'))
        max_open_trades = self.config.get('max_open_trades')

        floatfmt = ('s', 'd', '.2f', '.2f', '.8f', '.2f', 'd', '.1f', '.1f')
        tabular_data = []
        headers = ['Strategy', 'buy count', 'avg profit %', 'cum profit %',
                   'tot profit ' + stake_currency, 'tot profit %', 'avg duration',
                   'profit', 'loss']
        for strategy, results in all_results.items():
            tabular_data.append([
                strategy,
                len(results.index),
                results.profit_percent.mean() * 100.0,
                results.profit_percent.sum() * 100.0,
                results.profit_abs.sum(),
                results.profit_percent.sum() * 100.0 / max_open_trades,
                str(timedelta(
                    minutes=round(results.trade_duration.mean()))) if not results.empty else '0:00',
                len(results[results.profit_abs > 0]),
                len(results[results.profit_abs < 0])
            ])
        # Ignore type as floatfmt does allow tuples but mypy does not know that
        return tabulate(tabular_data, headers=headers,
                        floatfmt=floatfmt, tablefmt="pipe")  # type: ignore

    def _store_backtest_result(self, recordfilename: Path, results: DataFrame,
                               strategyname: Optional[str] = None) -> None:

        records = [(t.pair, t.profit_percent, t.open_time.timestamp(),
                    t.close_time.timestamp(), t.open_index - 1, t.trade_duration,
                    t.open_rate, t.close_rate, t.open_at_end, t.sell_reason.value)
                   for index, t in results.iterrows()]

        if records:
            if strategyname:
                # Inject strategyname to filename
                recordfilename = Path.joinpath(
                    recordfilename.parent,
                    f'{recordfilename.stem}-{strategyname}').with_suffix(recordfilename.suffix)
            logger.info(f'Dumping backtest results to {recordfilename}')
            file_dump_json(recordfilename, records)

    def _get_ticker_list(self, processed) -> Dict[str, DataFrame]:
        """
        Helper function to convert a processed tickerlist into a list for performance reasons.

        Used by backtest() - so keep this optimized for performance.
        """
        headers = ['date', 'buy', 'open', 'close', 'sell', 'low', 'high']
        ticker: Dict = {}
        # Create ticker dict
        for pair, pair_data in processed.items():
            pair_data['buy'], pair_data['sell'] = 0, 0  # cleanup from previous run

            ticker_data = self.strategy.advise_sell(
                self.strategy.advise_buy(pair_data, {'pair': pair}), {'pair': pair})[headers].copy()

            # to avoid using data from future, we buy/sell with signal from previous candle
            ticker_data.loc[:, 'buy'] = ticker_data['buy'].shift(1)
            ticker_data.loc[:, 'sell'] = ticker_data['sell'].shift(1)

            ticker_data.drop(ticker_data.head(1).index, inplace=True)

            # Convert from Pandas to list for performance reasons
            # (Looping Pandas is slow.)
            ticker[pair] = [x for x in ticker_data.itertuples()]
        return ticker

    def _get_sell_trade_entry(
            self, pair: str, buy_row: DataFrame,
            partial_ticker: List, trade_count_lock: Dict,
            stake_amount: float, max_open_trades: int) -> Optional[BacktestResult]:

        trade = Trade(
            pair=pair,
            open_rate=buy_row.open,
            open_date=buy_row.date,
            stake_amount=stake_amount,
            amount=stake_amount / buy_row.open,
            fee_open=self.fee,
            fee_close=self.fee,
            is_open=True,
        )
        logger.debug(f"{pair} - Backtesting emulates creation of new trade: {trade}.")
        # calculate win/lose forwards from buy point
        for sell_row in partial_ticker:
            if max_open_trades > 0:
                # Increase trade_count_lock for every iteration
                trade_count_lock[sell_row.date] = trade_count_lock.get(sell_row.date, 0) + 1

            sell = self.strategy.should_sell(trade, sell_row.open, sell_row.date, sell_row.buy,
                                             sell_row.sell, low=sell_row.low, high=sell_row.high)
            if sell.sell_flag:
                trade_dur = int((sell_row.date - buy_row.date).total_seconds() // 60)
                # Special handling if high or low hit STOP_LOSS or ROI
                if sell.sell_type in (SellType.STOP_LOSS, SellType.TRAILING_STOP_LOSS):
                    # Set close_rate to stoploss
                    closerate = trade.stop_loss
                elif sell.sell_type == (SellType.ROI):
                    roi = self.strategy.min_roi_reached_entry(trade_dur)
                    if roi is not None:
                        # - (Expected abs profit + open_rate + open_fee) / (fee_close -1)
                        closerate = - (trade.open_rate * roi + trade.open_rate *
                                       (1 + trade.fee_open)) / (trade.fee_close - 1)

                        # Use the maximum between closerate and low as we
                        # cannot sell outside of a candle.
                        # Applies when using {"xx": -1} as roi to force sells after xx minutes
                        closerate = max(closerate, sell_row.low)
                    else:
                        # This should not be reached...
                        closerate = sell_row.open
                else:
                    closerate = sell_row.open

                return BacktestResult(pair=pair,
                                      profit_percent=trade.calc_profit_percent(rate=closerate),
                                      profit_abs=trade.calc_profit(rate=closerate),
                                      open_time=buy_row.date,
                                      close_time=sell_row.date,
                                      trade_duration=trade_dur,
                                      open_index=buy_row.Index,
                                      close_index=sell_row.Index,
                                      open_at_end=False,
                                      open_rate=buy_row.open,
                                      close_rate=closerate,
                                      sell_reason=sell.sell_type
                                      )
        if partial_ticker:
            # no sell condition found - trade stil open at end of backtest period
            sell_row = partial_ticker[-1]
            bt_res = BacktestResult(pair=pair,
                                    profit_percent=trade.calc_profit_percent(rate=sell_row.open),
                                    profit_abs=trade.calc_profit(rate=sell_row.open),
                                    open_time=buy_row.date,
                                    close_time=sell_row.date,
                                    trade_duration=int((
                                        sell_row.date - buy_row.date).total_seconds() // 60),
                                    open_index=buy_row.Index,
                                    close_index=sell_row.Index,
                                    open_at_end=True,
                                    open_rate=buy_row.open,
                                    close_rate=sell_row.open,
                                    sell_reason=SellType.FORCE_SELL
                                    )
            logger.debug(f"{pair} - Force selling still open trade, "
                         f"profit percent: {bt_res.profit_percent}, "
                         f"profit abs: {bt_res.profit_abs}")

            return bt_res
        return None

    def backtest(self, args: Dict) -> DataFrame:
        """
        Implements backtesting functionality

        NOTE: This method is used by Hyperopt at each iteration. Please keep it optimized.
        Of course try to not have ugly code. By some accessor are sometime slower than functions.
        Avoid, logging on this method

        :param args: a dict containing:
            stake_amount: btc amount to use for each trade
            processed: a processed dictionary with format {pair, data}
            max_open_trades: maximum number of concurrent trades (default: 0, disabled)
            position_stacking: do we allow position stacking? (default: False)
        :return: DataFrame
        """
        # Arguments are long and noisy, so this is commented out.
        # Uncomment if you need to debug the backtest() method.
#        logger.debug(f"Start backtest, args: {args}")
        processed = args['processed']
        stake_amount = args['stake_amount']
        max_open_trades = args.get('max_open_trades', 0)
        position_stacking = args.get('position_stacking', False)
        start_date = args['start_date']
        end_date = args['end_date']
        trades = []
        trade_count_lock: Dict = {}

        # Dict of ticker-lists for performance (looping lists is a lot faster than dataframes)
        ticker: Dict = self._get_ticker_list(processed)

        lock_pair_until: Dict = {}
        # Indexes per pair, so some pairs are allowed to have a missing start.
        indexes: Dict = {}
        tmp = start_date + timedelta(minutes=self.ticker_interval_mins)

        # Loop timerange and get candle for each pair at that point in time
        while tmp < end_date:

            for i, pair in enumerate(ticker):
                if pair not in indexes:
                    indexes[pair] = 0

                try:
                    row = ticker[pair][indexes[pair]]
                except IndexError:
                    # missing Data for one pair at the end.
                    # Warnings for this are shown during data loading
                    continue

                # Waits until the time-counter reaches the start of the data for this pair.
                if row.date > tmp.datetime:
                    continue

                indexes[pair] += 1

                if row.buy == 0 or row.sell == 1:
                    continue  # skip rows where no buy signal or that would immediately sell off

                if (not position_stacking and pair in lock_pair_until
                        and row.date <= lock_pair_until[pair]):
                    # without positionstacking, we can only have one open trade per pair.
                    continue

                if max_open_trades > 0:
                    # Check if max_open_trades has already been reached for the given date
                    if not trade_count_lock.get(row.date, 0) < max_open_trades:
                        continue
                    trade_count_lock[row.date] = trade_count_lock.get(row.date, 0) + 1

                # since indexes has been incremented before, we need to go one step back to
                # also check the buying candle for sell conditions.
                trade_entry = self._get_sell_trade_entry(pair, row, ticker[pair][indexes[pair]-1:],
                                                         trade_count_lock, stake_amount,
                                                         max_open_trades)

                if trade_entry:
                    logger.debug(f"{pair} - Locking pair till "
                                 f"close_time={trade_entry.close_time}")
                    lock_pair_until[pair] = trade_entry.close_time
                    trades.append(trade_entry)
                else:
                    # Set lock_pair_until to end of testing period if trade could not be closed
                    lock_pair_until[pair] = end_date.datetime

            # Move time one configured time_interval ahead.
            tmp += timedelta(minutes=self.ticker_interval_mins)
        return DataFrame.from_records(trades, columns=BacktestResult._fields)

    def start(self) -> None:
        """
        Run a backtesting end-to-end
        :return: None
        """
        data: Dict[str, Any] = {}
        pairs = self.config['exchange']['pair_whitelist']
        logger.info('Using stake_currency: %s ...', self.config['stake_currency'])
        logger.info('Using stake_amount: %s ...', self.config['stake_amount'])

        timerange = TimeRange.parse_timerange(None if self.config.get(
            'timerange') is None else str(self.config.get('timerange')))
        data = history.load_data(
            datadir=Path(self.config['datadir']),
            pairs=pairs,
            ticker_interval=self.ticker_interval,
            timerange=timerange,
        )

        if not data:
            logger.critical("No data found. Terminating.")
            return
        # Use max_open_trades in backtesting, except --disable-max-market-positions is set
        if self.config.get('use_max_market_positions', True):
            max_open_trades = self.config['max_open_trades']
        else:
            logger.info('Ignoring max_open_trades (--disable-max-market-positions was used) ...')
            max_open_trades = 0
        all_results = {}

        min_date, max_date = history.get_timeframe(data)

        logger.info(
            'Backtesting with data from %s up to %s (%s days)..',
            min_date.isoformat(),
            max_date.isoformat(),
            (max_date - min_date).days
        )

        for strat in self.strategylist:
            logger.info("Running backtesting for Strategy %s", strat.get_strategy_name())
            self._set_strategy(strat)

            # need to reprocess data every time to populate signals
            preprocessed = self.strategy.tickerdata_to_dataframe(data)

            # Execute backtest and print results
            all_results[self.strategy.get_strategy_name()] = self.backtest(
                {
                    'stake_amount': self.config.get('stake_amount'),
                    'processed': preprocessed,
                    'max_open_trades': max_open_trades,
                    'position_stacking': self.config.get('position_stacking', False),
                    'start_date': min_date,
                    'end_date': max_date,
                }
            )

        for strategy, results in all_results.items():

            if self.config.get('export', False):
                self._store_backtest_result(Path(self.config['exportfilename']), results,
                                            strategy if len(self.strategylist) > 1 else None)

            print(f"Result for strategy {strategy}")
            print(' BACKTESTING REPORT '.center(133, '='))
            print(self._generate_text_table(data, results))

            print(' SELL REASON STATS '.center(133, '='))
            print(self._generate_text_table_sell_reason(data, results))

            print(' LEFT OPEN TRADES REPORT '.center(133, '='))
            print(self._generate_text_table(data, results.loc[results.open_at_end], True))
            print()
        if len(all_results) > 1:
            # Print Strategy summary table
            print(' Strategy Summary '.center(133, '='))
            print(self._generate_text_table_strategy(all_results))
            print('\nFor more details, please look at the detail tables above')
