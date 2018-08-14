# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logic
"""
import logging
import operator
from abc import ABC, abstractmethod
from argparse import Namespace
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from enum import Enum

import arrow
from pandas import DataFrame
from tabulate import tabulate

from freqtrade import DependencyException, constants
from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration
from freqtrade.exchange import Exchange
from freqtrade.misc import file_dump_json
import freqtrade.optimize as optimize
from freqtrade.strategy.interface import SellType
from freqtrade.strategy.resolver import IStrategy, StrategyResolver

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


class OptimizeType(Enum):
    BACKTEST = "backtest"
    BACKSLAP = "backslap"
    HYPEROPT = "hyperopt"


class IOptimize(ABC):
    """
    Backtesting Abstract class, this class contains all the logic to run a backtest

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
        if self.config.get('strategy_list', None):
            # Force one interval
            self.ticker_interval = str(self.config.get('ticker_interval'))
            for strat in list(self.config['strategy_list']):
                stratconf = deepcopy(self.config)
                stratconf['strategy'] = strat
                self.strategylist.append(StrategyResolver(stratconf).strategy)

        else:
            # only one strategy
            strat = StrategyResolver(self.config).strategy

            self.strategylist.append(StrategyResolver(self.config).strategy)
        # Load one strategy
        self._set_strategy(self.strategylist[0])

        self.exchange = Exchange(self.config)
        self.fee = self.exchange.get_fee()

    def _set_strategy(self, strategy):
        """
        Load strategy into backtesting
        """
        self.strategy = strategy
        self.ticker_interval = self.config.get('ticker_interval')
        self.tickerdata_to_dataframe = strategy.tickerdata_to_dataframe
        self.advise_buy = strategy.advise_buy
        self.advise_sell = strategy.advise_sell

    def _get_timeframe(self, data: Dict[str, DataFrame]) -> Tuple[arrow.Arrow, arrow.Arrow]:
        """
        Get the maximum timeframe for the given backtest data
        :param data: dictionary with preprocessed backtesting data
        :return: tuple containing min_date, max_date
        """
        timeframe = [
            (arrow.get(frame['date'].min()), arrow.get(frame['date'].max()))
            for frame in data.values()
        ]
        return min(timeframe, key=operator.itemgetter(0))[0], \
            max(timeframe, key=operator.itemgetter(1))[1]

    def _generate_text_table(self, data: Dict[str, Dict], results: DataFrame) -> str:
        """
        Generates and returns a text table for the given backtest data and the results dataframe
        :return: pretty printed table with tabulate as str
        """
        stake_currency = str(self.config.get('stake_currency'))

        floatfmt = ('s', 'd', '.2f', '.2f', '.8f', 'd', '.1f', '.1f')
        tabular_data = []
        headers = ['pair', 'buy count', 'avg profit %', 'cum profit %',
                   'total profit ' + stake_currency, 'avg duration', 'profit', 'loss']
        for pair in data:
            result = results[results.pair == pair]
            tabular_data.append([
                pair,
                len(result.index),
                result.profit_percent.mean() * 100.0,
                result.profit_percent.sum() * 100.0,
                result.profit_abs.sum(),
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
            str(timedelta(
                minutes=round(results.trade_duration.mean()))) if not results.empty else '0:00',
            len(results[results.profit_abs > 0]),
            len(results[results.profit_abs < 0])
        ])
        return tabulate(tabular_data, headers=headers, floatfmt=floatfmt, tablefmt="pipe")

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

        floatfmt = ('s', 'd', '.2f', '.2f', '.8f', 'd', '.1f', '.1f')
        tabular_data = []
        headers = ['Strategy', 'buy count', 'avg profit %', 'cum profit %',
                   'total profit ' + stake_currency, 'avg duration', 'profit', 'loss']
        for strategy, results in all_results.items():
            tabular_data.append([
                strategy,
                len(results.index),
                results.profit_percent.mean() * 100.0,
                results.profit_percent.sum() * 100.0,
                results.profit_abs.sum(),
                str(timedelta(
                    minutes=round(results.trade_duration.mean()))) if not results.empty else '0:00',
                len(results[results.profit_abs > 0]),
                len(results[results.profit_abs < 0])
            ])
        return tabulate(tabular_data, headers=headers, floatfmt=floatfmt, tablefmt="pipe")

    def _store_backtest_result(self, recordfilename: str, results: DataFrame,
                               strategyname: Optional[str] = None) -> None:

        records = [(t.pair, t.profit_percent, t.open_time.timestamp(),
                    t.close_time.timestamp(), t.open_index - 1, t.trade_duration,
                    t.open_rate, t.close_rate, t.open_at_end, t.sell_reason.value)
                   for index, t in results.iterrows()]

        if records:
            if strategyname:
                # Inject strategyname to filename
                recname = Path(recordfilename)
                recordfilename = str(Path.joinpath(
                    recname.parent, f'{recname.stem}-{strategyname}').with_suffix(recname.suffix))
            logger.info('Dumping backtest results to %s', recordfilename)
            file_dump_json(recordfilename, records)

    def start(self) -> None:
        """
        Run a backtesting end-to-end
        :return: None
        """
        data = {}
        pairs = self.config['exchange']['pair_whitelist']
        logger.info('Using stake_currency: %s ...', self.config['stake_currency'])
        logger.info('Using stake_amount: %s ...', self.config['stake_amount'])

        if self.config.get('live'):
            logger.info('Downloading data for all pairs in whitelist ...')
            for pair in pairs:
                data[pair] = self.exchange.get_candle_history(pair, self.ticker_interval)
        else:
            logger.info('Using local backtesting data (using whitelist in given config) ...')

            timerange = Arguments.parse_timerange(None if self.config.get(
                'timerange') is None else str(self.config.get('timerange')))
            data = optimize.load_data(
                self.config['datadir'],
                pairs=pairs,
                ticker_interval=self.ticker_interval,
                refresh_pairs=self.config.get('refresh_pairs', False),
                exchange=self.exchange,
                timerange=timerange
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

        for strat in self.strategylist:
            logger.info("Running backtesting for Strategy %s", strat.get_strategy_name())
            self._set_strategy(strat)

            # need to reprocess data every time to populate signals
            preprocessed = self.tickerdata_to_dataframe(data)

            # Print timeframe
            min_date, max_date = self._get_timeframe(preprocessed)
            logger.info(
                'Measuring data from %s up to %s (%s days)..',
                min_date.isoformat(),
                max_date.isoformat(),
                (max_date - min_date).days
            )

            # Execute backtest and print results
            all_results[self.strategy.get_strategy_name()] = self.run(
                {
                    'stake_amount': self.config.get('stake_amount'),
                    'processed': preprocessed,
                    'max_open_trades': max_open_trades,
                    'position_stacking': self.config.get('position_stacking', False),
                }
            )

        for strategy, results in all_results.items():

            if self.config.get('export', False):
                self._store_backtest_result(self.config['exportfilename'], results,
                                            strategy if len(self.strategylist) > 1 else None)

            print(f"Result for strategy {strategy}")
            print(f' {self._optimizetype.value.upper()} REPORT '.center(119, '='))
            print(self._generate_text_table(data, results))

            print(' SELL REASON STATS '.center(119, '='))
            print(self._generate_text_table_sell_reason(data, results))

            print(' LEFT OPEN TRADES REPORT '.center(119, '='))
            print(self._generate_text_table(data, results.loc[results.open_at_end]))
            print()
        if len(all_results) > 1:
            # Print Strategy summary table
            print(' Strategy Summary '.center(119, '='))
            print(self._generate_text_table_strategy(all_results))
            print('\nFor more details, please look at the detail tables above')

    @abstractmethod
    def run(self, args: Dict) -> DataFrame:
        """
        Runs backtesting functionality.

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


def setup_configuration(args: Namespace) -> Dict[str, Any]:
    """
    Prepare the configuration for the backtesting
    :param args: Cli args from Arguments()
    :return: Configuration
    """
    configuration = Configuration(args)
    config = configuration.get_config()

    # Ensure we do not use Exchange credentials
    config['exchange']['key'] = ''
    config['exchange']['secret'] = ''

    if config['stake_amount'] == constants.UNLIMITED_STAKE_AMOUNT:
        raise DependencyException('stake amount could not be "%s" for backtesting' %
                                  constants.UNLIMITED_STAKE_AMOUNT)

    return config
