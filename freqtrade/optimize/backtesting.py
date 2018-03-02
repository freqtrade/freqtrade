# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logic
"""

from typing import Dict, Tuple, Any
import arrow
from pandas import DataFrame, Series
from tabulate import tabulate

import freqtrade.optimize as optimize
from freqtrade.arguments import Arguments
from freqtrade.exchange import Bittrex
from freqtrade.configuration import Configuration
from freqtrade import exchange
from freqtrade.analyze import Analyze
from freqtrade.logger import Logger
from freqtrade.misc import file_dump_json
from freqtrade.persistence import Trade

from memory_profiler import profile

class Backtesting(object):
    """
    Backtesting class, this class contains all the logic to run a backtest

    To run a backtest:
    backtesting = Backtesting(config)
    backtesting.start()
    """
    def __init__(self, config: Dict[str, Any]) -> None:

        # Init the logger
        self.logging = Logger(name=__name__, level=config['loglevel'])
        self.logger = self.logging.get_logger()

        self.config = config
        self.analyze = None
        self.ticker_interval = None
        self.tickerdata_to_dataframe = None
        self.populate_buy_trend = None
        self.populate_sell_trend = None
        self._init()

    def _init(self) -> None:
        """
        Init objects required for backtesting
        :return: None
        """
        self.analyze = Analyze(self.config)
        self.ticker_interval = self.analyze.strategy.ticker_interval
        self.tickerdata_to_dataframe = self.analyze.tickerdata_to_dataframe
        self.populate_buy_trend = self.analyze.populate_buy_trend
        self.populate_sell_trend = self.analyze.populate_sell_trend
        exchange._API = Bittrex({'key': '', 'secret': ''})

    @staticmethod
    def get_timeframe(data: Dict[str, DataFrame]) -> Tuple[arrow.Arrow, arrow.Arrow]:
        """
        Get the maximum timeframe for the given backtest data
        :param data: dictionary with preprocessed backtesting data
        :return: tuple containing min_date, max_date
        """
        all_dates = Series([])
        for pair_data in data.values():
            all_dates = all_dates.append(pair_data['date'])
        all_dates.sort_values(inplace=True)
        return arrow.get(all_dates.iloc[0]), arrow.get(all_dates.iloc[-1])

    def _generate_text_table(self, data: Dict[str, Dict], results: DataFrame) -> str:
        """
        Generates and returns a text table for the given backtest data and the results dataframe
        :return: pretty printed table with tabulate as str
        """
        stake_currency = self.config.get('stake_currency')
        ticker_interval = self.ticker_interval

        floatfmt = ('s', 'd', '.2f', '.8f', '.1f')
        tabular_data = []
        headers = ['pair', 'buy count', 'avg profit %',
                   'total profit ' + stake_currency, 'avg duration', 'profit', 'loss']
        for pair in data:
            result = results[results.currency == pair]
            tabular_data.append([
                pair,
                len(result.index),
                result.profit_percent.mean() * 100.0,
                result.profit_BTC.sum(),
                result.duration.mean() * ticker_interval,
                len(result[result.profit_BTC > 0]),
                len(result[result.profit_BTC < 0])
            ])

        # Append Total
        tabular_data.append([
            'TOTAL',
            len(results.index),
            results.profit_percent.mean() * 100.0,
            results.profit_BTC.sum(),
            results.duration.mean() * ticker_interval,
            len(results[results.profit_BTC > 0]),
            len(results[results.profit_BTC < 0])
        ])
        return tabulate(tabular_data, headers=headers, floatfmt=floatfmt)

    def _get_sell_trade_entry(self, pair, row, buy_subset, ticker, trade_count_lock, args):
        stake_amount = args['stake_amount']
        max_open_trades = args.get('max_open_trades', 0)
        trade = Trade(
            open_rate=row.close,
            open_date=row.date,
            stake_amount=stake_amount,
            amount=stake_amount / row.open,
            fee=exchange.get_fee()
        )

        # calculate win/lose forwards from buy point
        sell_subset = ticker[ticker.date > row.date][['close', 'date', 'sell']]
        for row2 in sell_subset.itertuples(index=True):
            if max_open_trades > 0:
                # Increase trade_count_lock for every iteration
                trade_count_lock[row2.date] = trade_count_lock.get(row2.date, 0) + 1

            # Buy is on is in the buy_subset there is a row that matches the date
            # of the sell event
            buy_signal = not buy_subset[buy_subset.date == row2.date].empty
            if(
                    self.analyze.should_sell(
                        trade=trade,
                        rate=row2.close,
                        date=row2.date,
                        buy=buy_signal,
                        sell=row2.sell
                    )
            ):
                return \
                    row2, \
                    (
                        pair,
                        trade.calc_profit_percent(rate=row2.close),
                        trade.calc_profit(rate=row2.close),
                        row2.Index - row.Index
                    ),\
                    row2.date
        return None

    def backtest(self, args) -> DataFrame:
        """
        Implements backtesting functionality

        NOTE: This method is used by Hyperopt at each iteration. Please keep it optimized.
        Of course try to not have ugly code. By some accessor are sometime slower than functions.
        Avoid, logging on this method

        :param args: a dict containing:
            stake_amount: btc amount to use for each trade
            processed: a processed dictionary with format {pair, data}
            max_open_trades: maximum number of concurrent trades (default: 0, disabled)
            realistic: do we try to simulate realistic trades? (default: True)
            sell_profit_only: sell if profit only
            use_sell_signal: act on sell-signal
            stoploss: use stoploss
        :return: DataFrame
        """
        processed = args['processed']
        max_open_trades = args.get('max_open_trades', 0)
        realistic = args.get('realistic', True)
        record = args.get('record', None)
        records = []
        trades = []
        trade_count_lock = {}
        for pair, pair_data in processed.items():
            pair_data['buy'], pair_data['sell'] = 0, 0
            ticker = self.populate_sell_trend(
                self.populate_buy_trend(pair_data)
            )
            # for each buy point
            lock_pair_until = None
            headers = ['buy', 'open', 'close', 'date', 'sell']
            buy_subset = ticker[(ticker.buy == 1) & (ticker.sell == 0)][headers]
            for row in buy_subset.itertuples(index=True):
                if realistic:
                    if lock_pair_until is not None and row.date <= lock_pair_until:
                        continue
                if max_open_trades > 0:
                    # Check if max_open_trades has already been reached for the given date
                    if not trade_count_lock.get(row.date, 0) < max_open_trades:
                        continue

                if max_open_trades > 0:
                    # Increase lock
                    trade_count_lock[row.date] = trade_count_lock.get(row.date, 0) + 1

                ret = self._get_sell_trade_entry(
                    pair=pair,
                    row=row,
                    buy_subset=buy_subset,
                    ticker=ticker,
                    trade_count_lock=trade_count_lock,
                    args=args
                )

                if ret:
                    row2, trade_entry, next_date = ret
                    lock_pair_until = next_date
                    trades.append(trade_entry)
                    if record:
                        # Note, need to be json.dump friendly
                        # record a tuple of pair, current_profit_percent,
                        # entry-date, duration
                        records.append((pair, trade_entry[1],
                                        row.date.strftime('%s'),
                                        row2.date.strftime('%s'),
                                        row.Index, trade_entry[3]))
        # For now export inside backtest(), maybe change so that backtest()
        # returns a tuple like: (dataframe, records, logs, etc)
        if record and record.find('trades') >= 0:
            self.logger.info('Dumping backtest results')
            file_dump_json('backtest-result.json', records)
        labels = ['currency', 'profit_percent', 'profit_BTC', 'duration']
        return DataFrame.from_records(trades, columns=labels)

    @profile(precision=10)
    def start(self) -> None:
        """
        Run a backtesting end-to-end
        :return: None
        """
        data = {}
        pairs = self.config['exchange']['pair_whitelist']

        if self.config.get('live'):
            self.logger.info('Downloading data for all pairs in whitelist ...')
            for pair in pairs:
                data[pair] = exchange.get_ticker_history(pair, self.ticker_interval)
        else:
            self.logger.info('Using local backtesting data (using whitelist in given config) ...')
            self.logger.info('Using stake_currency: %s ...', self.config['stake_currency'])
            self.logger.info('Using stake_amount: %s ...', self.config['stake_amount'])

            timerange = Arguments.parse_timerange(self.config.get('timerange'))
            data = optimize.load_data(
                self.config['datadir'],
                pairs=pairs,
                ticker_interval=self.ticker_interval,
                refresh_pairs=self.config.get('refresh_pairs', False),
                timerange=timerange
            )

        max_open_trades = self.config.get('max_open_trades', 0)
        preprocessed = self.tickerdata_to_dataframe(data)

        # Print timeframe
        min_date, max_date = self.get_timeframe(preprocessed)

        import pprint
        pprint.pprint(min_date)
        pprint.pprint(max_date)
        self.logger.info(
            'Measuring data from %s up to %s (%s days)..',
            min_date.isoformat(),
            max_date.isoformat(),
            (max_date - min_date).days
        )

        # Execute backtest and print results
        sell_profit_only = self.config.get('experimental', {}).get('sell_profit_only', False)
        use_sell_signal = self.config.get('experimental', {}).get('use_sell_signal', False)
        results = self.backtest(
            {
                'stake_amount': self.config.get('stake_amount'),
                'processed': preprocessed,
                'max_open_trades': max_open_trades,
                'realistic': self.config.get('realistic_simulation', False),
                'sell_profit_only': sell_profit_only,
                'use_sell_signal': use_sell_signal,
                'stoploss': self.analyze.strategy.stoploss,
                'record': self.config.get('export')
            }
        )

        self.logging.set_format('%(message)s')
        self.logger.info(
            '\n==================================== '
            'BACKTESTING REPORT'
            ' ====================================\n'
            '%s',
            self._generate_text_table(
                data,
                results
            )
        )


def setup_configuration(args) -> Dict[str, Any]:
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

    return config


def start(args) -> None:
    """
    Start Backtesting script
    :param args: Cli args from Arguments()
    :return: None
    """

    # Initialize logger
    logger = Logger(name=__name__).get_logger()
    logger.info('Starting freqtrade in Backtesting mode')

    # Initialize configuration
    config = setup_configuration(args)

    # Initialize backtesting object
    backtesting = Backtesting(config)
    backtesting.start()
