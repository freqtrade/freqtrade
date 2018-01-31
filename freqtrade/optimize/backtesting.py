# pragma pylint: disable=missing-docstring,W0212

import logging
from typing import Dict, Tuple

import arrow
from pandas import DataFrame, Series
from tabulate import tabulate

import freqtrade.misc as misc
import freqtrade.optimize as optimize
from freqtrade import exchange
from freqtrade.analyze import populate_buy_trend, populate_sell_trend
from freqtrade.exchange import Bittrex
from freqtrade.main import should_sell
from freqtrade.persistence import Trade
from freqtrade.strategy.strategy import Strategy

logger = logging.getLogger(__name__)


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


def generate_text_table(
        data: Dict[str, Dict], results: DataFrame, stake_currency, ticker_interval) -> str:
    """
    Generates and returns a text table for the given backtest data and the results dataframe
    :return: pretty printed table with tabulate as str
    """
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


def get_sell_trade_entry(pair, row, buy_subset, ticker, trade_count_lock, args):
    stake_amount = args['stake_amount']
    max_open_trades = args.get('max_open_trades', 0)
    trade = Trade(open_rate=row.close,
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
        if(should_sell(trade, row2.close, row2.date, buy_signal, row2.sell)):
            return row2, (pair,
                          trade.calc_profit_percent(rate=row2.close),
                          trade.calc_profit(rate=row2.close),
                          row2.Index - row.Index
                          ), row2.date
    return None


def backtest(args) -> DataFrame:
    """
    Implements backtesting functionality
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
    trade_count_lock: dict = {}
    exchange._API = Bittrex({'key': '', 'secret': ''})
    for pair, pair_data in processed.items():
        pair_data['buy'], pair_data['sell'] = 0, 0
        ticker = populate_sell_trend(populate_buy_trend(pair_data))
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

            ret = get_sell_trade_entry(pair, row, buy_subset, ticker,
                                       trade_count_lock, args)
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
        logger.info('Dumping backtest results')
        misc.file_dump_json('backtest-result.json', records)
    labels = ['currency', 'profit_percent', 'profit_BTC', 'duration']
    return DataFrame.from_records(trades, columns=labels)


def start(args):
    # Initialize logger
    logging.basicConfig(
        level=args.loglevel,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    exchange._API = Bittrex({'key': '', 'secret': ''})

    logger.info('Using config: %s ...', args.config)
    config = misc.load_config(args.config)
    ticker_interval = config.get('ticker_interval', args.ticker_interval)
    logger.info('Using ticker_interval: %s ...', ticker_interval)

    data = {}
    pairs = config['exchange']['pair_whitelist']
    if args.live:
        logger.info('Downloading data for all pairs in whitelist ...')
        for pair in pairs:
            data[pair] = exchange.get_ticker_history(pair, ticker_interval)
    else:
        logger.info('Using local backtesting data (using whitelist in given config) ...')
        logger.info('Using stake_currency: %s ...', config['stake_currency'])
        logger.info('Using stake_amount: %s ...', config['stake_amount'])

        timerange = misc.parse_timerange(args.timerange)
        data = optimize.load_data(args.datadir, pairs=pairs, ticker_interval=args.ticker_interval,
                                  refresh_pairs=args.refresh_pairs,
                                  timerange=timerange)
    max_open_trades = 0
    if args.realistic_simulation:
        logger.info('Using max_open_trades: %s ...', config['max_open_trades'])
        max_open_trades = config['max_open_trades']

    # init the strategy to use
    config.update({'strategy': args.strategy})
    strategy = Strategy()
    strategy.init(config)

    # Monkey patch config
    from freqtrade import main
    main._CONF = config

    preprocessed = optimize.tickerdata_to_dataframe(data)
    # Print timeframe
    min_date, max_date = get_timeframe(preprocessed)
    logger.info('Measuring data from %s up to %s (%s days)..',
                min_date.isoformat(),
                max_date.isoformat(),
                (max_date-min_date).days)
    # Execute backtest and print results
    sell_profit_only = config.get('experimental', {}).get('sell_profit_only', False)
    use_sell_signal = config.get('experimental', {}).get('use_sell_signal', False)
    results = backtest({'stake_amount': config['stake_amount'],
                        'processed': preprocessed,
                        'max_open_trades': max_open_trades,
                        'realistic': args.realistic_simulation,
                        'sell_profit_only': sell_profit_only,
                        'use_sell_signal': use_sell_signal,
                        'stoploss': strategy.stoploss,
                        'record': args.export
                        })
    logger.info(
        '\n==================================== BACKTESTING REPORT ====================================\n%s',  # noqa
        generate_text_table(data, results, config['stake_currency'], args.ticker_interval)
    )
