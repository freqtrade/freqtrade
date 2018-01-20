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
from freqtrade.main import min_roi_reached
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


def get_timeframe(data: Dict[str, DataFrame]) -> Tuple[arrow.Arrow, arrow.Arrow]:
    """
    Get the maximum timeframe for the given backtest data
    :param data: dictionary with preprocessed backtesting data
    :return: tuple containing min_date, max_date
    """
    all_dates = Series([])
    for pair, pair_data in data.items():
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
            result.profit.sum(),
            result.loss.sum()
        ])

    # Append Total
    tabular_data.append([
        'TOTAL',
        len(results.index),
        results.profit_percent.mean() * 100.0,
        results.profit_BTC.sum(),
        results.duration.mean() * ticker_interval,
        results.profit.sum(),
        results.loss.sum()
    ])
    return tabulate(tabular_data, headers=headers, floatfmt=floatfmt)


def get_trade_entry(pair, row, ticker, trade_count_lock, args):
    stake_amount = args['stake_amount']
    max_open_trades = args.get('max_open_trades', 0)
    sell_profit_only = args.get('sell_profit_only', False)
    stoploss = args.get('stoploss', -1)
    use_sell_signal = args.get('use_sell_signal', False)
    trade = Trade(open_rate=row.close,
                  open_date=row.date,
                  stake_amount=stake_amount,
                  amount=stake_amount / row.open,
                  fee=exchange.get_fee()
                  )

    # calculate win/lose forwards from buy point
    sell_subset = ticker[row.Index + 1:][['close', 'date', 'sell']]
    for row2 in sell_subset.itertuples(index=True):
        if max_open_trades > 0:
            # Increase trade_count_lock for every iteration
            trade_count_lock[row2.date] = trade_count_lock.get(row2.date, 0) + 1

        current_profit_percent = trade.calc_profit_percent(rate=row2.close)
        if (sell_profit_only and current_profit_percent < 0):
            continue
        if min_roi_reached(trade, row2.close, row2.date) or \
            (row2.sell == 1 and use_sell_signal) or \
                current_profit_percent <= stoploss:
            current_profit_btc = trade.calc_profit(rate=row2.close)
            return row2, (pair,
                          current_profit_percent,
                          current_profit_btc,
                          row2.Index - row.Index,
                          current_profit_btc > 0,
                          current_profit_btc < 0
                          )


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
        buy_subset = ticker[ticker.buy == 1][['buy', 'open', 'close', 'date', 'sell']]
        for row in buy_subset.itertuples(index=True):
            if realistic:
                if lock_pair_until is not None and row.Index <= lock_pair_until:
                    continue
            if max_open_trades > 0:
                # Check if max_open_trades has already been reached for the given date
                if not trade_count_lock.get(row.date, 0) < max_open_trades:
                    continue

            if max_open_trades > 0:
                # Increase lock
                trade_count_lock[row.date] = trade_count_lock.get(row.date, 0) + 1

            ret = get_trade_entry(pair, row, ticker,
                                  trade_count_lock, args)
            if ret:
                row2, trade_entry = ret
                lock_pair_until = row2.Index
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
    labels = ['currency', 'profit_percent', 'profit_BTC', 'duration', 'profit', 'loss']
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

    logger.info('Using ticker_interval: %s ...', args.ticker_interval)

    data = {}
    pairs = config['exchange']['pair_whitelist']
    if args.live:
        logger.info('Downloading data for all pairs in whitelist ...')
        for pair in pairs:
            data[pair] = exchange.get_ticker_history(pair, args.ticker_interval)
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

    # Monkey patch config
    from freqtrade import main
    main._CONF = config

    preprocessed = optimize.tickerdata_to_dataframe(data)
    # Print timeframe
    min_date, max_date = get_timeframe(preprocessed)
    logger.info('Measuring data from %s up to %s ...', min_date.isoformat(), max_date.isoformat())
    # Execute backtest and print results
    sell_profit_only = config.get('experimental', {}).get('sell_profit_only', False)
    use_sell_signal = config.get('experimental', {}).get('use_sell_signal', False)
    results = backtest({'stake_amount': config['stake_amount'],
                        'processed': preprocessed,
                        'max_open_trades': max_open_trades,
                        'realistic': args.realistic_simulation,
                        'sell_profit_only': sell_profit_only,
                        'use_sell_signal': use_sell_signal,
                        'stoploss': config.get('stoploss'),
                        'record': args.export
                        })
    logger.info(
        '\n==================================== BACKTESTING REPORT ====================================\n%s',  # noqa
        generate_text_table(data, results, config['stake_currency'], args.ticker_interval)
    )
