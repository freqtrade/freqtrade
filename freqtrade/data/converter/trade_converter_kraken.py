import logging
from pathlib import Path

import pandas as pd

from freqtrade.constants import DATETIME_PRINT_FORMAT, DEFAULT_TRADES_COLUMNS, Config
from freqtrade.data.converter.trade_converter import (trades_convert_types,
                                                      trades_df_remove_duplicates)
from freqtrade.data.history import get_datahandler
from freqtrade.enums import TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.resolvers import ExchangeResolver


logger = logging.getLogger(__name__)

KRAKEN_CSV_TRADE_COLUMNS = ['timestamp', 'price', 'amount']


def import_kraken_trades_from_csv(config: Config, convert_to: str):
    """
    Import kraken trades from csv
    """
    if config['exchange']['name'] != 'kraken':
        raise OperationalException('This function is only for the kraken exchange.')

    datadir: Path = config['datadir']
    data_handler = get_datahandler(datadir, data_format=convert_to)

    tradesdir: Path = config['datadir'] / 'trades_csv'
    exchange = ExchangeResolver.load_exchange(config, validate=False)
    # iterate through directories in this directory
    data_symbols = {p.stem for p in tradesdir.rglob('*.csv')}

    # create pair/filename mapping
    markets = {
        (m['symbol'], m['altname']) for m in exchange.markets.values()
        if m.get('altname') in data_symbols
    }
    logger.info(f"Found csv files for {', '.join(data_symbols)}.")

    if pairs_raw := config.get('pairs'):
        pairs = expand_pairlist(pairs_raw, [m[0] for m in markets])
        markets = {m for m in markets if m[0] in pairs}
        if not markets:
            logger.info(f"No data found for pairs {', '.join(pairs_raw)}.")
            return
    logger.info(f"Converting pairs: {', '.join(m[0] for m in markets)}.")

    for pair, name in markets:
        logger.debug(f"Converting pair {pair}, files */{name}.csv")
        dfs = []
        # Load and combine all csv files for this pair
        for f in tradesdir.rglob(f"{name}.csv"):
            df = pd.read_csv(f, names=KRAKEN_CSV_TRADE_COLUMNS)
            if not df.empty:
                dfs.append(df)

        # Load existing trades data
        if not dfs:
            # edgecase, can only happen if the file was deleted between the above glob and here
            logger.info(f"No data found for pair {pair}")
            continue

        trades = pd.concat(dfs, ignore_index=True)
        del dfs

        trades.loc[:, 'timestamp'] = trades['timestamp'] * 1e3
        trades.loc[:, 'cost'] = trades['price'] * trades['amount']
        for col in DEFAULT_TRADES_COLUMNS:
            if col not in trades.columns:
                trades.loc[:, col] = ''
        trades = trades[DEFAULT_TRADES_COLUMNS]
        trades = trades_convert_types(trades)

        trades_df = trades_df_remove_duplicates(trades)
        del trades
        logger.info(f"{pair}: {len(trades_df)} trades, from "
                    f"{trades_df['date'].min():{DATETIME_PRINT_FORMAT}} to "
                    f"{trades_df['date'].max():{DATETIME_PRINT_FORMAT}}")

        data_handler.trades_store(pair, trades_df, TradingMode.SPOT)
