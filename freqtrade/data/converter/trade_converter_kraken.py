import logging
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.constants import DATETIME_PRINT_FORMAT, DEFAULT_TRADES_COLUMNS, Config
from freqtrade.data.converter.trade_converter import (trades_convert_types,
                                                      trades_df_remove_duplicates)
from freqtrade.resolvers import ExchangeResolver


logger = logging.getLogger(__name__)

KRAKEN_CSV_TRADE_COLUMNS = ['timestamp', 'price', 'amount']


def import_kraken_trades_from_csv(config: Config, convert_to: str):
    """
    Import kraken trades from csv
    """
    if config['exchange']['name'] != 'kraken':
        raise ValueError('This function is only for kraken exchange')

    from freqtrade.data.history.idatahandler import get_datahandler
    datadir: Path = config['datadir']
    data_handler = get_datahandler(datadir, data_format=convert_to)

    tradesdir: Path = config['datadir'] / 'trades_csv'
    exchange = ExchangeResolver.load_exchange(config, validate=False)
    # iterate through directories in this directory
    data_symbols = {p.stem for p in tradesdir.rglob('*.csv')}
    print(data_symbols)

    # create pair/filename mapping
    markets = {
        (m['symbol'], m['altname']) for m in exchange.markets.values()
        if m.get('altname') in data_symbols
    }

    for pair, name in markets:
        dfs = []
        # Load and combine all csv files for this pair
        for f in tradesdir.rglob(f"{name}.csv"):
            # print(pair, f)
            df = pd.read_csv(f, names=KRAKEN_CSV_TRADE_COLUMNS)
            dfs.append(df)

        if not dfs:
            continue
        trades = pd.concat(dfs, ignore_index=True)

        trades.loc[:, 'timestamp'] = trades['timestamp'] * 1e3
        trades.loc[:, 'cost'] = trades['price'] * trades['amount']
        for col in DEFAULT_TRADES_COLUMNS:
            if col not in trades.columns:
                trades[col] = np.nan

        trades = trades[DEFAULT_TRADES_COLUMNS]
        trades = trades_convert_types(trades)

        trades_df = trades_df_remove_duplicates(trades)
        logger.info(f"{pair}: {len(trades_df)} trades, from "
                    f"{trades_df['date'].min():{DATETIME_PRINT_FORMAT}} to "
                    f"{trades_df['date'].max():{DATETIME_PRINT_FORMAT}}")

        data_handler.trades_store(pair, trades_df)
