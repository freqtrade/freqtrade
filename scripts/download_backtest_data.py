#!/usr/bin/env python3
"""
This script generates json files with pairs history data
"""
import arrow
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from freqtrade.configuration import Arguments, TimeRange
from freqtrade.configuration import Configuration
from freqtrade.configuration.arguments import ARGS_DOWNLOADER
from freqtrade.configuration.check_exchange import check_exchange
from freqtrade.configuration.load_config import load_config_file
from freqtrade.data.history import download_pair_history
from freqtrade.exchange import Exchange
from freqtrade.misc import deep_merge_dicts

import logging

logger = logging.getLogger('download_backtest_data')

DEFAULT_DL_PATH = 'user_data/data'

# Do not read the default config if config is not specified
# in the command line options explicitely
arguments = Arguments(sys.argv[1:], 'Download backtest data',
                      no_default_config=True)
arguments._build_args(optionlist=ARGS_DOWNLOADER)
args = arguments._parse_args()

# Use bittrex as default exchange
exchange_name = args.exchange or 'bittrex'

pairs: List = []

configuration = Configuration(args)
config: Dict[str, Any] = {}

if args.config:
    # Now expecting a list of config filenames here, not a string
    for path in args.config:
        logger.info(f"Using config: {path}...")
        # Merge config options, overwriting old values
        config = deep_merge_dicts(load_config_file(path), config)

    config['stake_currency'] = ''
    # Ensure we do not use Exchange credentials
    config['exchange']['dry_run'] = True
    config['exchange']['key'] = ''
    config['exchange']['secret'] = ''

    pairs = config['exchange']['pair_whitelist']

    if config.get('ticker_interval'):
        timeframes = args.timeframes or [config.get('ticker_interval')]
    else:
        timeframes = args.timeframes or ['1m', '5m']

else:
    config = {
        'stake_currency': '',
        'dry_run': True,
        'exchange': {
            'name': exchange_name,
            'key': '',
            'secret': '',
            'pair_whitelist': [],
            'ccxt_async_config': {
                'enableRateLimit': True,
                'rateLimit': 200
            }
        }
    }
    timeframes = args.timeframes or ['1m', '5m']

configuration._process_logging_options(config)

if args.config and args.exchange:
    logger.warning("The --exchange option is ignored, "
                   "using exchange settings from the configuration file.")

# Check if the exchange set by the user is supported
check_exchange(config)

configuration._process_datadir_options(config)

dl_path = Path(config['datadir'])

pairs_file = Path(args.pairs_file) if args.pairs_file else dl_path.joinpath('pairs.json')

if not pairs or args.pairs_file:
    logger.info(f'Reading pairs file "{pairs_file}".')
    # Download pairs from the pairs file if no config is specified
    # or if pairs file is specified explicitely
    if not pairs_file.exists():
        sys.exit(f'No pairs file found with path "{pairs_file}".')

    with pairs_file.open() as file:
        pairs = list(set(json.load(file)))

    pairs.sort()

timerange = TimeRange()
if args.days:
    time_since = arrow.utcnow().shift(days=-args.days).strftime("%Y%m%d")
    timerange = arguments.parse_timerange(f'{time_since}-')

logger.info(f'About to download pairs: {pairs}, intervals: {timeframes} to {dl_path}')

pairs_not_available = []

try:
    # Init exchange
    exchange = Exchange(config)

    for pair in pairs:
        if pair not in exchange._api.markets:
            pairs_not_available.append(pair)
            logger.info(f"Skipping pair {pair}...")
            continue
        for ticker_interval in timeframes:
            pair_print = pair.replace('/', '_')
            filename = f'{pair_print}-{ticker_interval}.json'
            dl_file = dl_path.joinpath(filename)
            if args.erase and dl_file.exists():
                logger.info(
                    f'Deleting existing data for pair {pair}, interval {ticker_interval}.')
                dl_file.unlink()

            logger.info(f'Downloading pair {pair}, interval {ticker_interval}.')
            download_pair_history(datadir=dl_path, exchange=exchange,
                                  pair=pair, ticker_interval=str(ticker_interval),
                                  timerange=timerange)

except KeyboardInterrupt:
    sys.exit("SIGINT received, aborting ...")

finally:
    if pairs_not_available:
        logger.info(
            f"Pairs [{','.join(pairs_not_available)}] not available "
            f"on exchange {config['exchange']['name']}.")
