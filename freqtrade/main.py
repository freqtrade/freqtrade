#!/usr/bin/env python3
"""
Main Freqtrade bot script.
Read the documentation to know what cli arguments you need.
"""

import logging
import sys
from typing import Dict
from freqtrade.configuration import Configuration
from freqtrade.arguments import Arguments
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.logger import Logger
from freqtrade import (__version__)

logger = Logger(name='freqtrade').get_logger()


def main(sysargv: Dict) -> None:
    """
    This function will initiate the bot and start the trading loop.
    :return: None
    """
    arguments = Arguments(
        sysargv,
        'Simple High Frequency Trading Bot for crypto currencies'
    )
    args = arguments.get_parsed_arg()

    # A subcommand has been issued.
    # Means if Backtesting or Hyperopt have been called we exit the bot
    if hasattr(args, 'func'):
        args.func(args)
        return 0

    logger.info(
        'Starting freqtrade %s (loglevel=%s)',
        __version__,
        logging.getLevelName(args.loglevel)
    )

    try:
        # Load and validate configuration
        configuration = Configuration(args)

        # Init the bot
        freqtrade = FreqtradeBot(configuration.get_config())

        state = None
        while 1:
            state = freqtrade.worker(old_state=state)

    except KeyboardInterrupt:
        logger.info('SIGINT received, aborting ...')
    except BaseException:
        logger.exception('Fatal exception!')
    finally:
        freqtrade.clean()
        sys.exit(0)


def set_loggers() -> None:
    """
    Set the logger level for Third party libs
    :return: None
    """
    logging.getLogger('requests.packages.urllib3').setLevel(logging.INFO)
    logging.getLogger('telegram').setLevel(logging.INFO)


if __name__ == '__main__':
    set_loggers()
    main(sys.argv[1:])
