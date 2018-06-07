#!/usr/bin/env python3
"""
Main Freqtrade bot script.
Read the documentation to know what cli arguments you need.
"""
import logging
import sys
from typing import List

from freqtrade import OperationalException
from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration
from freqtrade.freqtradebot import FreqtradeBot

logger = logging.getLogger('freqtrade')


def main(sysargv: List[str]) -> None:
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
        return

    freqtrade = None
    return_code = 1
    try:
        # Load and validate configuration
        config = Configuration(args).get_config()

        # Init the bot
        freqtrade = FreqtradeBot(config)

        state = None
        while 1:
            state = freqtrade.worker(old_state=state)

    except KeyboardInterrupt:
        logger.info('SIGINT received, aborting ...')
        return_code = 0
    except OperationalException as e:
        logger.error(str(e))
        return_code = 2
    except BaseException:
        logger.exception('Fatal exception!')
    finally:
        if freqtrade:
            freqtrade.clean()
        sys.exit(return_code)


def set_loggers() -> None:
    """
    Set the logger level for Third party libs
    :return: None
    """
    logging.getLogger('requests.packages.urllib3').setLevel(logging.INFO)
    logging.getLogger('ccxt.base.exchange').setLevel(logging.INFO)
    logging.getLogger('telegram').setLevel(logging.INFO)


if __name__ == '__main__':
    set_loggers()
    main(sys.argv[1:])
