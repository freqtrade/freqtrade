#!/usr/bin/env python3
"""
Main Freqtrade bot script.
Read the documentation to know what cli arguments you need.
"""

import sys
# check min. python version
if sys.version_info < (3, 6):
    sys.exit("Freqtrade requires Python version >= 3.6")

# flake8: noqa E402
import logging
from argparse import Namespace
from typing import List

from freqtrade import OperationalException
from freqtrade.arguments import Arguments
from freqtrade.configuration import set_loggers
from freqtrade.worker import Worker


logger = logging.getLogger('freqtrade')


def main(sysargv: List[str]) -> None:
    """
    This function will initiate the bot and start the trading loop.
    :return: None
    """
    set_loggers()
    arguments = Arguments(
        sysargv,
        'Free, open source crypto trading bot'
    )
    args: Namespace = arguments.get_parsed_arg()

    # A subcommand has been issued.
    # Means if Backtesting or Hyperopt have been called we exit the bot
    if hasattr(args, 'func'):
        args.func(args)
        return

    worker = None
    return_code = 1
    try:
        set_loggers()

        worker = None
        return_code = 1

        arguments = Arguments(
            sysargv,
            'Free, open source crypto trading bot'
        )
        args: Namespace = arguments.get_parsed_arg()

        # A subcommand has been issued.
        # Means if Backtesting or Hyperopt have been called we exit the bot
        if hasattr(args, 'func'):
            args.func(args)
            # TODO: fetch return_code as returned by the command function here
            return_code = 0
        else:
            # Load and run worker
            worker = Worker(args)
            worker.run()

    except KeyboardInterrupt:
        logger.info('SIGINT received, aborting ...')
        return_code = 0
    except OperationalException as e:
        logger.error(str(e))
        return_code = 2
    except BaseException:
        logger.exception('Fatal exception!')
    finally:
        if worker:
            worker.exit()
        sys.exit(return_code)


if __name__ == '__main__':
    main(sys.argv[1:])
