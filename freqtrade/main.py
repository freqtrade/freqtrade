#!/usr/bin/env python3
"""
Main Freqtrade bot script.
Read the documentation to know what cli arguments you need.
"""
import logging
import sys
from argparse import Namespace
from typing import List

from freqtrade import OperationalException
from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration, set_loggers
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.state import State
from freqtrade.rpc import RPCMessageType

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
        while True:
            state = freqtrade.worker(old_state=state)
            if state == State.RELOAD_CONF:
                freqtrade = reconfigure(freqtrade, args)

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
            freqtrade.rpc.send_msg({
                'type': RPCMessageType.STATUS_NOTIFICATION,
                'status': 'process died'
            })
            freqtrade.cleanup()
        sys.exit(return_code)


def reconfigure(freqtrade: FreqtradeBot, args: Namespace) -> FreqtradeBot:
    """
    Cleans up current instance, reloads the configuration and returns the new instance
    """
    # Clean up current modules
    freqtrade.cleanup()

    # Create new instance
    freqtrade = FreqtradeBot(Configuration(args).get_config())
    freqtrade.rpc.send_msg({
        'type': RPCMessageType.STATUS_NOTIFICATION,
        'status': 'config reloaded'
    })
    return freqtrade


if __name__ == '__main__':
    set_loggers()
    main(sys.argv[1:])
