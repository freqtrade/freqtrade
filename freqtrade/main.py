#!/usr/bin/env python3
"""
Main Freqtrade bot script.
Read the documentation to know what cli arguments you need.
"""
import logging
import sys
import time
from argparse import Namespace
from typing import Any, Callable, List
import sdnotify

from freqtrade import (constants, OperationalException, __version__)
from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration, set_loggers
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
        'Free, open source crypto trading bot'
    )
    args = arguments.get_parsed_arg()

    # A subcommand has been issued.
    # Means if Backtesting or Hyperopt have been called we exit the bot
    if hasattr(args, 'func'):
        args.func(args)
        return

    return_code = 1
    try:
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
        if worker is not None:
            worker.exit()
        sys.exit(return_code)


class Worker(object):
    """
    Freqtradebot worker class
    """

    def __init__(self, args: Namespace) -> None:
        """
        Init all variables and objects the bot needs to work
        """
        logger.info('Starting worker %s', __version__)

        self._args = args
        self._init()

        # Tell systemd that we completed initialization phase
        if self._sd_notify:
            logger.debug("sd_notify: READY=1")
            self._sd_notify.notify("READY=1")

    def _init(self):
        """
        Also called from the _reconfigure() method.
        """
        # Load configuration
        self._config = Configuration(self._args, None).get_config()

        # Import freqtradebot here in order to avoid python circular
        # dependency error, damn!
        from freqtrade.freqtradebot import FreqtradeBot

        # Init the instance of the bot
        self.freqtrade = FreqtradeBot(self._config, self)

        # Set initial bot state
        initial_state = self._config.get('initial_state')
        if initial_state:
            self._state = State[initial_state.upper()]
        else:
            self._state = State.STOPPED

        self._throttle_secs = self._config.get('internals', {}).get(
            'process_throttle_secs',
            constants.PROCESS_THROTTLE_SECS
        )

        self._sd_notify = sdnotify.SystemdNotifier() if \
            self._config.get('internals', {}).get('sd_notify', False) else None

    @property
    def state(self) -> State:
        return self._state

    @state.setter
    def state(self, value: State):
        self._state = value

    def run(self):
        state = None
        while True:
            state = self._worker(old_state=state, throttle_secs=self._throttle_secs)
            if state == State.RELOAD_CONF:
                self.freqtrade = self._reconfigure()

    def _worker(self, old_state: State, throttle_secs: float) -> State:
        """
        Trading routine that must be run at each loop
        :param old_state: the previous service state from the previous call
        :return: current service state
        """
        state = self._state

        # Log state transition
        if state != old_state:
            self.freqtrade.rpc.send_msg({
                'type': RPCMessageType.STATUS_NOTIFICATION,
                'status': f'{state.name.lower()}'
            })
            logger.info('Changing state to: %s', state.name)
            if state == State.RUNNING:
                self.freqtrade.rpc.startup_messages(self._config, self.freqtrade.pairlists)

        if state == State.STOPPED:
            # Ping systemd watchdog before sleeping in the stopped state
            if self._sd_notify:
                logger.debug("sd_notify: WATCHDOG=1\\nSTATUS=State: STOPPED.")
                self._sd_notify.notify("WATCHDOG=1\nSTATUS=State: STOPPED.")

            time.sleep(throttle_secs)

        elif state == State.RUNNING:
            # Ping systemd watchdog before throttling
            if self._sd_notify:
                logger.debug("sd_notify: WATCHDOG=1\\nSTATUS=State: RUNNING.")
                self._sd_notify.notify("WATCHDOG=1\nSTATUS=State: RUNNING.")

            self._throttle(func=self._process, min_secs=throttle_secs)

        return state

    def _throttle(self, func: Callable[..., Any], min_secs: float, *args, **kwargs) -> Any:
        """
        Throttles the given callable that it
        takes at least `min_secs` to finish execution.
        :param func: Any callable
        :param min_secs: minimum execution time in seconds
        :return: Any
        """
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = max(min_secs - (end - start), 0.0)
        logger.debug('Throttling %s for %.2f seconds', func.__name__, duration)
        time.sleep(duration)
        return result

    def _process(self) -> bool:
        return self.freqtrade.process()

    def _reconfigure(self):
        """
        Cleans up current freqtradebot instance, reloads the configuration and
        returns the new instance
        """
        # Tell systemd that we initiated reconfiguration
        if self._sd_notify:
            logger.debug("sd_notify: RELOADING=1")
            self._sd_notify.notify("RELOADING=1")

        # Clean up current freqtrade modules
        self.freqtrade.cleanup()

        # Load and validate config and create new instance of the bot
        self._init()

        self.freqtrade.rpc.send_msg({
            'type': RPCMessageType.STATUS_NOTIFICATION,
            'status': 'config reloaded'
        })

        # Tell systemd that we completed reconfiguration
        if self._sd_notify:
            logger.debug("sd_notify: READY=1")
            self._sd_notify.notify("READY=1")

    def exit(self):
        # Tell systemd that we are exiting now
        if self._sd_notify:
            logger.debug("sd_notify: STOPPING=1")
            self._sd_notify.notify("STOPPING=1")

        if self.freqtrade:
            self.freqtrade.rpc.send_msg({
                'type': RPCMessageType.STATUS_NOTIFICATION,
                'status': 'process died'
            })
            self.freqtrade.cleanup()


if __name__ == '__main__':
    set_loggers()
    main(sys.argv[1:])
