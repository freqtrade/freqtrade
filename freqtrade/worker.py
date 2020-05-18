"""
Main Freqtrade worker class.
"""
import logging
import time
import traceback
from os import getpid
from typing import Any, Callable, Dict, Optional

import sdnotify

from freqtrade import __version__, constants
from freqtrade.configuration import Configuration
from freqtrade.exceptions import OperationalException, TemporaryError
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.state import State

logger = logging.getLogger(__name__)


class Worker:
    """
    Freqtradebot worker class
    """

    def __init__(self, args: Dict[str, Any], config: Dict[str, Any] = None) -> None:
        """
        Init all variables and objects the bot needs to work
        """
        logger.info(f"Starting worker {__version__}")

        self._args = args
        self._config = config
        self._init(False)

        self.last_throttle_start_time: float = 0
        self._heartbeat_msg: float = 0

        # Tell systemd that we completed initialization phase
        self._notify("READY=1")

    def _init(self, reconfig: bool) -> None:
        """
        Also called from the _reconfigure() method (with reconfig=True).
        """
        if reconfig or self._config is None:
            # Load configuration
            self._config = Configuration(self._args, None).get_config()

        # Init the instance of the bot
        self.freqtrade = FreqtradeBot(self._config)

        internals_config = self._config.get('internals', {})
        self._throttle_secs = internals_config.get('process_throttle_secs',
                                                   constants.PROCESS_THROTTLE_SECS)
        self._heartbeat_interval = internals_config.get('heartbeat_interval', 60)

        self._sd_notify = sdnotify.SystemdNotifier() if \
            self._config.get('internals', {}).get('sd_notify', False) else None

    def _notify(self, message: str) -> None:
        """
        Removes the need to verify in all occurances if sd_notify is enabled
        :param message: Message to send to systemd if it's enabled.
        """
        if self._sd_notify:
            logger.debug(f"sd_notify: {message}")
            self._sd_notify.notify(message)

    def run(self) -> None:
        state = None
        while True:
            state = self._worker(old_state=state)
            if state == State.RELOAD_CONF:
                self._reconfigure()

    def _worker(self, old_state: Optional[State]) -> State:
        """
        The main routine that runs each throttling iteration and handles the states.
        :param old_state: the previous service state from the previous call
        :return: current service state
        """
        state = self.freqtrade.state

        # Log state transition
        if state != old_state:
            self.freqtrade.notify_status(f'{state.name.lower()}')

            logger.info(f"Changing state to: {state.name}")
            if state == State.RUNNING:
                self.freqtrade.startup()

            # Reset heartbeat timestamp to log the heartbeat message at
            # first throttling iteration when the state changes
            self._heartbeat_msg = 0

        if state == State.STOPPED:
            # Ping systemd watchdog before sleeping in the stopped state
            self._notify("WATCHDOG=1\nSTATUS=State: STOPPED.")

            self._throttle(func=self._process_stopped, throttle_secs=self._throttle_secs)

        elif state == State.RUNNING:
            # Ping systemd watchdog before throttling
            self._notify("WATCHDOG=1\nSTATUS=State: RUNNING.")

            self._throttle(func=self._process_running, throttle_secs=self._throttle_secs)

        if self._heartbeat_interval:
            now = time.time()
            if (now - self._heartbeat_msg) > self._heartbeat_interval:
                logger.info(f"Bot heartbeat. PID={getpid()}, "
                            f"version='{__version__}', state='{state.name}'")
                self._heartbeat_msg = now

        return state

    def _throttle(self, func: Callable[..., Any], throttle_secs: float, *args, **kwargs) -> Any:
        """
        Throttles the given callable that it
        takes at least `min_secs` to finish execution.
        :param func: Any callable
        :param throttle_secs: throttling interation execution time limit in seconds
        :return: Any (result of execution of func)
        """
        self.last_throttle_start_time = time.time()
        logger.debug("========================================")
        result = func(*args, **kwargs)
        time_passed = time.time() - self.last_throttle_start_time
        sleep_duration = max(throttle_secs - time_passed, 0.0)
        logger.debug(f"Throttling with '{func.__name__}()': sleep for {sleep_duration:.2f} s, "
                     f"last iteration took {time_passed:.2f} s.")
        time.sleep(sleep_duration)
        return result

    def _process_stopped(self) -> None:
        self.freqtrade.process_stopped()

    def _process_running(self) -> None:
        try:
            self.freqtrade.process()
        except TemporaryError as error:
            logger.warning(f"Error: {error}, retrying in {constants.RETRY_TIMEOUT} seconds...")
            time.sleep(constants.RETRY_TIMEOUT)
        except OperationalException:
            tb = traceback.format_exc()
            hint = 'Issue `/start` if you think it is safe to restart.'

            self.freqtrade.notify_status(f'OperationalException:\n```\n{tb}```{hint}')

            logger.exception('OperationalException. Stopping trader ...')
            self.freqtrade.state = State.STOPPED

    def _reconfigure(self) -> None:
        """
        Cleans up current freqtradebot instance, reloads the configuration and
        replaces it with the new instance
        """
        # Tell systemd that we initiated reconfiguration
        self._notify("RELOADING=1")

        # Clean up current freqtrade modules
        self.freqtrade.cleanup()

        # Load and validate config and create new instance of the bot
        self._init(True)

        self.freqtrade.notify_status('config reloaded')

        # Tell systemd that we completed reconfiguration
        self._notify("READY=1")

    def exit(self) -> None:
        # Tell systemd that we are exiting now
        self._notify("STOPPING=1")

        if self.freqtrade:
            self.freqtrade.notify_status('process died')
            self.freqtrade.cleanup()
