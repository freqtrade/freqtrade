import os
import signal
import time
import logging
from multiprocessing import Value

logger = logging.getLogger('freqtrade.watchdog')

WATCHDOG_TIMEOUT = 300
KILL_TIMEOUT = 60


class Watchdog:

    def __init__(self, timeout=WATCHDOG_TIMEOUT, kill_timeout=KILL_TIMEOUT):
        self.shared_heartbeat = Value('d', 0.0)
        self.kill_signal = None
        self.timeout = timeout
        self.kill_timeout = kill_timeout
        self.heartbeat()

    def heartbeat(self) -> None:
        logger.debug("Heartbeat")
        self.shared_heartbeat.value = time.time()

    def exit_gracefully(self, signum, frame):
        logger.warning("Kill signal: {}".format(signum))
        self.kill_signal = signum

    def try_kill(self, pid):
        os.kill(pid, signal.SIGINT)
        for count in range(0, self.kill_timeout):
            try:
                pid, err_code = os.waitpid(pid, os.WNOHANG)
                if pid != 0 or err_code != 0:
                    return True
                time.sleep(1)
            except OSError:
                return True
        return False

    def kill(self, pid):
        logger.info("Stopping pid {}".format(pid))
        if pid:
            if self.try_kill(pid):
                logger.info("Process finished gracefully")
            else:
                logger.warning("Process not responded, kill by SIGTERM")
                os.kill(pid, signal.SIGTERM)
                os.wait()

    def start(self) -> bool:
        pid = os.fork()
        if pid != 0:
            # In watchdog proces, run it
            if not self.run(pid):
                # Got exit signal
                return False
            else:
                # Forked new children, continue to main
                self.heartbeat()
                return True
        else:
            # In children process, continue to main
            return True

    def run(self, pid) -> bool:
        logger.info("Watchdog started")
        self.orig_SIGINT = signal.signal(signal.SIGINT, self.exit_gracefully)
        self.orig_SIGTERM = signal.signal(signal.SIGTERM, self.exit_gracefully)
        try:
            while True:
                if self.kill_signal:
                    raise KeyboardInterrupt()

                timeout = time.time() - self.shared_heartbeat.value

                if timeout > self.timeout:
                    logger.warning("Kill process due to timeout: {}".format(timeout))
                    if not pid:
                        return False
                    self.kill(pid)
                    new_pid = os.fork()
                    if new_pid == 0:
                        logger.info("New children forked")
                        signal.signal(signal.SIGINT, self.orig_SIGINT)
                        signal.signal(signal.SIGTERM, self.orig_SIGTERM)
                        return True
                    else:
                        pid = new_pid

                time.sleep(1)

        except Exception as ex:
            logger.exception(ex)
            self.kill(pid)
            return False

        except KeyboardInterrupt:
            logger.info("Watchdog stopped")
            self.kill(pid)
            return False
