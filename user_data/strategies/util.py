import subprocess
import threading
from user_data.strategies.config import Config
import time
import watchdog

from user_data.strategies._429_watcher import _429_Watcher


def execute(mode, coin, brain):
    if Config.IS_PARALLEL_EXECUTION:
        threading.Thread(target=_perform_execute, args=(mode, coin, brain)).start()
    else:
        _perform_execute(mode, coin, brain)


def _perform_execute(mode, coin, brain):
    subprocess.call("python3 " + Config.EXECUTION_PATH + "launcher.py " + mode + " " + coin + " " + brain, shell=True)


def _perform_back_test(date_time, coin, brain):
    date = str(date_time)
    date = date.replace(" ", "#")
    subprocess.call(
        "python3 " + Config.EXECUTION_PATH + "back_tester.py " + date + " " + coin + " " + brain + " " + Config.BACKTEST_DUP + " " + Config.BACKTEST_MAX_COUNT_DUP,
        shell=True)


def back_test(date_time, coin, brain):
    time.sleep(Config.BACKTEST_THROTTLE_SECOND)
    if Config.IS_PARALLEL_EXECUTION:
        threading.Thread(target=_perform_back_test, args=(date_time, coin, brain)).start()
    else:
        _perform_back_test(date_time, coin, brain)


def create_429_watcher(is_test_mode):
    print("create_429_watcher:...")
    event_handler = _429_Watcher(is_test_mode)
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, path=Config._429_DIRECTORY, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
