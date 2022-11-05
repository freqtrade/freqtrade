import threading
import watchdog
import os
import time
from wao.brain_config import *
from wao._429_watcher import _429_Watcher
from wao.error_watcher import Error_Watcher
import pickle

from execution.config import Config
from execution.romeo import Romeo, RomeoExitPriceType
from execution.backtest_signal import BacktestSignal


def write_to_backtest_table(timestamp, coin, brain, time_out_hours, dup, type):
    print("STEP [1]++++++++++++++++++++++++++++++++++++" + ", write_to_backtest_table")
    BACKTEST_SIGNAL_LIST.append(BacktestSignal(brain, coin, type, time_out_hours, dup, timestamp=timestamp))
    pickle.dump(BACKTEST_SIGNAL_LIST, open(BACKTEST_SIGNAL_LIST_PICKLE_FILE_PATH, 'wb'))


def perform_execute_buy(coin, brain, time_out_hours, dup):
    is_test_mode = False
    if MODE == Config.MODE_TEST:
        is_test_mode = True
    elif MODE == Config.MODE_PROD:
        is_test_mode = False

    Config.COIN = coin
    Config.BRAIN = brain
    Config.ROMEO_SS_TIMEOUT_HOURS = time_out_hours
    Config.ROMEO_D_UP_PERCENTAGE = dup

    romeo = Romeo.instance(is_test_mode, True)
    ROMEO_POOL[coin] = romeo
    romeo.start()


def perform_execute_sell(coin):
    if Config.IS_SS_ENABLED:
        romeo = ROMEO_POOL.get(coin)
        if romeo is not None:
            romeo.perform_sell_signal(RomeoExitPriceType.SS)


def perform_create_429_watcher():
    print("perform_create_429_watcher: watching:- " + str(_429_DIRECTORY))
    event_handler = _429_Watcher()
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, path=_429_DIRECTORY, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def perform_create_error_watcher():
    print("perform_create_error_watcher: watching:- " + str(_WAO_LOGS_DIRECTORY))
    event_handler = Error_Watcher()
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, path=_WAO_LOGS_DIRECTORY, recursive=True)
    # Start the observer
    observer.start()
    try:
        while True:
            # Set the thread sleep time
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def create_watchers():
    if Config.ENABLE_429_SOLUTION:
        __create_429_directory()
        __create_429_watcher()
    if IS_ERROR_WATCHER_ENABLED:
        __create_error_watcher()


def clear_cumulative_value():
    # delete cumulative file
    _delete_file(CUMULATIVE_PROFIT_FILE_PATH)
    _delete_file(CUMULATIVE_PROFIT_BINANCE_FILE_PATH)
    _delete_file(INITIAL_ACCOUNT_BALANCE_BINANCE_FILE_PATH)


def _delete_file(file_name):
    if os.path.isfile(file_name):
        os.remove(file_name)


def create_initial_account_balance_binance_file():
    file_path = INITIAL_ACCOUNT_BALANCE_BINANCE_FILE_PATH
    if not os.path.exists(file_path):
        with open(file_path, 'w+') as file:
            file.write("")
        file.close()


def __create_429_directory():
    print("create_429_directory:..." + _429_DIRECTORY + "...")
    if not os.path.exists(_429_DIRECTORY):
        os.mkdir(_429_DIRECTORY)


def __create_error_watcher():
    threading.Thread(target=perform_create_error_watcher).start()


def __create_429_watcher():
    threading.Thread(target=perform_create_429_watcher).start()


def delete_backtest_table_file():
    file_name = BACKTEST_SIGNAL_LIST_PICKLE_FILE_PATH
    if os.path.isfile(file_name):
        os.remove(file_name)


def is_romeo_alive(coin):
    return ROMEO_POOL.get(coin) is not None


def remove_from_pool(coin):
    if is_romeo_alive(coin):
        del ROMEO_POOL[coin]
