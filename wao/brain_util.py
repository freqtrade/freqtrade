import sys
import threading
import watchdog
import os
import time
import datetime
from wao.brain_config import BrainConfig
from wao._429_watcher import _429_Watcher
import pickle

sys.path.append(BrainConfig.EXECUTION_PATH)
from config import Config
# from romeo import Romeo, RomeoExitPriceType
from backtest_signal import BacktestSignal


def write_to_backtest_table(timestamp, coin, brain, time_out_hours, type):
    print("STEP [1]++++++++++++++++++++++++++++++++++++" + ", write_to_backtest_table")
    BrainConfig.BACKTEST_SIGNAL_LIST.append(BacktestSignal(brain, coin, type, time_out_hours, timestamp=timestamp))
    pickle.dump(BrainConfig.BACKTEST_SIGNAL_LIST, open(BrainConfig.BACKTEST_SIGNAL_LIST_PICKLE_FILE_PATH, 'wb'))


def perform_execute_buy(coin, brain, romeo_pool, time_out_hours):
    is_test_mode = False
    if BrainConfig.MODE == Config.MODE_TEST:
        is_test_mode = True
    elif BrainConfig.MODE == Config.MODE_PROD:
        is_test_mode = False

    Config.COIN = coin
    Config.BRAIN = brain
    Config.ROMEO_SS_TIMEOUT_HOURS = time_out_hours

    # romeo = Romeo.instance(is_test_mode, True)
    # romeo_pool[coin] = romeo
    # romeo.start()


def perform_execute_sell(coin, romeo_pool):
    if Config.IS_SS_ENABLED:
        romeo = romeo_pool.get(coin)
        # if romeo is not None:
        #     romeo.perform_sell_signal(RomeoExitPriceType.SS)


def perform_create_429_watcher():
    print("perform_create_429_watcher: watching:- " + str(BrainConfig._429_DIRECTORY))
    event_handler = _429_Watcher()
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, path=BrainConfig._429_DIRECTORY, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def setup_429():
    if Config.ENABLE_429_SOLUTION:
        __create_429_directory()
        __create_429_watcher()


def clear_cumulative_value():
    # delete cumulative file
    file_name = BrainConfig.CUMULATIVE_PROFIT_FILE_PATH
    if os.path.isfile(file_name):
        os.remove(file_name)


def __create_429_directory():
    print("create_429_directory:..." + BrainConfig._429_DIRECTORY + "...")
    if not os.path.exists(BrainConfig._429_DIRECTORY):
        os.mkdir(BrainConfig._429_DIRECTORY)


def __create_429_watcher():
    threading.Thread(target=perform_create_429_watcher).start()


def delete_backtest_table_file():
    file_name = BrainConfig.BACKTEST_SIGNAL_LIST_PICKLE_FILE_PATH
    if os.path.isfile(file_name):
        os.remove(file_name)
