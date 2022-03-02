EXECUTION_PATH = '/root/workspace2/execution'
import subprocess
import threading
from user_data.strategies.config import Config
import time
import sys
sys.path.append(EXECUTION_PATH)
from config import Config as ExecutionConfig
from back_tester import get_unix_timestamp, get_month_from_timestamp, get_year_from_timestamp
from romeo import Romeo

def execute(mode, coin, brain):
    if Config.IS_PARALLEL_EXECUTION:
        threading.Thread(target=_perform_execute, args=(mode, coin, brain)).start()
    else:
        _perform_execute(mode, coin, brain)


def _perform_execute(mode, coin, brain):
    subprocess.call("python3 " + Config.EXECUTION_PATH + "launcher.py " + mode + " " + coin + " " + brain, shell=True)


def _perform_back_test(date_time, coin, brain):
    date = str(date_time)
    date = date.replace(" ", ", ")
    ExecutionConfig.COIN = coin
    ExecutionConfig.BRAIN = brain
    ExecutionConfig.ROMEO_D_UP_PERCENTAGE = float(Config.BACKTEST_DUP)
    ExecutionConfig.ROMEO_D_UP_MAX = int(Config.BACKTEST_MAX_COUNT_DUP)
    ExecutionConfig.BACKTEST_SIGNAL_TIMESTAMP = get_unix_timestamp(date.split("+", 1)[0])
    ExecutionConfig.BACKTEST_MONTH_INDEX = get_month_from_timestamp()
    ExecutionConfig.BACKTEST_YEAR = get_year_from_timestamp()
    ExecutionConfig.IS_BACKTEST = True
    print("back_tester: Config.BACKTEST_SIGNAL_TIMESTAMP = " + str(
        ExecutionConfig.BACKTEST_SIGNAL_TIMESTAMP) + " Config.BACKTEST_MONTH_INDEX = " + str(
        ExecutionConfig.BACKTEST_MONTH_INDEX) + " Config.COIN = " + str(ExecutionConfig.COIN) + " Config.BRAIN = " + str(
        ExecutionConfig.BRAIN) + " Config.ROMEO_D_UP_PERCENTAGE = " + str(
        ExecutionConfig.ROMEO_D_UP_PERCENTAGE) + " Config.ROMEO_D_UP_MAX = " + str(ExecutionConfig.ROMEO_D_UP_MAX))

    Romeo.instance(True)

def test_romeo_start_parallel(date_time, coin, brain):
    threading.Thread(target=test_romeo_start, args=(date_time, coin, brain)).start()

def test_romeo_start(date_time, coin, brain):
    date = str(date_time)
    date = date.replace(" ", ", ")
    ExecutionConfig.COIN = coin
    ExecutionConfig.BRAIN = brain
    ExecutionConfig.ROMEO_D_UP_PERCENTAGE = float(Config.BACKTEST_DUP)
    ExecutionConfig.ROMEO_D_UP_MAX = int(Config.BACKTEST_MAX_COUNT_DUP)
    ExecutionConfig.BACKTEST_SIGNAL_TIMESTAMP = get_unix_timestamp(date.split("+", 1)[0])
    ExecutionConfig.BACKTEST_MONTH_INDEX = get_month_from_timestamp()
    ExecutionConfig.BACKTEST_YEAR = get_year_from_timestamp()
    ExecutionConfig.IS_BACKTEST = True
    print("back_tester: Config.BACKTEST_SIGNAL_TIMESTAMP = " + str(
        ExecutionConfig.BACKTEST_SIGNAL_TIMESTAMP) + " Config.BACKTEST_MONTH_INDEX = " + str(
        ExecutionConfig.BACKTEST_MONTH_INDEX) + " Config.COIN = " + str(ExecutionConfig.COIN) + " Config.BRAIN = " + str(
        ExecutionConfig.BRAIN) + " Config.ROMEO_D_UP_PERCENTAGE = " + str(
        ExecutionConfig.ROMEO_D_UP_PERCENTAGE) + " Config.ROMEO_D_UP_MAX = " + str(ExecutionConfig.ROMEO_D_UP_MAX))

    Romeo.instance(True).start()

def test_romeo_sell(date_time, coin, brain):
    date = str(date_time)
    date = date.replace(" ", ", ")
    ExecutionConfig.COIN = coin
    ExecutionConfig.BRAIN = brain
    ExecutionConfig.ROMEO_D_UP_PERCENTAGE = float(Config.BACKTEST_DUP)
    ExecutionConfig.ROMEO_D_UP_MAX = int(Config.BACKTEST_MAX_COUNT_DUP)
    ExecutionConfig.BACKTEST_SIGNAL_TIMESTAMP = get_unix_timestamp(date.split("+", 1)[0])
    ExecutionConfig.BACKTEST_MONTH_INDEX = get_month_from_timestamp()
    ExecutionConfig.BACKTEST_YEAR = get_year_from_timestamp()
    ExecutionConfig.IS_BACKTEST = True
    print("back_tester: Config.BACKTEST_SIGNAL_TIMESTAMP = " + str(
        ExecutionConfig.BACKTEST_SIGNAL_TIMESTAMP) + " Config.BACKTEST_MONTH_INDEX = " + str(
        ExecutionConfig.BACKTEST_MONTH_INDEX) + " Config.COIN = " + str(ExecutionConfig.COIN) + " Config.BRAIN = " + str(
        ExecutionConfig.BRAIN) + " Config.ROMEO_D_UP_PERCENTAGE = " + str(
        ExecutionConfig.ROMEO_D_UP_PERCENTAGE) + " Config.ROMEO_D_UP_MAX = " + str(ExecutionConfig.ROMEO_D_UP_MAX))

    Romeo.instance(True).perform_sell_signal()



def back_test(date_time, coin, brain):
    time.sleep(Config.BACKTEST_THROTTLE_SECOND)
    if Config.IS_PARALLEL_EXECUTION:
        threading.Thread(target=_perform_back_test, args=(date_time, coin, brain)).start()
    else:
        _perform_back_test(date_time, coin, brain)
