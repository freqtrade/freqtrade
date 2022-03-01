import subprocess
import threading
from user_data.strategies.config import Config
import time


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
    # subprocess.call(
    #     "python3 " + Config.EXECUTION_PATH + "back_tester.py " + date + " " + coin + " " + brain + " " + Config.BACKTEST_DUP + " " + Config.BACKTEST_MAX_COUNT_DUP,
    #     shell=True)

    #todo: uncomment line 26-41, import execution, pass the variables from line 22 above and test if it works directly
    # date = sys.argv[1].replace("#", ", ")
    # Config.COIN = sys.argv[2]
    # Config.BRAIN = sys.argv[3]
    # Config.ROMEO_D_UP_PERCENTAGE = float(sys.argv[4])
    # Config.ROMEO_D_UP_MAX = int(sys.argv[5])
    # Config.BACKTEST_SIGNAL_TIMESTAMP = get_unix_timestamp(date.split("+", 1)[0])
    # Config.BACKTEST_MONTH_INDEX = get_month_from_timestamp()
    # Config.BACKTEST_YEAR = get_year_from_timestamp()
    # Config.IS_BACKTEST = True
    # print("back_tester: Config.BACKTEST_SIGNAL_TIMESTAMP = " + str(
    #     Config.BACKTEST_SIGNAL_TIMESTAMP) + " Config.BACKTEST_MONTH_INDEX = " + str(
    #     Config.BACKTEST_MONTH_INDEX) + " Config.COIN = " + str(Config.COIN) + " Config.BRAIN = " + str(
    #     Config.BRAIN) + " Config.ROMEO_D_UP_PERCENTAGE = " + str(
    #     Config.ROMEO_D_UP_PERCENTAGE) + " Config.ROMEO_D_UP_MAX = " + str(Config.ROMEO_D_UP_MAX))
    #
    # Romeo.instance(True)


def back_test(date_time, coin, brain):
    time.sleep(Config.BACKTEST_THROTTLE_SECOND)
    if Config.IS_PARALLEL_EXECUTION:
        threading.Thread(target=_perform_back_test, args=(date_time, coin, brain)).start()
    else:
        _perform_back_test(date_time, coin, brain)
