import sys
import threading
import watchdog
import os
import time
import datetime
from wao.brain_config import BrainConfig
from wao._429_watcher import _429_Watcher
from wao.brain_util import perform_back_test_buy
import pickle
import threading

sys.path.append(BrainConfig.EXECUTION_PATH)
from config import Config
from romeo import Romeo, RomeoExitPriceType

print("STEP [2]++++++++++++++++++++++++++++++++++++" + ", read_from_backtest_table")
file_pi2 = open(BrainConfig.BACKTEST_TABLE_FILE_PATH, 'r')
backtest_execution_list = pickle.load(file_pi2)
print("STEP [2]++++++++++++++++++++++++++++++++++++" + ", backtest_execution_list.size=" + str(
    len(backtest_execution_list)))


def __buy_back_test(date_time, coin, brain, timeout_hours):
    if BrainConfig.IS_PARALLEL_EXECUTION:
        threading.Thread(target=perform_back_test_buy,
                         args=(date_time, coin, brain, timeout_hours)).start()
    else:
        perform_back_test_buy(date_time, coin, brain, timeout_hours)


for backtest_execution in backtest_execution_list:
    print("STEP [2]++++++++++++++++++++++++++++++++++++ execution.type=" + str(backtest_execution.type))

    # backtest_execution_stack: first element is buy and second is sell
    stack = []
    if backtest_execution.type == "buy":
        if len(stack) == 0:
            stack.append(backtest_execution)
        elif len(stack) == 1:
            stack.pop()
            print("STEP [2]++++++++++++++++++" + " warning: buy signal replaced: stack was not empty")
            stack.append(backtest_execution)
        else:
            print("STEP [2]++++++++++++++++++" + " Error: buy signal ignored: stack was full")

    if backtest_execution.type == "sell":
        if len(stack) == 0:
            print("STEP [2]++++++++++++++++++" + " warning: sell signal ignored: stack was empty")
        elif len(stack) == 1:
            stack.append(backtest_execution)
        else:
            print("STEP [2]++++++++++++++++++" + " Error: sell signal ignored: stack was full")

    if len(stack) == 2:
        execution = stack[0]
        __buy_back_test(execution.timestamp, execution.coin, execution.brain, execution.timeout_hours)



