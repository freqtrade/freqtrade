import sys
import threading
import watchdog
import os
import time
import datetime
from wao.brain_config import BrainConfig
from wao._429_watcher import _429_Watcher
from wao.brain_util import perform_back_test_buy, perform_back_test_sell, clear_execution_state, is_execution_state_open
import pickle
import threading
from wao.backtest_execution import BacktestExecution

print("STEP [2]++++++++++++++++++++++++++++++++++++" + ", read_from_backtest_table")
backtest_execution_list = pickle.load(open(BrainConfig.BACKTEST_EXECUTION_LIST_FILE_PATH, 'r'))
print("STEP [2]++++++++++++++++++++++++++++++++++++" + ", backtest_execution_list.size=" + str(
    len(backtest_execution_list)))


def __buy_back_test(date_time, coin, brain, timeout_hours):
    if BrainConfig.IS_PARALLEL_EXECUTION:
        threading.Thread(target=perform_back_test_buy,
                         args=(date_time, coin, brain, timeout_hours)).start()
    else:
        perform_back_test_buy(date_time, coin, brain, timeout_hours)


# backtest_execution_stack: first element is buy and second is sell
stack = []
index = 0
for backtest_execution in backtest_execution_list:

    print("STEP [3]++++++++++++++++++++++++++++++++++++ execution.type=" + str(backtest_execution.type) + str(
        index + 1) + " of " + str(len(backtest_execution_list)))

    while is_execution_state_open() and index > 0:
        print("STEP [3]++++++++++++++++ is_execution_state_open: True  index=" + str(index) + ", waiting...")
        time.sleep(1)

    clear_execution_state()

    if backtest_execution.type == "buy":
        if len(stack) == 0:
            stack.append(backtest_execution)
        elif len(stack) == 1:
            stack.pop()
            print("STEP [3]++++++++++++++++++" + " warning: buy signal replaced: stack was not empty")
            stack.append(backtest_execution)
        else:
            print("STEP [3]++++++++++++++++++" + " Error: buy signal ignored: stack was full")

    if backtest_execution.type == "sell":
        if len(stack) == 0:
            print("STEP [3]++++++++++++++++++" + " warning: sell signal ignored: stack was empty")
        elif len(stack) == 1:
            stack.append(backtest_execution)
        else:
            print("STEP [3]++++++++++++++++++" + " Error: sell signal ignored: stack was full")

    if len(stack) == 2:
        print("STEP [4]++++++++++++++++++++++++++++++++++++ stack" + str(stack))

        buy_execution = stack[0]
        __buy_back_test(buy_execution.timestamp, buy_execution.coin, buy_execution.brain, buy_execution.timeout_hours)

        sell_execution = stack[1]
        perform_back_test_sell(sell_execution.timestamp)

        stack = []

    index += 1
