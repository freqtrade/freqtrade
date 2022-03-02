from wao.util import _perform_execute, _perform_back_test

EXECUTION_PATH = '/root/workspace2/execution'
import subprocess
import threading
from wao.config import Config
import time
import sys

sys.path.append(EXECUTION_PATH)
from config import Config as ExecutionConfig
from back_tester import get_unix_timestamp, get_month_from_timestamp, get_year_from_timestamp
from romeo import Romeo


class StrategyController:
    romeo_pool = {}

    def __init__(self):
        pass

    def back_test(self, date_time, coin, brain):
        time.sleep(Config.BACKTEST_THROTTLE_SECOND)
        if Config.IS_PARALLEL_EXECUTION:
            threading.Thread(target=_perform_back_test, args=(date_time, coin, brain, self.romeo_pool)).start()
        else:
            _perform_back_test(date_time, coin, brain, self.romeo_pool)

    def execute(self, mode, coin, brain):
        if Config.IS_PARALLEL_EXECUTION:
            threading.Thread(target=_perform_execute, args=(mode, coin, brain, self.romeo_pool)).start()
        else:
            _perform_execute(mode, coin, brain, self.romeo_pool)
