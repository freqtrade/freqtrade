from wao.util import _perform_execute, _perform_back_test
import threading
from wao.config import Config
from wao.util import setup_429
from wao.notifier import send_start_deliminator_message
import time


class StrategyController:
    romeo_pool = {}

    def __init__(self, brain):
        setup_429()
        if Config.IS_BACKTEST:
            send_start_deliminator_message(brain, Config.BACKTEST_COIN,
                                           Config.BACKTEST_MONTH_LIST[Config.BACKTEST_DATA_CLEANER_MONTH_INDEX],
                                           Config.BACKTEST_DATA_CLEANER_YEAR, Config.BACKTEST_DUP,
                                           Config.BACKTEST_MAX_COUNT_DUP)

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

    def perform_sell_signal(self, coin):
        romeo = self.romeo_pool.get(coin)
        if romeo is not None:
            romeo.perform_sell_signal()

    def remove_from_pool(self, coin):
        romeo = self.romeo_pool.get(coin)
        if romeo is not None:
            del self.romeo_pool[coin]



            
