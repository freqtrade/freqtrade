from wao.brain_util import perform_execute_buy, perform_execute_sell, perform_back_test_buy, perform_back_test_sell, \
    write_to_backtest_table
import threading
from wao.brain_config import BrainConfig
from wao.brain_util import setup_429
from wao.notifier import send_start_deliminator_message
import time
import os


class WAOStrategyController:

    def __init__(self, brain, time_out_hours):
        # romeo_pool: key=coin, value=romeo_instance
        self.romeo_pool = {}
        self.brain = brain
        self.time_out_hours = time_out_hours
        print("WAOStrategyController: __init__: is_backtest=" + str(BrainConfig.IS_BACKTEST))
        setup_429()
        if BrainConfig.IS_BACKTEST:
            send_start_deliminator_message(self.brain, BrainConfig.BACKTEST_COIN,
                                           BrainConfig.BACKTEST_MONTH_LIST[
                                               BrainConfig.BACKTEST_DATA_CLEANER_MONTH_INDEX],
                                           BrainConfig.BACKTEST_DATA_CLEANER_YEAR, BrainConfig.BACKTEST_DUP,
                                           BrainConfig.BACKTEST_MAX_COUNT_DUP)
            # delete cumulative file
            file_name = BrainConfig.CUMULATIVE_PROFIT_FILE_PATH
            if os.path.isfile(file_name):
                os.remove(file_name)

    def on_buy_signal(self, current_time, coin):
        print("WAOStrategyController: on_buy_signal: current_time=" + str(current_time) + ", coin=" + str(coin) +
              ", brain=" + str(self.brain))
        if BrainConfig.IS_BACKTEST:
            write_to_backtest_table(current_time, coin, "buy")
        else:
            self.__buy_execute(coin)

    def on_sell_signal(self, sell_reason, current_time, coin):
        print("WAOStrategyController: on_sell_signal: sell_reason=" + str(sell_reason) + ", current_time=" + str(
            current_time) + ", coin=" + str(coin) + ", brain=" + str(self.brain))
        if BrainConfig.IS_BACKTEST:
            write_to_backtest_table(current_time, coin, "sell")
        else:
            perform_execute_sell(coin, self.romeo_pool)
            self.__remove_from_pool(coin)

    def __buy_execute(self, coin):
        if BrainConfig.IS_PARALLEL_EXECUTION:
            threading.Thread(target=perform_execute_buy,
                             args=(coin, self.brain, self.romeo_pool, self.time_out_hours)).start()
        else:
            perform_execute_buy(coin, self.brain, self.romeo_pool, self.time_out_hours)

    def __remove_from_pool(self, coin):
        if self.is_romeo_alive(coin):
            del self.romeo_pool[coin]

    def is_romeo_alive(self, coin):
        return self.romeo_pool.get(coin) is not None
