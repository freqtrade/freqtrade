from wao.brain_util import perform_execute_buy, perform_execute_sell, write_to_backtest_table, clear_cumulative_value
import threading
from wao.brain_config import BrainConfig
from wao.brain_util import setup_429
from wao.notifier import send_start_deliminator_message
import sys
import os

sys.path.append(BrainConfig.EXECUTION_PATH)
from config import Config


class WAOStrategyController:

    def __init__(self, brain, time_out_hours):
        self.brain = brain
        self.time_out_hours = time_out_hours
        print("WAOStrategyController: __init__: is_backtest=" + str(BrainConfig.IS_BACKTEST))
        setup_429()
        clear_cumulative_value()
        if BrainConfig.IS_BACKTEST:
            send_start_deliminator_message(self.brain,
                                           BrainConfig.BACKTEST_MONTH_LIST[
                                               BrainConfig.BACKTEST_DATA_CLEANER_MONTH_INDEX],
                                           BrainConfig.BACKTEST_DATA_CLEANER_YEAR)

    def on_buy_signal(self, current_time, coin):
        print("WAOStrategyController: on_buy_signal: current_time=" + str(current_time) + ", coin=" + str(coin) +
              ", brain=" + str(self.brain))
        if BrainConfig.IS_BACKTEST:
            write_to_backtest_table(current_time, coin, self.brain, self.time_out_hours, "buy")
        else:
            self.__buy_execute(coin)

    def on_sell_signal(self, sell_reason, current_time, coin):
        print("WAOStrategyController: on_sell_signal: sell_reason=" + str(sell_reason) + ", current_time=" + str(
            current_time) + ", coin=" + str(coin) + ", brain=" + str(self.brain))
        if BrainConfig.IS_BACKTEST:
            write_to_backtest_table(current_time, coin, self.brain, self.time_out_hours, "sell")
        else:
            perform_execute_sell(coin)
            self.__remove_from_pool(coin)

    def __buy_execute(self, coin):
        if Config.IS_PARALLEL_EXECUTION:
            threading.Thread(target=perform_execute_buy,
                             args=(coin, self.brain, self.time_out_hours)).start()
        else:
            perform_execute_buy(coin, self.brain, self.time_out_hours)

    def __remove_from_pool(self, coin):
        if self.is_romeo_alive(coin):
            del BrainConfig.ROMEO_POOL[coin]

    def is_romeo_alive(self, coin):
        return

    @Config.bus.on(Config.EVENT_BUS_EXECUTION_SELF_COMPLETE)
    def on_execution_self_complete(coin):
        print("on_execution_self_complete: " + coin)
        if BrainConfig.ROMEO_POOL.get(coin) is not None:
            del BrainConfig.ROMEO_POOL[coin]


