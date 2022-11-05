from wao.brain_util import perform_execute_buy, perform_execute_sell, write_to_backtest_table, clear_cumulative_value, create_initial_account_balance_binance_file, is_romeo_alive, remove_from_pool
import threading
from wao.brain_config import *
from wao.brain_util import create_watchers
from wao.notifier import send_start_deliminator_message

from execution.config import Config


class WAOStrategyController:

    def __init__(self, time_out_hours, dup):
        self.time_out_hours = time_out_hours
        self.dup = dup
        print("WAOStrategyController: __init__: is_backtest=" + str(IS_BACKTEST))
        create_watchers()
        clear_cumulative_value()
        create_initial_account_balance_binance_file()
        Config.IS_SCHEDULE_ORDER = IS_SCHEDULE_ORDER
        Config.IS_LIMIT_STOP_ORDER_ENABLED = IS_LIMIT_STOP_ORDER_ENABLED
        Config.WORKSPACE_NORMAL = WORKSPACE_NORMAL
        if IS_BACKTEST:
            send_start_deliminator_message(BRAIN,
                                           BACKTEST_MONTH_LIST[
                                               BACKTEST_DATA_CLEANER_MONTH_INDEX],
                                           BACKTEST_DATA_CLEANER_YEAR)

    def on_buy_signal(self, current_time, coin):
        print("WAOStrategyController: on_buy_signal: current_time=" + str(current_time) + ", coin=" + str(coin) +
              ", brain=" + str(BRAIN))

        if is_romeo_alive(coin):
            print("WAOStrategyController: on_buy_signal: warning: alive romeo detected: ignoring buy_signal!")
            return

        if IS_BACKTEST:
            write_to_backtest_table(current_time, coin, BRAIN, self.time_out_hours, self.dup, "buy")
        else:
            self.__buy_execute(coin)

    def on_sell_signal(self, sell_reason, current_time, coin):
        print("WAOStrategyController: on_sell_signal: sell_reason=" + str(sell_reason) + ", current_time=" + str(
            current_time) + ", coin=" + str(coin) + ", brain=" + str(BRAIN))

        if not is_romeo_alive(coin):
            print("WAOStrategyController: on_sell_signal: warning: romeo not alive: empty pool: ignoring sell_signal!")
            return

        if IS_BACKTEST:
            write_to_backtest_table(current_time, coin, BRAIN, self.time_out_hours, self.dup, "sell")
        else:
            perform_execute_sell(coin)
            remove_from_pool(coin)

    def __buy_execute(self, coin):
        if Config.IS_PARALLEL_EXECUTION:
            threading.Thread(target=perform_execute_buy,
                             args=(coin, BRAIN, self.time_out_hours, self.dup)).start()
        else:
            perform_execute_buy(coin, BRAIN, self.time_out_hours, self.dup)

    @Config.bus.on(Config.EVENT_BUS_EXECUTION_SELF_COMPLETE)
    def on_execution_self_complete(coin):
        print("on_execution_self_complete: " + coin)
        remove_from_pool(coin)


