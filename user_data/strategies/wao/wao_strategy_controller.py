from wao.brain_util import perform_execute_buy, perform_execute_sell, write_to_backtest_table, clear_cumulative_value, create_initial_account_balance_binance_file, is_romeo_alive, remove_from_pool
import threading
from wao.brain_config import BrainConfig
from wao.brain_util import create_watchers
from execution.config import Config
from execution.notifier import Notifier


class WAOStrategyController:

    def __init__(self, time_out_hours, dup):
        Config.BRAIN = BrainConfig.BRAIN
        Config.ROMEO_SS_TIMEOUT_HOURS = time_out_hours
        self.dup = dup
        self.notifier = Notifier(BrainConfig.MODE)
        self.execution_index = 0
        print("WAOStrategyController: __init__: is_backtest=" + str(BrainConfig.IS_BACKTEST) +
              ", system.version=" + str(Config.VERSION) + ", brain=" + Config.BRAIN +
              ", system_ss_timeout_hour=" + str(Config.ROMEO_SS_TIMEOUT_HOURS))
        create_watchers(self.notifier)
        clear_cumulative_value()
        create_initial_account_balance_binance_file()
        Config.IS_SCHEDULE_ORDER = BrainConfig.IS_SCHEDULE_ORDER
        Config.IS_LIMIT_STOP_ORDER_ENABLED = BrainConfig.IS_LIMIT_STOP_ORDER_ENABLED
        Config.WORKSPACE_NORMAL = BrainConfig.WORKSPACE_NORMAL
        if BrainConfig.IS_BACKTEST:
            self.__send_start_deliminator_message(BrainConfig.BRAIN,
                                        BrainConfig.BACKTEST_MONTH_LIST[BrainConfig.BACKTEST_DATA_CLEANER_MONTH_INDEX],
                                        BrainConfig.BACKTEST_DATA_CLEANER_YEAR)

    def on_buy_signal(self, current_time, coin):
        print("WAOStrategyController: on_buy_signal: current_time=" + str(current_time) + ", coin=" + str(coin) +
              ", brain=" + str(BrainConfig.BRAIN))

        if is_romeo_alive(coin):
            print("WAOStrategyController: on_buy_signal: warning: alive romeo detected: ignoring buy_signal!")
            return

        if BrainConfig.IS_BACKTEST:
            write_to_backtest_table(current_time, coin, self.dup, "buy")
        else:
            self.__buy_execute(coin)

    def on_sell_signal(self, sell_reason, current_time, coin):
        print("WAOStrategyController: on_sell_signal: sell_reason=" + str(sell_reason) + ", current_time=" + str(
            current_time) + ", coin=" + str(coin) + ", brain=" + str(BrainConfig.BRAIN))

        if not is_romeo_alive(coin):
            print("WAOStrategyController: on_sell_signal: warning: romeo not alive: empty pool: ignoring sell_signal!")
            return

        if BrainConfig.IS_BACKTEST:
            write_to_backtest_table(current_time, coin, self.dup, "sell")
        else:
            perform_execute_sell(coin)
            remove_from_pool(coin)

    def __buy_execute(self, coin):
        self.execution_index += 1
        if Config.IS_PARALLEL_EXECUTION:
            threading.Thread(target=perform_execute_buy,
                             args=(coin, self.dup, self.execution_index)).start()
        else:
            perform_execute_buy(coin, self.dup, self.execution_index)

    def __send_start_deliminator_message(self, brain, month, year):
        print("WAOStrategyController: send_start_deliminator_message: ")
        text = "========" + str(brain) + " " + str(month) + " " + str(year) + "=======>"
        self.notifier.post_request(text)

    @Config.bus.on(Config.EVENT_BUS_EXECUTION_SELF_COMPLETE)
    def on_execution_self_complete(coin):
        print("on_execution_self_complete: " + coin)
        remove_from_pool(coin)





