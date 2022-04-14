from wao.brain_util import perform_execute_buy, perform_execute_sell, perform_back_test_buy, perform_back_test_sell
import threading
from wao.brain_config import BrainConfig
from wao.brain_util import setup_429
from wao.notifier import send_start_deliminator_message
import time


class StrategyController:
    # romeo_pool: key=coin+brain, value=romeo_instance
    romeo_pool = {}
    brain = ""

    def setup(self, brain):
        self.brain = brain
        setup_429()
        if BrainConfig.IS_BACKTEST:
            send_start_deliminator_message(brain, BrainConfig.BACKTEST_COIN,
                                           BrainConfig.BACKTEST_MONTH_LIST[
                                               BrainConfig.BACKTEST_DATA_CLEANER_MONTH_INDEX],
                                           BrainConfig.BACKTEST_DATA_CLEANER_YEAR, BrainConfig.BACKTEST_DUP,
                                           BrainConfig.BACKTEST_MAX_COUNT_DUP)

    def on_buy_signal(self, current_time, mode, coin):
        print("StrategyController: on_buy_signal: current_time=" + str(current_time) + ", mode=" + str(
            mode) + ", coin=" + str(coin) + ", brain=" + str(self.brain))
        if BrainConfig.IS_BACKTEST:
            self.__buy_back_test(current_time, coin)
        else:
            self.__buy_execute(mode, coin)

    def on_sell_signal(self, sell_reason, current_time, mode, coin):
        print("StrategyController: on_sell_signal: sell_reason=" + str(sell_reason) + ", current_time=" + str(
            current_time) + ", mode=" + str(mode) + ", coin=" + str(coin) + ", brain=" + str(self.brain))
        if sell_reason == 'sell_signal' or sell_reason == 'roi':
            if BrainConfig.IS_BACKTEST:
                perform_back_test_sell(current_time)
            else:
                perform_execute_sell(coin, brain, self.romeo_pool)

        self.__remove_from_pool(coin, brain)

    def __buy_back_test(self, date_time, coin):
        time.sleep(BrainConfig.BACKTEST_THROTTLE_SECOND)
        if BrainConfig.IS_PARALLEL_EXECUTION:
            threading.Thread(target=perform_back_test_buy, args=(date_time, coin, self.brain, self.romeo_pool)).start()
        else:
            perform_back_test_buy(date_time, coin, self.brain, self.romeo_pool)

    def __buy_execute(self, mode, coin):
        if BrainConfig.IS_PARALLEL_EXECUTION:
            threading.Thread(target=perform_execute_buy, args=(mode, coin, self.brain, self.romeo_pool)).start()
        else:
            perform_execute_buy(mode, coin, self.brain, self.romeo_pool)

    def __remove_from_pool(self, coin, brain):
        romeo = self.romeo_pool.get(coin + brain)
        if romeo is not None:
            del self.romeo_pool[coin + brain]
