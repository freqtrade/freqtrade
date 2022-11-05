from datetime import datetime
from typing import Optional
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


from brain_util import perform_execute_buy, perform_execute_sell, write_to_backtest_table, clear_cumulative_value, create_initial_account_balance_binance_file, is_romeo_alive, remove_from_pool
import threading
from brain_config import BrainConfig
from brain_util import create_watchers
from notifier import send_start_deliminator_message
from execution.config import Config



class WAOStrategyController:

    def __init__(self, time_out_hours, dup):
        self.time_out_hours = time_out_hours
        self.dup = dup
        print("WAOStrategyController: __init__: is_backtest=" + str(BrainConfig.IS_BACKTEST))
        create_watchers()
        clear_cumulative_value()
        create_initial_account_balance_binance_file()
        Config.IS_SCHEDULE_ORDER = BrainConfig.IS_SCHEDULE_ORDER
        Config.IS_LIMIT_STOP_ORDER_ENABLED = BrainConfig.IS_LIMIT_STOP_ORDER_ENABLED
        Config.WORKSPACE_NORMAL = BrainConfig.WORKSPACE_NORMAL
        if BrainConfig.IS_BACKTEST:
            send_start_deliminator_message(BrainConfig.BRAIN,
                                           BrainConfig.BACKTEST_MONTH_LIST[
                                               BrainConfig.BACKTEST_DATA_CLEANER_MONTH_INDEX],
                                           BrainConfig.BACKTEST_DATA_CLEANER_YEAR)

    def on_buy_signal(self, current_time, coin):
        print("WAOStrategyController: on_buy_signal: current_time=" + str(current_time) + ", coin=" + str(coin) +
              ", brain=" + str(BrainConfig.BRAIN))

        if is_romeo_alive(coin):
            print("WAOStrategyController: on_buy_signal: warning: alive romeo detected: ignoring buy_signal!")
            return

        if BrainConfig.IS_BACKTEST:
            write_to_backtest_table(current_time, coin, BrainConfig.BRAIN, self.time_out_hours, self.dup, "buy")
        else:
            self.__buy_execute(coin)

    def on_sell_signal(self, sell_reason, current_time, coin):
        print("WAOStrategyController: on_sell_signal: sell_reason=" + str(sell_reason) + ", current_time=" + str(
            current_time) + ", coin=" + str(coin) + ", brain=" + str(BrainConfig.BRAIN))

        if not is_romeo_alive(coin):
            print("WAOStrategyController: on_sell_signal: warning: romeo not alive: empty pool: ignoring sell_signal!")
            return

        if BrainConfig.IS_BACKTEST:
            write_to_backtest_table(current_time, coin, BrainConfig.BRAIN, self.time_out_hours, self.dup, "sell")
        else:
            perform_execute_sell(coin)
            remove_from_pool(coin)

    def __buy_execute(self, coin):
        if Config.IS_PARALLEL_EXECUTION:
            threading.Thread(target=perform_execute_buy,
                             args=(coin, BrainConfig.BRAIN, self.time_out_hours, self.dup)).start()
        else:
            perform_execute_buy(coin, BrainConfig.BRAIN, self.time_out_hours, self.dup)

    @Config.bus.on(Config.EVENT_BUS_EXECUTION_SELF_COMPLETE)
    def on_execution_self_complete(coin):
        print("on_execution_self_complete: " + coin)
        remove_from_pool(coin)


class WAOStrategy(IStrategy):
    # Optional order type mapping
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def __init__(self, config: dict, time_out_hours, dup):
        super().__init__(config)
        self.controller = WAOStrategyController(time_out_hours, dup)

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            **kwargs) -> bool:
        """
        Called right before placing a buy order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be bought.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in target (quote) currency that's going to be traded.
        :param rate: Rate that's going to be used when using limit orders
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param current_time: datetime object, containing the current datetime
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the buy-order is placed on the exchange.
            False aborts the process
        """
        coin = pair.split("/")[0]

        # is_same_coin_trade_open = self.controller.is_romeo_alive(coin)
        # print("WAOStrategy: confirm_trade_entry: is_same_coin_trade_open="+str(is_same_coin_trade_open))
        #
        # if is_same_coin_trade_open:
        #     return False
        # else:
        #     self.controller.on_buy_signal(current_time, coin)
        #     return True
        self.controller.on_buy_signal(current_time, coin)
        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing a regular sell order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be sold.
        :param trade: trade object.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in quote currency.
        :param rate: Rate that's going to be used when using limit orders
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param sell_reason: Sell reason.
            Can be any of ['roi', 'stop_loss', 'stoploss_on_exchange', 'trailing_stop_loss',
                           'sell_signal', 'force_sell', 'emergency_sell']
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the sell-order is placed on the exchange.
            False aborts the process
        """
        coin = pair.split("/")[0]

        self.controller.on_sell_signal(sell_reason, current_time, coin)
        return True
