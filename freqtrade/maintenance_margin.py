from typing import List

from freqtrade.enums import TradingMode
from freqtrade.leverage import liquidation_price
from freqtrade.persistence import Trade


class MaintenanceMargin:

    trades: List[Trade]
    exchange_name: str
    trading_mode: TradingMode

    @property
    def margin_level(self):
        # This is the current value of all assets,
        # and if you pass below liq_level, you are liquidated
        # TODO: Add args to formula
        return liquidation_price(
            trading_mode=self.trading_mode,
            exchange_name=self.exchange_name
        )

    @property
    def liq_level(self):    # This may be a constant value and may not need a function
        # TODO-lev: The is the value that you are liquidated at
        return              # If constant, would need to be recalculated after each new trade

    def __init__(self, exchange_name: str, trading_mode: TradingMode):
        self.exchange_name = exchange_name
        self.trading_mode = trading_mode
        return

    def add_new_trade(self, trade):
        self.trades.append(trade)

    def remove_trade(self, trade):
        self.trades.remove(trade)

    # ? def update_trade_pric(self):

    def sell_all(self):
        # TODO-lev
        return

    def run(self):
        # TODO-lev: implement a thread that constantly updates with every price change,
        # TODO-lev: must update at least every few seconds or so
        # while true:
        #   if self.margin_level <= self.liq_level:
        #       self.sell_all()
        return
