from freqtrade.enums import LiqFormula, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import Trade


class MaintenanceMargin:

    trades: list[Trade]
    formula: LiqFormula

    @property
    def margin_level(self):
        return self.formula()  # TODO: Add args to formula

    @property
    def liq_level(self):    # This may be a constant value and may not need a function
        return              # If constant, would need to be recalculated after each new trade

    def __init__(self, formula: LiqFormula, trading_mode: TradingMode):
        if (
            trading_mode != TradingMode.CROSS_MARGIN or
            trading_mode != TradingMode.CROSS_FUTURES
        ):
            raise OperationalException("Maintenance margin should only be used for cross trading")
        return

    def add_new_trade(self, trade):
        return

    def remove_trade(self, trade):
        return

    # ? def update_trade_pric(self):

    def sell_all(self):
        return

    def run(self):
        # TODO-mg: implement a thread that constantly updates with every price change,
        # TODO-mg: must update at least every second
        # while true:
        #   if self.margin_level <= self.liq_level:
        #       self.sell_all()
        return
