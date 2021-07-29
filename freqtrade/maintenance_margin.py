from freqtrade.enums import MaintenanceMarginFormula
from freqtrade.persistence import Trade


class MaintenanceMargin:

    trades: list[Trade]
    formula: MaintenanceMarginFormula

    @property
    def margin_level(self):
        return self.formula()  # TODO: Add args to formula

    @property
    def liq_level(self):    # This may be a constant value and may not need a function
        return

    def __init__(self, formula: MaintenanceMarginFormula):
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
