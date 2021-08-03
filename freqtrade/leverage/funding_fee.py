from datetime import datetime, timedelta
from typing import List

import schedule

from freqtrade.persistence import Trade


class FundingFee:

    trades: List[Trade]
    # Binance
    begin_times = [
        # TODO-lev: Make these UTC time
        "23:59:45",
        "07:59:45",
        "15:59:45",
    ]

    # FTX
    # begin_times = every hour

    def _is_time_between(self, begin_time, end_time):
        # If check time is not given, default to current UTC time
        check_time = datetime.utcnow().time()
        if begin_time < end_time:
            return check_time >= begin_time and check_time <= end_time
        else:  # crosses midnight
            return check_time >= begin_time or check_time <= end_time

    def _apply_funding_fees(self, num_of: int = 1):
        if num_of == 0:
            return
        for trade in self.trades:
            trade.adjust_funding_fee(self._calculate(trade.amount) * num_of)

    def _calculate(self, amount):
        # TODO-futures: implement
        # TODO-futures: Check how other exchages do it and adjust accordingly
        # https://www.binance.com/en/support/faq/360033525031
        # mark_price =
        # contract_size = maybe trade.amount
        # funding_rate =  # https://www.binance.com/en/futures/funding-history/0
        # nominal_value = mark_price * contract_size
        # adjustment = nominal_value * funding_rate
        # return adjustment

        # FTX - paid in USD(always)
        # position size * TWAP of((future - index) / index) / 24
        # https: // help.ftx.com/hc/en-us/articles/360027946571-Funding
        return

    def initial_funding_fee(self, amount) -> float:
        # A funding fee interval is applied immediately if within 30s of an iterval
        # May only exist on binance
        for begin_string in self.begin_times:
            begin_time = datetime.strptime(begin_string, "%H:%M:%S")
            end_time = (begin_time + timedelta(seconds=30))
            if self._is_time_between(begin_time.time(), end_time.time()):
                return self._calculate(amount)
        return 0.0

    def start(self):
        for interval in self.begin_times:
            schedule.every().day.at(interval).do(self._apply_funding_fees())

        # https://stackoverflow.com/a/30393162/6331353
        # TODO-futures: Put schedule.run_pending() somewhere in the bot_loop

    def reboot(self):
        # TODO-futures Find out how many begin_times have passed since last funding_fee added
        amount_missed = 0
        self.apply_funding_fees(num_of=amount_missed)
        self.start()

    def add_new_trade(self, trade):
        self.trades.append(trade)

    def remove_trade(self, trade):
        self.trades.remove(trade)
