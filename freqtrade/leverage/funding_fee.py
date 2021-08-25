from datetime import datetime, timedelta
from typing import List

import schedule

from freqtrade.exchange import Exchange
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
    exchange: Exchange

    # FTX
    # begin_times = every hour

    def __init__(self, exchange: Exchange):
        self.exchange = exchange

    def _is_time_between(self, begin_time, end_time):
        # If check time is not given, default to current UTC time
        check_time = datetime.utcnow().time()
        if begin_time < end_time:
            return check_time >= begin_time and check_time <= end_time
        else:  # crosses midnight
            return check_time >= begin_time or check_time <= end_time

    def _apply_current_funding_fees(self):
        funding_rates = self.exchange.fetch_funding_rates()

        for trade in self.trades:
            funding_rate = funding_rates[trade.pair]
            self._apply_fee_to_trade(funding_rate, trade)

    def _apply_fee_to_trade(self, funding_rate: dict, trade: Trade):

        amount = trade.amount
        mark_price = funding_rate['markPrice']
        rate = funding_rate['fundingRate']
        # index_price = funding_rate['indexPrice']
        # interest_rate = funding_rate['interestRate']

        funding_fee = self.exchange.get_funding_fee(
            amount,
            mark_price,
            rate,
            # interest_rate
            # index_price,
        )

        trade.adjust_funding_fee(funding_fee)

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
