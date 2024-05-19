# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

from strategy_test_v2 import StrategyTestV2

from freqtrade.strategy import BooleanParameter, DecimalParameter, IntParameter, RealParameter


class HyperoptableStrategyV2(StrategyTestV2):
    """
    Default Strategy provided by freqtrade bot.
    Please do not modify this strategy, it's  intended for internal use only.
    Please look at the SampleStrategy in the user_data/strategy directory
    or strategy repository https://github.com/freqtrade/freqtrade-strategies
    for samples and inspiration.
    """

    buy_params = {
        "buy_rsi": 35,
        # Intentionally not specified, so "default" is tested
        # 'buy_plusdi': 0.4
    }

    sell_params = {
        # Sell parameters
        "sell_rsi": 74,
        "sell_minusdi": 0.4,
    }

    buy_plusdi = RealParameter(low=0, high=1, default=0.5, space="buy")
    sell_rsi = IntParameter(low=50, high=100, default=70, space="sell")
    sell_minusdi = DecimalParameter(
        low=0, high=1, default=0.5001, decimals=3, space="sell", load=False
    )
    protection_enabled = BooleanParameter(default=True)
    protection_cooldown_lookback = IntParameter([0, 50], default=30)

    @property
    def protections(self):
        prot = []
        if self.protection_enabled.value:
            prot.append(
                {
                    "method": "CooldownPeriod",
                    "stop_duration_candles": self.protection_cooldown_lookback.value,
                }
            )
        return prot

    bot_loop_started = False

    def bot_loop_start(self, **kwargs):
        self.bot_loop_started = True

    def bot_start(self, **kwargs) -> None:
        """
        Parameters can also be defined here ...
        """
        self.buy_rsi = IntParameter([0, 50], default=30, space="buy")
