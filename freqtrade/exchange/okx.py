import logging
from typing import Dict, List, Tuple

from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


class Okx(Exchange):
    """Okx exchange class.

    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    _ft_has: Dict = {
        "ohlcv_candle_limit": 300,
        "mark_ohlcv_timeframe": "4h",
        "funding_fee_timeframe": "8h",
        "can_fetch_multiple_tiers": False,
    }

    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.MARGIN, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED),
    ]

    def _lev_prep(
        self,
        pair: str,
        leverage: float,
        side: str  # buy or sell
    ):
        if self.trading_mode != TradingMode.SPOT:
            if self.margin_mode is None:
                raise OperationalException(
                    f"{self.name}.margin_mode must be set for {self.trading_mode.value}"
                )
            self._api.set_leverage(
                leverage,
                pair,
                params={
                    "mgnMode": self.margin_mode.value,
                    "posSide": "long" if side == "buy" else "short",
                })

    def get_max_pair_stake_amount(
        self,
        pair: str,
        price: float,
        leverage: float = 1.0
    ) -> float:

        if self.trading_mode == TradingMode.SPOT:
            return float('inf')  # Not actually inf, but this probably won't matter for SPOT

        if pair not in self._leverage_tiers:
            return float('inf')

        pair_tiers = self._leverage_tiers[pair]
        return pair_tiers[-1]['max'] / leverage

    def load_leverage_tiers(self) -> Dict[str, List[Dict]]:
        # * This is slow(~45s) on Okex, must make 90-some api calls to load all linear swap markets
        if self.trading_mode == TradingMode.FUTURES:
            markets = self.markets
            symbols = []

            for symbol, market in markets.items():
                if (self.market_is_future(market)
                        and market['quote'] == self._config['stake_currency']):
                    symbols.append(symbol)

            tiers: Dict[str, List[Dict]] = {}

            for symbol in symbols:
                res = self._api.fetchLeverageTiers(symbol)
                tiers[symbol] = []
                for tier in res[symbol]:
                    tiers[symbol].append(self.parse_leverage_tier(tier))

            return tiers
        else:
            return {}
