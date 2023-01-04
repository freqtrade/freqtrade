""" Bybit exchange subclass """
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_utils import timeframe_to_msecs


logger = logging.getLogger(__name__)


class Bybit(Exchange):
    """
    Bybit exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """

    _ft_has: Dict = {
        "ohlcv_candle_limit": 1000,
        "ccxt_futures_name": "linear",
        "ohlcv_has_history": False,
    }
    _ft_has_futures: Dict = {
        "ohlcv_candle_limit": 200,
        "ohlcv_has_history": True,
        "mark_ohlcv_timeframe": "4h",
    }

    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.FUTURES, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED)
    ]

    @property
    def _ccxt_config(self) -> Dict:
        # Parameters to add directly to ccxt sync/async initialization.
        # ccxt defaults to swap mode.
        config = {}
        if self.trading_mode == TradingMode.SPOT:
            config.update({
                "options": {
                    "defaultType": "spot"
                }
            })
        config.update(super()._ccxt_config)
        return config

    def market_is_future(self, market: Dict[str, Any]) -> bool:
        main = super().market_is_future(market)
        # For ByBit, we'll only support USDT markets for now.
        return (
            main and market['settle'] == 'USDT'
        )

    async def _fetch_funding_rate_history(
        self,
        pair: str,
        timeframe: str,
        limit: int,
        since_ms: Optional[int] = None,
    ) -> List[List]:
        """
        Fetch funding rate history
        Necessary workaround until https://github.com/ccxt/ccxt/issues/15990 is fixed.
        """
        params = {}
        if since_ms:
            until = since_ms + (timeframe_to_msecs(timeframe) * self._ft_has['ohlcv_candle_limit'])
            params.update({'until': until})
        # Funding rate
        data = await self._api_async.fetch_funding_rate_history(
            pair, since=since_ms,
            params=params)
        # Convert funding rate to candle pattern
        data = [[x['timestamp'], x['fundingRate'], 0, 0, 0, 0] for x in data]
        return data

    def _lev_prep(self, pair: str, leverage: float, side: BuySell):
        if self.trading_mode != TradingMode.SPOT:
            params = {'leverage': leverage}
            self.set_margin_mode(pair, self.margin_mode, accept_fail=True, params=params)
            self._set_leverage(leverage, pair, accept_fail=True)

    def _get_params(
        self,
        side: BuySell,
        ordertype: str,
        leverage: float,
        reduceOnly: bool,
        time_in_force: str = 'GTC',
    ) -> Dict:
        params = super()._get_params(
            side=side,
            ordertype=ordertype,
            leverage=leverage,
            reduceOnly=reduceOnly,
            time_in_force=time_in_force,
        )
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode:
            params['position_idx'] = 0
        return params

    def dry_run_liquidation_price(
        self,
        pair: str,
        open_rate: float,   # Entry price of position
        is_short: bool,
        amount: float,
        stake_amount: float,
        leverage: float,
        wallet_balance: float,  # Or margin balance
        mm_ex_1: float = 0.0,  # (Binance) Cross only
        upnl_ex_1: float = 0.0,  # (Binance) Cross only
    ) -> Optional[float]:
        """
        Important: Must be fetching data from cached values as this is used by backtesting!
        PERPETUAL:
         bybit:
          https://www.bybithelp.com/HelpCenterKnowledge/bybitHC_Article?language=en_US&id=000001067

        Long:
        Liquidation Price = (
            Entry Price * (1 - Initial Margin Rate + Maintenance Margin Rate)
            - Extra Margin Added/ Contract)
        Short:
        Liquidation Price = (
            Entry Price * (1 + Initial Margin Rate - Maintenance Margin Rate)
            + Extra Margin Added/ Contract)

        Implementation Note: Extra margin is currently not used.

        :param pair: Pair to calculate liquidation price for
        :param open_rate: Entry price of position
        :param is_short: True if the trade is a short, false otherwise
        :param amount: Absolute value of position size incl. leverage (in base currency)
        :param stake_amount: Stake amount - Collateral in settle currency.
        :param leverage: Leverage used for this position.
        :param trading_mode: SPOT, MARGIN, FUTURES, etc.
        :param margin_mode: Either ISOLATED or CROSS
        :param wallet_balance: Amount of margin_mode in the wallet being used to trade
            Cross-Margin Mode: crossWalletBalance
            Isolated-Margin Mode: isolatedWalletBalance
        """

        market = self.markets[pair]
        mm_ratio, _ = self.get_maintenance_ratio_and_amt(pair, stake_amount)

        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:

            if market['inverse']:
                raise OperationalException(
                    "Freqtrade does not yet support inverse contracts")
            initial_margin_rate = 1 / leverage

            # See docstring - ignores extra margin!
            if is_short:
                return open_rate * (1 + initial_margin_rate - mm_ratio)
            else:
                return open_rate * (1 - initial_margin_rate + mm_ratio)

        else:
            raise OperationalException(
                "Freqtrade only supports isolated futures for leverage trading")

    def _get_funding_fees_from_exchange(self, pair: str, since: Union[datetime, int]) -> float:
        """
        Returns the sum of all funding fees that were exchanged for a pair within a timeframe
        Dry-run handling happens as part of _calculate_funding_fees.
        :param pair: (e.g. ADA/USDT)
        :param since: The earliest time of consideration for calculating funding fees,
            in unix time or as a datetime
        """
        # TODO: Workaround for bybit, which has no funding-fees
        return 0
