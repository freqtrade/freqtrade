""" Bybit exchange subclass """
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import ccxt

from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
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
        "ohlcv_candle_limit": 200,
        "ohlcv_has_history": False,
    }
    _ft_has_futures: Dict = {
        "ohlcv_has_history": True,
        "mark_ohlcv_timeframe": "4h",
        "funding_fee_timeframe": "8h",
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "stop_price_type_field": "triggerBy",
        "stop_price_type_value_mapping": {
            PriceType.LAST: "LastPrice",
            PriceType.MARK: "MarkPrice",
            PriceType.INDEX: "IndexPrice",
        },
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

    @retrier
    def additional_exchange_init(self) -> None:
        """
        Additional exchange initialization logic.
        .api will be available at this point.
        Must be overridden in child methods if required.
        """
        try:
            if self.trading_mode == TradingMode.FUTURES and not self._config['dry_run']:
                position_mode = self._api.set_position_mode(False)
                self._log_exchange_response('set_position_mode', position_mode)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Error in additional_exchange_init due to {e.__class__.__name__}. Message: {e}'
                ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

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

    def get_funding_fees(
            self, pair: str, amount: float, is_short: bool, open_date: datetime) -> float:
        """
        Fetch funding fees, either from the exchange (live) or calculates them
        based on funding rate/mark price history
        :param pair: The quote/base pair of the trade
        :param is_short: trade direction
        :param amount: Trade amount
        :param open_date: Open date of the trade
        :return: funding fee since open_date
        :raises: ExchangeError if something goes wrong.
        """
        # Bybit does not provide "applied" funding fees per position.
        if self.trading_mode == TradingMode.FUTURES:
            return self._fetch_and_calculate_funding_fees(
                    pair, amount, is_short, open_date)
        return 0.0
