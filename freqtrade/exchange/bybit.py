""" Bybit exchange subclass """
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import ccxt

from freqtrade.constants import BuySell
from freqtrade.enums import CandleType, MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, ExchangeError, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.util.datetime_helpers import dt_now, dt_ts


logger = logging.getLogger(__name__)


class Bybit(Exchange):
    """
    Bybit exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """
    unified_account = False

    _ft_has: Dict = {
        "ohlcv_candle_limit": 1000,
        "ohlcv_has_history": True,
        "order_time_in_force": ["GTC", "FOK", "IOC", "PO"],
    }
    _ft_has_futures: Dict = {
        "ohlcv_has_history": True,
        "mark_ohlcv_timeframe": "4h",
        "funding_fee_timeframe": "8h",
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        # bybit response parsing fails to populate stopLossPrice
        "stop_price_prop": "stopPrice",
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
            if not self._config['dry_run']:
                if self.trading_mode == TradingMode.FUTURES:
                    position_mode = self._api.set_position_mode(False)
                    self._log_exchange_response('set_position_mode', position_mode)
                is_unified = self._api.is_unified_enabled()
                # Returns a tuple of bools, first for margin, second for Account
                if is_unified and len(is_unified) > 1 and is_unified[1]:
                    self.unified_account = True
                    logger.info("Bybit: Unified account.")
                    raise OperationalException("Bybit: Unified account is not supported. "
                                               "Please use a standard (sub)account.")
                else:
                    self.unified_account = False
                    logger.info("Bybit: Standard account.")
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Error in additional_exchange_init due to {e.__class__.__name__}. Message: {e}'
                ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def ohlcv_candle_limit(
            self, timeframe: str, candle_type: CandleType, since_ms: Optional[int] = None) -> int:

        if candle_type in (CandleType.FUNDING_RATE):
            return 200

        return super().ohlcv_candle_limit(timeframe, candle_type, since_ms)

    def _lev_prep(self, pair: str, leverage: float, side: BuySell, accept_fail: bool = False):
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
            try:
                return self._fetch_and_calculate_funding_fees(
                        pair, amount, is_short, open_date)
            except ExchangeError:
                logger.warning(f"Could not update funding fees for {pair}.")
        return 0.0

    def fetch_orders(self, pair: str, since: datetime, params: Optional[Dict] = None) -> List[Dict]:
        """
        Fetch all orders for a pair "since"
        :param pair: Pair for the query
        :param since: Starting time for the query
        """
        # On bybit, the distance between since and "until" can't exceed 7 days.
        # we therefore need to split the query into multiple queries.
        orders = []

        while since < dt_now():
            until = since + timedelta(days=7, minutes=-1)
            orders += super().fetch_orders(pair, since, params={'until': dt_ts(until)})
            since = until

        return orders

    def fetch_order(self, order_id: str, pair: str, params: Dict = {}) -> Dict:
        order = super().fetch_order(order_id, pair, params)
        if (
            order.get('status') == 'canceled'
            and order.get('filled') == 0.0
            and order.get('remaining') == 0.0
        ):
            # Canceled orders will have "remaining=0" on bybit.
            order['remaining'] = None
        return order
