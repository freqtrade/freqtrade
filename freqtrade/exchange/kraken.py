""" Kraken exchange subclass """
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import ccxt
from pandas import DataFrame

from freqtrade.enums import Collateral, TradingMode
from freqtrade.exceptions import (DDosProtection, InsufficientFundsError, InvalidOrderException,
                                  OperationalException, TemporaryError)
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier


logger = logging.getLogger(__name__)


class Kraken(Exchange):

    _params: Dict = {"trading_agreement": "agree"}
    _ft_has: Dict = {
        "stoploss_on_exchange": True,
        "ohlcv_candle_limit": 720,
        "trades_pagination": "id",
        "trades_pagination_arg": "since",
        "mark_ohlcv_timeframe": "4h",
    }

    _supported_trading_mode_collateral_pairs: List[Tuple[TradingMode, Collateral]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.MARGIN, Collateral.CROSS),
        # (TradingMode.FUTURES, Collateral.CROSS)
    ]

    def market_is_tradable(self, market: Dict[str, Any]) -> bool:
        """
        Check if the market symbol is tradable by Freqtrade.
        Default checks + check if pair is darkpool pair.
        """
        parent_check = super().market_is_tradable(market)

        return (parent_check and
                market.get('darkpool', False) is False)

    @retrier
    def get_balances(self) -> dict:
        if self._config['dry_run']:
            return {}

        try:
            balances = self._api.fetch_balance()
            # Remove additional info from ccxt results
            balances.pop("info", None)
            balances.pop("free", None)
            balances.pop("total", None)
            balances.pop("used", None)

            orders = self._api.fetch_open_orders()
            order_list = [(x["symbol"].split("/")[0 if x["side"] == "sell" else 1],
                           x["remaining"] if x["side"] == "sell" else x["remaining"] * x["price"],
                           # Don't remove the below comment, this can be important for debugging
                           # x["side"], x["amount"],
                           ) for x in orders]
            for bal in balances:
                if not isinstance(balances[bal], dict):
                    continue
                balances[bal]['used'] = sum(order[1] for order in order_list if order[0] == bal)
                balances[bal]['free'] = balances[bal]['total'] - balances[bal]['used']

            return balances
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get balance due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def stoploss_adjust(self, stop_loss: float, order: Dict, side: str) -> bool:
        """
        Verify stop_loss against stoploss-order value (limit or price)
        Returns True if adjustment is necessary.
        """
        return (order['type'] in ('stop-loss', 'stop-loss-limit') and (
                (side == "sell" and stop_loss > float(order['price'])) or
                (side == "buy" and stop_loss < float(order['price']))
                ))

    @retrier(retries=0)
    def stoploss(self, pair: str, amount: float, stop_price: float,
                 order_types: Dict, side: str, leverage: float) -> Dict:
        """
        Creates a stoploss market order.
        Stoploss market orders is the only stoploss type supported by kraken.
        """
        params = self._params.copy()

        if order_types.get('stoploss', 'market') == 'limit':
            ordertype = "stop-loss-limit"
            limit_price_pct = order_types.get('stoploss_on_exchange_limit_ratio', 0.99)
            if side == "sell":
                limit_rate = stop_price * limit_price_pct
            else:
                limit_rate = stop_price * (2 - limit_price_pct)
            params['price2'] = self.price_to_precision(pair, limit_rate)
        else:
            ordertype = "stop-loss"

        stop_price = self.price_to_precision(pair, stop_price)

        if self._config['dry_run']:
            dry_order = self.create_dry_run_order(
                pair, ordertype, side, amount, stop_price, leverage)
            return dry_order

        try:
            amount = self.amount_to_precision(pair, amount)

            order = self._api.create_order(symbol=pair, type=ordertype, side=side,
                                           amount=amount, price=stop_price, params=params)
            self._log_exchange_response('create_stoploss_order', order)
            logger.info('stoploss order added for %s. '
                        'stop price: %s.', pair, stop_price)
            return order
        except ccxt.InsufficientFunds as e:
            raise InsufficientFundsError(
                f'Insufficient funds to create {ordertype} {side} order on market {pair}. '
                f'Tried to create stoploss with amount {amount} at stoploss {stop_price}. '
                f'Message: {e}') from e
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f'Could not create {ordertype} {side} order on market {pair}. '
                f'Tried to create stoploss with amount {amount} at stoploss {stop_price}. '
                f'Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not place {side} order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _set_leverage(
        self,
        leverage: float,
        pair: Optional[str] = None,
        trading_mode: Optional[TradingMode] = None
    ):
        """
        Kraken set's the leverage as an option in the order object, so we need to
        add it to params
        """
        return

    def _get_params(self, ordertype: str, leverage: float, time_in_force: str = 'gtc') -> Dict:
        params = super()._get_params(ordertype, leverage, time_in_force)
        if leverage > 1.0:
            params['leverage'] = leverage
        return params

    def _calculate_funding_fees(
        self,
        funding_rates: DataFrame,
        mark_rates: DataFrame,
        amount: float,
        open_date: datetime,
        close_date: Optional[datetime] = None,
        time_in_ratio: Optional[float] = None
    ) -> float:
        """
        # ! This method will always error when run by Freqtrade because time_in_ratio is never
        # ! passed to _get_funding_fee. For kraken futures to work in dry run and backtesting
        # ! functionality must be added that passes the parameter time_in_ratio to
        # ! _get_funding_fee when using Kraken
        calculates the sum of all funding fees that occurred for a pair during a futures trade
        :param funding_rates: Dataframe containing Funding rates (Type FUNDING_RATE)
        :param mark_rates: Dataframe containing Mark rates (Type mark_ohlcv_price)
        :param amount: The quantity of the trade
        :param open_date: The date and time that the trade started
        :param close_date: The date and time that the trade ended
        :param time_in_ratio: Not used by most exchange classes
        """
        if not time_in_ratio:
            raise OperationalException(
                f"time_in_ratio is required for {self.name}._get_funding_fee")
        fees: float = 0

        df = funding_rates.merge(mark_rates, on='date', how="inner", suffixes=["_fund", "_mark"])
        if not df.empty:
            df = df[(df['date'] >= open_date) & (df['date'] <= close_date)]
            fees = sum(df['open_fund'] * df['open_mark'] * amount * time_in_ratio)

        return fees
