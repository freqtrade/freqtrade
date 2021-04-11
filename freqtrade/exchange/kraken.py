""" Kraken exchange subclass """
import logging
from typing import Any, Dict

import ccxt

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
    }

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
                           # Don't remove the below comment, this can be important for debuggung
                           # x["side"], x["amount"],
                           ) for x in orders]
            for bal in balances:
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

    def stoploss_adjust(self, stop_loss: float, order: Dict) -> bool:
        """
        Verify stop_loss against stoploss-order value (limit or price)
        Returns True if adjustment is necessary.
        """
        return (order['type'] in ('stop-loss', 'stop-loss-limit')
                and stop_loss > float(order['price']))

    @retrier(retries=0)
    def stoploss(self, pair: str, amount: float, stop_price: float, order_types: Dict) -> Dict:
        """
        Creates a stoploss market order.
        Stoploss market orders is the only stoploss type supported by kraken.
        """
        params = self._params.copy()

        if order_types.get('stoploss', 'market') == 'limit':
            ordertype = "stop-loss-limit"
            limit_price_pct = order_types.get('stoploss_on_exchange_limit_ratio', 0.99)
            limit_rate = stop_price * limit_price_pct
            params['price2'] = self.price_to_precision(pair, limit_rate)
        else:
            ordertype = "stop-loss"

        stop_price = self.price_to_precision(pair, stop_price)

        if self._config['dry_run']:
            dry_order = self.create_dry_run_order(
                pair, ordertype, "sell", amount, stop_price)
            return dry_order

        try:
            amount = self.amount_to_precision(pair, amount)

            order = self._api.create_order(symbol=pair, type=ordertype, side='sell',
                                           amount=amount, price=stop_price, params=params)
            logger.info('stoploss order added for %s. '
                        'stop price: %s.', pair, stop_price)
            return order
        except ccxt.InsufficientFunds as e:
            raise InsufficientFundsError(
                f'Insufficient funds to create {ordertype} sell order on market {pair}. '
                f'Tried to create stoploss with amount {amount} at stoploss {stop_price}. '
                f'Message: {e}') from e
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f'Could not create {ordertype} sell order on market {pair}. '
                f'Tried to create stoploss with amount {amount} at stoploss {stop_price}. '
                f'Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not place sell order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e
