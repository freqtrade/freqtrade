""" FTX exchange subclass """
import logging
from typing import Dict

import ccxt

from freqtrade.exceptions import (DependencyException, InvalidOrderException,
                                  OperationalException, TemporaryError)
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier

logger = logging.getLogger(__name__)


class Ftx(Exchange):

    _ft_has: Dict = {
        "stoploss_on_exchange": True,
        "ohlcv_candle_limit": 1500,
    }

    def stoploss_adjust(self, stop_loss: float, order: Dict) -> bool:
        """
        Verify stop_loss against stoploss-order value (limit or price)
        Returns True if adjustment is necessary.
        """
        return order['type'] == 'stop' and stop_loss > float(order['price'])

    def stoploss(self, pair: str, amount: float, stop_price: float, order_types: Dict) -> Dict:
        """
        Creates a stoploss order.
        depending on order_types.stoploss configuration, uses 'market' or limit order.

        Limit orders are defined by having orderPrice set, otherwise a market order is used.
        """
        limit_price_pct = order_types.get('stoploss_on_exchange_limit_ratio', 0.99)
        limit_rate = stop_price * limit_price_pct

        ordertype = "stop"

        stop_price = self.price_to_precision(pair, stop_price)

        if self._config['dry_run']:
            dry_order = self.dry_run_order(
                pair, ordertype, "sell", amount, stop_price)
            return dry_order

        try:
            params = self._params.copy()
            if order_types.get('stoploss', 'market') == 'limit':
                # set orderPrice to place limit order, otherwise it's a market order
                params['orderPrice'] = limit_rate

            amount = self.amount_to_precision(pair, amount)

            order = self._api.create_order(symbol=pair, type=ordertype, side='sell',
                                           amount=amount, price=stop_price, params=params)
            logger.info('stoploss order added for %s. '
                        'stop price: %s.', pair, stop_price)
            return order
        except ccxt.InsufficientFunds as e:
            raise DependencyException(
                f'Insufficient funds to create {ordertype} sell order on market {pair}. '
                f'Tried to create stoploss with amount {amount} at stoploss {stop_price}. '
                f'Message: {e}') from e
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f'Could not create {ordertype} sell order on market {pair}. '
                f'Tried to create stoploss with amount {amount} at stoploss {stop_price}. '
                f'Message: {e}') from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not place sell order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_stoploss_order(self, order_id: str, pair: str) -> Dict:
        if self._config['dry_run']:
            try:
                order = self._dry_run_open_orders[order_id]
                return order
            except KeyError as e:
                # Gracefully handle errors with dry-run orders.
                raise InvalidOrderException(
                    f'Tried to get an invalid dry-run-order (id: {order_id}). Message: {e}') from e
        try:
            orders = self._api.fetch_orders(pair, None, params={'type': 'stop'})

            order = [order for order in orders if order['id'] == order_id]
            if len(order) == 1:
                return order[0]
            else:
                raise InvalidOrderException(f"Could not get stoploss order for id {order_id}")

        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f'Tried to get an invalid order (id: {order_id}). Message: {e}') from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def cancel_stoploss_order(self, order_id: str, pair: str) -> Dict:
        if self._config['dry_run']:
            return {}
        try:
            return self._api.cancel_order(order_id, pair, params={'type': 'stop'})
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f'Could not cancel order. Message: {e}') from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not cancel order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e
