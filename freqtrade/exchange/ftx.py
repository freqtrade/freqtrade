""" FTX exchange subclass """
import logging
from typing import Any, Dict

import ccxt

from freqtrade.exceptions import (DDosProtection, InsufficientFundsError, InvalidOrderException,
                                  OperationalException, TemporaryError)
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import API_FETCH_ORDER_RETRY_COUNT, retrier
from freqtrade.misc import safe_value_fallback2


logger = logging.getLogger(__name__)


class Ftx(Exchange):

    _ft_has: Dict = {
        "stoploss_on_exchange": True,
        "ohlcv_candle_limit": 1500,
    }

    def market_is_tradable(self, market: Dict[str, Any]) -> bool:
        """
        Check if the market symbol is tradable by Freqtrade.
        Default checks + check if pair is spot pair (no futures trading yet).
        """
        parent_check = super().market_is_tradable(market)

        return (parent_check and
                market.get('spot', False) is True)

    def stoploss_adjust(self, stop_loss: float, order: Dict) -> bool:
        """
        Verify stop_loss against stoploss-order value (limit or price)
        Returns True if adjustment is necessary.
        """
        return order['type'] == 'stop' and stop_loss > float(order['price'])

    @retrier(retries=0)
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
            dry_order = self.create_dry_run_order(
                pair, ordertype, "sell", amount, stop_price)
            return dry_order

        try:
            params = self._params.copy()
            if order_types.get('stoploss', 'market') == 'limit':
                # set orderPrice to place limit order, otherwise it's a market order
                params['orderPrice'] = limit_rate

            params['stopPrice'] = stop_price
            amount = self.amount_to_precision(pair, amount)

            order = self._api.create_order(symbol=pair, type=ordertype, side='sell',
                                           amount=amount, params=params)
            self._log_exchange_response('create_stoploss_order', order)
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

    @retrier(retries=API_FETCH_ORDER_RETRY_COUNT)
    def fetch_stoploss_order(self, order_id: str, pair: str) -> Dict:
        if self._config['dry_run']:
            return self.fetch_dry_run_order(order_id)

        try:
            orders = self._api.fetch_orders(pair, None, params={'type': 'stop'})

            order = [order for order in orders if order['id'] == order_id]
            self._log_exchange_response('fetch_stoploss_order', order)
            if len(order) == 1:
                if order[0].get('status') == 'closed':
                    # Trigger order was triggered ...
                    real_order_id = order[0].get('info', {}).get('orderId')

                    order1 = self._api.fetch_order(real_order_id, pair)
                    self._log_exchange_response('fetch_stoploss_order1', order1)
                    # Fake type to stop - as this was really a stop order.
                    order1['id_stop'] = order1['id']
                    order1['id'] = order_id
                    order1['type'] = 'stop'
                    order1['status_stop'] = 'triggered'
                    return order1
                return order[0]
            else:
                raise InvalidOrderException(f"Could not get stoploss order for id {order_id}")

        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f'Tried to get an invalid order (id: {order_id}). Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
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
            order = self._api.cancel_order(order_id, pair, params={'type': 'stop'})
            self._log_exchange_response('cancel_stoploss_order', order)
            return order
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f'Could not cancel order. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not cancel order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def get_order_id_conditional(self, order: Dict[str, Any]) -> str:
        if order['type'] == 'stop':
            return safe_value_fallback2(order, order, 'id_stop', 'id')
        return order['id']
