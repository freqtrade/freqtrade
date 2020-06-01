""" Binance exchange subclass """
import logging
from typing import Dict

import ccxt

from freqtrade.exceptions import (DependencyException, InvalidOrderException,
                                  OperationalException, TemporaryError)
from freqtrade.exchange import Exchange

logger = logging.getLogger(__name__)


class Binance(Exchange):

    _ft_has: Dict = {
        "stoploss_on_exchange": True,
        "order_time_in_force": ['gtc', 'fok', 'ioc'],
        "trades_pagination": "id",
        "trades_pagination_arg": "fromId",
    }

    def fetch_l2_order_book(self, pair: str, limit: int = 100) -> dict:
        """
        get order book level 2 from exchange

        20180619: binance support limits but only on specific range
        """
        limit_range = [5, 10, 20, 50, 100, 500, 1000]
        # get next-higher step in the limit_range list
        limit = min(list(filter(lambda x: limit <= x, limit_range)))

        return super().fetch_l2_order_book(pair, limit)

    def stoploss_adjust(self, stop_loss: float, order: Dict) -> bool:
        """
        Verify stop_loss against stoploss-order value (limit or price)
        Returns True if adjustment is necessary.
        """
        return order['type'] == 'stop_loss_limit' and stop_loss > float(order['info']['stopPrice'])

    def stoploss(self, pair: str, amount: float, stop_price: float, order_types: Dict) -> Dict:
        """
        creates a stoploss limit order.
        this stoploss-limit is binance-specific.
        It may work with a limited number of other exchanges, but this has not been tested yet.
        """
        # Limit price threshold: As limit price should always be below stop-price
        limit_price_pct = order_types.get('stoploss_on_exchange_limit_ratio', 0.99)
        rate = stop_price * limit_price_pct

        ordertype = "stop_loss_limit"

        stop_price = self.price_to_precision(pair, stop_price)

        # Ensure rate is less than stop price
        if stop_price <= rate:
            raise OperationalException(
                'In stoploss limit order, stop price should be more than limit price')

        if self._config['dry_run']:
            dry_order = self.dry_run_order(
                pair, ordertype, "sell", amount, stop_price)
            return dry_order

        try:
            params = self._params.copy()
            params.update({'stopPrice': stop_price})

            amount = self.amount_to_precision(pair, amount)

            rate = self.price_to_precision(pair, rate)

            order = self._api.create_order(symbol=pair, type=ordertype, side='sell',
                                           amount=amount, price=rate, params=params)
            logger.info('stoploss limit order added for %s. '
                        'stop price: %s. limit: %s', pair, stop_price, rate)
            return order
        except ccxt.InsufficientFunds as e:
            raise DependencyException(
                f'Insufficient funds to create {ordertype} sell order on market {pair}.'
                f'Tried to sell amount {amount} at rate {rate}. '
                f'Message: {e}') from e
        except ccxt.InvalidOrder as e:
            # Errors:
            # `binance Order would trigger immediately.`
            raise InvalidOrderException(
                f'Could not create {ordertype} sell order on market {pair}. '
                f'Tried to sell amount {amount} at rate {rate}. '
                f'Message: {e}') from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not place sell order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e
