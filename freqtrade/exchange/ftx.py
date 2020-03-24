""" FTX exchange subclass """
import logging
from typing import Dict

import ccxt

from freqtrade.exceptions import (DependencyException, InvalidOrderException,
                                  OperationalException, TemporaryError)
from freqtrade.exchange import Exchange

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
        Creates a stoploss market order.
        Stoploss market orders is the only stoploss type supported by kraken.
        """

        ordertype = "stop"

        stop_price = self.price_to_precision(pair, stop_price)

        if self._config['dry_run']:
            dry_order = self.dry_run_order(
                pair, ordertype, "sell", amount, stop_price)
            return dry_order

        try:
            params = self._params.copy()

            amount = self.amount_to_precision(pair, amount)

            order = self._api.create_order(symbol=pair, type=ordertype, side='sell',
                                           amount=amount, price=stop_price, params=params)
            logger.info('stoploss order added for %s. '
                        'stop price: %s.', pair, stop_price)
            return order
        except ccxt.InsufficientFunds as e:
            raise DependencyException(
                f'Insufficient funds to create {ordertype} sell order on market {pair}.'
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
