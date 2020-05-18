""" Kraken exchange subclass """
import logging
from typing import Dict

import ccxt

from freqtrade.exceptions import (DependencyException, InvalidOrderException,
                                  OperationalException, TemporaryError)
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier

logger = logging.getLogger(__name__)


class Kraken(Exchange):

    _params: Dict = {"trading_agreement": "agree"}
    _ft_has: Dict = {
        "stoploss_on_exchange": True,
        "trades_pagination": "id",
        "trades_pagination_arg": "since",
    }

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
                           x["remaining"],
                           # Don't remove the below comment, this can be important for debuggung
                           # x["side"], x["amount"],
                           ) for x in orders]
            for bal in balances:
                balances[bal]['used'] = sum(order[1] for order in order_list if order[0] == bal)
                balances[bal]['free'] = balances[bal]['total'] - balances[bal]['used']

            return balances
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
        return order['type'] == 'stop-loss' and stop_loss > float(order['price'])

    def stoploss(self, pair: str, amount: float, stop_price: float, order_types: Dict) -> Dict:
        """
        Creates a stoploss market order.
        Stoploss market orders is the only stoploss type supported by kraken.
        """

        ordertype = "stop-loss"

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
