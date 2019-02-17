""" Kraken exchange subclass """
import logging
from random import randint
from typing import Dict

import arrow
import ccxt

from freqtrade import OperationalException, DependencyException, TemporaryError
from freqtrade.exchange import Exchange

logger = logging.getLogger(__name__)


class Kraken(Exchange):

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        self._params = {"trading_agreement": "agree"}

    def buy(self, pair: str, ordertype: str, amount: float,
            rate: float, time_in_force) -> Dict:
        if self._conf['dry_run']:
            order_id = f'dry_run_buy_{randint(0, 10**6)}'
            self._dry_run_open_orders[order_id] = {
                'pair': pair,
                'price': rate,
                'amount': amount,
                'type': ordertype,
                'side': 'buy',
                'remaining': 0.0,
                'datetime': arrow.utcnow().isoformat(),
                'status': 'closed',
                'fee': None
            }
            return {'id': order_id}

        try:
            # Set the precision for amount and price(rate) as accepted by the exchange
            amount = self.symbol_amount_prec(pair, amount)
            rate = self.symbol_price_prec(pair, rate) if ordertype != 'market' else None

            params = self._params.copy()
            if time_in_force != 'gtc':
                params.update({'timeInForce': time_in_force})

            return self._api.create_order(pair, ordertype, 'buy',
                                          amount, rate, params)

        except ccxt.InsufficientFunds as e:
            raise DependencyException(
                f'Insufficient funds to create limit buy order on market {pair}.'
                f'Tried to buy amount {amount} at rate {rate} (total {rate*amount}).'
                f'Message: {e}')
        except ccxt.InvalidOrder as e:
            raise DependencyException(
                f'Could not create limit buy order on market {pair}.'
                f'Tried to buy amount {amount} at rate {rate} (total {rate*amount}).'
                f'Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not place buy order due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    def sell(self, pair: str, ordertype: str, amount: float,
             rate: float, time_in_force='gtc') -> Dict:
        if self._conf['dry_run']:
            order_id = f'dry_run_sell_{randint(0, 10**6)}'
            self._dry_run_open_orders[order_id] = {
                'pair': pair,
                'price': rate,
                'amount': amount,
                'type': ordertype,
                'side': 'sell',
                'remaining': 0.0,
                'datetime': arrow.utcnow().isoformat(),
                'status': 'closed'
            }
            return {'id': order_id}

        try:
            # Set the precision for amount and price(rate) as accepted by the exchange
            amount = self.symbol_amount_prec(pair, amount)
            rate = self.symbol_price_prec(pair, rate) if ordertype != 'market' else None

            params = self._params.copy()
            if time_in_force != 'gtc':
                params.update({'timeInForce': time_in_force})

            return self._api.create_order(pair, ordertype, 'sell',
                                          amount, rate, params)

        except ccxt.InsufficientFunds as e:
            raise DependencyException(
                f'Insufficient funds to create limit sell order on market {pair}.'
                f'Tried to sell amount {amount} at rate {rate} (total {rate*amount}).'
                f'Message: {e}')
        except ccxt.InvalidOrder as e:
            raise DependencyException(
                f'Could not create limit sell order on market {pair}.'
                f'Tried to sell amount {amount} at rate {rate} (total {rate*amount}).'
                f'Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not place sell order due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    def stoploss_limit(self, pair: str, amount: float, stop_price: float, rate: float) -> Dict:
        """
        creates a stoploss limit order.
        NOTICE: it is not supported by all exchanges. only binance is tested for now.
        """

        # Set the precision for amount and price(rate) as accepted by the exchange
        amount = self.symbol_amount_prec(pair, amount)
        rate = self.symbol_price_prec(pair, rate)
        stop_price = self.symbol_price_prec(pair, stop_price)

        # Ensure rate is less than stop price
        if stop_price <= rate:
            raise OperationalException(
                'In stoploss limit order, stop price should be more than limit price')

        if self._conf['dry_run']:
            order_id = f'dry_run_buy_{randint(0, 10**6)}'
            self._dry_run_open_orders[order_id] = {
                'info': {},
                'id': order_id,
                'pair': pair,
                'price': stop_price,
                'amount': amount,
                'type': 'stop_loss_limit',
                'side': 'sell',
                'remaining': amount,
                'datetime': arrow.utcnow().isoformat(),
                'status': 'open',
                'fee': None
            }
            return self._dry_run_open_orders[order_id]

        try:

            params = self._params.copy()
            params.update({'stopPrice': stop_price})

            order = self._api.create_order(pair, 'stop_loss_limit', 'sell',
                                           amount, rate, params)
            logger.info('stoploss limit order added for %s. '
                        'stop price: %s. limit: %s' % (pair, stop_price, rate))
            return order

        except ccxt.InsufficientFunds as e:
            raise DependencyException(
                f'Insufficient funds to place stoploss limit order on market {pair}. '
                f'Tried to put a stoploss amount {amount} with '
                f'stop {stop_price} and limit {rate} (total {rate*amount}).'
                f'Message: {e}')
        except ccxt.InvalidOrder as e:
            raise DependencyException(
                f'Could not place stoploss limit order on market {pair}.'
                f'Tried to place stoploss amount {amount} with '
                f'stop {stop_price} and limit {rate} (total {rate*amount}).'
                f'Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not place stoploss limit order due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)
