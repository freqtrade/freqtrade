""" Binance exchange subclass """
import logging
from typing import Dict, Optional

import ccxt

from freqtrade.exceptions import (DDosProtection, InsufficientFundsError, InvalidOrderException,
                                  OperationalException, TemporaryError)
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier


logger = logging.getLogger(__name__)


class Binance(Exchange):

    _ft_has: Dict = {
        "stoploss_on_exchange": True,
        "order_time_in_force": ['gtc', 'fok', 'ioc'],
        "time_in_force_parameter": "timeInForce",
        "ohlcv_candle_limit": 1000,
        "trades_pagination": "id",
        "trades_pagination_arg": "fromId",
        "l2_limit_range": [5, 10, 20, 50, 100, 500, 1000],
    }

    def stoploss_adjust(self, stop_loss: float, order: Dict, side: str) -> bool:
        """
        Verify stop_loss against stoploss-order value (limit or price)
        Returns True if adjustment is necessary.
        :param side: "buy" or "sell"
        """
        # TODO-lev: Short support
        return order['type'] == 'stop_loss_limit' and stop_loss > float(order['info']['stopPrice'])

    @retrier(retries=0)
    def stoploss(self, pair: str, amount: float,
                 stop_price: float, order_types: Dict, side: str) -> Dict:
        """
        creates a stoploss limit order.
        this stoploss-limit is binance-specific.
        It may work with a limited number of other exchanges, but this has not been tested yet.
        :param side: "buy" or "sell"
        """
        # TODO-lev: Short support
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
            dry_order = self.create_dry_run_order(
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
            self._log_exchange_response('create_stoploss_order', order)
            return order
        except ccxt.InsufficientFunds as e:
            raise InsufficientFundsError(
                f'Insufficient funds to create {ordertype} sell order on market {pair}. '
                f'Tried to sell amount {amount} at rate {rate}. '
                f'Message: {e}') from e
        except ccxt.InvalidOrder as e:
            # Errors:
            # `binance Order would trigger immediately.`
            raise InvalidOrderException(
                f'Could not create {ordertype} sell order on market {pair}. '
                f'Tried to sell amount {amount} at rate {rate}. '
                f'Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not place sell order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _apply_leverage_to_stake_amount(self, stake_amount: float, leverage: float):
        return stake_amount / leverage

    def fill_leverage_brackets(self):
        """
            Assigns property _leverage_brackets to a dictionary of information about the leverage
            allowed on each pair
        """
        leverage_brackets = self._api.load_leverage_brackets()
        for pair, brackets in leverage_brackets.items:
            self.leverage_brackets[pair] = [
                [
                    min_amount,
                    float(margin_req)
                ] for [
                    min_amount,
                    margin_req
                ] in brackets
            ]

    def get_max_leverage(self, pair: Optional[str], nominal_value: Optional[float]) -> float:
        """
            Returns the maximum leverage that a pair can be traded at
            :param pair: The base/quote currency pair being traded
            :nominal_value: The total value of the trade in quote currency (collateral + debt)
        """
        pair_brackets = self._leverage_brackets[pair]
        max_lev = 1.0
        for [min_amount, margin_req] in pair_brackets:
            print(nominal_value, min_amount)
            if nominal_value >= min_amount:
                max_lev = 1/margin_req
        return max_lev

    def set_leverage(self, pair, leverage):
        """
            Binance Futures must set the leverage before making a futures trade, in order to not
            have the same leverage on every trade
        """
        self._api.set_leverage(symbol=pair, leverage=leverage)
