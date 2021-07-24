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

    def stoploss_adjust(self, stop_loss: float, order: Dict) -> bool:
        """
        Verify stop_loss against stoploss-order value (limit or price)
        Returns True if adjustment is necessary.
        """
        return order['type'] == 'stop_loss_limit' and stop_loss > float(order['info']['stopPrice'])

    @retrier(retries=0)
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

    def transfer(self, asset: str, amount: float, frm: str, to: str, pair: Optional[str]):
        res = self._api.sapi_post_margin_isolated_transfer({
            "asset": asset,
            "amount": amount,
            "transFrom": frm,
            "transTo": to,
            "symbol": pair
        })
        logger.info(f"Transfer response: {res}")

    def borrow(self, asset: str, amount: float, pair: str):
        res = self._api.sapi_post_margin_loan({
            "asset": asset,
            "isIsolated": True,
            "symbol": pair,
            "amount": amount
        })  # borrow from binance
        logger.info(f"Borrow response: {res}")

    def repay(self, asset: str, amount: float, pair: str):
        res = self._api.sapi_post_margin_repay({
            "asset": asset,
            "isIsolated": True,
            "symbol": pair,
            "amount": amount
        })  # borrow from binance
        logger.info(f"Borrow response: {res}")

    def setup_leveraged_enter(
        self,
        pair: str,
        leverage: float,
        amount: float,
        quote_currency: Optional[str],
        is_short: Optional[bool]
    ):
        if not quote_currency or not is_short:
            raise OperationalException(
                "quote_currency and is_short are required arguments to setup_leveraged_enter"
                " when trading with leverage on binance"
            )
        open_rate = 2  # TODO-mg: get the real open_rate, or real stake_amount
        stake_amount = amount * open_rate
        if is_short:
            borrowed = stake_amount * ((leverage-1)/leverage)
        else:
            borrowed = amount

        self.transfer(  # Transfer to isolated margin
            asset=quote_currency,
            amount=stake_amount,
            frm='SPOT',
            to='ISOLATED_MARGIN',
            pair=pair
        )

        self.borrow(
            asset=quote_currency,
            amount=borrowed,
            pair=pair
        )  # borrow from binance

    def complete_leveraged_exit(
        self,
        pair: str,
        leverage: float,
        amount: float,
        quote_currency: Optional[str],
        is_short: Optional[bool]
    ):

        if not quote_currency or not is_short:
            raise OperationalException(
                "quote_currency and is_short are required arguments to setup_leveraged_enter"
                " when trading with leverage on binance"
            )

        open_rate = 2  # TODO-mg: get the real open_rate, or real stake_amount
        stake_amount = amount * open_rate
        if is_short:
            borrowed = stake_amount * ((leverage-1)/leverage)
        else:
            borrowed = amount

        self.repay(
            asset=quote_currency,
            amount=borrowed,
            pair=pair
        )  # repay binance

        self.transfer(  # Transfer to isolated margin
            asset=quote_currency,
            amount=stake_amount,
            frm='ISOLATED_MARGIN',
            to='SPOT',
            pair=pair
        )

    def apply_leverage_to_stake_amount(self, stake_amount: float, leverage: float):
        return stake_amount / leverage
