""" Kraken exchange subclass """
import logging
from typing import Dict

import ccxt

from freqtrade import OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange import retrier

logger = logging.getLogger(__name__)


class Kraken(Exchange):

    _params: Dict = {"trading_agreement": "agree"}
    _ft_has: Dict = {
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
