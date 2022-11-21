""" Gate.io exchange subclass """
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange
from freqtrade.misc import safe_value_fallback2


logger = logging.getLogger(__name__)


class Gateio(Exchange):
    """
    Gate.io exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """

    _ft_has: Dict = {
        "ohlcv_candle_limit": 1000,
        "order_time_in_force": ['GTC', 'IOC'],
        "stoploss_order_types": {"limit": "limit"},
        "stoploss_on_exchange": True,
    }

    _ft_has_futures: Dict = {
        "needs_trading_fees": True,
        "fee_cost_in_contracts": False,  # Set explicitly to false for clarity
        "order_props_in_contracts": ['amount', 'filled', 'remaining'],
    }

    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.MARGIN, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED)
    ]

    def validate_ordertypes(self, order_types: Dict) -> None:

        if self.trading_mode != TradingMode.FUTURES:
            if any(v == 'market' for k, v in order_types.items()):
                raise OperationalException(
                    f'Exchange {self.name} does not support market orders.')

    def _get_params(
            self,
            side: BuySell,
            ordertype: str,
            leverage: float,
            reduceOnly: bool,
            time_in_force: str = 'GTC',
            ) -> Dict:
        params = super()._get_params(
            side=side,
            ordertype=ordertype,
            leverage=leverage,
            reduceOnly=reduceOnly,
            time_in_force=time_in_force,
        )
        if ordertype == 'market' and self.trading_mode == TradingMode.FUTURES:
            params['type'] = 'market'
            param = self._ft_has.get('time_in_force_parameter', '')
            params.update({param: 'IOC'})
        return params

    def get_trades_for_order(self, order_id: str, pair: str, since: datetime,
                             params: Optional[Dict] = None) -> List:
        trades = super().get_trades_for_order(order_id, pair, since, params)

        if self.trading_mode == TradingMode.FUTURES:
            # Futures usually don't contain fees in the response.
            # As such, futures orders on gateio will not contain a fee, which causes
            # a repeated "update fee" cycle and wrong calculations.
            # Therefore we patch the response with fees if it's not available.
            # An alternative also contianing fees would be
            # privateFuturesGetSettleAccountBook({"settle": "usdt"})
            pair_fees = self._trading_fees.get(pair, {})
            if pair_fees:
                for idx, trade in enumerate(trades):
                    fee = trade.get('fee', {})
                    if fee and fee.get('cost') is None:
                        takerOrMaker = trade.get('takerOrMaker', 'taker')
                        if pair_fees.get(takerOrMaker) is not None:
                            trades[idx]['fee'] = {
                                'currency': self.get_pair_quote_currency(pair),
                                'cost': trade['cost'] * pair_fees[takerOrMaker],
                                'rate': pair_fees[takerOrMaker],
                            }
        return trades

    def get_order_id_conditional(self, order: Dict[str, Any]) -> str:
        if self.trading_mode == TradingMode.FUTURES:
            return safe_value_fallback2(order, order, 'id_stop', 'id')
        return order['id']

    def fetch_stoploss_order(self, order_id: str, pair: str, params: Dict = {}) -> Dict:
        order = self.fetch_order(
            order_id=order_id,
            pair=pair,
            params={'stop': True}
        )
        if self.trading_mode == TradingMode.FUTURES:
            if order['status'] == 'closed':
                # Places a real order - which we need to fetch explicitly.
                new_orderid = order.get('info', {}).get('trade_id')
                if new_orderid:
                    order1 = self.fetch_order(order_id=new_orderid, pair=pair, params=params)
                    order1['id_stop'] = order1['id']
                    order1['id'] = order_id
                    order1['stopPrice'] = order.get('stopPrice')

                    return order1
        return order

    def cancel_stoploss_order(self, order_id: str, pair: str, params: Dict = {}) -> Dict:
        return self.cancel_order(
            order_id=order_id,
            pair=pair,
            params={'stop': True}
        )
