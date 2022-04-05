""" Gate.io exchange subclass """
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange


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
        "ohlcv_volume_currency": "quote",
        "stoploss_order_types": {"limit": "limit"},
        "stoploss_on_exchange": True,
    }

    _ft_has_futures: Dict = {
        "needs_trading_fees": True
    }

    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.MARGIN, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED)
    ]

    def validate_ordertypes(self, order_types: Dict) -> None:
        super().validate_ordertypes(order_types)

        if self.trading_mode != TradingMode.FUTURES:
            if any(v == 'market' for k, v in order_types.items()):
                raise OperationalException(
                    f'Exchange {self.name} does not support market orders.')

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
                    if trade.get('fee', {}).get('cost') is None:
                        takerOrMaker = trade.get('takerOrMaker', 'taker')
                        if pair_fees.get(takerOrMaker) is not None:
                            trades[idx]['fee'] = {
                                'currency': self.get_pair_quote_currency(pair),
                                'cost': trade['cost'] * pair_fees[takerOrMaker],
                                'rate': pair_fees[takerOrMaker],
                            }
        return trades

    def fetch_stoploss_order(self, order_id: str, pair: str, params={}) -> Dict:
        return self.fetch_order(
            order_id=order_id,
            pair=pair,
            params={'stop': True}
        )

    def cancel_stoploss_order(self, order_id: str, pair: str, params={}) -> Dict:
        return self.cancel_order(
            order_id=order_id,
            pair=pair,
            params={'stop': True}
        )

    def stoploss_adjust(self, stop_loss: float, order: Dict, side: str) -> bool:
        """
        Verify stop_loss against stoploss-order value (limit or price)
        Returns True if adjustment is necessary.
        """
        return ((side == "sell" and stop_loss > float(order['stopPrice'])) or
                (side == "buy" and stop_loss < float(order['stopPrice'])))
