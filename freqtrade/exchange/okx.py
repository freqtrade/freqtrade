import logging
from typing import Dict, List, Optional, Tuple

import ccxt

from freqtrade.constants import BuySell
from freqtrade.enums import CandleType, MarginMode, TradingMode
from freqtrade.enums.pricetype import PriceType
from freqtrade.exceptions import DDosProtection, OperationalException, TemporaryError
from freqtrade.exchange import Exchange, date_minus_candles
from freqtrade.exchange.common import retrier


logger = logging.getLogger(__name__)


class Okx(Exchange):
    """Okx exchange class.

    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    _ft_has: Dict = {
        "ohlcv_candle_limit": 100,  # Warning, special case with data prior to X months
        "mark_ohlcv_timeframe": "4h",
        "funding_fee_timeframe": "8h",
    }
    _ft_has_futures: Dict = {
        "tickers_have_quoteVolume": False,
        "fee_cost_in_contracts": True,
        "stop_price_type_field": "tpTriggerPxType",
        "stop_price_type_value_mapping": {
            PriceType.LAST: "last",
            PriceType.MARK: "index",
            PriceType.INDEX: "mark",
            },
    }

    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.MARGIN, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED),
    ]

    net_only = True

    _ccxt_params: Dict = {'options': {'brokerId': 'ffb5405ad327SUDE'}}

    def ohlcv_candle_limit(
            self, timeframe: str, candle_type: CandleType, since_ms: Optional[int] = None) -> int:
        """
        Exchange ohlcv candle limit
        OKX has the following behaviour:
        * 300 candles for uptodate data
        * 100 candles for historic data
        * 100 candles for additional candles (not futures or spot).
        :param timeframe: Timeframe to check
        :param candle_type: Candle-type
        :param since_ms: Starting timestamp
        :return: Candle limit as integer
        """
        if (
            candle_type in (CandleType.FUTURES, CandleType.SPOT) and
            (not since_ms or since_ms > (date_minus_candles(timeframe, 300).timestamp() * 1000))
        ):
            return 300

        return super().ohlcv_candle_limit(timeframe, candle_type, since_ms)

    @retrier
    def additional_exchange_init(self) -> None:
        """
        Additional exchange initialization logic.
        .api will be available at this point.
        Must be overridden in child methods if required.
        """
        try:
            if self.trading_mode == TradingMode.FUTURES and not self._config['dry_run']:
                accounts = self._api.fetch_accounts()
                self._log_exchange_response('fetch_accounts', accounts)
                if len(accounts) > 0:
                    self.net_only = accounts[0].get('info', {}).get('posMode') == 'net_mode'
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Error in additional_exchange_init due to {e.__class__.__name__}. Message: {e}'
                ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _get_posSide(self, side: BuySell, reduceOnly: bool):
        if self.net_only:
            return 'net'
        if not reduceOnly:
            # Enter
            return 'long' if side == 'buy' else 'short'
        else:
            # Exit
            return 'long' if side == 'sell' else 'short'

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
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode:
            params['tdMode'] = self.margin_mode.value
            params['posSide'] = self._get_posSide(side, reduceOnly)
        return params

    @retrier
    def _lev_prep(self, pair: str, leverage: float, side: BuySell):
        if self.trading_mode != TradingMode.SPOT and self.margin_mode is not None:
            try:
                # TODO-lev: Test me properly (check mgnMode passed)
                res = self._api.set_leverage(
                    leverage=leverage,
                    symbol=pair,
                    params={
                        "mgnMode": self.margin_mode.value,
                        "posSide": self._get_posSide(side, False),
                    })
                self._log_exchange_response('set_leverage', res)

            except ccxt.DDoSProtection as e:
                raise DDosProtection(e) from e
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                raise TemporaryError(
                    f'Could not set leverage due to {e.__class__.__name__}. Message: {e}') from e
            except ccxt.BaseError as e:
                raise OperationalException(e) from e

    def get_max_pair_stake_amount(
        self,
        pair: str,
        price: float,
        leverage: float = 1.0
    ) -> float:

        if self.trading_mode == TradingMode.SPOT:
            return float('inf')  # Not actually inf, but this probably won't matter for SPOT

        if pair not in self._leverage_tiers:
            return float('inf')

        pair_tiers = self._leverage_tiers[pair]
        return pair_tiers[-1]['maxNotional'] / leverage
