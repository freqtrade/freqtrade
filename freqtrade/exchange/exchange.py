# pragma pylint: disable=W0603
"""
Cryptocurrency Exchanges support
"""
import asyncio
import http
import inspect
import logging
from copy import deepcopy
from datetime import datetime, timezone
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

import arrow
import ccxt
import ccxt.async_support as ccxt_async
from cachetools import TTLCache
from ccxt.base.decimal_to_precision import (ROUND_DOWN, ROUND_UP, TICK_SIZE, TRUNCATE,
                                            decimal_to_precision)
from pandas import DataFrame

from freqtrade.constants import DEFAULT_AMOUNT_RESERVE_PERCENT, ListPairsWithTimeframes
from freqtrade.data.converter import ohlcv_to_dataframe, trades_dict_to_list
from freqtrade.exceptions import (DDosProtection, ExchangeError, InsufficientFundsError,
                                  InvalidOrderException, OperationalException, RetryableOrderError,
                                  TemporaryError)
from freqtrade.exchange.common import (API_FETCH_ORDER_RETRY_COUNT, BAD_EXCHANGES,
                                       EXCHANGE_HAS_OPTIONAL, EXCHANGE_HAS_REQUIRED, retrier,
                                       retrier_async)
from freqtrade.misc import deep_merge_dicts, safe_value_fallback2
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist


CcxtModuleType = Any


logger = logging.getLogger(__name__)


# Workaround for adding samesite support to pre 3.8 python
# Only applies to python3.7, and only on certain exchanges (kraken)
# Replicates the fix from starlette (which is actually causing this problem)
http.cookies.Morsel._reserved["samesite"] = "SameSite"  # type: ignore


class Exchange:

    _config: Dict = {}

    # Parameters to add directly to ccxt sync/async initialization.
    _ccxt_config: Dict = {}

    # Parameters to add directly to buy/sell calls (like agreeing to trading agreement)
    _params: Dict = {}

    # Dict to specify which options each exchange implements
    # This defines defaults, which can be selectively overridden by subclasses using _ft_has
    # or by specifying them in the configuration.
    _ft_has_default: Dict = {
        "stoploss_on_exchange": False,
        "order_time_in_force": ["gtc"],
        "ohlcv_candle_limit": 500,
        "ohlcv_partial_candle": True,
        "trades_pagination": "time",  # Possible are "time" or "id"
        "trades_pagination_arg": "since",
        "l2_limit_range": None,
        "l2_limit_range_required": True,  # Allow Empty L2 limit (kucoin)
    }
    _ft_has: Dict = {}

    def __init__(self, config: Dict[str, Any], validate: bool = True) -> None:
        """
        Initializes this module with the given config,
        it does basic validation whether the specified exchange and pairs are valid.
        :return: None
        """
        self._api: ccxt.Exchange = None
        self._api_async: ccxt_async.Exchange = None
        self._markets: Dict = {}

        self._config.update(config)

        # Holds last candle refreshed time of each pair
        self._pairs_last_refresh_time: Dict[Tuple[str, str], int] = {}
        # Timestamp of last markets refresh
        self._last_markets_refresh: int = 0

        # Cache for 10 minutes ...
        self._fetch_tickers_cache: TTLCache = TTLCache(maxsize=1, ttl=60 * 10)

        # Holds candles
        self._klines: Dict[Tuple[str, str], DataFrame] = {}

        # Holds all open sell orders for dry_run
        self._dry_run_open_orders: Dict[str, Any] = {}

        if config['dry_run']:
            logger.info('Instance is running with dry_run enabled')
        logger.info(f"Using CCXT {ccxt.__version__}")
        exchange_config = config['exchange']

        # Deep merge ft_has with default ft_has options
        self._ft_has = deep_merge_dicts(self._ft_has, deepcopy(self._ft_has_default))
        if exchange_config.get('_ft_has_params'):
            self._ft_has = deep_merge_dicts(exchange_config.get('_ft_has_params'),
                                            self._ft_has)
            logger.info("Overriding exchange._ft_has with config params, result: %s", self._ft_has)

        # Assign this directly for easy access
        self._ohlcv_partial_candle = self._ft_has['ohlcv_partial_candle']

        self._trades_pagination = self._ft_has['trades_pagination']
        self._trades_pagination_arg = self._ft_has['trades_pagination_arg']

        # Initialize ccxt objects
        ccxt_config = self._ccxt_config.copy()
        ccxt_config = deep_merge_dicts(exchange_config.get('ccxt_config', {}), ccxt_config)
        ccxt_config = deep_merge_dicts(exchange_config.get('ccxt_sync_config', {}), ccxt_config)

        self._api = self._init_ccxt(exchange_config, ccxt_kwargs=ccxt_config)

        ccxt_async_config = self._ccxt_config.copy()
        ccxt_async_config = deep_merge_dicts(exchange_config.get('ccxt_config', {}),
                                             ccxt_async_config)
        ccxt_async_config = deep_merge_dicts(exchange_config.get('ccxt_async_config', {}),
                                             ccxt_async_config)
        self._api_async = self._init_ccxt(
            exchange_config, ccxt_async, ccxt_kwargs=ccxt_async_config)

        logger.info('Using Exchange "%s"', self.name)

        if validate:
            # Check if timeframe is available
            self.validate_timeframes(config.get('timeframe'))

            # Initial markets load
            self._load_markets()

            # Check if all pairs are available
            self.validate_stakecurrency(config['stake_currency'])
            if not exchange_config.get('skip_pair_validation'):
                self.validate_pairs(config['exchange']['pair_whitelist'])
            self.validate_ordertypes(config.get('order_types', {}))
            self.validate_order_time_in_force(config.get('order_time_in_force', {}))
            self.validate_required_startup_candles(config.get('startup_candle_count', 0),
                                                   config.get('timeframe', ''))

        # Converts the interval provided in minutes in config to seconds
        self.markets_refresh_interval: int = exchange_config.get(
            "markets_refresh_interval", 60) * 60

    def __del__(self):
        """
        Destructor - clean up async stuff
        """
        self.close()

    def close(self):
        logger.debug("Exchange object destroyed, closing async loop")
        if self._api_async and inspect.iscoroutinefunction(self._api_async.close):
            asyncio.get_event_loop().run_until_complete(self._api_async.close())

    def _init_ccxt(self, exchange_config: Dict[str, Any], ccxt_module: CcxtModuleType = ccxt,
                   ccxt_kwargs: dict = None) -> ccxt.Exchange:
        """
        Initialize ccxt with given config and return valid
        ccxt instance.
        """
        # Find matching class for the given exchange name
        name = exchange_config['name']

        if not is_exchange_known_ccxt(name, ccxt_module):
            raise OperationalException(f'Exchange {name} is not supported by ccxt')

        ex_config = {
            'apiKey': exchange_config.get('key'),
            'secret': exchange_config.get('secret'),
            'password': exchange_config.get('password'),
            'uid': exchange_config.get('uid', ''),
        }
        if ccxt_kwargs:
            logger.info('Applying additional ccxt config: %s', ccxt_kwargs)
            ex_config.update(ccxt_kwargs)
        try:

            api = getattr(ccxt_module, name.lower())(ex_config)
        except (KeyError, AttributeError) as e:
            raise OperationalException(f'Exchange {name} is not supported') from e
        except ccxt.BaseError as e:
            raise OperationalException(f"Initialization of ccxt failed. Reason: {e}") from e

        self.set_sandbox(api, exchange_config, name)

        return api

    @property
    def name(self) -> str:
        """exchange Name (from ccxt)"""
        return self._api.name

    @property
    def id(self) -> str:
        """exchange ccxt id"""
        return self._api.id

    @property
    def timeframes(self) -> List[str]:
        return list((self._api.timeframes or {}).keys())

    @property
    def markets(self) -> Dict:
        """exchange ccxt markets"""
        if not self._markets:
            logger.info("Markets were not loaded. Loading them now..")
            self._load_markets()
        return self._markets

    @property
    def precisionMode(self) -> str:
        """exchange ccxt precisionMode"""
        return self._api.precisionMode

    def ohlcv_candle_limit(self, timeframe: str) -> int:
        """
        Exchange ohlcv candle limit
        Uses ohlcv_candle_limit_per_timeframe if the exchange has different limts
        per timeframe (e.g. bittrex), otherwise falls back to ohlcv_candle_limit
        :param timeframe: Timeframe to check
        :return: Candle limit as integer
        """
        return int(self._ft_has.get('ohlcv_candle_limit_per_timeframe', {}).get(
            timeframe, self._ft_has.get('ohlcv_candle_limit')))

    def get_markets(self, base_currencies: List[str] = None, quote_currencies: List[str] = None,
                    pairs_only: bool = False, active_only: bool = False) -> Dict[str, Any]:
        """
        Return exchange ccxt markets, filtered out by base currency and quote currency
        if this was requested in parameters.

        TODO: consider moving it to the Dataprovider
        """
        markets = self.markets
        if not markets:
            raise OperationalException("Markets were not loaded.")

        if base_currencies:
            markets = {k: v for k, v in markets.items() if v['base'] in base_currencies}
        if quote_currencies:
            markets = {k: v for k, v in markets.items() if v['quote'] in quote_currencies}
        if pairs_only:
            markets = {k: v for k, v in markets.items() if self.market_is_tradable(v)}
        if active_only:
            markets = {k: v for k, v in markets.items() if market_is_active(v)}
        return markets

    def get_quote_currencies(self) -> List[str]:
        """
        Return a list of supported quote currencies
        """
        markets = self.markets
        return sorted(set([x['quote'] for _, x in markets.items()]))

    def get_pair_quote_currency(self, pair: str) -> str:
        """
        Return a pair's quote currency
        """
        return self.markets.get(pair, {}).get('quote', '')

    def get_pair_base_currency(self, pair: str) -> str:
        """
        Return a pair's quote currency
        """
        return self.markets.get(pair, {}).get('base', '')

    def market_is_tradable(self, market: Dict[str, Any]) -> bool:
        """
        Check if the market symbol is tradable by Freqtrade.
        By default, checks if it's splittable by `/` and both sides correspond to base / quote
        """
        symbol_parts = market['symbol'].split('/')
        return (len(symbol_parts) == 2 and
                len(symbol_parts[0]) > 0 and
                len(symbol_parts[1]) > 0 and
                symbol_parts[0] == market.get('base') and
                symbol_parts[1] == market.get('quote')
                )

    def klines(self, pair_interval: Tuple[str, str], copy: bool = True) -> DataFrame:
        if pair_interval in self._klines:
            return self._klines[pair_interval].copy() if copy else self._klines[pair_interval]
        else:
            return DataFrame()

    def set_sandbox(self, api: ccxt.Exchange, exchange_config: dict, name: str) -> None:
        if exchange_config.get('sandbox'):
            if api.urls.get('test'):
                api.urls['api'] = api.urls['test']
                logger.info("Enabled Sandbox API on %s", name)
            else:
                logger.warning(
                    f"No Sandbox URL in CCXT for {name}, exiting. Please check your config.json")
                raise OperationalException(f'Exchange {name} does not provide a sandbox api')

    def _load_async_markets(self, reload: bool = False) -> None:
        try:
            if self._api_async:
                asyncio.get_event_loop().run_until_complete(
                    self._api_async.load_markets(reload=reload))

        except (asyncio.TimeoutError, ccxt.BaseError) as e:
            logger.warning('Could not load async markets. Reason: %s', e)
            return

    def _load_markets(self) -> None:
        """ Initialize markets both sync and async """
        try:
            self._markets = self._api.load_markets()
            self._load_async_markets()
            self._last_markets_refresh = arrow.utcnow().int_timestamp
        except ccxt.BaseError:
            logger.exception('Unable to initialize markets.')

    def reload_markets(self) -> None:
        """Reload markets both sync and async if refresh interval has passed """
        # Check whether markets have to be reloaded
        if (self._last_markets_refresh > 0) and (
                self._last_markets_refresh + self.markets_refresh_interval
                > arrow.utcnow().int_timestamp):
            return None
        logger.debug("Performing scheduled market reload..")
        try:
            self._markets = self._api.load_markets(reload=True)
            # Also reload async markets to avoid issues with newly listed pairs
            self._load_async_markets(reload=True)
            self._last_markets_refresh = arrow.utcnow().int_timestamp
        except ccxt.BaseError:
            logger.exception("Could not reload markets.")

    def validate_stakecurrency(self, stake_currency: str) -> None:
        """
        Checks stake-currency against available currencies on the exchange.
        :param stake_currency: Stake-currency to validate
        :raise: OperationalException if stake-currency is not available.
        """
        quote_currencies = self.get_quote_currencies()
        if stake_currency not in quote_currencies:
            raise OperationalException(
                f"{stake_currency} is not available as stake on {self.name}. "
                f"Available currencies are: {', '.join(quote_currencies)}")

    def validate_pairs(self, pairs: List[str]) -> None:
        """
        Checks if all given pairs are tradable on the current exchange.
        :param pairs: list of pairs
        :raise: OperationalException if one pair is not available
        :return: None
        """

        if not self.markets:
            logger.warning('Unable to validate pairs (assuming they are correct).')
            return
        extended_pairs = expand_pairlist(pairs, list(self.markets), keep_invalid=True)
        invalid_pairs = []
        for pair in extended_pairs:
            # Note: ccxt has BaseCurrency/QuoteCurrency format for pairs
            if self.markets and pair not in self.markets:
                raise OperationalException(
                    f'Pair {pair} is not available on {self.name}. '
                    f'Please remove {pair} from your whitelist.')

                # From ccxt Documentation:
                # markets.info: An associative array of non-common market properties,
                # including fees, rates, limits and other general market information.
                # The internal info array is different for each particular market,
                # its contents depend on the exchange.
                # It can also be a string or similar ... so we need to verify that first.
            elif (isinstance(self.markets[pair].get('info', None), dict)
                  and self.markets[pair].get('info', {}).get('IsRestricted', False)):
                # Warn users about restricted pairs in whitelist.
                # We cannot determine reliably if Users are affected.
                logger.warning(f"Pair {pair} is restricted for some users on this exchange."
                               f"Please check if you are impacted by this restriction "
                               f"on the exchange and eventually remove {pair} from your whitelist.")
            if (self._config['stake_currency'] and
                    self.get_pair_quote_currency(pair) != self._config['stake_currency']):
                invalid_pairs.append(pair)
        if invalid_pairs:
            raise OperationalException(
                f"Stake-currency '{self._config['stake_currency']}' not compatible with "
                f"pair-whitelist. Please remove the following pairs: {invalid_pairs}")

    def get_valid_pair_combination(self, curr_1: str, curr_2: str) -> str:
        """
        Get valid pair combination of curr_1 and curr_2 by trying both combinations.
        """
        for pair in [f"{curr_1}/{curr_2}", f"{curr_2}/{curr_1}"]:
            if pair in self.markets and self.markets[pair].get('active'):
                return pair
        raise ExchangeError(f"Could not combine {curr_1} and {curr_2} to get a valid pair.")

    def validate_timeframes(self, timeframe: Optional[str]) -> None:
        """
        Check if timeframe from config is a supported timeframe on the exchange
        """
        if not hasattr(self._api, "timeframes") or self._api.timeframes is None:
            # If timeframes attribute is missing (or is None), the exchange probably
            # has no fetchOHLCV method.
            # Therefore we also show that.
            raise OperationalException(
                f"The ccxt library does not provide the list of timeframes "
                f"for the exchange \"{self.name}\" and this exchange "
                f"is therefore not supported. ccxt fetchOHLCV: {self.exchange_has('fetchOHLCV')}")

        if timeframe and (timeframe not in self.timeframes):
            raise OperationalException(
                f"Invalid timeframe '{timeframe}'. This exchange supports: {self.timeframes}")

        if timeframe and timeframe_to_minutes(timeframe) < 1:
            raise OperationalException("Timeframes < 1m are currently not supported by Freqtrade.")

    def validate_ordertypes(self, order_types: Dict) -> None:
        """
        Checks if order-types configured in strategy/config are supported
        """
        if any(v == 'market' for k, v in order_types.items()):
            if not self.exchange_has('createMarketOrder'):
                raise OperationalException(
                    f'Exchange {self.name} does not support market orders.')

        if (order_types.get("stoploss_on_exchange")
                and not self._ft_has.get("stoploss_on_exchange", False)):
            raise OperationalException(
                f'On exchange stoploss is not supported for {self.name}.'
            )

    def validate_order_time_in_force(self, order_time_in_force: Dict) -> None:
        """
        Checks if order time in force configured in strategy/config are supported
        """
        if any(v not in self._ft_has["order_time_in_force"]
               for k, v in order_time_in_force.items()):
            raise OperationalException(
                f'Time in force policies are not supported for {self.name} yet.')

    def validate_required_startup_candles(self, startup_candles: int, timeframe: str) -> None:
        """
        Checks if required startup_candles is more than ohlcv_candle_limit().
        Requires a grace-period of 5 candles - so a startup-period up to 494 is allowed by default.
        """
        candle_limit = self.ohlcv_candle_limit(timeframe)
        if startup_candles + 5 > candle_limit:
            raise OperationalException(
                f"This strategy requires {startup_candles} candles to start. "
                f"{self.name} only provides {candle_limit} for {timeframe}.")

    def exchange_has(self, endpoint: str) -> bool:
        """
        Checks if exchange implements a specific API endpoint.
        Wrapper around ccxt 'has' attribute
        :param endpoint: Name of endpoint (e.g. 'fetchOHLCV', 'fetchTickers')
        :return: bool
        """
        return endpoint in self._api.has and self._api.has[endpoint]

    def amount_to_precision(self, pair: str, amount: float) -> float:
        '''
        Returns the amount to buy or sell to a precision the Exchange accepts
        Reimplementation of ccxt internal methods - ensuring we can test the result is correct
        based on our definitions.
        '''
        if self.markets[pair]['precision']['amount']:
            amount = float(decimal_to_precision(amount, rounding_mode=TRUNCATE,
                                                precision=self.markets[pair]['precision']['amount'],
                                                counting_mode=self.precisionMode,
                                                ))

        return amount

    def price_to_precision(self, pair: str, price: float) -> float:
        '''
        Returns the price rounded up to the precision the Exchange accepts.
        Partial Reimplementation of ccxt internal method decimal_to_precision(),
        which does not support rounding up
        TODO: If ccxt supports ROUND_UP for decimal_to_precision(), we could remove this and
        align with amount_to_precision().
        Rounds up
        '''
        if self.markets[pair]['precision']['price']:
            # price = float(decimal_to_precision(price, rounding_mode=ROUND,
            #                                    precision=self.markets[pair]['precision']['price'],
            #                                    counting_mode=self.precisionMode,
            #                                    ))
            if self.precisionMode == TICK_SIZE:
                precision = self.markets[pair]['precision']['price']
                missing = price % precision
                if missing != 0:
                    price = price - missing + precision
            else:
                symbol_prec = self.markets[pair]['precision']['price']
                big_price = price * pow(10, symbol_prec)
                price = ceil(big_price) / pow(10, symbol_prec)
        return price

    def price_get_one_pip(self, pair: str, price: float) -> float:
        """
        Get's the "1 pip" value for this pair.
        Used in PriceFilter to calculate the 1pip movements.
        """
        precision = self.markets[pair]['precision']['price']
        if self.precisionMode == TICK_SIZE:
            return precision
        else:
            return 1 / pow(10, precision)

    def get_min_pair_stake_amount(self, pair: str, price: float,
                                  stoploss: float) -> Optional[float]:
        try:
            market = self.markets[pair]
        except KeyError:
            raise ValueError(f"Can't get market information for symbol {pair}")

        if 'limits' not in market:
            return None

        min_stake_amounts = []
        limits = market['limits']
        if ('cost' in limits and 'min' in limits['cost']
                and limits['cost']['min'] is not None):
            min_stake_amounts.append(limits['cost']['min'])

        if ('amount' in limits and 'min' in limits['amount']
                and limits['amount']['min'] is not None):
            min_stake_amounts.append(limits['amount']['min'] * price)

        if not min_stake_amounts:
            return None

        # reserve some percent defined in config (5% default) + stoploss
        amount_reserve_percent = 1.0 + self._config.get('amount_reserve_percent',
                                                        DEFAULT_AMOUNT_RESERVE_PERCENT)
        amount_reserve_percent = (
          amount_reserve_percent / (1 - abs(stoploss)) if abs(stoploss) != 1 else 1.5
        )
        # it should not be more than 50%
        amount_reserve_percent = max(min(amount_reserve_percent, 1.5), 1)

        # The value returned should satisfy both limits: for amount (base currency) and
        # for cost (quote, stake currency), so max() is used here.
        # See also #2575 at github.
        return max(min_stake_amounts) * amount_reserve_percent

    def create_dry_run_order(self, pair: str, ordertype: str, side: str, amount: float,
                             rate: float, params: Dict = {}) -> Dict[str, Any]:
        order_id = f'dry_run_{side}_{datetime.now().timestamp()}'
        _amount = self.amount_to_precision(pair, amount)
        dry_order = {
            'id': order_id,
            'symbol': pair,
            'price': rate,
            'average': rate,
            'amount': _amount,
            'cost': _amount * rate,
            'type': ordertype,
            'side': side,
            'remaining': _amount,
            'datetime': arrow.utcnow().isoformat(),
            'timestamp': int(arrow.utcnow().int_timestamp * 1000),
            'status': "closed" if ordertype == "market" else "open",
            'fee': None,
            'info': {}
        }
        self._store_dry_order(dry_order, pair)
        # Copy order and close it - so the returned order is open unless it's a market order
        return dry_order

    def _store_dry_order(self, dry_order: Dict, pair: str) -> None:
        closed_order = dry_order.copy()
        if closed_order['type'] in ["market", "limit"]:
            closed_order.update({
                'status': 'closed',
                'filled': closed_order['amount'],
                'remaining': 0,
                'fee': {
                    'currency': self.get_pair_quote_currency(pair),
                    'cost': dry_order['cost'] * self.get_fee(pair),
                    'rate': self.get_fee(pair)
                }
            })
        if closed_order["type"] in ["stop_loss_limit", "stop-loss-limit"]:
            closed_order["info"].update({"stopPrice": closed_order["price"]})
        self._dry_run_open_orders[closed_order["id"]] = closed_order

    def create_order(self, pair: str, ordertype: str, side: str, amount: float,
                     rate: float, params: Dict = {}) -> Dict:
        try:
            # Set the precision for amount and price(rate) as accepted by the exchange
            amount = self.amount_to_precision(pair, amount)
            needs_price = (ordertype != 'market'
                           or self._api.options.get("createMarketBuyOrderRequiresPrice", False))
            rate_for_order = self.price_to_precision(pair, rate) if needs_price else None

            return self._api.create_order(pair, ordertype, side,
                                          amount, rate_for_order, params)

        except ccxt.InsufficientFunds as e:
            raise InsufficientFundsError(
                f'Insufficient funds to create {ordertype} {side} order on market {pair}. '
                f'Tried to {side} amount {amount} at rate {rate}.'
                f'Message: {e}') from e
        except ccxt.InvalidOrder as e:
            raise ExchangeError(
                f'Could not create {ordertype} {side} order on market {pair}. '
                f'Tried to {side} amount {amount} at rate {rate}. '
                f'Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not place {side} order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def buy(self, pair: str, ordertype: str, amount: float,
            rate: float, time_in_force: str) -> Dict:

        if self._config['dry_run']:
            dry_order = self.create_dry_run_order(pair, ordertype, "buy", amount, rate)
            return dry_order

        params = self._params.copy()
        if time_in_force != 'gtc' and ordertype != 'market':
            params.update({'timeInForce': time_in_force})

        return self.create_order(pair, ordertype, 'buy', amount, rate, params)

    def sell(self, pair: str, ordertype: str, amount: float,
             rate: float, time_in_force: str = 'gtc') -> Dict:

        if self._config['dry_run']:
            dry_order = self.create_dry_run_order(pair, ordertype, "sell", amount, rate)
            return dry_order

        params = self._params.copy()
        if time_in_force != 'gtc' and ordertype != 'market':
            params.update({'timeInForce': time_in_force})

        return self.create_order(pair, ordertype, 'sell', amount, rate, params)

    def stoploss_adjust(self, stop_loss: float, order: Dict) -> bool:
        """
        Verify stop_loss against stoploss-order value (limit or price)
        Returns True if adjustment is necessary.
        """
        raise OperationalException(f"stoploss is not implemented for {self.name}.")

    def stoploss(self, pair: str, amount: float, stop_price: float, order_types: Dict) -> Dict:
        """
        creates a stoploss order.
        The precise ordertype is determined by the order_types dict or exchange default.
        Since ccxt does not unify stoploss-limit orders yet, this needs to be implemented in each
        exchange's subclass.
        The exception below should never raise, since we disallow
        starting the bot in validate_ordertypes()
        Note: Changes to this interface need to be applied to all sub-classes too.
        """

        raise OperationalException(f"stoploss is not implemented for {self.name}.")

    @retrier
    def get_balance(self, currency: str) -> float:

        # ccxt exception is already handled by get_balances
        balances = self.get_balances()
        balance = balances.get(currency)
        if balance is None:
            raise TemporaryError(
                f'Could not get {currency} balance due to malformed exchange response: {balances}')
        return balance['free']

    @retrier
    def get_balances(self) -> dict:

        try:
            balances = self._api.fetch_balance()
            # Remove additional info from ccxt results
            balances.pop("info", None)
            balances.pop("free", None)
            balances.pop("total", None)
            balances.pop("used", None)

            return balances
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get balance due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_tickers(self, cached: bool = False) -> Dict:
        """
        :param cached: Allow cached result
        :return: fetch_tickers result
        """
        if cached:
            tickers = self._fetch_tickers_cache.get('fetch_tickers')
            if tickers:
                return tickers
        try:
            tickers = self._api.fetch_tickers()
            self._fetch_tickers_cache['fetch_tickers'] = tickers
            return tickers
        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching tickers in batch. '
                f'Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not load tickers due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def fetch_ticker(self, pair: str) -> dict:
        try:
            if (pair not in self.markets or
                    self.markets[pair].get('active', False) is False):
                raise ExchangeError(f"Pair {pair} not available")
            data = self._api.fetch_ticker(pair)
            return data
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not load ticker due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def get_historic_ohlcv(self, pair: str, timeframe: str,
                           since_ms: int) -> List:
        """
        Get candle history using asyncio and returns the list of candles.
        Handles all async work for this.
        Async over one pair, assuming we get `self.ohlcv_candle_limit()` candles per call.
        :param pair: Pair to download
        :param timeframe: Timeframe to get data for
        :param since_ms: Timestamp in milliseconds to get history from
        :return: List with candle (OHLCV) data
        """
        return asyncio.get_event_loop().run_until_complete(
            self._async_get_historic_ohlcv(pair=pair, timeframe=timeframe,
                                           since_ms=since_ms))

    def get_historic_ohlcv_as_df(self, pair: str, timeframe: str,
                                 since_ms: int) -> DataFrame:
        """
        Minimal wrapper around get_historic_ohlcv - converting the result into a dataframe
        :param pair: Pair to download
        :param timeframe: Timeframe to get data for
        :param since_ms: Timestamp in milliseconds to get history from
        :return: OHLCV DataFrame
        """
        ticks = self.get_historic_ohlcv(pair, timeframe, since_ms=since_ms)
        return ohlcv_to_dataframe(ticks, timeframe, pair=pair, fill_missing=True,
                                  drop_incomplete=self._ohlcv_partial_candle)

    async def _async_get_historic_ohlcv(self, pair: str,
                                        timeframe: str,
                                        since_ms: int) -> List:
        """
        Download historic ohlcv
        """

        one_call = timeframe_to_msecs(timeframe) * self.ohlcv_candle_limit(timeframe)
        logger.debug(
            "one_call: %s msecs (%s)",
            one_call,
            arrow.utcnow().shift(seconds=one_call // 1000).humanize(only_distance=True)
        )
        input_coroutines = [self._async_get_candle_history(
            pair, timeframe, since) for since in
            range(since_ms, arrow.utcnow().int_timestamp * 1000, one_call)]

        results = await asyncio.gather(*input_coroutines, return_exceptions=True)

        # Combine gathered results
        data: List = []
        for res in results:
            if isinstance(res, Exception):
                logger.warning("Async code raised an exception: %s", res.__class__.__name__)
                continue
            # Deconstruct tuple if it's not an exception
            p, _, new_data = res
            if p == pair:
                data.extend(new_data)
        # Sort data again after extending the result - above calls return in "async order"
        data = sorted(data, key=lambda x: x[0])
        logger.info("Downloaded data for %s with length %s.", pair, len(data))
        return data

    def refresh_latest_ohlcv(self, pair_list: ListPairsWithTimeframes, *,
                             since_ms: Optional[int] = None, cache: bool = True
                             ) -> Dict[Tuple[str, str], DataFrame]:
        """
        Refresh in-memory OHLCV asynchronously and set `_klines` with the result
        Loops asynchronously over pair_list and downloads all pairs async (semi-parallel).
        Only used in the dataprovider.refresh() method.
        :param pair_list: List of 2 element tuples containing pair, interval to refresh
        :param since_ms: time since when to download, in milliseconds
        :param cache: Assign result to _klines. Usefull for one-off downloads like for pairlists
        :return: Dict of [{(pair, timeframe): Dataframe}]
        """
        logger.debug("Refreshing candle (OHLCV) data for %d pairs", len(pair_list))

        input_coroutines = []

        # Gather coroutines to run
        for pair, timeframe in set(pair_list):
            if (((pair, timeframe) not in self._klines)
                    or self._now_is_time_to_refresh(pair, timeframe)):
                input_coroutines.append(self._async_get_candle_history(pair, timeframe,
                                                                       since_ms=since_ms))
            else:
                logger.debug(
                    "Using cached candle (OHLCV) data for pair %s, timeframe %s ...",
                    pair, timeframe
                )

        results = asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*input_coroutines, return_exceptions=True))

        results_df = {}
        # handle caching
        for res in results:
            if isinstance(res, Exception):
                logger.warning("Async code raised an exception: %s", res.__class__.__name__)
                continue
            # Deconstruct tuple (has 3 elements)
            pair, timeframe, ticks = res
            # keeping last candle time as last refreshed time of the pair
            if ticks:
                self._pairs_last_refresh_time[(pair, timeframe)] = ticks[-1][0] // 1000
            # keeping parsed dataframe in cache
            ohlcv_df = ohlcv_to_dataframe(
                    ticks, timeframe, pair=pair, fill_missing=True,
                    drop_incomplete=self._ohlcv_partial_candle)
            results_df[(pair, timeframe)] = ohlcv_df
            if cache:
                self._klines[(pair, timeframe)] = ohlcv_df
        return results_df

    def _now_is_time_to_refresh(self, pair: str, timeframe: str) -> bool:
        # Timeframe in seconds
        interval_in_sec = timeframe_to_seconds(timeframe)

        return not ((self._pairs_last_refresh_time.get((pair, timeframe), 0)
                     + interval_in_sec) >= arrow.utcnow().int_timestamp)

    @retrier_async
    async def _async_get_candle_history(self, pair: str, timeframe: str,
                                        since_ms: Optional[int] = None) -> Tuple[str, str, List]:
        """
        Asynchronously get candle history data using fetch_ohlcv
        returns tuple: (pair, timeframe, ohlcv_list)
        """
        try:
            # Fetch OHLCV asynchronously
            s = '(' + arrow.get(since_ms // 1000).isoformat() + ') ' if since_ms is not None else ''
            logger.debug(
                "Fetching pair %s, interval %s, since %s %s...",
                pair, timeframe, since_ms, s
            )

            data = await self._api_async.fetch_ohlcv(pair, timeframe=timeframe,
                                                     since=since_ms,
                                                     limit=self.ohlcv_candle_limit(timeframe))

            # Some exchanges sort OHLCV in ASC order and others in DESC.
            # Ex: Bittrex returns the list of OHLCV in ASC order (oldest first, newest last)
            # while GDAX returns the list of OHLCV in DESC order (newest first, oldest last)
            # Only sort if necessary to save computing time
            try:
                if data and data[0][0] > data[-1][0]:
                    data = sorted(data, key=lambda x: x[0])
            except IndexError:
                logger.exception("Error loading %s. Result was %s.", pair, data)
                return pair, timeframe, []
            logger.debug("Done fetching pair %s, interval %s ...", pair, timeframe)
            return pair, timeframe, data

        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching historical '
                f'candle (OHLCV) data. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not fetch historical candle (OHLCV) data '
                                 f'for pair {pair} due to {e.__class__.__name__}. '
                                 f'Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(f'Could not fetch historical candle (OHLCV) data '
                                       f'for pair {pair}. Message: {e}') from e

    @retrier_async
    async def _async_fetch_trades(self, pair: str,
                                  since: Optional[int] = None,
                                  params: Optional[dict] = None) -> List[List]:
        """
        Asyncronously gets trade history using fetch_trades.
        Handles exchange errors, does one call to the exchange.
        :param pair: Pair to fetch trade data for
        :param since: Since as integer timestamp in milliseconds
        returns: List of dicts containing trades
        """
        try:
            # fetch trades asynchronously
            if params:
                logger.debug("Fetching trades for pair %s, params: %s ", pair, params)
                trades = await self._api_async.fetch_trades(pair, params=params, limit=1000)
            else:
                logger.debug(
                    "Fetching trades for pair %s, since %s %s...",
                    pair,  since,
                    '(' + arrow.get(since // 1000).isoformat() + ') ' if since is not None else ''
                )
                trades = await self._api_async.fetch_trades(pair, since=since, limit=1000)
            return trades_dict_to_list(trades)
        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching historical trade data.'
                f'Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load trade history due to {e.__class__.__name__}. '
                                 f'Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(f'Could not fetch trade data. Msg: {e}') from e

    async def _async_get_trade_history_id(self, pair: str,
                                          until: int,
                                          since: Optional[int] = None,
                                          from_id: Optional[str] = None) -> Tuple[str, List[List]]:
        """
        Asyncronously gets trade history using fetch_trades
        use this when exchange uses id-based iteration (check `self._trades_pagination`)
        :param pair: Pair to fetch trade data for
        :param since: Since as integer timestamp in milliseconds
        :param until: Until as integer timestamp in milliseconds
        :param from_id: Download data starting with ID (if id is known). Ignores "since" if set.
        returns tuple: (pair, trades-list)
        """

        trades: List[List] = []

        if not from_id:
            # Fetch first elements using timebased method to get an ID to paginate on
            # Depending on the Exchange, this can introduce a drift at the start of the interval
            # of up to an hour.
            # e.g. Binance returns the "last 1000" candles within a 1h time interval
            # - so we will miss the first trades.
            t = await self._async_fetch_trades(pair, since=since)
            # DEFAULT_TRADES_COLUMNS: 0 -> timestamp
            # DEFAULT_TRADES_COLUMNS: 1 -> id
            from_id = t[-1][1]
            trades.extend(t[:-1])
        while True:
            t = await self._async_fetch_trades(pair,
                                               params={self._trades_pagination_arg: from_id})
            if t:
                # Skip last id since its the key for the next call
                trades.extend(t[:-1])
                if from_id == t[-1][1] or t[-1][0] > until:
                    logger.debug(f"Stopping because from_id did not change. "
                                 f"Reached {t[-1][0]} > {until}")
                    # Reached the end of the defined-download period - add last trade as well.
                    trades.extend(t[-1:])
                    break

                from_id = t[-1][1]
            else:
                break

        return (pair, trades)

    async def _async_get_trade_history_time(self, pair: str, until: int,
                                            since: Optional[int] = None) -> Tuple[str, List[List]]:
        """
        Asyncronously gets trade history using fetch_trades,
        when the exchange uses time-based iteration (check `self._trades_pagination`)
        :param pair: Pair to fetch trade data for
        :param since: Since as integer timestamp in milliseconds
        :param until: Until as integer timestamp in milliseconds
        returns tuple: (pair, trades-list)
        """

        trades: List[List] = []
        # DEFAULT_TRADES_COLUMNS: 0 -> timestamp
        # DEFAULT_TRADES_COLUMNS: 1 -> id
        while True:
            t = await self._async_fetch_trades(pair, since=since)
            if t:
                since = t[-1][0]
                trades.extend(t)
                # Reached the end of the defined-download period
                if until and t[-1][0] > until:
                    logger.debug(
                        f"Stopping because until was reached. {t[-1][0]} > {until}")
                    break
            else:
                break

        return (pair, trades)

    async def _async_get_trade_history(self, pair: str,
                                       since: Optional[int] = None,
                                       until: Optional[int] = None,
                                       from_id: Optional[str] = None) -> Tuple[str, List[List]]:
        """
        Async wrapper handling downloading trades using either time or id based methods.
        """

        logger.debug(f"_async_get_trade_history(), pair: {pair}, "
                     f"since: {since}, until: {until}, from_id: {from_id}")

        if until is None:
            until = ccxt.Exchange.milliseconds()
            logger.debug(f"Exchange milliseconds: {until}")

        if self._trades_pagination == 'time':
            return await self._async_get_trade_history_time(
                pair=pair, since=since, until=until)
        elif self._trades_pagination == 'id':
            return await self._async_get_trade_history_id(
                pair=pair, since=since, until=until, from_id=from_id
            )
        else:
            raise OperationalException(f"Exchange {self.name} does use neither time, "
                                       f"nor id based pagination")

    def get_historic_trades(self, pair: str,
                            since: Optional[int] = None,
                            until: Optional[int] = None,
                            from_id: Optional[str] = None) -> Tuple[str, List]:
        """
        Get trade history data using asyncio.
        Handles all async work and returns the list of candles.
        Async over one pair, assuming we get `self.ohlcv_candle_limit()` candles per call.
        :param pair: Pair to download
        :param since: Timestamp in milliseconds to get history from
        :param until: Timestamp in milliseconds. Defaults to current timestamp if not defined.
        :param from_id: Download data starting with ID (if id is known)
        :returns List of trade data
        """
        if not self.exchange_has("fetchTrades"):
            raise OperationalException("This exchange does not suport downloading Trades.")

        return asyncio.get_event_loop().run_until_complete(
            self._async_get_trade_history(pair=pair, since=since,
                                          until=until, from_id=from_id))

    def check_order_canceled_empty(self, order: Dict) -> bool:
        """
        Verify if an order has been cancelled without being partially filled
        :param order: Order dict as returned from fetch_order()
        :return: True if order has been cancelled without being filled, False otherwise.
        """
        return (order.get('status') in ('closed', 'canceled', 'cancelled')
                and order.get('filled') == 0.0)

    @retrier
    def cancel_order(self, order_id: str, pair: str) -> Dict:
        if self._config['dry_run']:
            order = self._dry_run_open_orders.get(order_id)
            if order:
                order.update({'status': 'canceled', 'filled': 0.0, 'remaining': order['amount']})
                return order
            else:
                return {}

        try:
            return self._api.cancel_order(order_id, pair)
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f'Could not cancel order. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not cancel order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    # Assign method to cancel_stoploss_order to allow easy overriding in other classes
    cancel_stoploss_order = cancel_order

    def is_cancel_order_result_suitable(self, corder) -> bool:
        if not isinstance(corder, dict):
            return False

        required = ('fee', 'status', 'amount')
        return all(k in corder for k in required)

    def cancel_order_with_result(self, order_id: str, pair: str, amount: float) -> Dict:
        """
        Cancel order returning a result.
        Creates a fake result if cancel order returns a non-usable result
        and fetch_order does not work (certain exchanges don't return cancelled orders)
        :param order_id: Orderid to cancel
        :param pair: Pair corresponding to order_id
        :param amount: Amount to use for fake response
        :return: Result from either cancel_order if usable, or fetch_order
        """
        try:
            corder = self.cancel_order(order_id, pair)
            if self.is_cancel_order_result_suitable(corder):
                return corder
        except InvalidOrderException:
            logger.warning(f"Could not cancel order {order_id} for {pair}.")
        try:
            order = self.fetch_order(order_id, pair)
        except InvalidOrderException:
            logger.warning(f"Could not fetch cancelled order {order_id}.")
            order = {'fee': {}, 'status': 'canceled', 'amount': amount, 'info': {}}

        return order

    @retrier(retries=API_FETCH_ORDER_RETRY_COUNT)
    def fetch_order(self, order_id: str, pair: str) -> Dict:
        if self._config['dry_run']:
            try:
                order = self._dry_run_open_orders[order_id]
                return order
            except KeyError as e:
                # Gracefully handle errors with dry-run orders.
                raise InvalidOrderException(
                    f'Tried to get an invalid dry-run-order (id: {order_id}). Message: {e}') from e
        try:
            return self._api.fetch_order(order_id, pair)
        except ccxt.OrderNotFound as e:
            raise RetryableOrderError(
                f'Order not found (pair: {pair} id: {order_id}). Message: {e}') from e
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f'Tried to get an invalid order (pair: {pair} id: {order_id}). Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    # Assign method to fetch_stoploss_order to allow easy overriding in other classes
    fetch_stoploss_order = fetch_order

    def fetch_order_or_stoploss_order(self, order_id: str, pair: str,
                                      stoploss_order: bool = False) -> Dict:
        """
        Simple wrapper calling either fetch_order or fetch_stoploss_order depending on
        the stoploss_order parameter
        :param stoploss_order: If true, uses fetch_stoploss_order, otherwise fetch_order.
        """
        if stoploss_order:
            return self.fetch_stoploss_order(order_id, pair)
        return self.fetch_order(order_id, pair)

    @staticmethod
    def get_next_limit_in_list(limit: int, limit_range: Optional[List[int]],
                               range_required: bool = True):
        """
        Get next greater value in the list.
        Used by fetch_l2_order_book if the api only supports a limited range
        """
        if not limit_range:
            return limit

        result = min([x for x in limit_range if limit <= x] + [max(limit_range)])
        if not range_required and limit > result:
            # Range is not required - we can use None as parameter.
            return None
        return result

    @retrier
    def fetch_l2_order_book(self, pair: str, limit: int = 100) -> dict:
        """
        Get L2 order book from exchange.
        Can be limited to a certain amount (if supported).
        Returns a dict in the format
        {'asks': [price, volume], 'bids': [price, volume]}
        """
        limit1 = self.get_next_limit_in_list(limit, self._ft_has['l2_limit_range'],
                                             self._ft_has['l2_limit_range_required'])
        try:

            return self._api.fetch_l2_order_book(pair, limit1)
        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching order book.'
                f'Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get order book due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_trades_for_order(self, order_id: str, pair: str, since: datetime) -> List:
        """
        Fetch Orders using the "fetch_my_trades" endpoint and filter them by order-id.
        The "since" argument passed in is coming from the database and is in UTC,
        as timezone-native datetime object.
        From the python documentation:
            > Naive datetime instances are assumed to represent local time
        Therefore, calling "since.timestamp()" will get the UTC timestamp, after applying the
        transformation from local timezone to UTC.
        This works for timezones UTC+ since then the result will contain trades from a few hours
        instead of from the last 5 seconds, however fails for UTC- timezones,
        since we're then asking for trades with a "since" argument in the future.

        :param order_id order_id: Order-id as given when creating the order
        :param pair: Pair the order is for
        :param since: datetime object of the order creation time. Assumes object is in UTC.
        """
        if self._config['dry_run']:
            return []
        if not self.exchange_has('fetchMyTrades'):
            return []
        try:
            # Allow 5s offset to catch slight time offsets (discovered in #1185)
            # since needs to be int in milliseconds
            my_trades = self._api.fetch_my_trades(
                pair, int((since.replace(tzinfo=timezone.utc).timestamp() - 5) * 1000))
            matched_trades = [trade for trade in my_trades if trade['order'] == order_id]

            return matched_trades
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get trades due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_fee(self, symbol: str, type: str = '', side: str = '', amount: float = 1,
                price: float = 1, taker_or_maker: str = 'maker') -> float:
        try:
            if self._config['dry_run'] and self._config.get('fee', None) is not None:
                return self._config['fee']
            # validate that markets are loaded before trying to get fee
            if self._api.markets is None or len(self._api.markets) == 0:
                self._api.load_markets()

            return self._api.calculate_fee(symbol=symbol, type=type, side=side, amount=amount,
                                           price=price, takerOrMaker=taker_or_maker)['rate']
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get fee info due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @staticmethod
    def order_has_fee(order: Dict) -> bool:
        """
        Verifies if the passed in order dict has the needed keys to extract fees,
        and that these keys (currency, cost) are not empty.
        :param order: Order or trade (one trade) dict
        :return: True if the fee substructure contains currency and cost, false otherwise
        """
        if not isinstance(order, dict):
            return False
        return ('fee' in order and order['fee'] is not None
                and (order['fee'].keys() >= {'currency', 'cost'})
                and order['fee']['currency'] is not None
                and order['fee']['cost'] is not None
                )

    def calculate_fee_rate(self, order: Dict) -> Optional[float]:
        """
        Calculate fee rate if it's not given by the exchange.
        :param order: Order or trade (one trade) dict
        """
        if order['fee'].get('rate') is not None:
            return order['fee'].get('rate')
        fee_curr = order['fee']['currency']
        # Calculate fee based on order details
        if fee_curr in self.get_pair_base_currency(order['symbol']):
            # Base currency - divide by amount
            return round(
                order['fee']['cost'] / safe_value_fallback2(order, order, 'filled', 'amount'), 8)
        elif fee_curr in self.get_pair_quote_currency(order['symbol']):
            # Quote currency - divide by cost
            return round(order['fee']['cost'] / order['cost'], 8) if order['cost'] else None
        else:
            # If Fee currency is a different currency
            if not order['cost']:
                # If cost is None or 0.0 -> falsy, return None
                return None
            try:
                comb = self.get_valid_pair_combination(fee_curr, self._config['stake_currency'])
                tick = self.fetch_ticker(comb)

                fee_to_quote_rate = safe_value_fallback2(tick, tick, 'last', 'ask')
                return round((order['fee']['cost'] * fee_to_quote_rate) / order['cost'], 8)
            except ExchangeError:
                return None

    def extract_cost_curr_rate(self, order: Dict) -> Tuple[float, str, Optional[float]]:
        """
        Extract tuple of cost, currency, rate.
        Requires order_has_fee to run first!
        :param order: Order or trade (one trade) dict
        :return: Tuple with cost, currency, rate of the given fee dict
        """
        return (order['fee']['cost'],
                order['fee']['currency'],
                self.calculate_fee_rate(order))


def is_exchange_known_ccxt(exchange_name: str, ccxt_module: CcxtModuleType = None) -> bool:
    return exchange_name in ccxt_exchanges(ccxt_module)


def is_exchange_officially_supported(exchange_name: str) -> bool:
    return exchange_name in ['bittrex', 'binance', 'kraken']


def ccxt_exchanges(ccxt_module: CcxtModuleType = None) -> List[str]:
    """
    Return the list of all exchanges known to ccxt
    """
    return ccxt_module.exchanges if ccxt_module is not None else ccxt.exchanges


def available_exchanges(ccxt_module: CcxtModuleType = None) -> List[str]:
    """
    Return exchanges available to the bot, i.e. non-bad exchanges in the ccxt list
    """
    exchanges = ccxt_exchanges(ccxt_module)
    return [x for x in exchanges if validate_exchange(x)[0]]


def validate_exchange(exchange: str) -> Tuple[bool, str]:
    ex_mod = getattr(ccxt, exchange.lower())()
    if not ex_mod or not ex_mod.has:
        return False, ''
    missing = [k for k in EXCHANGE_HAS_REQUIRED if ex_mod.has.get(k) is not True]
    if missing:
        return False, f"missing: {', '.join(missing)}"

    missing_opt = [k for k in EXCHANGE_HAS_OPTIONAL if not ex_mod.has.get(k)]

    if exchange.lower() in BAD_EXCHANGES:
        return False, BAD_EXCHANGES.get(exchange.lower(), '')
    if missing_opt:
        return True, f"missing opt: {', '.join(missing_opt)}"

    return True, ''


def validate_exchanges(all_exchanges: bool) -> List[Tuple[str, bool, str]]:
    """
    :return: List of tuples with exchangename, valid, reason.
    """
    exchanges = ccxt_exchanges() if all_exchanges else available_exchanges()
    exchanges_valid = [
        (e, *validate_exchange(e)) for e in exchanges
    ]
    return exchanges_valid


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Translates the timeframe interval value written in the human readable
    form ('1m', '5m', '1h', '1d', '1w', etc.) to the number
    of seconds for one timeframe interval.
    """
    return ccxt.Exchange.parse_timeframe(timeframe)


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Same as timeframe_to_seconds, but returns minutes.
    """
    return ccxt.Exchange.parse_timeframe(timeframe) // 60


def timeframe_to_msecs(timeframe: str) -> int:
    """
    Same as timeframe_to_seconds, but returns milliseconds.
    """
    return ccxt.Exchange.parse_timeframe(timeframe) * 1000


def timeframe_to_prev_date(timeframe: str, date: datetime = None) -> datetime:
    """
    Use Timeframe and determine last possible candle.
    :param timeframe: timeframe in string format (e.g. "5m")
    :param date: date to use. Defaults to utcnow()
    :returns: date of previous candle (with utc timezone)
    """
    if not date:
        date = datetime.now(timezone.utc)

    new_timestamp = ccxt.Exchange.round_timeframe(timeframe, date.timestamp() * 1000,
                                                  ROUND_DOWN) // 1000
    return datetime.fromtimestamp(new_timestamp, tz=timezone.utc)


def timeframe_to_next_date(timeframe: str, date: datetime = None) -> datetime:
    """
    Use Timeframe and determine next candle.
    :param timeframe: timeframe in string format (e.g. "5m")
    :param date: date to use. Defaults to utcnow()
    :returns: date of next candle (with utc timezone)
    """
    if not date:
        date = datetime.now(timezone.utc)
    new_timestamp = ccxt.Exchange.round_timeframe(timeframe, date.timestamp() * 1000,
                                                  ROUND_UP) // 1000
    return datetime.fromtimestamp(new_timestamp, tz=timezone.utc)


def market_is_active(market: Dict) -> bool:
    """
    Return True if the market is active.
    """
    # "It's active, if the active flag isn't explicitly set to false. If it's missing or
    # true then it's true. If it's undefined, then it's most likely true, but not 100% )"
    # See https://github.com/ccxt/ccxt/issues/4874,
    # https://github.com/ccxt/ccxt/issues/4075#issuecomment-434760520
    return market.get('active', True) is not False
