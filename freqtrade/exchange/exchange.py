# pragma pylint: disable=W0603
"""
Cryptocurrency Exchanges support
"""
import asyncio
import inspect
import logging
from copy import deepcopy
from datetime import datetime, timezone
from math import ceil, floor
from random import randint
from typing import Any, Dict, List, Optional, Tuple

import arrow
import ccxt
import ccxt.async_support as ccxt_async
from ccxt.base.decimal_to_precision import ROUND_UP, ROUND_DOWN
from pandas import DataFrame

from freqtrade import (DependencyException, InvalidOrderException,
                       OperationalException, TemporaryError, constants)
from freqtrade.data.converter import parse_ticker_dataframe
from freqtrade.misc import deep_merge_dicts

logger = logging.getLogger(__name__)


API_RETRY_COUNT = 4
BAD_EXCHANGES = {
    "bitmex": "Various reasons.",
    "bitstamp": "Does not provide history. "
                "Details in https://github.com/freqtrade/freqtrade/issues/1983",
    "hitbtc": "This API cannot be used with Freqtrade. "
              "Use `hitbtc2` exchange id to access this exchange.",
    **dict.fromkeys([
        'adara',
        'anxpro',
        'bigone',
        'coinbase',
        'coinexchange',
        'coinmarketcap',
        'lykke',
        'xbtce',
        ], "Does not provide timeframes. ccxt fetchOHLCV: False"),
    **dict.fromkeys([
        'bcex',
        'bit2c',
        'bitbay',
        'bitflyer',
        'bitforex',
        'bithumb',
        'bitso',
        'bitstamp1',
        'bl3p',
        'braziliex',
        'btcbox',
        'btcchina',
        'btctradeim',
        'btctradeua',
        'bxinth',
        'chilebit',
        'coincheck',
        'coinegg',
        'coinfalcon',
        'coinfloor',
        'coingi',
        'coinmate',
        'coinone',
        'coinspot',
        'coolcoin',
        'crypton',
        'deribit',
        'exmo',
        'exx',
        'flowbtc',
        'foxbit',
        'fybse',
        # 'hitbtc',
        'ice3x',
        'independentreserve',
        'indodax',
        'itbit',
        'lakebtc',
        'latoken',
        'liquid',
        'livecoin',
        'luno',
        'mixcoins',
        'negociecoins',
        'nova',
        'paymium',
        'southxchange',
        'stronghold',
        'surbitcoin',
        'therock',
        'tidex',
        'vaultoro',
        'vbtc',
        'virwox',
        'yobit',
        'zaif',
        ], "Does not provide timeframes. ccxt fetchOHLCV: emulated"),
    }


def retrier_async(f):
    async def wrapper(*args, **kwargs):
        count = kwargs.pop('count', API_RETRY_COUNT)
        try:
            return await f(*args, **kwargs)
        except (TemporaryError, DependencyException) as ex:
            logger.warning('%s() returned exception: "%s"', f.__name__, ex)
            if count > 0:
                count -= 1
                kwargs.update({'count': count})
                logger.warning('retrying %s() still for %s times', f.__name__, count)
                return await wrapper(*args, **kwargs)
            else:
                logger.warning('Giving up retrying: %s()', f.__name__)
                raise ex
    return wrapper


def retrier(f):
    def wrapper(*args, **kwargs):
        count = kwargs.pop('count', API_RETRY_COUNT)
        try:
            return f(*args, **kwargs)
        except (TemporaryError, DependencyException) as ex:
            logger.warning('%s() returned exception: "%s"', f.__name__, ex)
            if count > 0:
                count -= 1
                kwargs.update({'count': count})
                logger.warning('retrying %s() still for %s times', f.__name__, count)
                return wrapper(*args, **kwargs)
            else:
                logger.warning('Giving up retrying: %s()', f.__name__)
                raise ex
    return wrapper


class Exchange:

    _config: Dict = {}

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

    }
    _ft_has: Dict = {}

    def __init__(self, config: dict) -> None:
        """
        Initializes this module with the given config,
        it does basic validation whether the specified exchange and pairs are valid.
        :return: None
        """
        self._api: ccxt.Exchange = None
        self._api_async: ccxt_async.Exchange = None

        self._config.update(config)

        self._cached_ticker: Dict[str, Any] = {}

        # Holds last candle refreshed time of each pair
        self._pairs_last_refresh_time: Dict[Tuple[str, str], int] = {}
        # Timestamp of last markets refresh
        self._last_markets_refresh: int = 0

        # Holds candles
        self._klines: Dict[Tuple[str, str], DataFrame] = {}

        # Holds all open sell orders for dry_run
        self._dry_run_open_orders: Dict[str, Any] = {}

        if config['dry_run']:
            logger.info('Instance is running with dry_run enabled')

        exchange_config = config['exchange']

        # Deep merge ft_has with default ft_has options
        self._ft_has = deep_merge_dicts(self._ft_has, deepcopy(self._ft_has_default))
        if exchange_config.get("_ft_has_params"):
            self._ft_has = deep_merge_dicts(exchange_config.get("_ft_has_params"),
                                            self._ft_has)
            logger.info("Overriding exchange._ft_has with config params, result: %s", self._ft_has)

        # Assign this directly for easy access
        self._ohlcv_candle_limit = self._ft_has['ohlcv_candle_limit']
        self._ohlcv_partial_candle = self._ft_has['ohlcv_partial_candle']

        self._trades_pagination = self._ft_has['trades_pagination']
        self._trades_pagination_arg = self._ft_has['trades_pagination_arg']

        # Initialize ccxt objects
        self._api = self._init_ccxt(
            exchange_config, ccxt_kwargs=exchange_config.get('ccxt_config'))
        self._api_async = self._init_ccxt(
            exchange_config, ccxt_async, ccxt_kwargs=exchange_config.get('ccxt_async_config'))

        logger.info('Using Exchange "%s"', self.name)

        # Check if timeframe is available
        self.validate_timeframes(config.get('ticker_interval'))

        # Converts the interval provided in minutes in config to seconds
        self.markets_refresh_interval: int = exchange_config.get(
            "markets_refresh_interval", 60) * 60
        # Initial markets load
        self._load_markets()

        # Check if all pairs are available
        self.validate_pairs(config['exchange']['pair_whitelist'])
        self.validate_ordertypes(config.get('order_types', {}))
        self.validate_order_time_in_force(config.get('order_time_in_force', {}))

    def __del__(self):
        """
        Destructor - clean up async stuff
        """
        logger.debug("Exchange object destroyed, closing async loop")
        if self._api_async and inspect.iscoroutinefunction(self._api_async.close):
            asyncio.get_event_loop().run_until_complete(self._api_async.close())

    def _init_ccxt(self, exchange_config: dict, ccxt_module=ccxt,
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
        if not self._api.markets:
            logger.warning("Markets were not loaded. Loading them now..")
            self._load_markets()
        return self._api.markets

    def klines(self, pair_interval: Tuple[str, str], copy=True) -> DataFrame:
        if pair_interval in self._klines:
            return self._klines[pair_interval].copy() if copy else self._klines[pair_interval]
        else:
            return DataFrame()

    def set_sandbox(self, api, exchange_config: dict, name: str):
        if exchange_config.get('sandbox'):
            if api.urls.get('test'):
                api.urls['api'] = api.urls['test']
                logger.info("Enabled Sandbox API on %s", name)
            else:
                logger.warning(name, "No Sandbox URL in CCXT, exiting. "
                                     "Please check your config.json")
                raise OperationalException(f'Exchange {name} does not provide a sandbox api')

    def _load_async_markets(self, reload=False) -> None:
        try:
            if self._api_async:
                asyncio.get_event_loop().run_until_complete(
                    self._api_async.load_markets(reload=reload))

        except ccxt.BaseError as e:
            logger.warning('Could not load async markets. Reason: %s', e)
            return

    def _load_markets(self) -> None:
        """ Initialize markets both sync and async """
        try:
            self._api.load_markets()
            self._load_async_markets()
            self._last_markets_refresh = arrow.utcnow().timestamp
        except ccxt.BaseError as e:
            logger.warning('Unable to initialize markets. Reason: %s', e)

    def _reload_markets(self) -> None:
        """Reload markets both sync and async, if refresh interval has passed"""
        # Check whether markets have to be reloaded
        if (self._last_markets_refresh > 0) and (
                self._last_markets_refresh + self.markets_refresh_interval
                > arrow.utcnow().timestamp):
            return None
        logger.debug("Performing scheduled market reload..")
        try:
            self._api.load_markets(reload=True)
            self._last_markets_refresh = arrow.utcnow().timestamp
        except ccxt.BaseError:
            logger.exception("Could not reload markets.")

    def validate_pairs(self, pairs: List[str]) -> None:
        """
        Checks if all given pairs are tradable on the current exchange.
        Raises OperationalException if one pair is not available.
        :param pairs: list of pairs
        :return: None
        """

        if not self.markets:
            logger.warning('Unable to validate pairs (assuming they are correct).')
            return

        for pair in pairs:
            # Note: ccxt has BaseCurrency/QuoteCurrency format for pairs
            # TODO: add a support for having coins in BTC/USDT format
            if self.markets and pair not in self.markets:
                raise OperationalException(
                    f'Pair {pair} is not available on {self.name}. '
                    f'Please remove {pair} from your whitelist.')
            elif self.markets[pair].get('info', {}).get('IsRestricted', False):
                # Warn users about restricted pairs in whitelist.
                # We cannot determine reliably if Users are affected.
                logger.warning(f"Pair {pair} is restricted for some users on this exchange."
                               f"Please check if you are impacted by this restriction "
                               f"on the exchange and eventually remove {pair} from your whitelist.")

    def get_valid_pair_combination(self, curr_1, curr_2) -> str:
        """
        Get valid pair combination of curr_1 and curr_2 by trying both combinations.
        """
        for pair in [f"{curr_1}/{curr_2}", f"{curr_2}/{curr_1}"]:
            if pair in self.markets and self.markets[pair].get('active'):
                return pair
        raise DependencyException(f"Could not combine {curr_1} and {curr_2} to get a valid pair.")

    def validate_timeframes(self, timeframe: Optional[str]) -> None:
        """
        Checks if ticker interval from config is a supported timeframe on the exchange
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
                f"Invalid ticker interval '{timeframe}'. This exchange supports: {self.timeframes}")

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

    def exchange_has(self, endpoint: str) -> bool:
        """
        Checks if exchange implements a specific API endpoint.
        Wrapper around ccxt 'has' attribute
        :param endpoint: Name of endpoint (e.g. 'fetchOHLCV', 'fetchTickers')
        :return: bool
        """
        return endpoint in self._api.has and self._api.has[endpoint]

    def symbol_amount_prec(self, pair, amount: float):
        '''
        Returns the amount to buy or sell to a precision the Exchange accepts
        Rounded down
        '''
        if self.markets[pair]['precision']['amount']:
            symbol_prec = self.markets[pair]['precision']['amount']
            big_amount = amount * pow(10, symbol_prec)
            amount = floor(big_amount) / pow(10, symbol_prec)
        return amount

    def symbol_price_prec(self, pair, price: float):
        '''
        Returns the price buying or selling with to the precision the Exchange accepts
        Rounds up
        '''
        if self.markets[pair]['precision']['price']:
            symbol_prec = self.markets[pair]['precision']['price']
            big_price = price * pow(10, symbol_prec)
            price = ceil(big_price) / pow(10, symbol_prec)
        return price

    def dry_run_order(self, pair: str, ordertype: str, side: str, amount: float,
                      rate: float, params: Dict = {}) -> Dict[str, Any]:
        order_id = f'dry_run_{side}_{randint(0, 10**6)}'
        dry_order = {
            "id": order_id,
            'pair': pair,
            'price': rate,
            'amount': amount,
            "cost": amount * rate,
            'type': ordertype,
            'side': side,
            'remaining': amount,
            'datetime': arrow.utcnow().isoformat(),
            'status': "closed" if ordertype == "market" else "open",
            'fee': None,
            "info": {}
        }
        self._store_dry_order(dry_order)
        # Copy order and close it - so the returned order is open unless it's a market order
        return dry_order

    def _store_dry_order(self, dry_order: Dict) -> None:
        closed_order = dry_order.copy()
        if closed_order["type"] in ["market", "limit"]:
            closed_order.update({
                "status": "closed",
                "filled": closed_order["amount"],
                "remaining": 0
                })
        if closed_order["type"] in ["stop_loss_limit"]:
            closed_order["info"].update({"stopPrice": closed_order["price"]})
        self._dry_run_open_orders[closed_order["id"]] = closed_order

    def create_order(self, pair: str, ordertype: str, side: str, amount: float,
                     rate: float, params: Dict = {}) -> Dict:
        try:
            # Set the precision for amount and price(rate) as accepted by the exchange
            amount = self.symbol_amount_prec(pair, amount)
            needs_price = (ordertype != 'market'
                           or self._api.options.get("createMarketBuyOrderRequiresPrice", False))
            rate = self.symbol_price_prec(pair, rate) if needs_price else None

            return self._api.create_order(pair, ordertype, side,
                                          amount, rate, params)

        except ccxt.InsufficientFunds as e:
            raise DependencyException(
                f'Insufficient funds to create {ordertype} {side} order on market {pair}.'
                f'Tried to {side} amount {amount} at rate {rate}.'
                f'Message: {e}') from e
        except ccxt.InvalidOrder as e:
            raise DependencyException(
                f'Could not create {ordertype} {side} order on market {pair}.'
                f'Tried to {side} amount {amount} at rate {rate}.'
                f'Message: {e}') from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not place {side} order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def buy(self, pair: str, ordertype: str, amount: float,
            rate: float, time_in_force) -> Dict:

        if self._config['dry_run']:
            dry_order = self.dry_run_order(pair, ordertype, "buy", amount, rate)
            return dry_order

        params = self._params.copy()
        if time_in_force != 'gtc' and ordertype != 'market':
            params.update({'timeInForce': time_in_force})

        return self.create_order(pair, ordertype, 'buy', amount, rate, params)

    def sell(self, pair: str, ordertype: str, amount: float,
             rate: float, time_in_force='gtc') -> Dict:

        if self._config['dry_run']:
            dry_order = self.dry_run_order(pair, ordertype, "sell", amount, rate)
            return dry_order

        params = self._params.copy()
        if time_in_force != 'gtc' and ordertype != 'market':
            params.update({'timeInForce': time_in_force})

        return self.create_order(pair, ordertype, 'sell', amount, rate, params)

    def stoploss_limit(self, pair: str, amount: float, stop_price: float, rate: float) -> Dict:
        """
        creates a stoploss limit order.
        Since ccxt does not unify stoploss-limit orders yet, this needs to be implemented in each
        exchange's subclass.
        The exception below should never raise, since we disallow
        starting the bot in validate_ordertypes()
        Note: Changes to this interface need to be applied to all sub-classes too.
        """

        raise OperationalException(f"stoploss_limit is not implemented for {self.name}.")

    @retrier
    def get_balance(self, currency: str) -> float:
        if self._config['dry_run']:
            return constants.DRY_RUN_WALLET

        # ccxt exception is already handled by get_balances
        balances = self.get_balances()
        balance = balances.get(currency)
        if balance is None:
            raise TemporaryError(
                f'Could not get {currency} balance due to malformed exchange response: {balances}')
        return balance['free']

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

            return balances
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get balance due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_tickers(self) -> Dict:
        try:
            return self._api.fetch_tickers()
        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching tickers in batch.'
                f'Message: {e}') from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not load tickers due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_ticker(self, pair: str, refresh: Optional[bool] = True) -> dict:
        if refresh or pair not in self._cached_ticker.keys():
            try:
                if pair not in self._api.markets or not self._api.markets[pair].get('active'):
                    raise DependencyException(f"Pair {pair} not available")
                data = self._api.fetch_ticker(pair)
                try:
                    self._cached_ticker[pair] = {
                        'bid': float(data['bid']),
                        'ask': float(data['ask']),
                    }
                except KeyError:
                    logger.debug("Could not cache ticker data for %s", pair)
                return data
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                raise TemporaryError(
                    f'Could not load ticker due to {e.__class__.__name__}. Message: {e}') from e
            except ccxt.BaseError as e:
                raise OperationalException(e) from e
        else:
            logger.info("returning cached ticker-data for %s", pair)
            return self._cached_ticker[pair]

    def get_historic_ohlcv(self, pair: str, ticker_interval: str,
                           since_ms: int) -> List:
        """
        Gets candle history using asyncio and returns the list of candles.
        Handles all async doing.
        Async over one pair, assuming we get `_ohlcv_candle_limit` candles per call.
        :param pair: Pair to download
        :param ticker_interval: Interval to get
        :param since_ms: Timestamp in milliseconds to get history from
        :returns List of tickers
        """
        return asyncio.get_event_loop().run_until_complete(
            self._async_get_historic_ohlcv(pair=pair, ticker_interval=ticker_interval,
                                           since_ms=since_ms))

    async def _async_get_historic_ohlcv(self, pair: str,
                                        ticker_interval: str,
                                        since_ms: int) -> List:

        one_call = timeframe_to_msecs(ticker_interval) * self._ohlcv_candle_limit
        logger.debug(
            "one_call: %s msecs (%s)",
            one_call,
            arrow.utcnow().shift(seconds=one_call // 1000).humanize(only_distance=True)
        )
        input_coroutines = [self._async_get_candle_history(
            pair, ticker_interval, since) for since in
            range(since_ms, arrow.utcnow().timestamp * 1000, one_call)]

        tickers = await asyncio.gather(*input_coroutines, return_exceptions=True)

        # Combine tickers
        data: List = []
        for p, ticker_interval, ticker in tickers:
            if p == pair:
                data.extend(ticker)
        # Sort data again after extending the result - above calls return in "async order"
        data = sorted(data, key=lambda x: x[0])
        logger.info("downloaded %s with length %s.", pair, len(data))
        return data

    def refresh_latest_ohlcv(self, pair_list: List[Tuple[str, str]]) -> List[Tuple[str, List]]:
        """
        Refresh in-memory ohlcv asynchronously and set `_klines` with the result
        Loops asynchronously over pair_list and downloads all pairs async (semi-parallel).
        :param pair_list: List of 2 element tuples containing pair, interval to refresh
        :return: Returns a List of ticker-dataframes.
        """
        logger.debug("Refreshing ohlcv data for %d pairs", len(pair_list))

        input_coroutines = []

        # Gather coroutines to run
        for pair, ticker_interval in set(pair_list):
            if (not ((pair, ticker_interval) in self._klines)
                    or self._now_is_time_to_refresh(pair, ticker_interval)):
                input_coroutines.append(self._async_get_candle_history(pair, ticker_interval))
            else:
                logger.debug(
                    "Using cached ohlcv data for pair %s, interval %s ...",
                    pair, ticker_interval
                )

        tickers = asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*input_coroutines, return_exceptions=True))

        # handle caching
        for res in tickers:
            if isinstance(res, Exception):
                logger.warning("Async code raised an exception: %s", res.__class__.__name__)
                continue
            pair = res[0]
            ticker_interval = res[1]
            ticks = res[2]
            # keeping last candle time as last refreshed time of the pair
            if ticks:
                self._pairs_last_refresh_time[(pair, ticker_interval)] = ticks[-1][0] // 1000
            # keeping parsed dataframe in cache
            self._klines[(pair, ticker_interval)] = parse_ticker_dataframe(
                ticks, ticker_interval, pair=pair, fill_missing=True,
                drop_incomplete=self._ohlcv_partial_candle)
        return tickers

    def _now_is_time_to_refresh(self, pair: str, ticker_interval: str) -> bool:
        # Calculating ticker interval in seconds
        interval_in_sec = timeframe_to_seconds(ticker_interval)

        return not ((self._pairs_last_refresh_time.get((pair, ticker_interval), 0)
                     + interval_in_sec) >= arrow.utcnow().timestamp)

    @retrier_async
    async def _async_get_candle_history(self, pair: str, ticker_interval: str,
                                        since_ms: Optional[int] = None) -> Tuple[str, str, List]:
        """
        Asynchronously gets candle histories using fetch_ohlcv
        returns tuple: (pair, ticker_interval, ohlcv_list)
        """
        try:
            # fetch ohlcv asynchronously
            s = '(' + arrow.get(since_ms // 1000).isoformat() + ') ' if since_ms is not None else ''
            logger.debug(
                "Fetching pair %s, interval %s, since %s %s...",
                pair, ticker_interval, since_ms, s
            )

            data = await self._api_async.fetch_ohlcv(pair, timeframe=ticker_interval,
                                                     since=since_ms)

            # Because some exchange sort Tickers ASC and other DESC.
            # Ex: Bittrex returns a list of tickers ASC (oldest first, newest last)
            # when GDAX returns a list of tickers DESC (newest first, oldest last)
            # Only sort if necessary to save computing time
            try:
                if data and data[0][0] > data[-1][0]:
                    data = sorted(data, key=lambda x: x[0])
            except IndexError:
                logger.exception("Error loading %s. Result was %s.", pair, data)
                return pair, ticker_interval, []
            logger.debug("Done fetching pair %s, interval %s ...", pair, ticker_interval)
            return pair, ticker_interval, data

        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching historical candlestick data.'
                f'Message: {e}') from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load ticker history due to {e.__class__.__name__}. '
                                 f'Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(f'Could not fetch ticker data. Msg: {e}') from e

    @retrier_async
    async def _async_fetch_trades(self, pair: str,
                                  since: Optional[int] = None,
                                  params: Optional[dict] = None) -> List[Dict]:
        """
        Asyncronously gets trade history using fetch_trades.
        Handles exchange errors, does one call to the exchange.
        :param pair: Pair to fetch trade data for
        :param since: Since as integer timestamp in milliseconds
        returns tuple: (pair, ticker_interval, ohlcv_list)
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
            return trades
        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching historical trade data.'
                f'Message: {e}') from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load trade history due to {e.__class__.__name__}. '
                                 f'Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(f'Could not fetch trade data. Msg: {e}') from e

    async def _async_get_trade_history_id(self, pair: str,
                                          until: int,
                                          since: Optional[int] = None,
                                          from_id: Optional[str] = None) -> Tuple[str, List[Dict]]:
        """
        Asyncronously gets trade history using fetch_trades
        use this when exchange uses id-based iteration
        :param pair: Pair to fetch trade data for
        :param since: Since as integer timestamp in milliseconds
        :param until: Until as integer timestamp in milliseconds
        :param from_id: Download data starting with ID (if id is known). Ignores "since" if set.
        returns tuple: (pair, ticker_interval, ohlcv_list)
        """
        if self._trades_pagination == 'time':
            raise OperationalException(f"Wrong method called to get trades for {self.name}")

        trades: List[Dict] = []

        if not from_id:
            # Fetch first elements using timebased method to get an ID to paginate on
            # Depending on the Exchange, this can introduce a drift at the start of the interval
            # of up to an hour.
            # e.g. Binance returns the "last 1000" candles within a 1h time interval
            # - so we will miss the first trades.
            t = await self._async_fetch_trades(pair, since=since)
            from_id = t[-1]['id']
            trades.extend(t[:-1])
        while True:
            t = await self._async_fetch_trades(pair,
                                               params={self._trades_pagination_arg: from_id})
            if len(t):
                # Skip last id since its the key for the next call
                trades.extend(t[:-1])
                if from_id == t[-1]['id'] or t[-1]['timestamp'] > until:
                    logger.debug(f"Stopping because from_id did not change. "
                                 f"Reached {t[-1]['timestamp']} > {until}")
                    # Reached the end of the defined-download period
                    break

                from_id = t[-1]['id']
            else:
                break

        return (pair, trades)

    async def _async_get_trade_history_time(self, pair: str, until: int,
                                            since: Optional[int] = None) -> Tuple[str, List]:
        """
        Asyncronously gets trade history using fetch_trades,
        when the exchange uses time-based iteration
        :param pair: Pair to fetch trade data for
        :param since: Since as integer timestamp in milliseconds
        :param until: Until as integer timestamp in milliseconds
        returns tuple: (pair, ticker_interval, ohlcv_list)
        """
        if self._trades_pagination != 'time':
            raise OperationalException(f"Wrong method called to get trades for {self.name}")

        trades: List[Dict] = []
        while True:
            t = await self._async_fetch_trades(pair, since=since)
            if len(t):
                since = t[-1]['timestamp']
                trades.extend(t)
                # Reached the end of the defined-download period
                if until and t[-1]['timestamp'] > until:
                    logger.debug(
                        f"Stopping because until was reached. {t[-1]['timestamp']} > {until}")
                    break
            else:
                break

        return (pair, trades)

    async def _async_get_trade_history(self, pair: str,
                                       since: Optional[int] = None,
                                       until: Optional[int] = None,
                                       from_id: Optional[str] = None) -> Tuple[str, List[Dict]]:
        """
        Async wrapper handling downloading trades using either time or id based methods.
        """
        if not self.exchange_has("fetchTrades"):
            # TODO: Maybe don't stop the bot ... ?
            raise OperationalException("This exchange does not suport downloading Trades.")
        try:
            if self._trades_pagination == 'time':
                return await self._async_get_trade_history_time(
                    pair=pair, since=since,
                    until=until or ccxt.Exchange.milliseconds())
            elif self._trades_pagination == 'id':
                return await self._async_get_trade_history_id(
                    pair=pair, since=since,
                    until=until or ccxt.Exchange.milliseconds(), from_id=from_id
                )
            else:
                raise OperationalException(f"Exchange {self.name} does use neither time, "
                                           f"nor id based pagination")

        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching historical trade data.'
                f'Message: {e}') from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load trade history due to {e.__class__.__name__}. '
                                 f'Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(f'Could not fetch trade data. Msg: {e}') from e

    def get_historic_trades(self, pair: str,
                            since: Optional[int] = None,
                            until: Optional[int] = None,
                            from_id: Optional[str] = None) -> Tuple[str, List]:
        """
        Gets candle history using asyncio and returns the list of candles.
        Handles all async doing.
        Async over one pair, assuming we get `_ohlcv_candle_limit` candles per call.
        :param pair: Pair to download
        :param ticker_interval: Interval to get
        :param since: Timestamp in milliseconds to get history from
        :param until: Timestamp in milliseconds. Defaults to current timestamp if not defined.
        :param from_id: Download data starting with ID (if id is known)
        :returns List of tickers
        """

        return asyncio.get_event_loop().run_until_complete(
            self._async_get_trade_history(pair=pair, since=since,
                                          until=until, from_id=from_id))

    @retrier
    def cancel_order(self, order_id: str, pair: str) -> None:
        if self._config['dry_run']:
            return

        try:
            return self._api.cancel_order(order_id, pair)
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f'Could not cancel order. Message: {e}') from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not cancel order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_order(self, order_id: str, pair: str) -> Dict:
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
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f'Tried to get an invalid order (id: {order_id}). Message: {e}') from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_order_book(self, pair: str, limit: int = 100) -> dict:
        """
        get order book level 2 from exchange

        Notes:
        20180619: bittrex doesnt support limits -.-
        """
        try:

            return self._api.fetch_l2_order_book(pair, limit)
        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching order book.'
                f'Message: {e}') from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get order book due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_trades_for_order(self, order_id: str, pair: str, since: datetime) -> List:
        if self._config['dry_run']:
            return []
        if not self.exchange_has('fetchMyTrades'):
            return []
        try:
            # Allow 5s offset to catch slight time offsets (discovered in #1185)
            # since needs to be int in milliseconds
            my_trades = self._api.fetch_my_trades(pair, int((since.timestamp() - 5) * 1000))
            matched_trades = [trade for trade in my_trades if trade['order'] == order_id]

            return matched_trades

        except ccxt.NetworkError as e:
            raise TemporaryError(
                f'Could not get trades due to networking error. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_fee(self, symbol='ETH/BTC', type='', side='', amount=1,
                price=1, taker_or_maker='maker') -> float:
        try:
            # validate that markets are loaded before trying to get fee
            if self._api.markets is None or len(self._api.markets) == 0:
                self._api.load_markets()

            return self._api.calculate_fee(symbol=symbol, type=type, side=side, amount=amount,
                                           price=price, takerOrMaker=taker_or_maker)['rate']
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get fee info due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def build_ohlcv(self, trades: List[Dict], timeframe: str, since: int = None,
                    limit: int = None) -> List:
        """
        Build ohlcv data from trade list.
        trade-list has to be in the ccxt format, which is a list of dicts containing at least:
            * timestamp
            * price
            * amount
        :param trades: List of Dicts
        :param timeframe: timeframe to convert to (e.g. "5m")
        :param since: start at a specific data, as oposed to the trades-list start date
        :param limit: Limit amount of candles
        :return: ohlcv data (as returned by ccxt.fetch_ohlcv)
        """
        return self._api.build_ohlcv(trades, timeframe, since, limit)


def is_exchange_bad(exchange_name: str) -> bool:
    return exchange_name in BAD_EXCHANGES


def get_exchange_bad_reason(exchange_name: str) -> str:
    return BAD_EXCHANGES.get(exchange_name, "")


def is_exchange_known_ccxt(exchange_name: str, ccxt_module=None) -> bool:
    return exchange_name in ccxt_exchanges(ccxt_module)


def is_exchange_officially_supported(exchange_name: str) -> bool:
    return exchange_name in ['bittrex', 'binance']


def ccxt_exchanges(ccxt_module=None) -> List[str]:
    """
    Return the list of all exchanges known to ccxt
    """
    return ccxt_module.exchanges if ccxt_module is not None else ccxt.exchanges


def available_exchanges(ccxt_module=None) -> List[str]:
    """
    Return exchanges available to the bot, i.e. non-bad exchanges in the ccxt list
    """
    exchanges = ccxt_exchanges(ccxt_module)
    return [x for x in exchanges if not is_exchange_bad(x)]


def timeframe_to_seconds(ticker_interval: str) -> int:
    """
    Translates the timeframe interval value written in the human readable
    form ('1m', '5m', '1h', '1d', '1w', etc.) to the number
    of seconds for one timeframe interval.
    """
    return ccxt.Exchange.parse_timeframe(ticker_interval)


def timeframe_to_minutes(ticker_interval: str) -> int:
    """
    Same as timeframe_to_seconds, but returns minutes.
    """
    return ccxt.Exchange.parse_timeframe(ticker_interval) // 60


def timeframe_to_msecs(ticker_interval: str) -> int:
    """
    Same as timeframe_to_seconds, but returns milliseconds.
    """
    return ccxt.Exchange.parse_timeframe(ticker_interval) * 1000


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
