# pragma pylint: disable=W0603
""" Cryptocurrency Exchanges support """
import logging
import inspect
from random import randint
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from math import floor, ceil

import arrow
import asyncio
import ccxt
import ccxt.async_support as ccxt_async
from pandas import DataFrame

from freqtrade import constants, OperationalException, DependencyException, TemporaryError
from freqtrade.data.converter import parse_ticker_dataframe

logger = logging.getLogger(__name__)

API_RETRY_COUNT = 4


# Urls to exchange markets, insert quote and base with .format()
_EXCHANGE_URLS = {
    ccxt.bittrex.__name__: '/Market/Index?MarketName={quote}-{base}',
    ccxt.binance.__name__: '/tradeDetail.html?symbol={base}_{quote}',
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


class Exchange(object):

    _conf: Dict = {}
    _params: Dict = {}

    # Dict to specify which options each exchange implements
    # TODO: this should be merged with attributes from subclasses
    # To avoid having to copy/paste this to all subclasses.
    _ft_has = {
        "stoploss_on_exchange": False,
    }

    def __init__(self, config: dict) -> None:
        """
        Initializes this module with the given config,
        it does basic validation whether the specified exchange and pairs are valid.
        :return: None
        """
        self._conf.update(config)

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
        self._api: ccxt.Exchange = self._init_ccxt(
            exchange_config, ccxt_kwargs=exchange_config.get('ccxt_config'))
        self._api_async: ccxt_async.Exchange = self._init_ccxt(
            exchange_config, ccxt_async, ccxt_kwargs=exchange_config.get('ccxt_async_config'))

        logger.info('Using Exchange "%s"', self.name)

        # Converts the interval provided in minutes in config to seconds
        self.markets_refresh_interval: int = exchange_config.get(
            "markets_refresh_interval", 60) * 60
        # Initial markets load
        self._load_markets()

        # Check if all pairs are available
        self.validate_pairs(config['exchange']['pair_whitelist'])
        self.validate_ordertypes(config.get('order_types', {}))
        self.validate_order_time_in_force(config.get('order_time_in_force', {}))
        self.validate_trailing_stoploss(config)

        if config.get('ticker_interval'):
            # Check if timeframe is available
            self.validate_timeframes(config['ticker_interval'])

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

        if name not in ccxt_module.exchanges:
            raise OperationalException(f'Exchange {name} is not supported')

        ex_config = {
            'apiKey': exchange_config.get('key'),
            'secret': exchange_config.get('secret'),
            'password': exchange_config.get('password'),
            'uid': exchange_config.get('uid', ''),
            'enableRateLimit': exchange_config.get('ccxt_rate_limit', True)
        }
        if ccxt_kwargs:
            logger.info('Applying additional ccxt config: %s', ccxt_kwargs)
            ex_config.update(ccxt_kwargs)
        try:

            api = getattr(ccxt_module, name.lower())(ex_config)
        except (KeyError, AttributeError):
            raise OperationalException(f'Exchange {name} is not supported')

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
        self._api.load_markets(reload=True)
        self._last_markets_refresh = arrow.utcnow().timestamp

    def validate_pairs(self, pairs: List[str]) -> None:
        """
        Checks if all given pairs are tradable on the current exchange.
        Raises OperationalException if one pair is not available.
        :param pairs: list of pairs
        :return: None
        """

        if not self.markets:
            logger.warning('Unable to validate pairs (assuming they are correct).')
        #     return

        stake_cur = self._conf['stake_currency']
        for pair in pairs:
            # Note: ccxt has BaseCurrency/QuoteCurrency format for pairs
            # TODO: add a support for having coins in BTC/USDT format
            if not pair.endswith(stake_cur):
                raise OperationalException(
                    f'Pair {pair} not compatible with stake_currency: {stake_cur}')
            if self.markets and pair not in self.markets:
                raise OperationalException(
                    f'Pair {pair} is not available on {self.name}. '
                    f'Please remove {pair} from your whitelist.')

    def validate_timeframes(self, timeframe: List[str]) -> None:
        """
        Checks if ticker interval from config is a supported timeframe on the exchange
        """
        timeframes = self._api.timeframes
        if timeframe not in timeframes:
            raise OperationalException(
                f'Invalid ticker {timeframe}, this Exchange supports {timeframes}')

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
                'On exchange stoploss is not supported for %s.' % self.name
            )

    def validate_order_time_in_force(self, order_time_in_force: Dict) -> None:
        """
        Checks if order time in force configured in strategy/config are supported
        """
        if any(v != 'gtc' for k, v in order_time_in_force.items()):
            if self.name != 'Binance':
                raise OperationalException(
                    f'Time in force policies are not supporetd for  {self.name} yet.')

    def validate_trailing_stoploss(self, config) -> None:
        """
        Validates the trailing stoploss configuration
        """
        # Skip if trailing stoploss is not activated
        if not config.get('trailing_stop', False):
            return

        tsl_positive = float(config.get('trailing_stop_positive', 0))
        tsl_offset = float(config.get('trailing_stop_positive_offset', 0))
        tsl_only_offset = config.get('trailing_only_offset_is_reached', False)

        if tsl_only_offset:
            if tsl_positive == 0.0:
                raise OperationalException(
                    f'The config trailing_only_offset_is_reached need '
                    'trailing_stop_positive_offset to be more than 0 in your config.')
        if tsl_positive > 0 and 0 < tsl_offset <= tsl_positive:
            raise OperationalException(
                f'The config trailing_stop_positive_offset need '
                'to be greater than trailing_stop_positive_offset in your config.')

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
        dry_order = {  # TODO: additional entry should be added for stoploss limit
            "id": order_id,
            'pair': pair,
            'price': rate,
            'amount': amount,
            "cost": amount * rate,
            'type': ordertype,
            'side': side,
            'remaining': amount,
            'datetime': arrow.utcnow().isoformat(),
            'status': "open",
            'fee': None,
            "info": {}
        }
        self._store_dry_order(dry_order)
        return dry_order

    def _store_dry_order(self, dry_order: Dict) -> None:
        closed_order = dry_order.copy()
        if closed_order["type"] in ["market", "limit"]:
            closed_order.update({
                "status": "closed",
                "filled": closed_order["amount"],
                "remaining": 0
                })
        self._dry_run_open_orders[closed_order["id"]] = closed_order

    def create_order(self, pair: str, ordertype: str, side: str, amount: float,
                     rate: float, params: Dict = {}) -> Dict:
        try:
            # Set the precision for amount and price(rate) as accepted by the exchange
            amount = self.symbol_amount_prec(pair, amount)
            rate = self.symbol_price_prec(pair, rate) if ordertype != 'market' else None

            return self._api.create_order(pair, ordertype, side,
                                          amount, rate, params)

        except ccxt.InsufficientFunds as e:
            raise DependencyException(
                f'Insufficient funds to create {ordertype} {side} order on market {pair}.'
                f'Tried to {side} amount {amount} at rate {rate} (total {rate*amount}).'
                f'Message: {e}')
        except ccxt.InvalidOrder as e:
            raise DependencyException(
                f'Could not create {ordertype} {side} order on market {pair}.'
                f'Tried to {side} amount {amount} at rate {rate} (total {rate*amount}).'
                f'Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not place {side} order due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    def buy(self, pair: str, ordertype: str, amount: float,
            rate: float, time_in_force) -> Dict:

        if self._conf['dry_run']:
            dry_order = self.dry_run_order(pair, ordertype, "buy", amount, rate)
            return dry_order

        params = self._params.copy()
        if time_in_force != 'gtc' and ordertype != 'market':
            params.update({'timeInForce': time_in_force})

        return self.create_order(pair, ordertype, 'buy', amount, rate, params)

    def sell(self, pair: str, ordertype: str, amount: float,
             rate: float, time_in_force='gtc') -> Dict:

        if self._conf['dry_run']:
            dry_order = self.dry_run_order(pair, ordertype, "sell", amount, rate)
            return dry_order

        params = self._params.copy()
        if time_in_force != 'gtc' and ordertype != 'market':
            params.update({'timeInForce': time_in_force})

        return self.create_order(pair, ordertype, 'sell', amount, rate, params)

    def stoploss_limit(self, pair: str, amount: float, stop_price: float, rate: float) -> Dict:
        """
        creates a stoploss limit order.
        NOTICE: it is not supported by all exchanges. only binance is tested for now.
        TODO: implementation maybe needs to be moved to the binance subclass
        """
        ordertype = "stop_loss_limit"

        stop_price = self.symbol_price_prec(pair, stop_price)

        # Ensure rate is less than stop price
        if stop_price <= rate:
            raise OperationalException(
                'In stoploss limit order, stop price should be more than limit price')

        if self._conf['dry_run']:
            dry_order = self.dry_run_order(
                pair, ordertype, "sell", amount, stop_price)
            return dry_order

        params = self._params.copy()
        params.update({'stopPrice': stop_price})

        order = self.create_order(pair, ordertype, 'sell', amount, rate, params)
        logger.info('stoploss limit order added for %s. '
                    'stop price: %s. limit: %s' % (pair, stop_price, rate))
        return order

    @retrier
    def get_balance(self, currency: str) -> float:
        if self._conf['dry_run']:
            return 999.9

        # ccxt exception is already handled by get_balances
        balances = self.get_balances()
        balance = balances.get(currency)
        if balance is None:
            raise TemporaryError(
                f'Could not get {currency} balance due to malformed exchange response: {balances}')
        return balance['free']

    @retrier
    def get_balances(self) -> dict:
        if self._conf['dry_run']:
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
                f'Could not get balance due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    @retrier
    def get_tickers(self) -> Dict:
        try:
            return self._api.fetch_tickers()
        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching tickers in batch.'
                f'Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not load tickers due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    @retrier
    def get_ticker(self, pair: str, refresh: Optional[bool] = True) -> dict:
        if refresh or pair not in self._cached_ticker.keys():
            try:
                if pair not in self._api.markets:
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
                    f'Could not load ticker due to {e.__class__.__name__}. Message: {e}')
            except ccxt.BaseError as e:
                raise OperationalException(e)
        else:
            logger.info("returning cached ticker-data for %s", pair)
            return self._cached_ticker[pair]

    def get_history(self, pair: str, tick_interval: str,
                    since_ms: int) -> List:
        """
        Gets candle history using asyncio and returns the list of candles.
        Handles all async doing.
        """
        return asyncio.get_event_loop().run_until_complete(
            self._async_get_history(pair=pair, tick_interval=tick_interval,
                                    since_ms=since_ms))

    async def _async_get_history(self, pair: str,
                                 tick_interval: str,
                                 since_ms: int) -> List:
        # Assume exchange returns 500 candles
        _LIMIT = 500

        one_call = constants.TICKER_INTERVAL_MINUTES[tick_interval] * 60 * _LIMIT * 1000
        logger.debug("one_call: %s", one_call)
        input_coroutines = [self._async_get_candle_history(
            pair, tick_interval, since) for since in
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
        Refresh in-memory ohlcv asyncronously and set `_klines` with the result
        """
        logger.debug("Refreshing ohlcv data for %d pairs", len(pair_list))

        input_coroutines = []

        # Gather coroutines to run
        for pair, ticker_interval in set(pair_list):
            if (not ((pair, ticker_interval) in self._klines)
                    or self._now_is_time_to_refresh(pair, ticker_interval)):
                input_coroutines.append(self._async_get_candle_history(pair, ticker_interval))
            else:
                logger.debug("Using cached ohlcv data for %s, %s ...", pair, ticker_interval)

        tickers = asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*input_coroutines, return_exceptions=True))

        # handle caching
        for res in tickers:
            if isinstance(res, Exception):
                logger.warning("Async code raised an exception: %s", res.__class__.__name__)
                continue
            pair = res[0]
            tick_interval = res[1]
            ticks = res[2]
            # keeping last candle time as last refreshed time of the pair
            if ticks:
                self._pairs_last_refresh_time[(pair, tick_interval)] = ticks[-1][0] // 1000
            # keeping parsed dataframe in cache
            self._klines[(pair, tick_interval)] = parse_ticker_dataframe(
                ticks, tick_interval, fill_missing=True)
        return tickers

    def _now_is_time_to_refresh(self, pair: str, ticker_interval: str) -> bool:
        # Calculating ticker interval in seconds
        interval_in_sec = constants.TICKER_INTERVAL_MINUTES[ticker_interval] * 60

        return not ((self._pairs_last_refresh_time.get((pair, ticker_interval), 0)
                     + interval_in_sec) >= arrow.utcnow().timestamp)

    @retrier_async
    async def _async_get_candle_history(self, pair: str, tick_interval: str,
                                        since_ms: Optional[int] = None) -> Tuple[str, str, List]:
        """
        Asyncronously gets candle histories using fetch_ohlcv
        returns tuple: (pair, tick_interval, ohlcv_list)
        """
        try:
            # fetch ohlcv asynchronously
            logger.debug("fetching %s, %s since %s ...", pair, tick_interval, since_ms)

            data = await self._api_async.fetch_ohlcv(pair, timeframe=tick_interval,
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
                return pair, tick_interval, []
            logger.debug("done fetching %s, %s ...", pair, tick_interval)
            return pair, tick_interval, data

        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching historical candlestick data.'
                f'Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not load ticker history due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(f'Could not fetch ticker data. Msg: {e}')

    @retrier
    def cancel_order(self, order_id: str, pair: str) -> None:
        if self._conf['dry_run']:
            return

        try:
            return self._api.cancel_order(order_id, pair)
        except ccxt.InvalidOrder as e:
            raise DependencyException(
                f'Could not cancel order. Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not cancel order due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    @retrier
    def get_order(self, order_id: str, pair: str) -> Dict:
        if self._conf['dry_run']:
            order = self._dry_run_open_orders[order_id]
            return order
        try:
            return self._api.fetch_order(order_id, pair)
        except ccxt.InvalidOrder as e:
            raise DependencyException(
                f'Could not get order. Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get order due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

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
                f'Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get order book due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    @retrier
    def get_trades_for_order(self, order_id: str, pair: str, since: datetime) -> List:
        if self._conf['dry_run']:
            return []
        if not self.exchange_has('fetchMyTrades'):
            return []
        try:
            # Allow 5s offset to catch slight time offsets (discovered in #1185)
            my_trades = self._api.fetch_my_trades(pair, since.timestamp() - 5)
            matched_trades = [trade for trade in my_trades if trade['order'] == order_id]

            return matched_trades

        except ccxt.NetworkError as e:
            raise TemporaryError(
                f'Could not get trades due to networking error. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

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
                f'Could not get fee info due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)
