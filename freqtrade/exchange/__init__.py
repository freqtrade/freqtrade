# pragma pylint: disable=W0603
""" Cryptocurrency Exchanges support """
import logging
from random import randint
from typing import List, Dict, Any, Optional
from datetime import datetime

import ccxt
import arrow

from freqtrade import constants, OperationalException, DependencyException, TemporaryError

logger = logging.getLogger(__name__)

API_RETRY_COUNT = 4


# Urls to exchange markets, insert quote and base with .format()
_EXCHANGE_URLS = {
    ccxt.bittrex.__name__: '/Market/Index?MarketName={quote}-{base}',
    ccxt.binance.__name__: '/tradeDetail.html?symbol={base}_{quote}'
}


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


def init_ccxt(exchange_config: dict) -> ccxt.Exchange:
    """
    Initialize ccxt with given config and return valid
    ccxt instance.
    :param config: config to use
    :return: ccxt
    """
    # Find matching class for the given exchange name
    name = exchange_config['name']

    if name not in ccxt.exchanges:
        raise OperationalException(f'Exchange {name} is not supported')
    try:
        api = getattr(ccxt, name.lower())({
            'apiKey': exchange_config.get('key'),
            'secret': exchange_config.get('secret'),
            'password': exchange_config.get('password'),
            'uid': exchange_config.get('uid', ''),
            'enableRateLimit': True,
        })
    except (KeyError, AttributeError):
        raise OperationalException(f'Exchange {name} is not supported')

    return api


class Exchange(object):

    # Current selected exchange
    _API: ccxt.Exchange = None
    _CONF: Dict = {}
    _CACHED_TICKER: Dict[str, Any] = {}

    # Holds all open sell orders for dry_run
    _DRY_RUN_OPEN_ORDERS: Dict[str, Any] = {}

    def __init__(self, config: dict) -> None:
        """
        Initializes this module with the given config,
        it does basic validation whether the specified
        exchange and pairs are valid.
        :param config: config to use
        :return: None
        """
        self._API

        self._CONF.update(config)

        if config['dry_run']:
            logger.info('Instance is running with dry_run enabled')

        exchange_config = config['exchange']
        self._API = init_ccxt(exchange_config)

        logger.info('Using Exchange "%s"', self.get_name())

        # Check if all pairs are available
        self.validate_pairs(config['exchange']['pair_whitelist'])

    def get_name(self) -> str:
        return self._API.name

    def get_id(self) -> str:
        return self._API.id

    def validate_pairs(self, pairs: List[str]) -> None:
        """
        Checks if all given pairs are tradable on the current exchange.
        Raises OperationalException if one pair is not available.
        :param pairs: list of pairs
        :return: None
        """

        try:
            markets = self._API.load_markets()
        except ccxt.BaseError as e:
            logger.warning('Unable to validate pairs (assuming they are correct). Reason: %s', e)
            return

        stake_cur = self._CONF['stake_currency']
        for pair in pairs:
            # Note: ccxt has BaseCurrency/QuoteCurrency format for pairs
            # TODO: add a support for having coins in BTC/USDT format
            if not pair.endswith(stake_cur):
                raise OperationalException(
                    f'Pair {pair} not compatible with stake_currency: {stake_cur}')
            if pair not in markets:
                raise OperationalException(
                    f'Pair {pair} is not available at {self.get_name()}')

    def exchange_has(self, endpoint: str) -> bool:
        """
        Checks if exchange implements a specific API endpoint.
        Wrapper around ccxt 'has' attribute
        :param endpoint: Name of endpoint (e.g. 'fetchOHLCV', 'fetchTickers')
        :return: bool
        """
        return endpoint in self._API.has and self._API.has[endpoint]

    def buy(self, pair: str, rate: float, amount: float) -> Dict:
        if self._CONF['dry_run']:
            order_id = f'dry_run_buy_{randint(0, 10**6)}'
            self._DRY_RUN_OPEN_ORDERS[order_id] = {
                'pair': pair,
                'price': rate,
                'amount': amount,
                'type': 'limit',
                'side': 'buy',
                'remaining': 0.0,
                'datetime': arrow.utcnow().isoformat(),
                'status': 'closed',
                'fee': None
            }
            return {'id': order_id}

        try:
            return self._API.create_limit_buy_order(pair, amount, rate)
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

    def sell(self, pair: str, rate: float, amount: float) -> Dict:
        if self._CONF['dry_run']:
            order_id = f'dry_run_sell_{randint(0, 10**6)}'
            self._DRY_RUN_OPEN_ORDERS[order_id] = {
                'pair': pair,
                'price': rate,
                'amount': amount,
                'type': 'limit',
                'side': 'sell',
                'remaining': 0.0,
                'datetime': arrow.utcnow().isoformat(),
                'status': 'closed'
            }
            return {'id': order_id}

        try:
            return self._API.create_limit_sell_order(pair, amount, rate)
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

    @retrier
    def get_balance(self, currency: str) -> float:
        if self._CONF['dry_run']:
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
        if self._CONF['dry_run']:
            return {}

        try:
            balances = self._API.fetch_balance()
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
            return self._API.fetch_tickers()
        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._API.name} does not support fetching tickers in batch.'
                f'Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not load tickers due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    @retrier
    def get_ticker(self, pair: str, refresh: Optional[bool] = True) -> dict:
        if refresh or pair not in self._CACHED_TICKER.keys():
            try:
                data = self._API.fetch_ticker(pair)
                try:
                    self._CACHED_TICKER[pair] = {
                        'bid': float(data['bid']),
                        'ask': float(data['ask']),
                    }
                except KeyError:
                    logger.debug("Could not cache ticker data for %s", pair)
                return data
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                raise TemporaryError(
                    f'Could not load ticker history due to {e.__class__.__name__}. Message: {e}')
            except ccxt.BaseError as e:
                raise OperationalException(e)
        else:
            logger.info("returning cached ticker-data for %s", pair)
            return self._CACHED_TICKER[pair]

    @retrier
    def get_ticker_history(self, pair: str, tick_interval: str,
                           since_ms: Optional[int] = None) -> List[Dict]:
        try:
            # last item should be in the time interval [now - tick_interval, now]
            till_time_ms = arrow.utcnow().shift(
                            minutes=-constants.TICKER_INTERVAL_MINUTES[tick_interval]
                        ).timestamp * 1000
            # it looks as if some exchanges return cached data
            # and they update it one in several minute, so 10 mins interval
            # is necessary to skeep downloading of an empty array when all
            # chached data was already downloaded
            till_time_ms = min(till_time_ms, arrow.utcnow().shift(minutes=-10).timestamp * 1000)

            data: List[Dict[Any, Any]] = []
            while not since_ms or since_ms < till_time_ms:
                data_part = self._API.fetch_ohlcv(pair, timeframe=tick_interval, since=since_ms)

                # Because some exchange sort Tickers ASC and other DESC.
                # Ex: Bittrex returns a list of tickers ASC (oldest first, newest last)
                # when GDAX returns a list of tickers DESC (newest first, oldest last)
                data_part = sorted(data_part, key=lambda x: x[0])

                if not data_part:
                    break

                logger.debug('Downloaded data for %s time range [%s, %s]',
                             pair,
                             arrow.get(data_part[0][0] / 1000).format(),
                             arrow.get(data_part[-1][0] / 1000).format())

                data.extend(data_part)
                since_ms = data[-1][0] + 1

            return data
        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._API.name} does not support fetching historical candlestick data.'
                f'Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not load ticker history due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(f'Could not fetch ticker data. Msg: {e}')

    @retrier
    def cancel_order(self, order_id: str, pair: str) -> None:
        if self._CONF['dry_run']:
            return

        try:
            return self._API.cancel_order(order_id, pair)
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
        if self._CONF['dry_run']:
            order = self._DRY_RUN_OPEN_ORDERS[order_id]
            order.update({
                'id': order_id
            })
            return order
        try:
            return self._API.fetch_order(order_id, pair)
        except ccxt.InvalidOrder as e:
            raise DependencyException(
                f'Could not get order. Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get order due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    @retrier
    def get_trades_for_order(self, order_id: str, pair: str, since: datetime) -> List:
        if self._CONF['dry_run']:
            return []
        if not self.exchange_has('fetchMyTrades'):
            return []
        try:
            my_trades = self._API.fetch_my_trades(pair, since.timestamp())
            matched_trades = [trade for trade in my_trades if trade['order'] == order_id]

            return matched_trades

        except ccxt.NetworkError as e:
            raise TemporaryError(
                f'Could not get trades due to networking error. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    def get_pair_detail_url(self, pair: str) -> str:
        try:
            url_base = self._API.urls.get('www')
            base, quote = pair.split('/')

            return url_base + _EXCHANGE_URLS[self._API.id].format(base=base, quote=quote)
        except KeyError:
            logger.warning('Could not get exchange url for %s', self.get_name())
            return ""

    @retrier
    def get_markets(self) -> List[dict]:
        try:
            return self._API.fetch_markets()
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not load markets due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    @retrier
    def get_fee(self, symbol='ETH/BTC', type='', side='', amount=1,
                price=1, taker_or_maker='maker') -> float:
        try:
            # validate that markets are loaded before trying to get fee
            if self._API.markets is None or len(self._API.markets) == 0:
                self._API.load_markets()

            return self._API.calculate_fee(symbol=symbol, type=type, side=side, amount=amount,
                                           price=price, takerOrMaker=taker_or_maker)['rate']
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get fee info due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    def get_amount_lots(self, pair: str, amount: float) -> float:
        """
        get buyable amount rounding, ..
        """
        # validate that markets are loaded before trying to get fee
        if not self._API.markets:
            self._API.load_markets()
        return self._API.amount_to_lots(pair, amount)
