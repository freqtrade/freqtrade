import logging
import datetime
import json
import http
from typing import List, Dict, Optional

from binance.client import Client as _Binance
from binance.exceptions import BinanceAPIException
from binance.enums import *
from decimal import Decimal

from freqtrade import OperationalException
from freqtrade.exchange.interface import Exchange

logger = logging.getLogger(__name__)

_API: _Binance = None
_EXCHANGE_CONF: dict = {}
_CONF: dict = {}


class Binance(Exchange):
    """
    Binance API wrapper.
    """
    # Base URL and API endpoints
    BASE_URL: str = 'https://www.binance.com'

    def __init__(self, config: dict) -> None:
        global _API, _EXCHANGE_CONF, _CONF

        _EXCHANGE_CONF.update(config)

        _API = _Binance(_EXCHANGE_CONF['key'], _EXCHANGE_CONF['secret'])

    def _pair_to_symbol(self, pair, seperator='') -> str:
        """
        Turns freqtrade pair into Binance symbol
        - Freqtrade pair = <stake_currency>_<currency>
          i.e.: BTC_XALT
        - Binance symbol = <currency><stake_currency>
          i.e.: XALTBTC
        """

        pair_currencies = pair.split('_')

        return '{0}{1}{2}'.format(pair_currencies[1], seperator, pair_currencies[0])

    def _symbol_to_pair(self, symbol) -> str:
        """
        Turns Binance symbol into freqtrade pair
        - Freqtrade pair = <stake_currency>_<currency>
          i.e.: BTC_XALT
        - Binance symbol = <currency><stake_currency>
          i.e.: XALTBTC
        """
        stake = _EXCHANGE_CONF['stake_currency']

        symbol_stake_currency = symbol[-len(stake):]
        symbol_currency = symbol[:-len(stake)]

        return '{0}_{1}'.format(symbol_stake_currency, symbol_currency)

    @staticmethod
    def _handle_exception(excepter) -> None:
        """
        Validates the given Binance response/exception
        and raises a ContentDecodingError if a non-fatal issue happened.
        """
        # Could to alternate exception handling for specific exceptions/errors
        # See: http://python-binance.readthedocs.io/en/latest/binance.html#module-binance.exceptions
        if type(excepter) == http.client.RemoteDisconnected:
            logger.info(
                'Got HTTP error from Binance: %s' % excepter
            )
            return True

        if type(excepter) == json.decoder.JSONDecodeError:
            logger.info(
                'Got JSON error from Binance: %s' % excepter
            )
            return True

        if type(excepter) == BinanceAPIException:

            logger.info(
                'Got API error from Binance: %s' % excepter
            )

            return True

        raise type(excepter)(excepter.args)

    @property
    def fee(self) -> float:
        # 0.1 %: See https://support.binance.com/hc/en-us
        #             /articles/115000429332-Fee-Structure-on-Binance
        return 0.001

    def buy(self, pair: str, rate: float, amount: float) -> str:

        symbol = self._pair_to_symbol(pair)

        try:
            data = _API.order_limit_buy(
                       symbol=symbol,
                       quantity="{0:.8f}".format(amount),
                       price="{0:.8f}".format(rate))
        except Exception as e:
            Binance._handle_exception(e)
            raise OperationalException('{message} params=({pair}, {rate}, {amount})'.format(
                message=str(e),
                pair=pair,
                rate=Decimal(rate),
                amount=Decimal(amount)))

        return data['orderId']

    def sell(self, pair: str, rate: float, amount: float) -> str:

        symbol = self._pair_to_symbol(pair)

        try:
            data = _API.order_limit_sell(
                       symbol=symbol,
                       quantity="{0:.8f}".format(amount),
                       price="{0:.8f}".format(rate))
        except Exception as e:
            Binance._handle_exception(e)
            raise OperationalException(
                '{message} params=({pair}, {rate}, {amount})'.format(
                    message=str(e),
                    pair=pair,
                    rate=rate,
                    amount=amount))

        return data['orderId']

    def get_balance(self, currency: str) -> float:

        try:
            data = _API.get_asset_balance(asset=currency)
        except Exception as e:
            Binance._handle_exception(e)
            raise OperationalException('{message} params=({currency})'.format(
                message=str(e),
                currency=currency))

        return float(data['free'] or 0.0)

    def get_balances(self) -> List[Dict]:

        try:
            data = _API.get_account()
        except Exception as e:
            Binance._handle_exception(e)
            raise OperationalException('{message}'.format(message=str(e)))

        balances = data['balances']

        currency_balances = []
        for currency in balances:
            balance = {}

            if float(currency['free']) == 0 and float(currency['locked']) == 0:
                continue
            balance['Currency'] = currency.pop('asset')
            balance['Available'] = currency.pop('free')
            balance['Pending'] = currency.pop('locked')
            balance['Balance'] = float(balance['Available']) + float(balance['Pending'])

            currency_balances.append(balance)

        return currency_balances

    def get_ticker(self, pair: str, refresh: Optional[bool] = True) -> dict:

        symbol = self._pair_to_symbol(pair)

        try:
            data = _API.get_ticker(symbol=symbol)
        except Exception as e:
            Binance._handle_exception(e)
            raise OperationalException('{message} params=({pair})'.format(
                message=str(e),
                pair=pair))

        return {
            'bid': float(data['bidPrice']),
            'ask': float(data['askPrice']),
            'last': float(data['lastPrice']),
        }

    def get_ticker_history(self, pair: str, tick_interval: int) -> List[Dict]:

        INTERVAL_ENUM = eval('KLINE_INTERVAL_' + str(tick_interval) + 'MINUTE')

        if INTERVAL_ENUM in ['', None]:
            raise ValueError('Cannot parse tick_interval: {}'.format(tick_interval))

        symbol = self._pair_to_symbol(pair)

        try:
            data = _API.get_klines(symbol=symbol, interval=INTERVAL_ENUM)
        except Exception as e:
            Binance._handle_exception(e)
            raise OperationalException('{message} params=({pair})'.format(
                message=str(e),
                pair=pair))

        tick_data = []

        for tick in data:
            t = {}
            t['O'] = float(tick[1])
            t['H'] = float(tick[2])
            t['L'] = float(tick[3])
            t['C'] = float(tick[4])
            t['V'] = float(tick[5])
            t['T'] = datetime.datetime.fromtimestamp(int(tick[6])/1000).isoformat()
            t['BV'] = float(tick[7])

            tick_data.append(t)

        return tick_data

    def get_order(self, order_id: str, pair: str) -> Dict:

        symbol = self._pair_to_symbol(pair)

        try:
            data = _API.get_all_orders(symbol=symbol, orderId=order_id)
        except Exception as e:
            Binance._handle_exception(e)
            raise OperationalException(
                '{message} params=({symbol},{order_id})'.format(
                    message=str(e),
                    symbol=symbol,
                    order_id=order_id))

        order = {}

        for o in data:

            if o['orderId'] == int(order_id):

                order['id'] = o['orderId']
                order['type'] = "{}_{}".format(o['type'], o['side'])
                order['pair'] = self._symbol_to_pair(o['symbol'])
                order['opened'] = datetime.datetime.fromtimestamp(
                                      int(o['time'])/1000).isoformat()
                order['closed'] = datetime.datetime.fromtimestamp(
                                      int(o['time'])/1000).isoformat()\
                    if o['status'] == 'FILLED' else None
                order['rate'] = float(o['price'])
                order['amount'] = float(o['origQty'])
                order['remaining'] = int(float(o['origQty']) - float(o['executedQty']))

        return order

    def cancel_order(self, order_id: str, pair: str) -> None:

        symbol = self._pair_to_symbol(pair)

        try:
            data = _API.cancel_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            Binance._handle_exception(e)
            raise OperationalException('{message} params=({order_id})'.format(
                message=str(e),
                order_id=order_id))

        return data

    def get_pair_detail_url(self, pair: str) -> str:
        symbol = self._pair_to_symbol(pair, '_')
        return 'https://www.binance.com/indexSpa.html#/trade/index?symbol={}'.format(symbol)

    def get_markets(self) -> List[str]:
        try:
            data = _API.get_all_tickers()
        except Exception as e:
            Binance._handle_exception(e)
            raise OperationalException('{message}'.format(message=str(e)))

        markets = []

        stake = _EXCHANGE_CONF['stake_currency']

        for t in data:
            symbol = t['symbol']
            symbol_stake_currency = symbol[-len(stake):]

            if symbol_stake_currency == stake:
                pair = self._symbol_to_pair(symbol)
                markets.append(pair)

        return markets

    def get_market_summaries(self) -> List[Dict]:

        try:
            data = _API.get_ticker()
        except Exception as e:
            Binance._handle_exception(e)
            raise OperationalException('{message}'.format(message=str(e)))

        market_summaries = []

        for t in data:
            market = {}

            # Looks like this one is only one actually used
            market['MarketName'] = self._symbol_to_pair(t['symbol'])

            market['High'] = t['highPrice']
            market['Low'] = t['lowPrice']
            market['Volume'] = t['volume']
            market['Last'] = t['lastPrice']
            market['TimeStamp'] = t['closeTime']
            market['BaseVolume'] = t['volume']
            market['Bid'] = t['bidPrice']
            market['Ask'] = t['askPrice']
            market['OpenBuyOrders'] = None  # TODO: Implement me (or dont care)
            market['OpenSellOrders'] = None  # TODO: Implement me (or dont care)
            market['PrevDay'] = t['prevClosePrice']
            market['Created'] = None  # TODO: Implement me (or don't care)

            market_summaries.append(market)

        return market_summaries

    def get_trade_qty(self, pair: str) -> tuple:

        try:
            data = _API.get_exchange_info()
        except Exception as e:
            Binance._handle_exception(e)
            raise OperationalException('{message}'.format(message=str(e)))

        symbol = self._pair_to_symbol(pair)

        for s in data['symbols']:

            if symbol == s['symbol']:

                for f in s['filters']:

                    if f['filterType'] == 'LOT_SIZE':

                        return (float(f['minQty']), float(f['maxQty']), float(f['stepSize']))

        return (None, None, None)

    def get_wallet_health(self) -> List[Dict]:

        try:
            data = _API.get_exchange_info()
        except Exception as e:
            Binance._handle_exception(e)
            raise OperationalException('{message}'.format(message=str(e)))

        wallet_health = []

        for s in data['symbols']:
            wallet = {}
            wallet['Currency'] = s['baseAsset']
            wallet['IsActive'] = True if s['status'] == 'TRADING' else False
            wallet['LastChecked'] = None  # TODO
            wallet['Notice'] = s['status'] if s['status'] != 'TRADING' else ''

            wallet_health.append(wallet)

        return wallet_health
