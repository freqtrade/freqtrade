"""
This module contains class to define a RPC communications
"""
import logging
from abc import abstractmethod
from datetime import timedelta, datetime, date
from decimal import Decimal
from enum import Enum
from typing import Dict, Any, List, Optional

import arrow
import sqlalchemy as sql
from numpy import mean, nan_to_num
from pandas import DataFrame

from freqtrade import TemporaryError
from freqtrade.fiat_convert import CryptoToFiatConverter
from freqtrade.misc import shorten_date
from freqtrade.persistence import Trade
from freqtrade.state import State
from freqtrade.strategy.interface import SellType

logger = logging.getLogger(__name__)


class RPCMessageType(Enum):
    STATUS_NOTIFICATION = 'status'
    WARNING_NOTIFICATION = 'warning'
    CUSTOM_NOTIFICATION = 'custom'
    BUY_NOTIFICATION = 'buy'
    SELL_NOTIFICATION = 'sell'

    def __repr__(self):
        return self.value


class RPCException(Exception):
    """
    Should be raised with a rpc-formatted message in an _rpc_* method
    if the required state is wrong, i.e.:

    raise RPCException('*Status:* `no active trade`')
    """
    def __init__(self, message: str) -> None:
        super().__init__(self)
        self.message = message

    def __str__(self):
        return self.message


class RPC(object):
    """
    RPC class can be used to have extra feature, like bot data, and access to DB data
    """
    # Bind _fiat_converter if needed in each RPC handler
    _fiat_converter: Optional[CryptoToFiatConverter] = None

    def __init__(self, freqtrade) -> None:
        """
        Initializes all enabled rpc modules
        :param freqtrade: Instance of a freqtrade bot
        :return: None
        """
        self._freqtrade = freqtrade

    @property
    def name(self) -> str:
        """ Returns the lowercase name of the implementation """
        return self.__class__.__name__.lower()

    @abstractmethod
    def cleanup(self) -> None:
        """ Cleanup pending module resources """

    @abstractmethod
    def send_msg(self, msg: Dict[str, str]) -> None:
        """ Sends a message to all registered rpc modules """

    def _rpc_trade_status(self) -> List[Dict[str, Any]]:
        """
        Below follows the RPC backend it is prefixed with rpc_ to raise awareness that it is
        a remotely exposed function
        """
        # Fetch open trade
        trades = Trade.query.filter(Trade.is_open.is_(True)).all()
        if self._freqtrade.state != State.RUNNING:
            raise RPCException('trader is not running')
        elif not trades:
            raise RPCException('no active trade')
        else:
            results = []
            for trade in trades:
                order = None
                if trade.open_order_id:
                    order = self._freqtrade.exchange.get_order(trade.open_order_id, trade.pair)
                # calculate profit and send message to user
                current_rate = self._freqtrade.exchange.get_ticker(trade.pair, False)['bid']
                current_profit = trade.calc_profit_percent(current_rate)
                fmt_close_profit = (f'{round(trade.close_profit * 100, 2):.2f}%'
                                    if trade.close_profit else None)
                results.append(dict(
                    trade_id=trade.id,
                    pair=trade.pair,
                    market_url=self._freqtrade.exchange.get_pair_detail_url(trade.pair),
                    date=arrow.get(trade.open_date),
                    open_rate=trade.open_rate,
                    close_rate=trade.close_rate,
                    current_rate=current_rate,
                    amount=round(trade.amount, 8),
                    close_profit=fmt_close_profit,
                    current_profit=round(current_profit * 100, 2),
                    open_order='({} {} rem={:.8f})'.format(
                      order['type'], order['side'], order['remaining']
                    ) if order else None,
                ))
            return results

    def _rpc_status_table(self) -> DataFrame:
        trades = Trade.query.filter(Trade.is_open.is_(True)).all()
        if self._freqtrade.state != State.RUNNING:
            raise RPCException('trader is not running')
        elif not trades:
            raise RPCException('no active order')
        else:
            trades_list = []
            for trade in trades:
                # calculate profit and send message to user
                current_rate = self._freqtrade.exchange.get_ticker(trade.pair, False)['bid']
                trade_perc = (100 * trade.calc_profit_percent(current_rate))
                trades_list.append([
                    trade.id,
                    trade.pair,
                    shorten_date(arrow.get(trade.open_date).humanize(only_distance=True)),
                    f'{trade_perc:.2f}%'
                ])

            columns = ['ID', 'Pair', 'Since', 'Profit']
            df_statuses = DataFrame.from_records(trades_list, columns=columns)
            df_statuses = df_statuses.set_index(columns[0])
            return df_statuses

    def _rpc_daily_profit(
            self, timescale: int,
            stake_currency: str, fiat_display_currency: str) -> List[List[Any]]:
        today = datetime.utcnow().date()
        profit_days: Dict[date, Dict] = {}

        if not (isinstance(timescale, int) and timescale > 0):
            raise RPCException('timescale must be an integer greater than 0')

        for day in range(0, timescale):
            profitday = today - timedelta(days=day)
            trades = Trade.query \
                .filter(Trade.is_open.is_(False)) \
                .filter(Trade.close_date >= profitday)\
                .filter(Trade.close_date < (profitday + timedelta(days=1)))\
                .order_by(Trade.close_date)\
                .all()
            curdayprofit = sum(trade.calc_profit() for trade in trades)
            profit_days[profitday] = {
                'amount': f'{curdayprofit:.8f}',
                'trades': len(trades)
            }

        return [
            [
                key,
                '{value:.8f} {symbol}'.format(
                    value=float(value['amount']),
                    symbol=stake_currency
                ),
                '{value:.3f} {symbol}'.format(
                    value=self._fiat_converter.convert_amount(
                        value['amount'],
                        stake_currency,
                        fiat_display_currency
                    ) if self._fiat_converter else 0,
                    symbol=fiat_display_currency
                ),
                '{value} trade{s}'.format(
                    value=value['trades'],
                    s='' if value['trades'] < 2 else 's'
                ),
            ]
            for key, value in profit_days.items()
        ]

    def _rpc_trade_statistics(
            self, stake_currency: str, fiat_display_currency: str) -> Dict[str, Any]:
        """ Returns cumulative profit statistics """
        trades = Trade.query.order_by(Trade.id).all()

        profit_all_coin = []
        profit_all_percent = []
        profit_closed_coin = []
        profit_closed_percent = []
        durations = []

        for trade in trades:
            current_rate: float = 0.0

            if not trade.open_rate:
                continue
            if trade.close_date:
                durations.append((trade.close_date - trade.open_date).total_seconds())

            if not trade.is_open:
                profit_percent = trade.calc_profit_percent()
                profit_closed_coin.append(trade.calc_profit())
                profit_closed_percent.append(profit_percent)
            else:
                # Get current rate
                current_rate = self._freqtrade.exchange.get_ticker(trade.pair, False)['bid']
                profit_percent = trade.calc_profit_percent(rate=current_rate)

            profit_all_coin.append(
                trade.calc_profit(rate=Decimal(trade.close_rate or current_rate))
            )
            profit_all_percent.append(profit_percent)

        best_pair = Trade.session.query(
            Trade.pair, sql.func.sum(Trade.close_profit).label('profit_sum')
        ).filter(Trade.is_open.is_(False)) \
            .group_by(Trade.pair) \
            .order_by(sql.text('profit_sum DESC')).first()

        if not best_pair:
            raise RPCException('no closed trade')

        bp_pair, bp_rate = best_pair

        # Prepare data to display
        profit_closed_coin_sum = round(sum(profit_closed_coin), 8)
        profit_closed_percent = round(nan_to_num(mean(profit_closed_percent)) * 100, 2)
        profit_closed_fiat = self._fiat_converter.convert_amount(
            profit_closed_coin_sum,
            stake_currency,
            fiat_display_currency
        ) if self._fiat_converter else 0

        profit_all_coin_sum = round(sum(profit_all_coin), 8)
        profit_all_percent = round(nan_to_num(mean(profit_all_percent)) * 100, 2)
        profit_all_fiat = self._fiat_converter.convert_amount(
            profit_all_coin_sum,
            stake_currency,
            fiat_display_currency
        ) if self._fiat_converter else 0

        num = float(len(durations) or 1)
        return {
            'profit_closed_coin': profit_closed_coin_sum,
            'profit_closed_percent': profit_closed_percent,
            'profit_closed_fiat': profit_closed_fiat,
            'profit_all_coin': profit_all_coin_sum,
            'profit_all_percent': profit_all_percent,
            'profit_all_fiat': profit_all_fiat,
            'trade_count': len(trades),
            'first_trade_date': arrow.get(trades[0].open_date).humanize(),
            'latest_trade_date': arrow.get(trades[-1].open_date).humanize(),
            'avg_duration': str(timedelta(seconds=sum(durations) / num)).split('.')[0],
            'best_pair': bp_pair,
            'best_rate': round(bp_rate * 100, 2),
        }

    def _rpc_balance(self, fiat_display_currency: str) -> Dict:
        """ Returns current account balance per crypto """
        output = []
        total = 0.0
        for coin, balance in self._freqtrade.exchange.get_balances().items():
            if not balance['total']:
                continue

            if coin == 'BTC':
                rate = 1.0
            else:
                try:
                    if coin == 'USDT':
                        rate = 1.0 / self._freqtrade.exchange.get_ticker('BTC/USDT', False)['bid']
                    else:
                        rate = self._freqtrade.exchange.get_ticker(coin + '/BTC', False)['bid']
                except TemporaryError:
                    continue
            est_btc: float = rate * balance['total']
            total = total + est_btc
            output.append({
                'currency': coin,
                'available': balance['free'],
                'balance': balance['total'],
                'pending': balance['used'],
                'est_btc': est_btc,
            })
        if total == 0.0:
            raise RPCException('all balances are zero')

        symbol = fiat_display_currency
        value = self._fiat_converter.convert_amount(total, 'BTC',
                                                    symbol) if self._fiat_converter else 0
        return {
            'currencies': output,
            'total': total,
            'symbol': symbol,
            'value': value,
        }

    def _rpc_start(self) -> Dict[str, str]:
        """ Handler for start """
        if self._freqtrade.state == State.RUNNING:
            return {'status': 'already running'}

        self._freqtrade.state = State.RUNNING
        return {'status': 'starting trader ...'}

    def _rpc_stop(self) -> Dict[str, str]:
        """ Handler for stop """
        if self._freqtrade.state == State.RUNNING:
            self._freqtrade.state = State.STOPPED
            return {'status': 'stopping trader ...'}

        return {'status': 'already stopped'}

    def _rpc_reload_conf(self) -> Dict[str, str]:
        """ Handler for reload_conf. """
        self._freqtrade.state = State.RELOAD_CONF
        return {'status': 'reloading config ...'}

    def _rpc_forcesell(self, trade_id) -> None:
        """
        Handler for forcesell <id>.
        Sells the given trade at current price
        """
        def _exec_forcesell(trade: Trade) -> None:
            # Check if there is there is an open order
            if trade.open_order_id:
                order = self._freqtrade.exchange.get_order(trade.open_order_id, trade.pair)

                # Cancel open LIMIT_BUY orders and close trade
                if order and order['status'] == 'open' \
                        and order['type'] == 'limit' \
                        and order['side'] == 'buy':
                    self._freqtrade.exchange.cancel_order(trade.open_order_id, trade.pair)
                    trade.close(order.get('price') or trade.open_rate)
                    # Do the best effort, if we don't know 'filled' amount, don't try selling
                    if order['filled'] is None:
                        return
                    trade.amount = order['filled']

                # Ignore trades with an attached LIMIT_SELL order
                if order and order['status'] == 'open' \
                        and order['type'] == 'limit' \
                        and order['side'] == 'sell':
                    return

            # Get current rate and execute sell
            current_rate = self._freqtrade.exchange.get_ticker(trade.pair, False)['bid']
            self._freqtrade.execute_sell(trade, current_rate, SellType.FORCE_SELL)
        # ---- EOF def _exec_forcesell ----

        if self._freqtrade.state != State.RUNNING:
            raise RPCException('trader is not running')

        if trade_id == 'all':
            # Execute sell for all open orders
            for trade in Trade.query.filter(Trade.is_open.is_(True)).all():
                _exec_forcesell(trade)
            return

        # Query for trade
        trade = Trade.query.filter(
            sql.and_(
                Trade.id == trade_id,
                Trade.is_open.is_(True)
            )
        ).first()
        if not trade:
            logger.warning('forcesell: Invalid argument received')
            raise RPCException('invalid argument')

        _exec_forcesell(trade)
        Trade.session.flush()

    def _rpc_forcebuy(self, pair: str, price: Optional[float]) -> Optional[Trade]:
        """
        Handler for forcebuy <asset> <price>
        Buys a pair trade at the given or current price
        """

        if not self._freqtrade.config.get('forcebuy_enable', False):
            raise RPCException('Forcebuy not enabled.')

        if self._freqtrade.state != State.RUNNING:
            raise RPCException('trader is not running')

        # Check pair is in stake currency
        stake_currency = self._freqtrade.config.get('stake_currency')
        if not pair.endswith(stake_currency):
            raise RPCException(
                f'Wrong pair selected. Please pairs with stake {stake_currency} pairs only')
        # check if valid pair

        # check if pair already has an open pair
        trade = Trade.query.filter(Trade.is_open.is_(True)).filter(Trade.pair.is_(pair)).first()
        if trade:
            raise RPCException(f'position for {pair} already open - id: {trade.id}')

        # gen stake amount
        stakeamount = self._freqtrade._get_trade_stake_amount()

        # execute buy
        if self._freqtrade.execute_buy(pair, stakeamount, price):
            trade = Trade.query.filter(Trade.is_open.is_(True)).filter(Trade.pair.is_(pair)).first()
            return trade
        else:
            return None

    def _rpc_performance(self) -> List[Dict]:
        """
        Handler for performance.
        Shows a performance statistic from finished trades
        """
        if self._freqtrade.state != State.RUNNING:
            raise RPCException('trader is not running')

        pair_rates = Trade.session.query(Trade.pair,
                                         sql.func.sum(Trade.close_profit).label('profit_sum'),
                                         sql.func.count(Trade.pair).label('count')) \
            .filter(Trade.is_open.is_(False)) \
            .group_by(Trade.pair) \
            .order_by(sql.text('profit_sum DESC')) \
            .all()
        return [
            {'pair': pair, 'profit': round(rate * 100, 2), 'count': count}
            for pair, rate, count in pair_rates
        ]

    def _rpc_count(self) -> List[Trade]:
        """ Returns the number of trades running """
        if self._freqtrade.state != State.RUNNING:
            raise RPCException('trader is not running')

        return Trade.query.filter(Trade.is_open.is_(True)).all()
