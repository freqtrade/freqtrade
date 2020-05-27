"""
This module contains class to define a RPC communications
"""
import logging
from abc import abstractmethod
from datetime import date, datetime, timedelta
from enum import Enum
from math import isnan
from typing import Any, Dict, List, Optional, Tuple

import arrow
from numpy import NAN, mean

from freqtrade.exceptions import DependencyException, TemporaryError
from freqtrade.misc import shorten_date
from freqtrade.persistence import Trade
from freqtrade.rpc.fiat_convert import CryptoToFiatConverter
from freqtrade.state import State
from freqtrade.strategy.interface import SellType

logger = logging.getLogger(__name__)


class RPCMessageType(Enum):
    STATUS_NOTIFICATION = 'status'
    WARNING_NOTIFICATION = 'warning'
    CUSTOM_NOTIFICATION = 'custom'
    BUY_NOTIFICATION = 'buy'
    BUY_CANCEL_NOTIFICATION = 'buy_cancel'
    SELL_NOTIFICATION = 'sell'
    SELL_CANCEL_NOTIFICATION = 'sell_cancel'

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

    def __json__(self):
        return {
            'msg': self.message
        }


class RPC:
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

    def _rpc_show_config(self) -> Dict[str, Any]:
        """
        Return a dict of config options.
        Explicitly does NOT return the full config to avoid leakage of sensitive
        information via rpc.
        """
        config = self._freqtrade.config
        val = {
            'dry_run': config['dry_run'],
            'stake_currency': config['stake_currency'],
            'stake_amount': config['stake_amount'],
            'max_open_trades': config['max_open_trades'],
            'minimal_roi': config['minimal_roi'].copy(),
            'stoploss': config['stoploss'],
            'trailing_stop': config['trailing_stop'],
            'trailing_stop_positive': config.get('trailing_stop_positive'),
            'trailing_stop_positive_offset': config.get('trailing_stop_positive_offset'),
            'trailing_only_offset_is_reached': config.get('trailing_only_offset_is_reached'),
            'ticker_interval': config['ticker_interval'],
            'exchange': config['exchange']['name'],
            'strategy': config['strategy'],
            'forcebuy_enabled': config.get('forcebuy_enable', False),
            'state': str(self._freqtrade.state)
        }
        return val

    def _rpc_trade_status(self) -> List[Dict[str, Any]]:
        """
        Below follows the RPC backend it is prefixed with rpc_ to raise awareness that it is
        a remotely exposed function
        """
        # Fetch open trade
        trades = Trade.get_open_trades()
        if not trades:
            raise RPCException('no active trade')
        else:
            results = []
            for trade in trades:
                order = None
                if trade.open_order_id:
                    order = self._freqtrade.exchange.get_order(trade.open_order_id, trade.pair)
                # calculate profit and send message to user
                try:
                    current_rate = self._freqtrade.get_sell_rate(trade.pair, False)
                except DependencyException:
                    current_rate = NAN
                current_profit = trade.calc_profit_ratio(current_rate)
                fmt_close_profit = (f'{round(trade.close_profit * 100, 2):.2f}%'
                                    if trade.close_profit is not None else None)
                trade_dict = trade.to_json()
                trade_dict.update(dict(
                    base_currency=self._freqtrade.config['stake_currency'],
                    close_profit=trade.close_profit if trade.close_profit is not None else None,
                    close_profit_pct=fmt_close_profit,
                    current_rate=current_rate,
                    current_profit=current_profit,
                    current_profit_pct=round(current_profit * 100, 2),
                    open_order='({} {} rem={:.8f})'.format(
                        order['type'], order['side'], order['remaining']
                    ) if order else None,
                ))
                results.append(trade_dict)
            return results

    def _rpc_status_table(self, stake_currency: str,
                          fiat_display_currency: str) -> Tuple[List, List]:
        trades = Trade.get_open_trades()
        if not trades:
            raise RPCException('no active trade')
        else:
            trades_list = []
            for trade in trades:
                # calculate profit and send message to user
                try:
                    current_rate = self._freqtrade.get_sell_rate(trade.pair, False)
                except DependencyException:
                    current_rate = NAN
                trade_percent = (100 * trade.calc_profit_ratio(current_rate))
                trade_profit = trade.calc_profit(current_rate)
                profit_str = f'{trade_percent:.2f}%'
                if self._fiat_converter:
                    fiat_profit = self._fiat_converter.convert_amount(
                        trade_profit,
                        stake_currency,
                        fiat_display_currency
                    )
                    if fiat_profit and not isnan(fiat_profit):
                        profit_str += f" ({fiat_profit:.2f})"
                trades_list.append([
                    trade.id,
                    trade.pair + ('*' if (trade.open_order_id is not None
                                          and trade.close_rate_requested is None) else '')
                               + ('**' if (trade.close_rate_requested is not None) else ''),
                    shorten_date(arrow.get(trade.open_date).humanize(only_distance=True)),
                    profit_str
                ])
            profitcol = "Profit"
            if self._fiat_converter:
                profitcol += " (" + fiat_display_currency + ")"

            columns = ['ID', 'Pair', 'Since', profitcol]
            return trades_list, columns

    def _rpc_daily_profit(
            self, timescale: int,
            stake_currency: str, fiat_display_currency: str) -> Dict[str, Any]:
        today = datetime.utcnow().date()
        profit_days: Dict[date, Dict] = {}

        if not (isinstance(timescale, int) and timescale > 0):
            raise RPCException('timescale must be an integer greater than 0')

        for day in range(0, timescale):
            profitday = today - timedelta(days=day)
            trades = Trade.get_trades(trade_filter=[
                Trade.is_open.is_(False),
                Trade.close_date >= profitday,
                Trade.close_date < (profitday + timedelta(days=1))
            ]).order_by(Trade.close_date).all()
            curdayprofit = sum(trade.close_profit_abs for trade in trades)
            profit_days[profitday] = {
                'amount': f'{curdayprofit:.8f}',
                'trades': len(trades)
            }

        data = [
            {
                'date': key,
                'abs_profit': f'{float(value["amount"]):.8f}',
                'fiat_value': '{value:.3f}'.format(
                    value=self._fiat_converter.convert_amount(
                        value['amount'],
                        stake_currency,
                        fiat_display_currency
                    ) if self._fiat_converter else 0,
                ),
                'trade_count': f'{value["trades"]}',
            }
            for key, value in profit_days.items()
        ]
        return {
            'stake_currency': stake_currency,
            'fiat_display_currency': fiat_display_currency,
            'data': data
        }

    def _rpc_trade_history(self, limit: int) -> Dict:
        """ Returns the X last trades """
        if limit > 0:
            trades = Trade.get_trades().order_by(Trade.id.desc()).limit(limit)
        else:
            trades = Trade.get_trades().order_by(Trade.id.desc()).all()

        output = [trade.to_json() for trade in trades]

        return {
            "trades": output,
            "trades_count": len(output)
        }

    def _rpc_trade_statistics(
            self, stake_currency: str, fiat_display_currency: str) -> Dict[str, Any]:
        """ Returns cumulative profit statistics """
        trades = Trade.get_trades().order_by(Trade.id).all()

        profit_all_coin = []
        profit_all_ratio = []
        profit_closed_coin = []
        profit_closed_ratio = []
        durations = []

        for trade in trades:
            current_rate: float = 0.0

            if not trade.open_rate:
                continue
            if trade.close_date:
                durations.append((trade.close_date - trade.open_date).total_seconds())

            if not trade.is_open:
                profit_ratio = trade.close_profit
                profit_closed_coin.append(trade.close_profit_abs)
                profit_closed_ratio.append(profit_ratio)
            else:
                # Get current rate
                try:
                    current_rate = self._freqtrade.get_sell_rate(trade.pair, False)
                except DependencyException:
                    current_rate = NAN
                profit_ratio = trade.calc_profit_ratio(rate=current_rate)

            profit_all_coin.append(
                trade.calc_profit(rate=trade.close_rate or current_rate)
            )
            profit_all_ratio.append(profit_ratio)

        best_pair = Trade.get_best_pair()

        if not best_pair:
            raise RPCException('no closed trade')

        bp_pair, bp_rate = best_pair

        # Prepare data to display
        profit_closed_coin_sum = round(sum(profit_closed_coin), 8)
        profit_closed_percent = (round(mean(profit_closed_ratio) * 100, 2) if profit_closed_ratio
                                 else 0.0)
        profit_closed_fiat = self._fiat_converter.convert_amount(
            profit_closed_coin_sum,
            stake_currency,
            fiat_display_currency
        ) if self._fiat_converter else 0

        profit_all_coin_sum = round(sum(profit_all_coin), 8)
        profit_all_percent = round(mean(profit_all_ratio) * 100, 2) if profit_all_ratio else 0.0
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
            'first_trade_timestamp': int(trades[0].open_date.timestamp() * 1000),
            'latest_trade_date': arrow.get(trades[-1].open_date).humanize(),
            'latest_trade_timestamp': int(trades[-1].open_date.timestamp() * 1000),
            'avg_duration': str(timedelta(seconds=sum(durations) / num)).split('.')[0],
            'best_pair': bp_pair,
            'best_rate': round(bp_rate * 100, 2),
        }

    def _rpc_balance(self, stake_currency: str, fiat_display_currency: str) -> Dict:
        """ Returns current account balance per crypto """
        output = []
        total = 0.0
        try:
            tickers = self._freqtrade.exchange.get_tickers()
        except (TemporaryError, DependencyException):
            raise RPCException('Error getting current tickers.')

        self._freqtrade.wallets.update(require_update=False)

        for coin, balance in self._freqtrade.wallets.get_all_balances().items():
            if not balance.total:
                continue

            est_stake: float = 0
            if coin == stake_currency:
                rate = 1.0
                est_stake = balance.total
            else:
                try:
                    pair = self._freqtrade.exchange.get_valid_pair_combination(coin, stake_currency)
                    rate = tickers.get(pair, {}).get('bid', None)
                    if rate:
                        if pair.startswith(stake_currency):
                            rate = 1.0 / rate
                        est_stake = rate * balance.total
                except (TemporaryError, DependencyException):
                    logger.warning(f" Could not get rate for pair {coin}.")
                    continue
            total = total + (est_stake or 0)
            output.append({
                'currency': coin,
                'free': balance.free if balance.free is not None else 0,
                'balance': balance.total if balance.total is not None else 0,
                'used': balance.used if balance.used is not None else 0,
                'est_stake': est_stake or 0,
                'stake': stake_currency,
            })
        if total == 0.0:
            if self._freqtrade.config['dry_run']:
                raise RPCException('Running in Dry Run, balances are not available.')
            else:
                raise RPCException('All balances are zero.')

        symbol = fiat_display_currency
        value = self._fiat_converter.convert_amount(total, stake_currency,
                                                    symbol) if self._fiat_converter else 0
        return {
            'currencies': output,
            'total': total,
            'symbol': symbol,
            'value': value,
            'stake': stake_currency,
            'note': 'Simulated balances' if self._freqtrade.config['dry_run'] else ''
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

    def _rpc_stopbuy(self) -> Dict[str, str]:
        """
        Handler to stop buying, but handle open trades gracefully.
        """
        if self._freqtrade.state == State.RUNNING:
            # Set 'max_open_trades' to 0
            self._freqtrade.config['max_open_trades'] = 0

        return {'status': 'No more buy will occur from now. Run /reload_conf to reset.'}

    def _rpc_forcesell(self, trade_id: str) -> Dict[str, str]:
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
            current_rate = self._freqtrade.get_sell_rate(trade.pair, False)
            self._freqtrade.execute_sell(trade, current_rate, SellType.FORCE_SELL)
        # ---- EOF def _exec_forcesell ----

        if self._freqtrade.state != State.RUNNING:
            raise RPCException('trader is not running')

        with self._freqtrade._sell_lock:
            if trade_id == 'all':
                # Execute sell for all open orders
                for trade in Trade.get_open_trades():
                    _exec_forcesell(trade)
                Trade.session.flush()
                self._freqtrade.wallets.update()
                return {'result': 'Created sell orders for all open trades.'}

            # Query for trade
            trade = Trade.get_trades(
                trade_filter=[Trade.id == trade_id, Trade.is_open.is_(True), ]
            ).first()
            if not trade:
                logger.warning('forcesell: Invalid argument received')
                raise RPCException('invalid argument')

            _exec_forcesell(trade)
            Trade.session.flush()
            self._freqtrade.wallets.update()
            return {'result': f'Created sell order for trade {trade_id}.'}

    def _rpc_forcebuy(self, pair: str, price: Optional[float]) -> Optional[Trade]:
        """
        Handler for forcebuy <asset> <price>
        Buys a pair trade at the given or current price
        """

        if not self._freqtrade.config.get('forcebuy_enable', False):
            raise RPCException('Forcebuy not enabled.')

        if self._freqtrade.state != State.RUNNING:
            raise RPCException('trader is not running')

        # Check if pair quote currency equals to the stake currency.
        stake_currency = self._freqtrade.config.get('stake_currency')
        if not self._freqtrade.exchange.get_pair_quote_currency(pair) == stake_currency:
            raise RPCException(
                f'Wrong pair selected. Please pairs with stake {stake_currency} pairs only')
        # check if valid pair

        # check if pair already has an open pair
        trade = Trade.get_trades([Trade.is_open.is_(True), Trade.pair.is_(pair)]).first()
        if trade:
            raise RPCException(f'position for {pair} already open - id: {trade.id}')

        # gen stake amount
        stakeamount = self._freqtrade.get_trade_stake_amount(pair)

        # execute buy
        if self._freqtrade.execute_buy(pair, stakeamount, price):
            trade = Trade.get_trades([Trade.is_open.is_(True), Trade.pair.is_(pair)]).first()
            return trade
        else:
            return None

    def _rpc_performance(self) -> List[Dict[str, Any]]:
        """
        Handler for performance.
        Shows a performance statistic from finished trades
        """
        pair_rates = Trade.get_overall_performance()
        # Round and convert to %
        [x.update({'profit': round(x['profit'] * 100, 2)}) for x in pair_rates]
        return pair_rates

    def _rpc_count(self) -> Dict[str, float]:
        """ Returns the number of trades running """
        if self._freqtrade.state != State.RUNNING:
            raise RPCException('trader is not running')

        trades = Trade.get_open_trades()
        return {
            'current': len(trades),
            'max': float(self._freqtrade.config['max_open_trades']),
            'total_stake': sum((trade.open_rate * trade.amount) for trade in trades)
        }

    def _rpc_whitelist(self) -> Dict:
        """ Returns the currently active whitelist"""
        res = {'method': self._freqtrade.pairlists.name_list,
               'length': len(self._freqtrade.active_pair_whitelist),
               'whitelist': self._freqtrade.active_pair_whitelist
               }
        return res

    def _rpc_blacklist(self, add: List[str] = None) -> Dict:
        """ Returns the currently active blacklist"""
        if add:
            stake_currency = self._freqtrade.config.get('stake_currency')
            for pair in add:
                if (self._freqtrade.exchange.get_pair_quote_currency(pair) == stake_currency
                        and pair not in self._freqtrade.pairlists.blacklist):
                    self._freqtrade.pairlists.blacklist.append(pair)

        res = {'method': self._freqtrade.pairlists.name_list,
               'length': len(self._freqtrade.pairlists.blacklist),
               'blacklist': self._freqtrade.pairlists.blacklist,
               }
        return res

    def _rpc_edge(self) -> List[Dict[str, Any]]:
        """ Returns information related to Edge """
        if not self._freqtrade.edge:
            raise RPCException('Edge is not enabled.')
        return self._freqtrade.edge.accepted_pairs()
