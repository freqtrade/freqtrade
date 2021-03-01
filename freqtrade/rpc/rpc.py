"""
This module contains class to define a RPC communications
"""
import logging
from abc import abstractmethod
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from math import isnan
from typing import Any, Dict, List, Optional, Tuple, Union

import arrow
from numpy import NAN, inf, int64, mean
from pandas import DataFrame

from freqtrade.configuration.timerange import TimeRange
from freqtrade.constants import CANCEL_REASON, DATETIME_PRINT_FORMAT
from freqtrade.data.history import load_data
from freqtrade.exceptions import ExchangeError, PricingError
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_msecs
from freqtrade.loggers import bufferHandler
from freqtrade.misc import shorten_date
from freqtrade.persistence import PairLocks, Trade
from freqtrade.persistence.models import PairLock
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.rpc.fiat_convert import CryptoToFiatConverter
from freqtrade.state import State
from freqtrade.strategy.interface import SellType


logger = logging.getLogger(__name__)


class RPCMessageType(Enum):
    STATUS_NOTIFICATION = 'status'
    WARNING_NOTIFICATION = 'warning'
    STARTUP_NOTIFICATION = 'startup'
    BUY_NOTIFICATION = 'buy'
    BUY_CANCEL_NOTIFICATION = 'buy_cancel'
    SELL_NOTIFICATION = 'sell'
    SELL_CANCEL_NOTIFICATION = 'sell_cancel'

    def __repr__(self):
        return self.value

    def __str__(self):
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


class RPCHandler:

    def __init__(self, rpc: 'RPC', config: Dict[str, Any]) -> None:
        """
        Initializes RPCHandlers
        :param rpc: instance of RPC Helper class
        :param config: Configuration object
        :return: None
        """
        self._rpc = rpc
        self._config: Dict[str, Any] = config

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


class RPC:
    """
    RPC class can be used to have extra feature, like bot data, and access to DB data
    """
    # Bind _fiat_converter if needed
    _fiat_converter: Optional[CryptoToFiatConverter] = None

    def __init__(self, freqtrade) -> None:
        """
        Initializes all enabled rpc modules
        :param freqtrade: Instance of a freqtrade bot
        :return: None
        """
        self._freqtrade = freqtrade
        self._config: Dict[str, Any] = freqtrade.config
        if self._config.get('fiat_display_currency', None):
            self._fiat_converter = CryptoToFiatConverter()

    @staticmethod
    def _rpc_show_config(config, botstate: Union[State, str]) -> Dict[str, Any]:
        """
        Return a dict of config options.
        Explicitly does NOT return the full config to avoid leakage of sensitive
        information via rpc.
        """
        val = {
            'dry_run': config['dry_run'],
            'stake_currency': config['stake_currency'],
            'stake_amount': config['stake_amount'],
            'max_open_trades': (config['max_open_trades']
                                if config['max_open_trades'] != float('inf') else -1),
            'minimal_roi': config['minimal_roi'].copy() if 'minimal_roi' in config else {},
            'stoploss': config.get('stoploss'),
            'trailing_stop': config.get('trailing_stop'),
            'trailing_stop_positive': config.get('trailing_stop_positive'),
            'trailing_stop_positive_offset': config.get('trailing_stop_positive_offset'),
            'trailing_only_offset_is_reached': config.get('trailing_only_offset_is_reached'),
            'use_custom_stoploss': config.get('use_custom_stoploss'),
            'bot_name': config.get('bot_name', 'freqtrade'),
            'timeframe': config.get('timeframe'),
            'timeframe_ms': timeframe_to_msecs(config['timeframe']
                                               ) if 'timeframe' in config else '',
            'timeframe_min': timeframe_to_minutes(config['timeframe']
                                                  ) if 'timeframe' in config else '',
            'exchange': config['exchange']['name'],
            'strategy': config['strategy'],
            'forcebuy_enabled': config.get('forcebuy_enable', False),
            'ask_strategy': config.get('ask_strategy', {}),
            'bid_strategy': config.get('bid_strategy', {}),
            'state': str(botstate),
            'runmode': config['runmode'].value
        }
        return val

    def _rpc_trade_status(self, trade_ids: List[int] = []) -> List[Dict[str, Any]]:
        """
        Below follows the RPC backend it is prefixed with rpc_ to raise awareness that it is
        a remotely exposed function
        """
        # Fetch open trades
        if trade_ids:
            trades = Trade.get_trades(trade_filter=Trade.id.in_(trade_ids)).all()
        else:
            trades = Trade.get_open_trades()

        if not trades:
            raise RPCException('no active trade')
        else:
            results = []
            for trade in trades:
                order = None
                if trade.open_order_id:
                    order = self._freqtrade.exchange.fetch_order(trade.open_order_id, trade.pair)
                # calculate profit and send message to user
                try:
                    current_rate = self._freqtrade.get_sell_rate(trade.pair, False)
                except (ExchangeError, PricingError):
                    current_rate = NAN
                current_profit = trade.calc_profit_ratio(current_rate)
                current_profit_abs = trade.calc_profit(current_rate)
                # Calculate guaranteed profit (in case of trailing stop)
                stoploss_entry_dist = trade.calc_profit(trade.stop_loss)
                stoploss_entry_dist_ratio = trade.calc_profit_ratio(trade.stop_loss)
                # calculate distance to stoploss
                stoploss_current_dist = trade.stop_loss - current_rate
                stoploss_current_dist_ratio = stoploss_current_dist / current_rate

                trade_dict = trade.to_json()
                trade_dict.update(dict(
                    base_currency=self._freqtrade.config['stake_currency'],
                    close_profit=trade.close_profit if trade.close_profit is not None else None,
                    current_rate=current_rate,
                    current_profit=current_profit,  # Deprectated
                    current_profit_pct=round(current_profit * 100, 2),  # Deprectated
                    current_profit_abs=current_profit_abs,  # Deprectated
                    profit_ratio=current_profit,
                    profit_pct=round(current_profit * 100, 2),
                    profit_abs=current_profit_abs,

                    stoploss_current_dist=stoploss_current_dist,
                    stoploss_current_dist_ratio=round(stoploss_current_dist_ratio, 8),
                    stoploss_current_dist_pct=round(stoploss_current_dist_ratio * 100, 2),
                    stoploss_entry_dist=stoploss_entry_dist,
                    stoploss_entry_dist_ratio=round(stoploss_entry_dist_ratio, 8),
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
                except (PricingError, ExchangeError):
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
            curdayprofit = sum(
                trade.close_profit_abs for trade in trades if trade.close_profit_abs is not None)
            profit_days[profitday] = {
                'amount': curdayprofit,
                'trades': len(trades)
            }

        data = [
            {
                'date': key,
                'abs_profit': value["amount"],
                'fiat_value': self._fiat_converter.convert_amount(
                        value['amount'],
                        stake_currency,
                        fiat_display_currency
                    ) if self._fiat_converter else 0,
                'trade_count': value["trades"],
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
            trades = Trade.get_trades([Trade.is_open.is_(False)]).order_by(
                Trade.id.desc()).limit(limit)
        else:
            trades = Trade.get_trades([Trade.is_open.is_(False)]).order_by(Trade.id.desc()).all()

        output = [trade.to_json() for trade in trades]

        return {
            "trades": output,
            "trades_count": len(output)
        }

    def _rpc_stats(self) -> Dict[str, Any]:
        """
        Generate generic stats for trades in database
        """
        def trade_win_loss(trade):
            if trade.close_profit > 0:
                return 'wins'
            elif trade.close_profit < 0:
                return 'losses'
            else:
                return 'draws'
        trades = trades = Trade.get_trades([Trade.is_open.is_(False)])
        # Sell reason
        sell_reasons = {}
        for trade in trades:
            if trade.sell_reason not in sell_reasons:
                sell_reasons[trade.sell_reason] = {'wins': 0, 'losses': 0, 'draws': 0}
            sell_reasons[trade.sell_reason][trade_win_loss(trade)] += 1

        # Duration
        dur: Dict[str, List[int]] = {'wins': [], 'draws': [], 'losses': []}
        for trade in trades:
            if trade.close_date is not None and trade.open_date is not None:
                trade_dur = (trade.close_date - trade.open_date).total_seconds()
                dur[trade_win_loss(trade)].append(trade_dur)

        wins_dur = sum(dur['wins']) / len(dur['wins']) if len(dur['wins']) > 0 else 'N/A'
        draws_dur = sum(dur['draws']) / len(dur['draws']) if len(dur['draws']) > 0 else 'N/A'
        losses_dur = sum(dur['losses']) / len(dur['losses']) if len(dur['losses']) > 0 else 'N/A'

        durations = {'wins': wins_dur, 'draws': draws_dur, 'losses': losses_dur}
        return {'sell_reasons': sell_reasons, 'durations': durations}

    def _rpc_trade_statistics(
            self, stake_currency: str, fiat_display_currency: str) -> Dict[str, Any]:
        """ Returns cumulative profit statistics """
        trades = Trade.get_trades().order_by(Trade.id).all()

        profit_all_coin = []
        profit_all_ratio = []
        profit_closed_coin = []
        profit_closed_ratio = []
        durations = []
        winning_trades = 0
        losing_trades = 0

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
                if trade.close_profit >= 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
            else:
                # Get current rate
                try:
                    current_rate = self._freqtrade.get_sell_rate(trade.pair, False)
                except (PricingError, ExchangeError):
                    current_rate = NAN
                profit_ratio = trade.calc_profit_ratio(rate=current_rate)

            profit_all_coin.append(
                trade.calc_profit(rate=trade.close_rate or current_rate)
            )
            profit_all_ratio.append(profit_ratio)

        best_pair = Trade.get_best_pair()

        # Prepare data to display
        profit_closed_coin_sum = round(sum(profit_closed_coin), 8)
        profit_closed_ratio_mean = float(mean(profit_closed_ratio) if profit_closed_ratio else 0.0)
        profit_closed_ratio_sum = sum(profit_closed_ratio) if profit_closed_ratio else 0.0

        profit_closed_fiat = self._fiat_converter.convert_amount(
            profit_closed_coin_sum,
            stake_currency,
            fiat_display_currency
        ) if self._fiat_converter else 0

        profit_all_coin_sum = round(sum(profit_all_coin), 8)
        profit_all_ratio_mean = float(mean(profit_all_ratio) if profit_all_ratio else 0.0)
        profit_all_ratio_sum = sum(profit_all_ratio) if profit_all_ratio else 0.0
        profit_all_fiat = self._fiat_converter.convert_amount(
            profit_all_coin_sum,
            stake_currency,
            fiat_display_currency
        ) if self._fiat_converter else 0

        first_date = trades[0].open_date if trades else None
        last_date = trades[-1].open_date if trades else None
        num = float(len(durations) or 1)
        return {
            'profit_closed_coin': profit_closed_coin_sum,
            'profit_closed_percent': round(profit_closed_ratio_mean * 100, 2),  # DEPRECATED
            'profit_closed_percent_mean': round(profit_closed_ratio_mean * 100, 2),
            'profit_closed_ratio_mean': profit_closed_ratio_mean,
            'profit_closed_percent_sum': round(profit_closed_ratio_sum * 100, 2),
            'profit_closed_ratio_sum': profit_closed_ratio_sum,
            'profit_closed_fiat': profit_closed_fiat,
            'profit_all_coin': profit_all_coin_sum,
            'profit_all_percent': round(profit_all_ratio_mean * 100, 2),  # DEPRECATED
            'profit_all_percent_mean': round(profit_all_ratio_mean * 100, 2),
            'profit_all_ratio_mean': profit_all_ratio_mean,
            'profit_all_percent_sum': round(profit_all_ratio_sum * 100, 2),
            'profit_all_ratio_sum': profit_all_ratio_sum,
            'profit_all_fiat': profit_all_fiat,
            'trade_count': len(trades),
            'closed_trade_count': len([t for t in trades if not t.is_open]),
            'first_trade_date': arrow.get(first_date).humanize() if first_date else '',
            'first_trade_timestamp': int(first_date.timestamp() * 1000) if first_date else 0,
            'latest_trade_date': arrow.get(last_date).humanize() if last_date else '',
            'latest_trade_timestamp': int(last_date.timestamp() * 1000) if last_date else 0,
            'avg_duration': str(timedelta(seconds=sum(durations) / num)).split('.')[0],
            'best_pair': best_pair[0] if best_pair else '',
            'best_rate': round(best_pair[1] * 100, 2) if best_pair else 0,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
        }

    def _rpc_balance(self, stake_currency: str, fiat_display_currency: str) -> Dict:
        """ Returns current account balance per crypto """
        output = []
        total = 0.0
        try:
            tickers = self._freqtrade.exchange.get_tickers()
        except (ExchangeError):
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
                        if pair.startswith(stake_currency) and not pair.endswith(stake_currency):
                            rate = 1.0 / rate
                        est_stake = rate * balance.total
                except (ExchangeError):
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

    def _rpc_reload_config(self) -> Dict[str, str]:
        """ Handler for reload_config. """
        self._freqtrade.state = State.RELOAD_CONFIG
        return {'status': 'Reloading config ...'}

    def _rpc_stopbuy(self) -> Dict[str, str]:
        """
        Handler to stop buying, but handle open trades gracefully.
        """
        if self._freqtrade.state == State.RUNNING:
            # Set 'max_open_trades' to 0
            self._freqtrade.config['max_open_trades'] = 0

        return {'status': 'No more buy will occur from now. Run /reload_config to reset.'}

    def _rpc_forcesell(self, trade_id: str) -> Dict[str, str]:
        """
        Handler for forcesell <id>.
        Sells the given trade at current price
        """
        def _exec_forcesell(trade: Trade) -> None:
            # Check if there is there is an open order
            fully_canceled = False
            if trade.open_order_id:
                order = self._freqtrade.exchange.fetch_order(trade.open_order_id, trade.pair)

                if order['side'] == 'buy':
                    fully_canceled = self._freqtrade.handle_cancel_buy(
                        trade, order, CANCEL_REASON['FORCE_SELL'])

                if order['side'] == 'sell':
                    # Cancel order - so it is placed anew with a fresh price.
                    self._freqtrade.handle_cancel_sell(trade, order, CANCEL_REASON['FORCE_SELL'])

            if not fully_canceled:
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
                f'Wrong pair selected. Only pairs with stake-currency {stake_currency} allowed.')
        # check if valid pair

        # check if pair already has an open pair
        trade = Trade.get_trades([Trade.is_open.is_(True), Trade.pair == pair]).first()
        if trade:
            raise RPCException(f'position for {pair} already open - id: {trade.id}')

        # gen stake amount
        stakeamount = self._freqtrade.wallets.get_trade_stake_amount(
            pair, self._freqtrade.get_free_open_trades())

        # execute buy
        if self._freqtrade.execute_buy(pair, stakeamount, price):
            trade = Trade.get_trades([Trade.is_open.is_(True), Trade.pair == pair]).first()
            return trade
        else:
            return None

    def _rpc_delete(self, trade_id: int) -> Dict[str, Union[str, int]]:
        """
        Handler for delete <id>.
        Delete the given trade and close eventually existing open orders.
        """
        with self._freqtrade._sell_lock:
            c_count = 0
            trade = Trade.get_trades(trade_filter=[Trade.id == trade_id]).first()
            if not trade:
                logger.warning('delete trade: Invalid argument received')
                raise RPCException('invalid argument')

            # Try cancelling regular order if that exists
            if trade.open_order_id:
                try:
                    self._freqtrade.exchange.cancel_order(trade.open_order_id, trade.pair)
                    c_count += 1
                except (ExchangeError):
                    pass

            # cancel stoploss on exchange ...
            if (self._freqtrade.strategy.order_types.get('stoploss_on_exchange')
                    and trade.stoploss_order_id):
                try:
                    self._freqtrade.exchange.cancel_stoploss_order(trade.stoploss_order_id,
                                                                   trade.pair)
                    c_count += 1
                except (ExchangeError):
                    pass

            trade.delete()
            self._freqtrade.wallets.update()
            return {
                'result': 'success',
                'trade_id': trade_id,
                'result_msg': f'Deleted trade {trade_id}. Closed {c_count} open orders.',
                'cancel_order_count': c_count,
            }

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
            'max': (int(self._freqtrade.config['max_open_trades'])
                    if self._freqtrade.config['max_open_trades'] != float('inf') else -1),
            'total_stake': sum((trade.open_rate * trade.amount) for trade in trades)
        }

    def _rpc_locks(self) -> Dict[str, Any]:
        """ Returns the  current locks """

        locks = PairLocks.get_pair_locks(None)
        return {
            'lock_count': len(locks),
            'locks': [lock.to_json() for lock in locks]
        }

    def _rpc_delete_lock(self, lockid: Optional[int] = None,
                         pair: Optional[str] = None) -> Dict[str, Any]:
        """ Delete specific lock(s) """
        locks = []

        if pair:
            locks = PairLocks.get_pair_locks(pair)
        if lockid:
            locks = PairLock.query.filter(PairLock.id == lockid).all()

        for lock in locks:
            lock.active = False
            lock.lock_end_time = datetime.now(timezone.utc)

        # session is always the same
        PairLock.session.flush()

        return self._rpc_locks()

    def _rpc_whitelist(self) -> Dict:
        """ Returns the currently active whitelist"""
        res = {'method': self._freqtrade.pairlists.name_list,
               'length': len(self._freqtrade.active_pair_whitelist),
               'whitelist': self._freqtrade.active_pair_whitelist
               }
        return res

    def _rpc_blacklist(self, add: List[str] = None) -> Dict:
        """ Returns the currently active blacklist"""
        errors = {}
        if add:
            for pair in add:
                if pair not in self._freqtrade.pairlists.blacklist:
                    try:
                        expand_pairlist([pair], self._freqtrade.exchange.get_markets().keys())
                        self._freqtrade.pairlists.blacklist.append(pair)

                    except ValueError:
                        errors[pair] = {
                            'error_msg': f'Pair {pair} is not a valid wildcard.'}
                else:
                    errors[pair] = {
                        'error_msg': f'Pair {pair} already in pairlist.'}

        res = {'method': self._freqtrade.pairlists.name_list,
               'length': len(self._freqtrade.pairlists.blacklist),
               'blacklist': self._freqtrade.pairlists.blacklist,
               'blacklist_expanded': self._freqtrade.pairlists.expanded_blacklist,
               'errors': errors,
               }
        return res

    @staticmethod
    def _rpc_get_logs(limit: Optional[int]) -> Dict[str, Any]:
        """Returns the last X logs"""
        if limit:
            buffer = bufferHandler.buffer[-limit:]
        else:
            buffer = bufferHandler.buffer
        records = [[datetime.fromtimestamp(r.created).strftime(DATETIME_PRINT_FORMAT),
                   r.created * 1000, r.name, r.levelname,
                   r.message + ('\n' + r.exc_text if r.exc_text else '')]
                   for r in buffer]

        # Log format:
        # [logtime-formatted, logepoch, logger-name, loglevel, message \n + exception]
        # e.g. ["2020-08-27 11:35:01", 1598520901097.9397,
        #       "freqtrade.worker", "INFO", "Starting worker develop"]

        return {'log_count': len(records), 'logs': records}

    def _rpc_edge(self) -> List[Dict[str, Any]]:
        """ Returns information related to Edge """
        if not self._freqtrade.edge:
            raise RPCException('Edge is not enabled.')
        return self._freqtrade.edge.accepted_pairs()

    @staticmethod
    def _convert_dataframe_to_dict(strategy: str, pair: str, timeframe: str, dataframe: DataFrame,
                                   last_analyzed: datetime) -> Dict[str, Any]:
        has_content = len(dataframe) != 0
        buy_signals = 0
        sell_signals = 0
        if has_content:

            dataframe.loc[:, '__date_ts'] = dataframe.loc[:, 'date'].astype(int64) // 1000 // 1000
            # Move open to seperate column when signal for easy plotting
            if 'buy' in dataframe.columns:
                buy_mask = (dataframe['buy'] == 1)
                buy_signals = int(buy_mask.sum())
                dataframe.loc[buy_mask, '_buy_signal_open'] = dataframe.loc[buy_mask, 'open']
            if 'sell' in dataframe.columns:
                sell_mask = (dataframe['sell'] == 1)
                sell_signals = int(sell_mask.sum())
                dataframe.loc[sell_mask, '_sell_signal_open'] = dataframe.loc[sell_mask, 'open']
            dataframe = dataframe.replace([inf, -inf], NAN)
            dataframe = dataframe.replace({NAN: None})

        res = {
            'pair': pair,
            'timeframe': timeframe,
            'timeframe_ms': timeframe_to_msecs(timeframe),
            'strategy': strategy,
            'columns': list(dataframe.columns),
            'data': dataframe.values.tolist(),
            'length': len(dataframe),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'last_analyzed': last_analyzed,
            'last_analyzed_ts': int(last_analyzed.timestamp()),
            'data_start': '',
            'data_start_ts': 0,
            'data_stop': '',
            'data_stop_ts': 0,
        }
        if has_content:
            res.update({
                'data_start': str(dataframe.iloc[0]['date']),
                'data_start_ts': int(dataframe.iloc[0]['__date_ts']),
                'data_stop': str(dataframe.iloc[-1]['date']),
                'data_stop_ts': int(dataframe.iloc[-1]['__date_ts']),
            })
        return res

    def _rpc_analysed_dataframe(self, pair: str, timeframe: str,
                                limit: Optional[int]) -> Dict[str, Any]:

        _data, last_analyzed = self._freqtrade.dataprovider.get_analyzed_dataframe(
            pair, timeframe)
        _data = _data.copy()
        if limit:
            _data = _data.iloc[-limit:]
        return self._convert_dataframe_to_dict(self._freqtrade.config['strategy'],
                                               pair, timeframe, _data, last_analyzed)

    @staticmethod
    def _rpc_analysed_history_full(config, pair: str, timeframe: str,
                                   timerange: str) -> Dict[str, Any]:
        timerange_parsed = TimeRange.parse_timerange(timerange)

        _data = load_data(
            datadir=config.get("datadir"),
            pairs=[pair],
            timeframe=timeframe,
            timerange=timerange_parsed,
            data_format=config.get('dataformat_ohlcv', 'json'),
        )
        if pair not in _data:
            raise RPCException(f"No data for {pair}, {timeframe} in {timerange} found.")
        from freqtrade.resolvers.strategy_resolver import StrategyResolver
        strategy = StrategyResolver.load_strategy(config)
        df_analyzed = strategy.analyze_ticker(_data[pair], {'pair': pair})

        return RPC._convert_dataframe_to_dict(strategy.get_strategy_name(), pair, timeframe,
                                              df_analyzed, arrow.Arrow.utcnow().datetime)

    def _rpc_plot_config(self) -> Dict[str, Any]:

        return self._freqtrade.strategy.plot_config
