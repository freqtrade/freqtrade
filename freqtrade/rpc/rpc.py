"""
This module contains class to define a RPC communications
"""
import logging
from abc import abstractmethod
from datetime import date, datetime, timedelta, timezone
from math import isnan
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import psutil
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzlocal
from numpy import NAN, inf, int64, mean
from pandas import DataFrame, NaT
from sqlalchemy import func, select

from freqtrade import __version__
from freqtrade.configuration.timerange import TimeRange
from freqtrade.constants import CANCEL_REASON, Config
from freqtrade.data.history import load_data
from freqtrade.data.metrics import calculate_expectancy, calculate_max_drawdown
from freqtrade.enums import (CandleType, ExitCheckTuple, ExitType, MarketDirection, SignalDirection,
                             State, TradingMode)
from freqtrade.exceptions import ExchangeError, PricingError
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_msecs
from freqtrade.exchange.types import Tickers
from freqtrade.loggers import bufferHandler
from freqtrade.persistence import KeyStoreKeys, KeyValueStore, PairLocks, Trade
from freqtrade.persistence.models import PairLock
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.rpc.fiat_convert import CryptoToFiatConverter
from freqtrade.rpc.rpc_types import RPCSendMsg
from freqtrade.util import (decimals_per_coin, dt_humanize, dt_now, dt_ts_def, format_date,
                            shorten_date)
from freqtrade.wallets import PositionWallet, Wallet


logger = logging.getLogger(__name__)


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

    def __init__(self, rpc: 'RPC', config: Config) -> None:
        """
        Initializes RPCHandlers
        :param rpc: instance of RPC Helper class
        :param config: Configuration object
        :return: None
        """
        self._rpc = rpc
        self._config: Config = config

    @property
    def name(self) -> str:
        """ Returns the lowercase name of the implementation """
        return self.__class__.__name__.lower()

    @abstractmethod
    def cleanup(self) -> None:
        """ Cleanup pending module resources """

    @abstractmethod
    def send_msg(self, msg: RPCSendMsg) -> None:
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
        self._config: Config = freqtrade.config
        if self._config.get('fiat_display_currency'):
            self._fiat_converter = CryptoToFiatConverter()

    @staticmethod
    def _rpc_show_config(config, botstate: Union[State, str],
                         strategy_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Return a dict of config options.
        Explicitly does NOT return the full config to avoid leakage of sensitive
        information via rpc.
        """
        val = {
            'version': __version__,
            'strategy_version': strategy_version,
            'dry_run': config['dry_run'],
            'trading_mode': config.get('trading_mode', 'spot'),
            'short_allowed': config.get('trading_mode', 'spot') != 'spot',
            'stake_currency': config['stake_currency'],
            'stake_currency_decimals': decimals_per_coin(config['stake_currency']),
            'stake_amount': str(config['stake_amount']),
            'available_capital': config.get('available_capital'),
            'max_open_trades': (config.get('max_open_trades', 0)
                                if config.get('max_open_trades', 0) != float('inf') else -1),
            'minimal_roi': config['minimal_roi'].copy() if 'minimal_roi' in config else {},
            'stoploss': config.get('stoploss'),
            'stoploss_on_exchange': config.get('order_types',
                                               {}).get('stoploss_on_exchange', False),
            'trailing_stop': config.get('trailing_stop'),
            'trailing_stop_positive': config.get('trailing_stop_positive'),
            'trailing_stop_positive_offset': config.get('trailing_stop_positive_offset'),
            'trailing_only_offset_is_reached': config.get('trailing_only_offset_is_reached'),
            'unfilledtimeout': config.get('unfilledtimeout'),
            'use_custom_stoploss': config.get('use_custom_stoploss'),
            'order_types': config.get('order_types'),
            'bot_name': config.get('bot_name', 'freqtrade'),
            'timeframe': config.get('timeframe'),
            'timeframe_ms': timeframe_to_msecs(config['timeframe']
                                               ) if 'timeframe' in config else 0,
            'timeframe_min': timeframe_to_minutes(config['timeframe']
                                                  ) if 'timeframe' in config else 0,
            'exchange': config['exchange']['name'],
            'strategy': config['strategy'],
            'force_entry_enable': config.get('force_entry_enable', False),
            'exit_pricing': config.get('exit_pricing', {}),
            'entry_pricing': config.get('entry_pricing', {}),
            'state': str(botstate),
            'runmode': config['runmode'].value,
            'position_adjustment_enable': config.get('position_adjustment_enable', False),
            'max_entry_position_adjustment': (
                config.get('max_entry_position_adjustment', -1)
                if config.get('max_entry_position_adjustment') != float('inf')
                else -1)
        }
        return val

    def _rpc_trade_status(self, trade_ids: List[int] = []) -> List[Dict[str, Any]]:
        """
        Below follows the RPC backend it is prefixed with rpc_ to raise awareness that it is
        a remotely exposed function
        """
        # Fetch open trades
        if trade_ids:
            trades: Sequence[Trade] = Trade.get_trades(trade_filter=Trade.id.in_(trade_ids)).all()
        else:
            trades = Trade.get_open_trades()

        if not trades:
            raise RPCException('no active trade')
        else:
            results = []
            for trade in trades:
                current_profit_fiat: Optional[float] = None
                total_profit_fiat: Optional[float] = None

                # prepare open orders details
                oo_details: Optional[str] = ""
                oo_details_lst = [
                    f'({oo.order_type} {oo.side} rem={oo.safe_remaining:.8f})'
                    for oo in trade.open_orders
                    if oo.ft_order_side not in ['stoploss']
                ]
                oo_details = ', '.join(oo_details_lst)

                total_profit_abs = 0.0
                total_profit_ratio: Optional[float] = None
                # calculate profit and send message to user
                if trade.is_open:
                    try:
                        current_rate = self._freqtrade.exchange.get_rate(
                            trade.pair, side='exit', is_short=trade.is_short, refresh=False)
                    except (ExchangeError, PricingError):
                        current_rate = NAN
                    if len(trade.select_filled_orders(trade.entry_side)) > 0:

                        current_profit = current_profit_abs = current_profit_fiat = NAN
                        if not isnan(current_rate):
                            prof = trade.calculate_profit(current_rate)
                            current_profit = prof.profit_ratio
                            current_profit_abs = prof.profit_abs
                            total_profit_abs = prof.total_profit
                            total_profit_ratio = prof.total_profit_ratio
                    else:
                        current_profit = current_profit_abs = current_profit_fiat = 0.0

                else:
                    # Closed trade ...
                    current_rate = trade.close_rate
                    current_profit = trade.close_profit or 0.0
                    current_profit_abs = trade.close_profit_abs or 0.0

                # Calculate fiat profit
                if not isnan(current_profit_abs) and self._fiat_converter:
                    current_profit_fiat = self._fiat_converter.convert_amount(
                        current_profit_abs,
                        self._freqtrade.config['stake_currency'],
                        self._freqtrade.config['fiat_display_currency']
                    )
                    total_profit_fiat = self._fiat_converter.convert_amount(
                        total_profit_abs,
                        self._freqtrade.config['stake_currency'],
                        self._freqtrade.config['fiat_display_currency']
                    )

                # Calculate guaranteed profit (in case of trailing stop)
                stop_entry = trade.calculate_profit(trade.stop_loss)

                stoploss_entry_dist = stop_entry.profit_abs
                stoploss_entry_dist_ratio = stop_entry.profit_ratio

                # calculate distance to stoploss
                stoploss_current_dist = trade.stop_loss - current_rate
                stoploss_current_dist_ratio = stoploss_current_dist / current_rate

                trade_dict = trade.to_json()
                trade_dict.update(dict(
                    close_profit=trade.close_profit if not trade.is_open else None,
                    current_rate=current_rate,
                    profit_ratio=current_profit,
                    profit_pct=round(current_profit * 100, 2),
                    profit_abs=current_profit_abs,
                    profit_fiat=current_profit_fiat,
                    total_profit_abs=total_profit_abs,
                    total_profit_fiat=total_profit_fiat,
                    total_profit_ratio=total_profit_ratio,
                    stoploss_current_dist=stoploss_current_dist,
                    stoploss_current_dist_ratio=round(stoploss_current_dist_ratio, 8),
                    stoploss_current_dist_pct=round(stoploss_current_dist_ratio * 100, 2),
                    stoploss_entry_dist=stoploss_entry_dist,
                    stoploss_entry_dist_ratio=round(stoploss_entry_dist_ratio, 8),
                    open_orders=oo_details
                ))
                results.append(trade_dict)
            return results

    def _rpc_status_table(self, stake_currency: str,
                          fiat_display_currency: str) -> Tuple[List, List, float]:
        trades: List[Trade] = Trade.get_open_trades()
        nonspot = self._config.get('trading_mode', TradingMode.SPOT) != TradingMode.SPOT
        if not trades:
            raise RPCException('no active trade')
        else:
            trades_list = []
            fiat_profit_sum = NAN
            for trade in trades:
                # calculate profit and send message to user
                try:
                    current_rate = self._freqtrade.exchange.get_rate(
                        trade.pair, side='exit', is_short=trade.is_short, refresh=False)
                except (PricingError, ExchangeError):
                    current_rate = NAN
                    trade_profit = NAN
                    profit_str = f'{NAN:.2%}'
                else:
                    if trade.nr_of_successful_entries > 0:
                        profit = trade.calculate_profit(current_rate)
                        trade_profit = profit.profit_abs
                        profit_str = f'{profit.profit_ratio:.2%}'
                    else:
                        trade_profit = 0.0
                        profit_str = f'{0.0:.2f}'
                direction_str = ('S' if trade.is_short else 'L') if nonspot else ''
                if self._fiat_converter:
                    fiat_profit = self._fiat_converter.convert_amount(
                        trade_profit,
                        stake_currency,
                        fiat_display_currency
                    )
                    if not isnan(fiat_profit):
                        profit_str += f" ({fiat_profit:.2f})"
                        fiat_profit_sum = fiat_profit if isnan(fiat_profit_sum) \
                            else fiat_profit_sum + fiat_profit

                active_attempt_side_symbols = [
                    '*' if (oo and oo.ft_order_side == trade.entry_side) else '**'
                    for oo in trade.open_orders
                ]

                # exemple: '*.**.**' trying to enter, exit and exit with 3 different orders
                active_attempt_side_symbols_str = '.'.join(active_attempt_side_symbols)

                detail_trade = [
                    f'{trade.id} {direction_str}',
                    trade.pair + active_attempt_side_symbols_str,
                    shorten_date(dt_humanize(trade.open_date, only_distance=True)),
                    profit_str
                ]

                if self._config.get('position_adjustment_enable', False):
                    max_entry_str = ''
                    if self._config.get('max_entry_position_adjustment', -1) > 0:
                        max_entry_str = f"/{self._config['max_entry_position_adjustment'] + 1}"
                    filled_entries = trade.nr_of_successful_entries
                    detail_trade.append(f"{filled_entries}{max_entry_str}")
                trades_list.append(detail_trade)
            profitcol = "Profit"
            if self._fiat_converter:
                profitcol += " (" + fiat_display_currency + ")"

            columns = [
                'ID L/S' if nonspot else 'ID',
                'Pair',
                'Since',
                profitcol]
            if self._config.get('position_adjustment_enable', False):
                columns.append('# Entries')
            return trades_list, columns, fiat_profit_sum

    def _rpc_timeunit_profit(
            self, timescale: int,
            stake_currency: str, fiat_display_currency: str,
            timeunit: str = 'days') -> Dict[str, Any]:
        """
        :param timeunit: Valid entries are 'days', 'weeks', 'months'
        """
        start_date = datetime.now(timezone.utc).date()
        if timeunit == 'weeks':
            # weekly
            start_date = start_date - timedelta(days=start_date.weekday())  # Monday
        if timeunit == 'months':
            start_date = start_date.replace(day=1)

        def time_offset(step: int):
            if timeunit == 'months':
                return relativedelta(months=step)
            return timedelta(**{timeunit: step})

        if not (isinstance(timescale, int) and timescale > 0):
            raise RPCException('timescale must be an integer greater than 0')

        profit_units: Dict[date, Dict] = {}
        daily_stake = self._freqtrade.wallets.get_total_stake_amount()

        for day in range(0, timescale):
            profitday = start_date - time_offset(day)
            # Only query for necessary columns for performance reasons.
            trades = Trade.session.execute(
                select(Trade.close_profit_abs)
                .filter(Trade.is_open.is_(False),
                        Trade.close_date >= profitday,
                        Trade.close_date < (profitday + time_offset(1)))
                .order_by(Trade.close_date)
            ).all()

            curdayprofit = sum(
                trade.close_profit_abs for trade in trades if trade.close_profit_abs is not None)
            # Calculate this periods starting balance
            daily_stake = daily_stake - curdayprofit
            profit_units[profitday] = {
                'amount': curdayprofit,
                'daily_stake': daily_stake,
                'rel_profit': round(curdayprofit / daily_stake, 8) if daily_stake > 0 else 0,
                'trades': len(trades),
            }

        data = [
            {
                'date': key,
                'abs_profit': value["amount"],
                'starting_balance': value["daily_stake"],
                'rel_profit': value["rel_profit"],
                'fiat_value': self._fiat_converter.convert_amount(
                    value['amount'],
                    stake_currency,
                    fiat_display_currency
                ) if self._fiat_converter else 0,
                'trade_count': value["trades"],
            }
            for key, value in profit_units.items()
        ]
        return {
            'stake_currency': stake_currency,
            'fiat_display_currency': fiat_display_currency,
            'data': data
        }

    def _rpc_trade_history(self, limit: int, offset: int = 0, order_by_id: bool = False) -> Dict:
        """ Returns the X last trades """
        order_by: Any = Trade.id if order_by_id else Trade.close_date.desc()
        if limit:
            trades = Trade.session.scalars(
                Trade.get_trades_query([Trade.is_open.is_(False)])
                .order_by(order_by)
                .limit(limit)
                .offset(offset))
        else:
            trades = Trade.session.scalars(
                Trade.get_trades_query([Trade.is_open.is_(False)])
                .order_by(Trade.close_date.desc()))

        output = [trade.to_json() for trade in trades]
        total_trades = Trade.session.scalar(
            select(func.count(Trade.id)).filter(Trade.is_open.is_(False)))

        return {
            "trades": output,
            "trades_count": len(output),
            "offset": offset,
            "total_trades": total_trades,
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
        trades = Trade.get_trades([Trade.is_open.is_(False)], include_orders=False)
        # Duration
        dur: Dict[str, List[float]] = {'wins': [], 'draws': [], 'losses': []}
        # Exit reason
        exit_reasons = {}
        for trade in trades:
            if trade.exit_reason not in exit_reasons:
                exit_reasons[trade.exit_reason] = {'wins': 0, 'losses': 0, 'draws': 0}
            exit_reasons[trade.exit_reason][trade_win_loss(trade)] += 1

            if trade.close_date is not None and trade.open_date is not None:
                trade_dur = (trade.close_date - trade.open_date).total_seconds()
                dur[trade_win_loss(trade)].append(trade_dur)

        wins_dur = sum(dur['wins']) / len(dur['wins']) if len(dur['wins']) > 0 else None
        draws_dur = sum(dur['draws']) / len(dur['draws']) if len(dur['draws']) > 0 else None
        losses_dur = sum(dur['losses']) / len(dur['losses']) if len(dur['losses']) > 0 else None

        durations = {'wins': wins_dur, 'draws': draws_dur, 'losses': losses_dur}
        return {'exit_reasons': exit_reasons, 'durations': durations}

    def _rpc_trade_statistics(
            self, stake_currency: str, fiat_display_currency: str,
            start_date: datetime = datetime.fromtimestamp(0)) -> Dict[str, Any]:
        """ Returns cumulative profit statistics """
        trade_filter = ((Trade.is_open.is_(False) & (Trade.close_date >= start_date)) |
                        Trade.is_open.is_(True))
        trades: Sequence[Trade] = Trade.session.scalars(Trade.get_trades_query(
            trade_filter, include_orders=False).order_by(Trade.id)).all()

        profit_all_coin = []
        profit_all_ratio = []
        profit_closed_coin = []
        profit_closed_ratio = []
        durations = []
        winning_trades = 0
        losing_trades = 0
        winning_profit = 0.0
        losing_profit = 0.0

        for trade in trades:
            current_rate: float = 0.0

            if trade.close_date:
                durations.append((trade.close_date - trade.open_date).total_seconds())

            if not trade.is_open:
                profit_ratio = trade.close_profit or 0.0
                profit_abs = trade.close_profit_abs or 0.0
                profit_closed_coin.append(profit_abs)
                profit_closed_ratio.append(profit_ratio)
                if profit_ratio >= 0:
                    winning_trades += 1
                    winning_profit += profit_abs
                else:
                    losing_trades += 1
                    losing_profit += profit_abs
            else:
                # Get current rate
                try:
                    current_rate = self._freqtrade.exchange.get_rate(
                        trade.pair, side='exit', is_short=trade.is_short, refresh=False)
                except (PricingError, ExchangeError):
                    current_rate = NAN
                if isnan(current_rate):
                    profit_ratio = NAN
                    profit_abs = NAN
                else:
                    profit = trade.calculate_profit(trade.close_rate or current_rate)

                    profit_ratio = profit.profit_ratio
                    profit_abs = profit.total_profit

            profit_all_coin.append(profit_abs)
            profit_all_ratio.append(profit_ratio)

        closed_trade_count = len([t for t in trades if not t.is_open])

        best_pair = Trade.get_best_pair(start_date)
        trading_volume = Trade.get_trading_volume(start_date)

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
        # Doing the sum is not right - overall profit needs to be based on initial capital
        profit_all_ratio_sum = sum(profit_all_ratio) if profit_all_ratio else 0.0
        starting_balance = self._freqtrade.wallets.get_starting_balance()
        profit_closed_ratio_fromstart = 0
        profit_all_ratio_fromstart = 0
        if starting_balance:
            profit_closed_ratio_fromstart = profit_closed_coin_sum / starting_balance
            profit_all_ratio_fromstart = profit_all_coin_sum / starting_balance

        profit_factor = winning_profit / abs(losing_profit) if losing_profit else float('inf')

        winrate = (winning_trades / closed_trade_count) if closed_trade_count > 0 else 0

        trades_df = DataFrame([{'close_date': format_date(trade.close_date),
                                'close_date_dt': trade.close_date,
                                'profit_abs': trade.close_profit_abs}
                               for trade in trades if not trade.is_open and trade.close_date])

        expectancy, expectancy_ratio = calculate_expectancy(trades_df)

        max_drawdown_abs = 0.0
        max_drawdown = 0.0
        drawdown_start: Optional[datetime] = None
        drawdown_end: Optional[datetime] = None
        dd_high_val = dd_low_val = 0.0
        if len(trades_df) > 0:
            try:
                (max_drawdown_abs, drawdown_start, drawdown_end, dd_high_val, dd_low_val,
                 max_drawdown) = calculate_max_drawdown(
                    trades_df, value_col='profit_abs', date_col='close_date_dt',
                    starting_balance=starting_balance)
            except ValueError:
                # ValueError if no losing trade.
                pass

        profit_all_fiat = self._fiat_converter.convert_amount(
            profit_all_coin_sum,
            stake_currency,
            fiat_display_currency
        ) if self._fiat_converter else 0

        first_date = trades[0].open_date_utc if trades else None
        last_date = trades[-1].open_date_utc if trades else None
        num = float(len(durations) or 1)
        bot_start = KeyValueStore.get_datetime_value(KeyStoreKeys.BOT_START_TIME)
        return {
            'profit_closed_coin': profit_closed_coin_sum,
            'profit_closed_percent_mean': round(profit_closed_ratio_mean * 100, 2),
            'profit_closed_ratio_mean': profit_closed_ratio_mean,
            'profit_closed_percent_sum': round(profit_closed_ratio_sum * 100, 2),
            'profit_closed_ratio_sum': profit_closed_ratio_sum,
            'profit_closed_ratio': profit_closed_ratio_fromstart,
            'profit_closed_percent': round(profit_closed_ratio_fromstart * 100, 2),
            'profit_closed_fiat': profit_closed_fiat,
            'profit_all_coin': profit_all_coin_sum,
            'profit_all_percent_mean': round(profit_all_ratio_mean * 100, 2),
            'profit_all_ratio_mean': profit_all_ratio_mean,
            'profit_all_percent_sum': round(profit_all_ratio_sum * 100, 2),
            'profit_all_ratio_sum': profit_all_ratio_sum,
            'profit_all_ratio': profit_all_ratio_fromstart,
            'profit_all_percent': round(profit_all_ratio_fromstart * 100, 2),
            'profit_all_fiat': profit_all_fiat,
            'trade_count': len(trades),
            'closed_trade_count': closed_trade_count,
            'first_trade_date': format_date(first_date),
            'first_trade_humanized': dt_humanize(first_date) if first_date else '',
            'first_trade_timestamp': dt_ts_def(first_date, 0),
            'latest_trade_date': format_date(last_date),
            'latest_trade_humanized': dt_humanize(last_date) if last_date else '',
            'latest_trade_timestamp': dt_ts_def(last_date, 0),
            'avg_duration': str(timedelta(seconds=sum(durations) / num)).split('.')[0],
            'best_pair': best_pair[0] if best_pair else '',
            'best_rate': round(best_pair[1] * 100, 2) if best_pair else 0,  # Deprecated
            'best_pair_profit_ratio': best_pair[1] if best_pair else 0,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'profit_factor': profit_factor,
            'winrate': winrate,
            'expectancy': expectancy,
            'expectancy_ratio': expectancy_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_abs': max_drawdown_abs,
            'max_drawdown_start': format_date(drawdown_start),
            'max_drawdown_start_timestamp': dt_ts_def(drawdown_start),
            'max_drawdown_end': format_date(drawdown_end),
            'max_drawdown_end_timestamp': dt_ts_def(drawdown_end),
            'drawdown_high': dd_high_val,
            'drawdown_low': dd_low_val,
            'trading_volume': trading_volume,
            'bot_start_timestamp': dt_ts_def(bot_start, 0),
            'bot_start_date': format_date(bot_start),
        }

    def __balance_get_est_stake(
            self, coin: str, stake_currency: str, amount: float,
            balance: Wallet, tickers) -> Tuple[float, float]:
        est_stake = 0.0
        est_bot_stake = 0.0
        if coin == stake_currency:
            est_stake = balance.total
            if self._config.get('trading_mode', TradingMode.SPOT) != TradingMode.SPOT:
                # in Futures, "total" includes the locked stake, and therefore all positions
                est_stake = balance.free
            est_bot_stake = amount
        else:
            pair = self._freqtrade.exchange.get_valid_pair_combination(coin, stake_currency)
            rate: Optional[float] = tickers.get(pair, {}).get('last', None)
            if rate:
                if pair.startswith(stake_currency) and not pair.endswith(stake_currency):
                    rate = 1.0 / rate
                est_stake = rate * balance.total
                est_bot_stake = rate * amount

        return est_stake, est_bot_stake

    def _rpc_balance(self, stake_currency: str, fiat_display_currency: str) -> Dict:
        """ Returns current account balance per crypto """
        currencies: List[Dict] = []
        total = 0.0
        total_bot = 0.0
        try:
            tickers: Tickers = self._freqtrade.exchange.get_tickers(cached=True)
        except (ExchangeError):
            raise RPCException('Error getting current tickers.')

        open_trades: List[Trade] = Trade.get_open_trades()
        open_assets: Dict[str, Trade] = {t.safe_base_currency: t for t in open_trades}
        self._freqtrade.wallets.update(require_update=False)
        starting_capital = self._freqtrade.wallets.get_starting_balance()
        starting_cap_fiat = self._fiat_converter.convert_amount(
            starting_capital, stake_currency, fiat_display_currency) if self._fiat_converter else 0
        coin: str
        balance: Wallet
        for coin, balance in self._freqtrade.wallets.get_all_balances().items():
            if not balance.total:
                continue

            trade = open_assets.get(coin, None)
            is_bot_managed = coin == stake_currency or trade is not None
            trade_amount = trade.amount if trade else 0
            if coin == stake_currency:
                trade_amount = self._freqtrade.wallets.get_available_stake_amount()

            try:
                est_stake, est_stake_bot = self.__balance_get_est_stake(
                    coin, stake_currency, trade_amount, balance, tickers)
            except ValueError:
                continue

            total += est_stake

            if is_bot_managed:
                total_bot += est_stake_bot
            currencies.append({
                'currency': coin,
                'free': balance.free,
                'balance': balance.total,
                'used': balance.used,
                'bot_owned': trade_amount,
                'est_stake': est_stake or 0,
                'est_stake_bot': est_stake_bot if is_bot_managed else 0,
                'stake': stake_currency,
                'side': 'long',
                'leverage': 1,
                'position': 0,
                'is_bot_managed': is_bot_managed,
                'is_position': False,
            })
        symbol: str
        position: PositionWallet
        for symbol, position in self._freqtrade.wallets.get_all_positions().items():
            total += position.collateral
            total_bot += position.collateral

            currencies.append({
                'currency': symbol,
                'free': 0,
                'balance': 0,
                'used': 0,
                'position': position.position,
                'est_stake': position.collateral,
                'est_stake_bot': position.collateral,
                'stake': stake_currency,
                'leverage': position.leverage,
                'side': position.side,
                'is_bot_managed': True,
                'is_position': True
            })

        value = self._fiat_converter.convert_amount(
            total, stake_currency, fiat_display_currency) if self._fiat_converter else 0
        value_bot = self._fiat_converter.convert_amount(
            total_bot, stake_currency, fiat_display_currency) if self._fiat_converter else 0

        trade_count = len(Trade.get_trades_proxy())
        starting_capital_ratio = (total_bot / starting_capital) - 1 if starting_capital else 0.0
        starting_cap_fiat_ratio = (value_bot / starting_cap_fiat) - 1 if starting_cap_fiat else 0.0

        return {
            'currencies': currencies,
            'total': total,
            'total_bot': total_bot,
            'symbol': fiat_display_currency,
            'value': value,
            'value_bot': value_bot,
            'stake': stake_currency,
            'starting_capital': starting_capital,
            'starting_capital_ratio': starting_capital_ratio,
            'starting_capital_pct': round(starting_capital_ratio * 100, 2),
            'starting_capital_fiat': starting_cap_fiat,
            'starting_capital_fiat_ratio': starting_cap_fiat_ratio,
            'starting_capital_fiat_pct': round(starting_cap_fiat_ratio * 100, 2),
            'trade_count': trade_count,
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

    def _rpc_stopentry(self) -> Dict[str, str]:
        """
        Handler to stop buying, but handle open trades gracefully.
        """
        if self._freqtrade.state == State.RUNNING:
            # Set 'max_open_trades' to 0
            self._freqtrade.config['max_open_trades'] = 0
            self._freqtrade.strategy.max_open_trades = 0

        return {'status': 'No more entries will occur from now. Run /reload_config to reset.'}

    def _rpc_reload_trade_from_exchange(self, trade_id: int) -> Dict[str, str]:
        """
        Handler for reload_trade_from_exchange.
        Reloads a trade from it's orders, should manual interaction have happened.
        """
        trade = Trade.get_trades(trade_filter=[Trade.id == trade_id]).first()
        if not trade:
            raise RPCException(f"Could not find trade with id {trade_id}.")

        self._freqtrade.handle_onexchange_order(trade)
        return {'status': 'Reloaded from orders from exchange'}

    def __exec_force_exit(self, trade: Trade, ordertype: Optional[str],
                          amount: Optional[float] = None) -> bool:
        # Check if there is there are open orders
        trade_entry_cancelation_registry = []
        for oo in trade.open_orders:
            trade_entry_cancelation_res = {'order_id': oo.order_id, 'cancel_state': False}
            order = self._freqtrade.exchange.fetch_order(oo.order_id, trade.pair)

            if order['side'] == trade.entry_side:
                fully_canceled = self._freqtrade.handle_cancel_enter(
                    trade, order, oo, CANCEL_REASON['FORCE_EXIT'])
                trade_entry_cancelation_res['cancel_state'] = fully_canceled
                trade_entry_cancelation_registry.append(trade_entry_cancelation_res)

            if order['side'] == trade.exit_side:
                # Cancel order - so it is placed anew with a fresh price.
                self._freqtrade.handle_cancel_exit(
                    trade, order, oo, CANCEL_REASON['FORCE_EXIT'])

        if all(tocr['cancel_state'] is False for tocr in trade_entry_cancelation_registry):
            if trade.has_open_orders:
                # Order cancellation failed, so we can't exit.
                return False
            # Get current rate and execute sell
            current_rate = self._freqtrade.exchange.get_rate(
                trade.pair, side='exit', is_short=trade.is_short, refresh=True)
            exit_check = ExitCheckTuple(exit_type=ExitType.FORCE_EXIT)
            order_type = ordertype or self._freqtrade.strategy.order_types.get(
                "force_exit", self._freqtrade.strategy.order_types["exit"])
            sub_amount: Optional[float] = None
            if amount and amount < trade.amount:
                # Partial exit ...
                min_exit_stake = self._freqtrade.exchange.get_min_pair_stake_amount(
                    trade.pair, current_rate, trade.stop_loss_pct)
                remaining = (trade.amount - amount) * current_rate
                if remaining < min_exit_stake:
                    raise RPCException(f'Remaining amount of {remaining} would be too small.')
                sub_amount = amount

            self._freqtrade.execute_trade_exit(
                trade, current_rate, exit_check, ordertype=order_type,
                sub_trade_amt=sub_amount)

            return True
        return False

    def _rpc_force_exit(self, trade_id: str, ordertype: Optional[str] = None, *,
                        amount: Optional[float] = None) -> Dict[str, str]:
        """
        Handler for forceexit <id>.
        Sells the given trade at current price
        """

        if self._freqtrade.state != State.RUNNING:
            raise RPCException('trader is not running')

        with self._freqtrade._exit_lock:
            if trade_id == 'all':
                # Execute exit for all open orders
                for trade in Trade.get_open_trades():
                    self.__exec_force_exit(trade, ordertype)
                Trade.commit()
                self._freqtrade.wallets.update()
                return {'result': 'Created exit orders for all open trades.'}

            # Query for trade
            trade = Trade.get_trades(
                trade_filter=[Trade.id == trade_id, Trade.is_open.is_(True), ]
            ).first()
            if not trade:
                logger.warning('force_exit: Invalid argument received')
                raise RPCException('invalid argument')

            result = self.__exec_force_exit(trade, ordertype, amount)
            Trade.commit()
            self._freqtrade.wallets.update()
            if not result:
                raise RPCException('Failed to exit trade.')
            return {'result': f'Created exit order for trade {trade_id}.'}

    def _force_entry_validations(self, pair: str, order_side: SignalDirection):
        if not self._freqtrade.config.get('force_entry_enable', False):
            raise RPCException('Force_entry not enabled.')

        if self._freqtrade.state != State.RUNNING:
            raise RPCException('trader is not running')

        if order_side == SignalDirection.SHORT and self._freqtrade.trading_mode == TradingMode.SPOT:
            raise RPCException("Can't go short on Spot markets.")

        if pair not in self._freqtrade.exchange.get_markets(tradable_only=True):
            raise RPCException('Symbol does not exist or market is not active.')
        # Check if pair quote currency equals to the stake currency.
        stake_currency = self._freqtrade.config.get('stake_currency')
        if not self._freqtrade.exchange.get_pair_quote_currency(pair) == stake_currency:
            raise RPCException(
                f'Wrong pair selected. Only pairs with stake-currency {stake_currency} allowed.')

    def _rpc_force_entry(self, pair: str, price: Optional[float], *,
                         order_type: Optional[str] = None,
                         order_side: SignalDirection = SignalDirection.LONG,
                         stake_amount: Optional[float] = None,
                         enter_tag: Optional[str] = 'force_entry',
                         leverage: Optional[float] = None) -> Optional[Trade]:
        """
        Handler for forcebuy <asset> <price>
        Buys a pair trade at the given or current price
        """
        self._force_entry_validations(pair, order_side)

        # check if valid pair

        # check if pair already has an open pair
        trade: Optional[Trade] = Trade.get_trades(
            [Trade.is_open.is_(True), Trade.pair == pair]).first()
        is_short = (order_side == SignalDirection.SHORT)
        if trade:
            is_short = trade.is_short
            if not self._freqtrade.strategy.position_adjustment_enable:
                raise RPCException(f"position for {pair} already open - id: {trade.id}")
            if trade.has_open_orders:
                raise RPCException(f"position for {pair} already open - id: {trade.id} "
                                   f"and has open order {','.join(trade.open_orders_ids)}")
        else:
            if Trade.get_open_trade_count() >= self._config['max_open_trades']:
                raise RPCException("Maximum number of trades is reached.")

        if not stake_amount:
            # gen stake amount
            stake_amount = self._freqtrade.wallets.get_trade_stake_amount(
                pair, self._config['max_open_trades'])

        # execute buy
        if not order_type:
            order_type = self._freqtrade.strategy.order_types.get(
                'force_entry', self._freqtrade.strategy.order_types['entry'])
        with self._freqtrade._exit_lock:
            if self._freqtrade.execute_entry(pair, stake_amount, price,
                                             ordertype=order_type, trade=trade,
                                             is_short=is_short,
                                             enter_tag=enter_tag,
                                             leverage_=leverage,
                                             ):
                Trade.commit()
                trade = Trade.get_trades([Trade.is_open.is_(True), Trade.pair == pair]).first()
                return trade
            else:
                raise RPCException(f'Failed to enter position for {pair}.')

    def _rpc_cancel_open_order(self, trade_id: int):
        if self._freqtrade.state != State.RUNNING:
            raise RPCException('trader is not running')
        with self._freqtrade._exit_lock:
            # Query for trade
            trade = Trade.get_trades(
                trade_filter=[Trade.id == trade_id, Trade.is_open.is_(True), ]
            ).first()
            if not trade:
                logger.warning('cancel_open_order: Invalid trade_id received.')
                raise RPCException('Invalid trade_id.')
            if not trade.has_open_orders:
                logger.warning('cancel_open_order: No open order for trade_id.')
                raise RPCException('No open order for trade_id.')

            for open_order in trade.open_orders:
                try:
                    order = self._freqtrade.exchange.fetch_order(open_order.order_id, trade.pair)
                except ExchangeError as e:
                    logger.info(f"Cannot query order for {trade} due to {e}.", exc_info=True)
                    raise RPCException("Order not found.")
                self._freqtrade.handle_cancel_order(
                    order, open_order, trade, CANCEL_REASON['USER_CANCEL'])
            Trade.commit()

    def _rpc_delete(self, trade_id: int) -> Dict[str, Union[str, int]]:
        """
        Handler for delete <id>.
        Delete the given trade and close eventually existing open orders.
        """
        with self._freqtrade._exit_lock:
            c_count = 0
            trade = Trade.get_trades(trade_filter=[Trade.id == trade_id]).first()
            if not trade:
                logger.warning('delete trade: Invalid argument received')
                raise RPCException('invalid argument')

            # Try cancelling regular order if that exists
            for open_order in trade.open_orders:
                try:
                    self._freqtrade.exchange.cancel_order(open_order.order_id, trade.pair)
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

        return pair_rates

    def _rpc_enter_tag_performance(self, pair: Optional[str]) -> List[Dict[str, Any]]:
        """
        Handler for buy tag performance.
        Shows a performance statistic from finished trades
        """
        return Trade.get_enter_tag_performance(pair)

    def _rpc_exit_reason_performance(self, pair: Optional[str]) -> List[Dict[str, Any]]:
        """
        Handler for exit reason performance.
        Shows a performance statistic from finished trades
        """
        return Trade.get_exit_reason_performance(pair)

    def _rpc_mix_tag_performance(self, pair: Optional[str]) -> List[Dict[str, Any]]:
        """
        Handler for mix tag (enter_tag + exit_reason) performance.
        Shows a performance statistic from finished trades
        """
        mix_tags = Trade.get_mix_tag_performance(pair)

        return mix_tags

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
        locks: Sequence[PairLock] = []

        if pair:
            locks = PairLocks.get_pair_locks(pair)
        if lockid:
            locks = PairLock.session.scalars(select(PairLock).filter(PairLock.id == lockid)).all()

        for lock in locks:
            lock.active = False
            lock.lock_end_time = datetime.now(timezone.utc)

        Trade.commit()

        return self._rpc_locks()

    def _rpc_whitelist(self) -> Dict:
        """ Returns the currently active whitelist"""
        res = {'method': self._freqtrade.pairlists.name_list,
               'length': len(self._freqtrade.active_pair_whitelist),
               'whitelist': self._freqtrade.active_pair_whitelist
               }
        return res

    def _rpc_blacklist_delete(self, delete: List[str]) -> Dict:
        """ Removes pairs from currently active blacklist """
        errors = {}
        for pair in delete:
            if pair in self._freqtrade.pairlists.blacklist:
                self._freqtrade.pairlists.blacklist.remove(pair)
            else:
                errors[pair] = {
                    'error_msg': f"Pair {pair} is not in the current blacklist."
                }
        resp = self._rpc_blacklist()
        resp['errors'] = errors
        return resp

    def _rpc_blacklist(self, add: Optional[List[str]] = None) -> Dict:
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
        records = [[format_date(datetime.fromtimestamp(r.created)),
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
        signals = {
            'enter_long': 0,
            'exit_long': 0,
            'enter_short': 0,
            'exit_short': 0,
        }
        if has_content:

            dataframe.loc[:, '__date_ts'] = dataframe.loc[:, 'date'].view(int64) // 1000 // 1000
            # Move signal close to separate column when signal for easy plotting
            for sig_type in signals.keys():
                if sig_type in dataframe.columns:
                    mask = (dataframe[sig_type] == 1)
                    signals[sig_type] = int(mask.sum())
                    dataframe.loc[mask, f'_{sig_type}_signal_close'] = dataframe.loc[mask, 'close']

            # band-aid until this is fixed:
            # https://github.com/pandas-dev/pandas/issues/45836
            datetime_types = ['datetime', 'datetime64', 'datetime64[ns, UTC]']
            date_columns = dataframe.select_dtypes(include=datetime_types)
            for date_column in date_columns:
                # replace NaT with `None`
                dataframe[date_column] = dataframe[date_column].astype(object).replace({NaT: None})

            dataframe = dataframe.replace({inf: None, -inf: None, NAN: None})

        res = {
            'pair': pair,
            'timeframe': timeframe,
            'timeframe_ms': timeframe_to_msecs(timeframe),
            'strategy': strategy,
            'columns': list(dataframe.columns),
            'data': dataframe.values.tolist(),
            'length': len(dataframe),
            'buy_signals': signals['enter_long'],  # Deprecated
            'sell_signals': signals['exit_long'],  # Deprecated
            'enter_long_signals': signals['enter_long'],
            'exit_long_signals': signals['exit_long'],
            'enter_short_signals': signals['enter_short'],
            'exit_short_signals': signals['exit_short'],
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
        """ Analyzed dataframe in Dict form """

        _data, last_analyzed = self.__rpc_analysed_dataframe_raw(pair, timeframe, limit)
        return RPC._convert_dataframe_to_dict(self._freqtrade.config['strategy'],
                                              pair, timeframe, _data, last_analyzed)

    def __rpc_analysed_dataframe_raw(
        self,
        pair: str,
        timeframe: str,
        limit: Optional[int]
    ) -> Tuple[DataFrame, datetime]:
        """
        Get the dataframe and last analyze from the dataprovider

        :param pair: The pair to get
        :param timeframe: The timeframe of data to get
        :param limit: The amount of candles in the dataframe
        """
        _data, last_analyzed = self._freqtrade.dataprovider.get_analyzed_dataframe(
            pair, timeframe)
        _data = _data.copy()

        if limit:
            _data = _data.iloc[-limit:]

        return _data, last_analyzed

    def _ws_all_analysed_dataframes(
        self,
        pairlist: List[str],
        limit: Optional[int]
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Get the analysed dataframes of each pair in the pairlist.
        If specified, only return the most recent `limit` candles for
        each dataframe.

        :param pairlist: A list of pairs to get
        :param limit: If an integer, limits the size of dataframe
                      If a list of string date times, only returns those candles
        :returns: A generator of dictionaries with the key, dataframe, and last analyzed timestamp
        """
        timeframe = self._freqtrade.config['timeframe']
        candle_type = self._freqtrade.config.get('candle_type_def', CandleType.SPOT)

        for pair in pairlist:
            dataframe, last_analyzed = self.__rpc_analysed_dataframe_raw(pair, timeframe, limit)

            yield {
                "key": (pair, timeframe, candle_type),
                "df": dataframe,
                "la": last_analyzed
            }

    def _ws_request_analyzed_df(
        self,
        limit: Optional[int] = None,
        pair: Optional[str] = None
    ):
        """ Historical Analyzed Dataframes for WebSocket """
        pairlist = [pair] if pair else self._freqtrade.active_pair_whitelist

        return self._ws_all_analysed_dataframes(pairlist, limit)

    def _ws_request_whitelist(self):
        """ Whitelist data for WebSocket """
        return self._freqtrade.active_pair_whitelist

    @staticmethod
    def _rpc_analysed_history_full(config: Config, pair: str, timeframe: str,
                                   exchange) -> Dict[str, Any]:
        timerange_parsed = TimeRange.parse_timerange(config.get('timerange'))

        from freqtrade.data.converter import trim_dataframe
        from freqtrade.data.dataprovider import DataProvider
        from freqtrade.resolvers.strategy_resolver import StrategyResolver

        strategy = StrategyResolver.load_strategy(config)
        startup_candles = strategy.startup_candle_count

        _data = load_data(
            datadir=config["datadir"],
            pairs=[pair],
            timeframe=timeframe,
            timerange=timerange_parsed,
            data_format=config['dataformat_ohlcv'],
            candle_type=config.get('candle_type_def', CandleType.SPOT),
            startup_candles=startup_candles,
        )
        if pair not in _data:
            raise RPCException(
                f"No data for {pair}, {timeframe} in {config.get('timerange')} found.")

        strategy.dp = DataProvider(config, exchange=exchange, pairlists=None)
        strategy.ft_bot_start()

        df_analyzed = strategy.analyze_ticker(_data[pair], {'pair': pair})
        df_analyzed = trim_dataframe(df_analyzed, timerange_parsed, startup_candles=startup_candles)

        return RPC._convert_dataframe_to_dict(strategy.get_strategy_name(), pair, timeframe,
                                              df_analyzed.copy(), dt_now())

    def _rpc_plot_config(self) -> Dict[str, Any]:
        if (self._freqtrade.strategy.plot_config and
                'subplots' not in self._freqtrade.strategy.plot_config):
            self._freqtrade.strategy.plot_config['subplots'] = {}
        return self._freqtrade.strategy.plot_config

    @staticmethod
    def _rpc_plot_config_with_strategy(config: Config) -> Dict[str, Any]:

        from freqtrade.resolvers.strategy_resolver import StrategyResolver
        strategy = StrategyResolver.load_strategy(config)

        if (strategy.plot_config and 'subplots' not in strategy.plot_config):
            strategy.plot_config['subplots'] = {}
        return strategy.plot_config

    @staticmethod
    def _rpc_sysinfo() -> Dict[str, Any]:
        return {
            "cpu_pct": psutil.cpu_percent(interval=1, percpu=True),
            "ram_pct": psutil.virtual_memory().percent
        }

    def health(self) -> Dict[str, Optional[Union[str, int]]]:
        last_p = self._freqtrade.last_process
        if last_p is None:
            return {
                "last_process": None,
                "last_process_loc": None,
                "last_process_ts": None,
            }

        return {
            "last_process": str(last_p),
            "last_process_loc": format_date(last_p.astimezone(tzlocal())),
            "last_process_ts": int(last_p.timestamp()),
        }

    def _update_market_direction(self, direction: MarketDirection) -> None:
        self._freqtrade.strategy.market_direction = direction

    def _get_market_direction(self) -> MarketDirection:
        return self._freqtrade.strategy.market_direction
