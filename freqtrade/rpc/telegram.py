# pragma pylint: disable=unused-argument, unused-variable, protected-access, invalid-name

"""
This module manage Telegram communication
"""
import asyncio
import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import partial, wraps
from html import escape
from itertools import chain
from math import isnan
from threading import Thread
from typing import Any, Callable, Coroutine, Dict, List, Literal, Optional, Union

from tabulate import tabulate
from telegram import (CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton,
                      ReplyKeyboardMarkup, Update)
from telegram.constants import MessageLimit, ParseMode
from telegram.error import BadRequest, NetworkError, TelegramError
from telegram.ext import Application, CallbackContext, CallbackQueryHandler, CommandHandler
from telegram.helpers import escape_markdown

from freqtrade.__init__ import __version__
from freqtrade.constants import DUST_PER_COIN, Config
from freqtrade.enums import MarketDirection, RPCMessageType, SignalDirection, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.misc import chunks, plural
from freqtrade.persistence import Trade
from freqtrade.rpc import RPC, RPCException, RPCHandler
from freqtrade.rpc.rpc_types import RPCEntryMsg, RPCExitMsg, RPCOrderMsg, RPCSendMsg
from freqtrade.util import dt_humanize, fmt_coin, format_date, round_value


MAX_MESSAGE_LENGTH = MessageLimit.MAX_TEXT_LENGTH


logger = logging.getLogger(__name__)

logger.debug('Included module rpc.telegram ...')


def safe_async_db(func: Callable[..., Any]):
    """
    Decorator to safely handle sessions when switching async context
    :param func: function to decorate
    :return: decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """ Decorator logic """
        try:
            return func(*args, **kwargs)
        finally:
            Trade.session.remove()

    return wrapper


@dataclass
class TimeunitMappings:
    header: str
    message: str
    message2: str
    callback: str
    default: int
    dateformat: str


def authorized_only(command_handler: Callable[..., Coroutine[Any, Any, None]]):
    """
    Decorator to check if the message comes from the correct chat_id
    :param command_handler: Telegram CommandHandler
    :return: decorated function
    """

    @wraps(command_handler)
    async def wrapper(self, *args, **kwargs):
        """ Decorator logic """
        update = kwargs.get('update') or args[0]

        # Reject unauthorized messages
        if update.callback_query:
            cchat_id = int(update.callback_query.message.chat.id)
        else:
            cchat_id = int(update.message.chat_id)

        chat_id = int(self._config['telegram']['chat_id'])
        if cchat_id != chat_id:
            logger.info(f'Rejected unauthorized message from: {update.message.chat_id}')
            return wrapper
        # Rollback session to avoid getting data stored in a transaction.
        Trade.rollback()
        logger.debug(
            'Executing handler: %s for chat_id: %s',
            command_handler.__name__,
            chat_id
        )
        try:
            return await command_handler(self, *args, **kwargs)
        except RPCException as e:
            await self._send_msg(str(e))
        except BaseException:
            logger.exception('Exception occurred within Telegram module')
        finally:
            Trade.session.remove()

    return wrapper


class Telegram(RPCHandler):
    """  This class handles all telegram communication """

    def __init__(self, rpc: RPC, config: Config) -> None:
        """
        Init the Telegram call, and init the super class RPCHandler
        :param rpc: instance of RPC Helper class
        :param config: Configuration object
        :return: None
        """
        super().__init__(rpc, config)

        self._app: Application
        self._loop: asyncio.AbstractEventLoop
        self._init_keyboard()
        self._start_thread()

    def _start_thread(self):
        """
        Creates and starts the polling thread
        """
        self._thread = Thread(target=self._init, name='FTTelegram')
        self._thread.start()

    def _init_keyboard(self) -> None:
        """
        Validates the keyboard configuration from telegram config
        section.
        """
        self._keyboard: List[List[Union[str, KeyboardButton]]] = [
            ['/daily', '/profit', '/balance'],
            ['/status', '/status table', '/performance'],
            ['/count', '/start', '/stop', '/help']
        ]
        # do not allow commands with mandatory arguments and critical cmds
        # TODO: DRY! - its not good to list all valid cmds here. But otherwise
        #       this needs refactoring of the whole telegram module (same
        #       problem in _help()).
        valid_keys: List[str] = [
            r'/start$', r'/stop$', r'/status$', r'/status table$',
            r'/trades$', r'/performance$', r'/buys', r'/entries',
            r'/sells', r'/exits', r'/mix_tags',
            r'/daily$', r'/daily \d+$', r'/profit$', r'/profit \d+',
            r'/stats$', r'/count$', r'/locks$', r'/balance$',
            r'/stopbuy$', r'/stopentry$', r'/reload_config$', r'/show_config$',
            r'/logs$', r'/whitelist$', r'/whitelist(\ssorted|\sbaseonly)+$',
            r'/blacklist$', r'/bl_delete$',
            r'/weekly$', r'/weekly \d+$', r'/monthly$', r'/monthly \d+$',
            r'/forcebuy$', r'/forcelong$', r'/forceshort$',
            r'/forcesell$', r'/forceexit$',
            r'/edge$', r'/health$', r'/help$', r'/version$', r'/marketdir (long|short|even|none)$',
            r'/marketdir$'
        ]
        # Create keys for generation
        valid_keys_print = [k.replace('$', '') for k in valid_keys]

        # custom keyboard specified in config.json
        cust_keyboard = self._config['telegram'].get('keyboard', [])
        if cust_keyboard:
            combined = "(" + ")|(".join(valid_keys) + ")"
            # check for valid shortcuts
            invalid_keys = [b for b in chain.from_iterable(cust_keyboard)
                            if not re.match(combined, b)]
            if len(invalid_keys):
                err_msg = ('config.telegram.keyboard: Invalid commands for '
                           f'custom Telegram keyboard: {invalid_keys}'
                           f'\nvalid commands are: {valid_keys_print}')
                raise OperationalException(err_msg)
            else:
                self._keyboard = cust_keyboard
                logger.info('using custom keyboard from '
                            f'config.json: {self._keyboard}')

    def _init_telegram_app(self):
        return Application.builder().token(self._config['telegram']['token']).build()

    def _init(self) -> None:
        """
        Initializes this module with the given config,
        registers all known command handlers
        and starts polling for message updates
        Runs in a separate thread.
        """
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        self._app = self._init_telegram_app()

        # Register command handler and start telegram message polling
        handles = [
            CommandHandler('status', self._status),
            CommandHandler('profit', self._profit),
            CommandHandler('balance', self._balance),
            CommandHandler('start', self._start),
            CommandHandler('stop', self._stop),
            CommandHandler(['forcesell', 'forceexit', 'fx'], self._force_exit),
            CommandHandler(['forcebuy', 'forcelong'], partial(
                self._force_enter, order_side=SignalDirection.LONG)),
            CommandHandler('forceshort', partial(
                self._force_enter, order_side=SignalDirection.SHORT)),
            CommandHandler('reload_trade', self._reload_trade_from_exchange),
            CommandHandler('trades', self._trades),
            CommandHandler('delete', self._delete_trade),
            CommandHandler(['coo', 'cancel_open_order'], self._cancel_open_order),
            CommandHandler('performance', self._performance),
            CommandHandler(['buys', 'entries'], self._enter_tag_performance),
            CommandHandler(['sells', 'exits'], self._exit_reason_performance),
            CommandHandler('mix_tags', self._mix_tag_performance),
            CommandHandler('stats', self._stats),
            CommandHandler('daily', self._daily),
            CommandHandler('weekly', self._weekly),
            CommandHandler('monthly', self._monthly),
            CommandHandler('count', self._count),
            CommandHandler('locks', self._locks),
            CommandHandler(['unlock', 'delete_locks'], self._delete_locks),
            CommandHandler(['reload_config', 'reload_conf'], self._reload_config),
            CommandHandler(['show_config', 'show_conf'], self._show_config),
            CommandHandler(['stopbuy', 'stopentry'], self._stopentry),
            CommandHandler('whitelist', self._whitelist),
            CommandHandler('blacklist', self._blacklist),
            CommandHandler(['blacklist_delete', 'bl_delete'], self._blacklist_delete),
            CommandHandler('logs', self._logs),
            CommandHandler('edge', self._edge),
            CommandHandler('health', self._health),
            CommandHandler('help', self._help),
            CommandHandler('version', self._version),
            CommandHandler('marketdir', self._changemarketdir),
            CommandHandler('order', self._order),
            CommandHandler('list_custom_data', self._list_custom_data),
        ]
        callbacks = [
            CallbackQueryHandler(self._status_table, pattern='update_status_table'),
            CallbackQueryHandler(self._daily, pattern='update_daily'),
            CallbackQueryHandler(self._weekly, pattern='update_weekly'),
            CallbackQueryHandler(self._monthly, pattern='update_monthly'),
            CallbackQueryHandler(self._profit, pattern='update_profit'),
            CallbackQueryHandler(self._balance, pattern='update_balance'),
            CallbackQueryHandler(self._performance, pattern='update_performance'),
            CallbackQueryHandler(self._enter_tag_performance,
                                 pattern='update_enter_tag_performance'),
            CallbackQueryHandler(self._exit_reason_performance,
                                 pattern='update_exit_reason_performance'),
            CallbackQueryHandler(self._mix_tag_performance, pattern='update_mix_tag_performance'),
            CallbackQueryHandler(self._count, pattern='update_count'),
            CallbackQueryHandler(self._force_exit_inline, pattern=r"force_exit__\S+"),
            CallbackQueryHandler(self._force_enter_inline, pattern=r"force_enter__\S+"),
        ]
        for handle in handles:
            self._app.add_handler(handle)

        for callback in callbacks:
            self._app.add_handler(callback)

        logger.info(
            'rpc.telegram is listening for following commands: %s',
            [[x for x in sorted(h.commands)] for h in handles]
        )
        self._loop.run_until_complete(self._startup_telegram())

    async def _startup_telegram(self) -> None:
        await self._app.initialize()
        await self._app.start()
        if self._app.updater:
            await self._app.updater.start_polling(
                bootstrap_retries=-1,
                timeout=20,
                # read_latency=60,  # Assumed transmission latency
                drop_pending_updates=True,
                # stop_signals=[],  # Necessary as we don't run on the main thread
            )
            while True:
                await asyncio.sleep(10)
                if not self._app.updater.running:
                    break

    async def _cleanup_telegram(self) -> None:
        if self._app.updater:
            await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

    def cleanup(self) -> None:
        """
        Stops all running telegram threads.
        :return: None
        """
        # This can take up to `timeout` from the call to `start_polling`.
        asyncio.run_coroutine_threadsafe(self._cleanup_telegram(), self._loop)
        self._thread.join()

    def _exchange_from_msg(self, msg: RPCOrderMsg) -> str:
        """
        Extracts the exchange name from the given message.
        :param msg: The message to extract the exchange name from.
        :return: The exchange name.
        """
        return f"{msg['exchange']}{' (dry)' if self._config['dry_run'] else ''}"

    def _add_analyzed_candle(self, pair: str) -> str:
        candle_val = self._config['telegram'].get(
            'notification_settings', {}).get('show_candle', 'off')
        if candle_val != 'off':
            if candle_val == 'ohlc':
                analyzed_df, _ = self._rpc._freqtrade.dataprovider.get_analyzed_dataframe(
                    pair, self._config['timeframe'])
                candle = analyzed_df.iloc[-1].squeeze() if len(analyzed_df) > 0 else None
                if candle is not None:
                    return (
                        f"*Candle OHLC*: `{candle['open']}, {candle['high']}, "
                        f"{candle['low']}, {candle['close']}`\n"
                    )

        return ''

    def _format_entry_msg(self, msg: RPCEntryMsg) -> str:

        is_fill = msg['type'] in [RPCMessageType.ENTRY_FILL]
        emoji = '\N{CHECK MARK}' if is_fill else '\N{LARGE BLUE CIRCLE}'

        terminology = {
            '1_enter': 'New Trade',
            '1_entered': 'New Trade filled',
            'x_enter': 'Increasing position',
            'x_entered': 'Position increase filled',
        }

        key = f"{'x' if msg['sub_trade'] else '1'}_{'entered' if is_fill else 'enter'}"
        wording = terminology[key]

        message = (
            f"{emoji} *{self._exchange_from_msg(msg)}:*"
            f" {wording} (#{msg['trade_id']})\n"
            f"*Pair:* `{msg['pair']}`\n"
        )
        message += self._add_analyzed_candle(msg['pair'])
        message += f"*Enter Tag:* `{msg['enter_tag']}`\n" if msg.get('enter_tag') else ""
        message += f"*Amount:* `{round_value(msg['amount'], 8)}`\n"
        message += f"*Direction:* `{msg['direction']}"
        if msg.get('leverage') and msg.get('leverage', 1.0) != 1.0:
            message += f" ({msg['leverage']:.3g}x)"
        message += "`\n"
        message += f"*Open Rate:* `{round_value(msg['open_rate'], 8)} {msg['quote_currency']}`\n"
        if msg['type'] == RPCMessageType.ENTRY and msg['current_rate']:
            message += (
                f"*Current Rate:* `{round_value(msg['current_rate'], 8)} {msg['quote_currency']}`\n"
            )

        profit_fiat_extra = self.__format_profit_fiat(msg, 'stake_amount')  # type: ignore
        total = fmt_coin(msg['stake_amount'], msg['quote_currency'])

        message += f"*{'New ' if msg['sub_trade'] else ''}Total:* `{total}{profit_fiat_extra}`"

        return message

    def _format_exit_msg(self, msg: RPCExitMsg) -> str:
        duration = msg['close_date'].replace(
            microsecond=0) - msg['open_date'].replace(microsecond=0)
        duration_min = duration.total_seconds() / 60

        leverage_text = (f" ({msg['leverage']:.3g}x)"
                         if msg.get('leverage') and msg.get('leverage', 1.0) != 1.0
                         else "")

        profit_fiat_extra = self.__format_profit_fiat(msg, 'profit_amount')

        profit_extra = (
            f" ({msg['gain']}: {fmt_coin(msg['profit_amount'], msg['quote_currency'])}"
            f"{profit_fiat_extra})")

        is_fill = msg['type'] == RPCMessageType.EXIT_FILL
        is_sub_trade = msg.get('sub_trade')
        is_sub_profit = msg['profit_amount'] != msg.get('cumulative_profit')
        is_final_exit = msg.get('is_final_exit', False) and is_sub_profit
        profit_prefix = 'Sub ' if is_sub_trade else ''
        cp_extra = ''
        exit_wording = 'Exited' if is_fill else 'Exiting'
        if is_sub_trade or is_final_exit:
            cp_fiat = self.__format_profit_fiat(msg, 'cumulative_profit')

            if is_final_exit:
                profit_prefix = 'Sub '
                cp_extra = (
                    f"*Final Profit:* `{msg['final_profit_ratio']:.2%} "
                    f"({msg['cumulative_profit']:.8f} {msg['quote_currency']}{cp_fiat})`\n"
                )
            else:
                exit_wording = f"Partially {exit_wording.lower()}"
                if msg['cumulative_profit']:
                    cp_extra = (
                        f"*Cumulative Profit:* `"
                        f"{fmt_coin(msg['cumulative_profit'], msg['stake_currency'])}{cp_fiat}`\n"
                    )
        enter_tag = f"*Enter Tag:* `{msg['enter_tag']}`\n" if msg.get('enter_tag') else ""
        message = (
            f"{self._get_exit_emoji(msg)} *{self._exchange_from_msg(msg)}:* "
            f"{exit_wording} {msg['pair']} (#{msg['trade_id']})\n"
            f"{self._add_analyzed_candle(msg['pair'])}"
            f"*{f'{profit_prefix}Profit' if is_fill else f'Unrealized {profit_prefix}Profit'}:* "
            f"`{msg['profit_ratio']:.2%}{profit_extra}`\n"
            f"{cp_extra}"
            f"{enter_tag}"
            f"*Exit Reason:* `{msg['exit_reason']}`\n"
            f"*Direction:* `{msg['direction']}"
            f"{leverage_text}`\n"
            f"*Amount:* `{round_value(msg['amount'], 8)}`\n"
            f"*Open Rate:* `{fmt_coin(msg['open_rate'], msg['quote_currency'])}`\n"
        )
        if msg['type'] == RPCMessageType.EXIT and msg['current_rate']:
            message += f"*Current Rate:* `{fmt_coin(msg['current_rate'], msg['quote_currency'])}`\n"
            if msg['order_rate']:
                message += f"*Exit Rate:* `{fmt_coin(msg['order_rate'], msg['quote_currency'])}`"
        elif msg['type'] == RPCMessageType.EXIT_FILL:
            message += f"*Exit Rate:* `{fmt_coin(msg['close_rate'], msg['quote_currency'])}`"

        if is_sub_trade:
            stake_amount_fiat = self.__format_profit_fiat(msg, 'stake_amount')

            rem = fmt_coin(msg['stake_amount'], msg['quote_currency'])
            message += f"\n*Remaining:* `{rem}{stake_amount_fiat}`"
        else:
            message += f"\n*Duration:* `{duration} ({duration_min:.1f} min)`"
        return message

    def __format_profit_fiat(
            self,
            msg: RPCExitMsg,
            key: Literal['stake_amount', 'profit_amount', 'cumulative_profit']
    ) -> str:
        """
        Format Fiat currency to append to regular profit output
        """
        profit_fiat_extra = ''
        if self._rpc._fiat_converter and (fiat_currency := msg.get('fiat_currency')):
            profit_fiat = self._rpc._fiat_converter.convert_amount(
                    msg[key], msg['stake_currency'], fiat_currency)
            profit_fiat_extra = f" / {profit_fiat:.3f} {fiat_currency}"
        return profit_fiat_extra

    def compose_message(self, msg: RPCSendMsg) -> Optional[str]:
        if msg['type'] == RPCMessageType.ENTRY or msg['type'] == RPCMessageType.ENTRY_FILL:
            message = self._format_entry_msg(msg)

        elif msg['type'] == RPCMessageType.EXIT or msg['type'] == RPCMessageType.EXIT_FILL:
            message = self._format_exit_msg(msg)

        elif (
            msg['type'] == RPCMessageType.ENTRY_CANCEL
            or msg['type'] == RPCMessageType.EXIT_CANCEL
        ):
            message_side = 'enter' if msg['type'] == RPCMessageType.ENTRY_CANCEL else 'exit'
            message = (f"\N{WARNING SIGN} *{self._exchange_from_msg(msg)}:* "
                       f"Cancelling {'partial ' if msg.get('sub_trade') else ''}"
                       f"{message_side} Order for {msg['pair']} "
                       f"(#{msg['trade_id']}). Reason: {msg['reason']}.")

        elif msg['type'] == RPCMessageType.PROTECTION_TRIGGER:
            message = (
                f"*Protection* triggered due to {msg['reason']}. "
                f"`{msg['pair']}` will be locked until `{msg['lock_end_time']}`."
            )

        elif msg['type'] == RPCMessageType.PROTECTION_TRIGGER_GLOBAL:
            message = (
                f"*Protection* triggered due to {msg['reason']}. "
                f"*All pairs* will be locked until `{msg['lock_end_time']}`."
            )

        elif msg['type'] == RPCMessageType.STATUS:
            message = f"*Status:* `{msg['status']}`"

        elif msg['type'] == RPCMessageType.WARNING:
            message = f"\N{WARNING SIGN} *Warning:* `{msg['status']}`"
        elif msg['type'] == RPCMessageType.EXCEPTION:
            # Errors will contain exceptions, which are wrapped in tripple ticks.
            message = f"\N{WARNING SIGN} *ERROR:* \n {msg['status']}"

        elif msg['type'] == RPCMessageType.STARTUP:
            message = f"{msg['status']}"
        elif msg['type'] == RPCMessageType.STRATEGY_MSG:
            message = f"{msg['msg']}"
        else:
            logger.debug("Unknown message type: %s", msg['type'])
            return None
        return message

    def send_msg(self, msg: RPCSendMsg) -> None:
        """ Send a message to telegram channel """

        default_noti = 'on'

        msg_type = msg['type']
        noti = ''
        if msg['type'] == RPCMessageType.EXIT:
            sell_noti = self._config['telegram'] \
                .get('notification_settings', {}).get(str(msg_type), {})
            # For backward compatibility sell still can be string
            if isinstance(sell_noti, str):
                noti = sell_noti
            else:
                noti = sell_noti.get(str(msg['exit_reason']), default_noti)
        else:
            noti = self._config['telegram'] \
                .get('notification_settings', {}).get(str(msg_type), default_noti)

        if noti == 'off':
            logger.info(f"Notification '{msg_type}' not sent.")
            # Notification disabled
            return

        message = self.compose_message(deepcopy(msg))
        if message:
            asyncio.run_coroutine_threadsafe(
                self._send_msg(message, disable_notification=(noti == 'silent')),
                self._loop)

    def _get_exit_emoji(self, msg):
        """
        Get emoji for exit-messages
        """

        if float(msg['profit_ratio']) >= 0.05:
            return "\N{ROCKET}"
        elif float(msg['profit_ratio']) >= 0.0:
            return "\N{EIGHT SPOKED ASTERISK}"
        elif msg['exit_reason'] == "stop_loss":
            return "\N{WARNING SIGN}"
        else:
            return "\N{CROSS MARK}"

    def _prepare_order_details(self, filled_orders: List, quote_currency: str, is_open: bool):
        """
        Prepare details of trade with entry adjustment enabled
        """
        lines_detail: List[str] = []
        if len(filled_orders) > 0:
            first_avg = filled_orders[0]["safe_price"]
        order_nr = 0
        for order in filled_orders:
            lines: List[str] = []
            if order['is_open'] is True:
                continue
            order_nr += 1
            wording = 'Entry' if order['ft_is_entry'] else 'Exit'

            cur_entry_amount = order["filled"] or order["amount"]
            cur_entry_average = order["safe_price"]
            lines.append("  ")
            lines.append(f"*{wording} #{order_nr}:*")
            if order_nr == 1:
                lines.append(
                    f"*Amount:* {round_value(cur_entry_amount, 8)} "
                    f"({fmt_coin(order['cost'], quote_currency)})"
                )
                lines.append(f"*Average Price:* {round_value(cur_entry_average, 8)}")
            else:
                # TODO: This calculation ignores fees.
                price_to_1st_entry = ((cur_entry_average - first_avg) / first_avg)
                if is_open:
                    lines.append("({})".format(dt_humanize(order["order_filled_date"],
                                                           granularity=["day", "hour", "minute"])))
                lines.append(f"*Amount:* {round_value(cur_entry_amount, 8)} "
                             f"({fmt_coin(order['cost'], quote_currency)})")
                lines.append(f"*Average {wording} Price:* {round_value(cur_entry_average, 8)} "
                             f"({price_to_1st_entry:.2%} from 1st entry rate)")
                lines.append(f"*Order Filled:* {order['order_filled_date']}")

            lines_detail.append("\n".join(lines))

        return lines_detail

    @authorized_only
    async def _order(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /order.
        Returns the orders of the trade
        :param bot: telegram bot
        :param update: message update
        :return: None
        """

        trade_ids = []
        if context.args and len(context.args) > 0:
            trade_ids = [int(i) for i in context.args if i.isnumeric()]

        results = self._rpc._rpc_trade_status(trade_ids=trade_ids)
        for r in results:
            lines = [
                "*Order List for Trade #*`{trade_id}`"
            ]

            lines_detail = self._prepare_order_details(
                r['orders'], r['quote_currency'], r['is_open'])
            lines.extend(lines_detail if lines_detail else "")
            await self.__send_order_msg(lines, r)

    async def __send_order_msg(self, lines: List[str], r: Dict[str, Any]) -> None:
        """
        Send status message.
        """
        msg = ''

        for line in lines:
            if line:
                if (len(msg) + len(line) + 1) < MAX_MESSAGE_LENGTH:
                    msg += line + '\n'
                else:
                    await self._send_msg(msg.format(**r))
                    msg = "*Order List for Trade #*`{trade_id}` - continued\n" + line + '\n'

        await self._send_msg(msg.format(**r))

    @authorized_only
    async def _status(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /status.
        Returns the current TradeThread status
        :param bot: telegram bot
        :param update: message update
        :return: None
        """

        if context.args and 'table' in context.args:
            await self._status_table(update, context)
            return
        else:
            await self._status_msg(update, context)

    async def _status_msg(self, update: Update, context: CallbackContext) -> None:
        """
        handler for `/status` and `/status <id>`.

        """
        # Check if there's at least one numerical ID provided.
        # If so, try to get only these trades.
        trade_ids = []
        if context.args and len(context.args) > 0:
            trade_ids = [int(i) for i in context.args if i.isnumeric()]

        results = self._rpc._rpc_trade_status(trade_ids=trade_ids)
        position_adjust = self._config.get('position_adjustment_enable', False)
        max_entries = self._config.get('max_entry_position_adjustment', -1)
        for r in results:
            r['open_date_hum'] = dt_humanize(r['open_date'])
            r['num_entries'] = len([o for o in r['orders'] if o['ft_is_entry']])
            r['num_exits'] = len([o for o in r['orders'] if not o['ft_is_entry']
                                 and not o['ft_order_side'] == 'stoploss'])
            r['exit_reason'] = r.get('exit_reason', "")
            r['stake_amount_r'] = fmt_coin(r['stake_amount'], r['quote_currency'])
            r['max_stake_amount_r'] = fmt_coin(
                r['max_stake_amount'] or r['stake_amount'], r['quote_currency'])
            r['profit_abs_r'] = fmt_coin(r['profit_abs'], r['quote_currency'])
            r['realized_profit_r'] = fmt_coin(r['realized_profit'], r['quote_currency'])
            r['total_profit_abs_r'] = fmt_coin(
                r['total_profit_abs'], r['quote_currency'])
            lines = [
                "*Trade ID:* `{trade_id}`" +
                (" `(since {open_date_hum})`" if r['is_open'] else ""),
                "*Current Pair:* {pair}",
                f"*Direction:* {'`Short`' if r.get('is_short') else '`Long`'}"
                + " ` ({leverage}x)`" if r.get('leverage') else "",
                "*Amount:* `{amount} ({stake_amount_r})`",
                "*Total invested:* `{max_stake_amount_r}`" if position_adjust else "",
                "*Enter Tag:* `{enter_tag}`" if r['enter_tag'] else "",
                "*Exit Reason:* `{exit_reason}`" if r['exit_reason'] else "",
            ]

            if position_adjust:
                max_buy_str = (f"/{max_entries + 1}" if (max_entries > 0) else "")
                lines.extend([
                    "*Number of Entries:* `{num_entries}" + max_buy_str + "`",
                    "*Number of Exits:* `{num_exits}`"
                ])

            lines.extend([
                f"*Open Rate:* `{round_value(r['open_rate'], 8)}`",
                f"*Close Rate:* `{round_value(r['close_rate'], 8)}`" if r['close_rate'] else "",
                "*Open Date:* `{open_date}`",
                "*Close Date:* `{close_date}`" if r['close_date'] else "",
                f" \n*Current Rate:* `{round_value(r['current_rate'], 8)}`" if r['is_open'] else "",
                ("*Unrealized Profit:* " if r['is_open'] else "*Close Profit: *")
                + "`{profit_ratio:.2%}` `({profit_abs_r})`",
            ])

            if r['is_open']:
                if r.get('realized_profit'):
                    lines.extend([
                        "*Realized Profit:* `{realized_profit_ratio:.2%} ({realized_profit_r})`",
                        "*Total Profit:* `{total_profit_ratio:.2%} ({total_profit_abs_r})`"
                    ])

                # Append empty line to improve readability
                lines.append(" ")
                if (r['stop_loss_abs'] != r['initial_stop_loss_abs']
                        and r['initial_stop_loss_ratio'] is not None):
                    # Adding initial stoploss only if it is different from stoploss
                    lines.append("*Initial Stoploss:* `{initial_stop_loss_abs:.8f}` "
                                 "`({initial_stop_loss_ratio:.2%})`")

                # Adding stoploss and stoploss percentage only if it is not None
                lines.append(f"*Stoploss:* `{round_value(r['stop_loss_abs'], 8)}` " +
                             ("`({stop_loss_ratio:.2%})`" if r['stop_loss_ratio'] else ""))
                lines.append(f"*Stoploss distance:* `{round_value(r['stoploss_current_dist'], 8)}` "
                             "`({stoploss_current_dist_ratio:.2%})`")
                if r.get('open_orders'):
                    lines.append(
                        "*Open Order:* `{open_orders}`"
                        + ("- `{exit_order_status}`" if r['exit_order_status'] else ""))

            await self.__send_status_msg(lines, r)

    async def __send_status_msg(self, lines: List[str], r: Dict[str, Any]) -> None:
        """
        Send status message.
        """
        msg = ''

        for line in lines:
            if line:
                if (len(msg) + len(line) + 1) < MAX_MESSAGE_LENGTH:
                    msg += line + '\n'
                else:
                    await self._send_msg(msg.format(**r))
                    msg = "*Trade ID:* `{trade_id}` - continued\n" + line + '\n'

        await self._send_msg(msg.format(**r))

    @authorized_only
    async def _status_table(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /status table.
        Returns the current TradeThread status in table format
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        fiat_currency = self._config.get('fiat_display_currency', '')
        statlist, head, fiat_profit_sum = self._rpc._rpc_status_table(
            self._config['stake_currency'], fiat_currency)

        show_total = not isnan(fiat_profit_sum) and len(statlist) > 1
        max_trades_per_msg = 50
        """
        Calculate the number of messages of 50 trades per message
        0.99 is used to make sure that there are no extra (empty) messages
        As an example with 50 trades, there will be int(50/50 + 0.99) = 1 message
        """
        messages_count = max(int(len(statlist) / max_trades_per_msg + 0.99), 1)
        for i in range(0, messages_count):
            trades = statlist[i * max_trades_per_msg:(i + 1) * max_trades_per_msg]
            if show_total and i == messages_count - 1:
                # append total line
                trades.append(["Total", "", "", f"{fiat_profit_sum:.2f} {fiat_currency}"])

            message = tabulate(trades,
                               headers=head,
                               tablefmt='simple')
            if show_total and i == messages_count - 1:
                # insert separators line between Total
                lines = message.split("\n")
                message = "\n".join(lines[:-1] + [lines[1]] + [lines[-1]])
            await self._send_msg(f"<pre>{message}</pre>", parse_mode=ParseMode.HTML,
                                 reload_able=True, callback_path="update_status_table",
                                 query=update.callback_query)

    async def _timeunit_stats(self, update: Update, context: CallbackContext, unit: str) -> None:
        """
        Handler for /daily <n>
        Returns a daily profit (in BTC) over the last n days.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """

        vals = {
            'days': TimeunitMappings('Day', 'Daily', 'days', 'update_daily', 7, '%Y-%m-%d'),
            'weeks': TimeunitMappings('Monday', 'Weekly', 'weeks (starting from Monday)',
                                      'update_weekly', 8, '%Y-%m-%d'),
            'months': TimeunitMappings('Month', 'Monthly', 'months', 'update_monthly', 6, '%Y-%m'),
        }
        val = vals[unit]

        stake_cur = self._config['stake_currency']
        fiat_disp_cur = self._config.get('fiat_display_currency', '')
        try:
            timescale = int(context.args[0]) if context.args else val.default
        except (TypeError, ValueError, IndexError):
            timescale = val.default
        stats = self._rpc._rpc_timeunit_profit(
            timescale,
            stake_cur,
            fiat_disp_cur,
            unit
        )
        stats_tab = tabulate(
            [[f"{period['date']:{val.dateformat}} ({period['trade_count']})",
              f"{fmt_coin(period['abs_profit'], stats['stake_currency'])}",
              f"{period['fiat_value']:.2f} {stats['fiat_display_currency']}",
              f"{period['rel_profit']:.2%}",
              ] for period in stats['data']],
            headers=[
                f"{val.header} (count)",
                f'{stake_cur}',
                f'{fiat_disp_cur}',
                'Profit %',
                'Trades',
            ],
            tablefmt='simple')
        message = (
            f'<b>{val.message} Profit over the last {timescale} {val.message2}</b>:\n'
            f'<pre>{stats_tab}</pre>'
        )
        await self._send_msg(message, parse_mode=ParseMode.HTML, reload_able=True,
                             callback_path=val.callback, query=update.callback_query)

    @authorized_only
    async def _daily(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /daily <n>
        Returns a daily profit (in BTC) over the last n days.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        await self._timeunit_stats(update, context, 'days')

    @authorized_only
    async def _weekly(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /weekly <n>
        Returns a weekly profit (in BTC) over the last n weeks.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        await self._timeunit_stats(update, context, 'weeks')

    @authorized_only
    async def _monthly(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /monthly <n>
        Returns a monthly profit (in BTC) over the last n months.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        await self._timeunit_stats(update, context, 'months')

    @authorized_only
    async def _profit(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /profit.
        Returns a cumulative profit statistics.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        stake_cur = self._config['stake_currency']
        fiat_disp_cur = self._config.get('fiat_display_currency', '')

        start_date = datetime.fromtimestamp(0)
        timescale = None
        try:
            if context.args:
                timescale = int(context.args[0]) - 1
                today_start = datetime.combine(date.today(), datetime.min.time())
                start_date = today_start - timedelta(days=timescale)
        except (TypeError, ValueError, IndexError):
            pass

        stats = self._rpc._rpc_trade_statistics(
            stake_cur,
            fiat_disp_cur,
            start_date)
        profit_closed_coin = stats['profit_closed_coin']
        profit_closed_ratio_mean = stats['profit_closed_ratio_mean']
        profit_closed_percent = stats['profit_closed_percent']
        profit_closed_fiat = stats['profit_closed_fiat']
        profit_all_coin = stats['profit_all_coin']
        profit_all_ratio_mean = stats['profit_all_ratio_mean']
        profit_all_percent = stats['profit_all_percent']
        profit_all_fiat = stats['profit_all_fiat']
        trade_count = stats['trade_count']
        first_trade_date = f"{stats['first_trade_humanized']} ({stats['first_trade_date']})"
        latest_trade_date = f"{stats['latest_trade_humanized']} ({stats['latest_trade_date']})"
        avg_duration = stats['avg_duration']
        best_pair = stats['best_pair']
        best_pair_profit_ratio = stats['best_pair_profit_ratio']
        winrate = stats['winrate']
        expectancy = stats['expectancy']
        expectancy_ratio = stats['expectancy_ratio']

        if stats['trade_count'] == 0:
            markdown_msg = f"No trades yet.\n*Bot started:* `{stats['bot_start_date']}`"
        else:
            # Message to display
            if stats['closed_trade_count'] > 0:
                markdown_msg = ("*ROI:* Closed trades\n"
                                f"∙ `{fmt_coin(profit_closed_coin, stake_cur)} "
                                f"({profit_closed_ratio_mean:.2%}) "
                                f"({profit_closed_percent} \N{GREEK CAPITAL LETTER SIGMA}%)`\n"
                                f"∙ `{fmt_coin(profit_closed_fiat, fiat_disp_cur)}`\n")
            else:
                markdown_msg = "`No closed trade` \n"

            markdown_msg += (
                f"*ROI:* All trades\n"
                f"∙ `{fmt_coin(profit_all_coin, stake_cur)} "
                f"({profit_all_ratio_mean:.2%}) "
                f"({profit_all_percent} \N{GREEK CAPITAL LETTER SIGMA}%)`\n"
                f"∙ `{fmt_coin(profit_all_fiat, fiat_disp_cur)}`\n"
                f"*Total Trade Count:* `{trade_count}`\n"
                f"*Bot started:* `{stats['bot_start_date']}`\n"
                f"*{'First Trade opened' if not timescale else 'Showing Profit since'}:* "
                f"`{first_trade_date}`\n"
                f"*Latest Trade opened:* `{latest_trade_date}`\n"
                f"*Win / Loss:* `{stats['winning_trades']} / {stats['losing_trades']}`\n"
                f"*Winrate:* `{winrate:.2%}`\n"
                f"*Expectancy (Ratio):* `{expectancy:.2f} ({expectancy_ratio:.2f})`"
            )
            if stats['closed_trade_count'] > 0:
                markdown_msg += (
                    f"\n*Avg. Duration:* `{avg_duration}`\n"
                    f"*Best Performing:* `{best_pair}: {best_pair_profit_ratio:.2%}`\n"
                    f"*Trading volume:* `{fmt_coin(stats['trading_volume'], stake_cur)}`\n"
                    f"*Profit factor:* `{stats['profit_factor']:.2f}`\n"
                    f"*Max Drawdown:* `{stats['max_drawdown']:.2%} "
                    f"({fmt_coin(stats['max_drawdown_abs'], stake_cur)})`\n"
                    f"    from `{stats['max_drawdown_start']} "
                    f"({fmt_coin(stats['drawdown_high'], stake_cur)})`\n"
                    f"    to `{stats['max_drawdown_end']} "
                    f"({fmt_coin(stats['drawdown_low'], stake_cur)})`\n"
                )
        await self._send_msg(markdown_msg, reload_able=True, callback_path="update_profit",
                             query=update.callback_query)

    @authorized_only
    async def _stats(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /stats
        Show stats of recent trades
        """
        stats = self._rpc._rpc_stats()

        reason_map = {
            'roi': 'ROI',
            'stop_loss': 'Stoploss',
            'trailing_stop_loss': 'Trail. Stop',
            'stoploss_on_exchange': 'Stoploss',
            'exit_signal': 'Exit Signal',
            'force_exit': 'Force Exit',
            'emergency_exit': 'Emergency Exit',
        }
        exit_reasons_tabulate = [
            [
                reason_map.get(reason, reason),
                sum(count.values()),
                count['wins'],
                count['losses']
            ] for reason, count in stats['exit_reasons'].items()
        ]
        exit_reasons_msg = 'No trades yet.'
        for reason in chunks(exit_reasons_tabulate, 25):
            exit_reasons_msg = tabulate(
                reason,
                headers=['Exit Reason', 'Exits', 'Wins', 'Losses']
            )
            if len(exit_reasons_tabulate) > 25:
                await self._send_msg(f"```\n{exit_reasons_msg}```", ParseMode.MARKDOWN)
                exit_reasons_msg = ''

        durations = stats['durations']
        duration_msg = tabulate(
            [
                ['Wins', str(timedelta(seconds=durations['wins']))
                 if durations['wins'] is not None else 'N/A'],
                ['Losses', str(timedelta(seconds=durations['losses']))
                 if durations['losses'] is not None else 'N/A']
            ],
            headers=['', 'Avg. Duration']
        )
        msg = (f"""```\n{exit_reasons_msg}```\n```\n{duration_msg}```""")

        await self._send_msg(msg, ParseMode.MARKDOWN)

    @authorized_only
    async def _balance(self, update: Update, context: CallbackContext) -> None:
        """ Handler for /balance """
        full_result = context.args and 'full' in context.args
        result = self._rpc._rpc_balance(self._config['stake_currency'],
                                        self._config.get('fiat_display_currency', ''))

        balance_dust_level = self._config['telegram'].get('balance_dust_level', 0.0)
        if not balance_dust_level:
            balance_dust_level = DUST_PER_COIN.get(self._config['stake_currency'], 1.0)

        output = ''
        if self._config['dry_run']:
            output += "*Warning:* Simulated balances in Dry Mode.\n"
        starting_cap = fmt_coin(result['starting_capital'], self._config['stake_currency'])
        output += f"Starting capital: `{starting_cap}`"
        starting_cap_fiat = fmt_coin(
            result['starting_capital_fiat'], self._config['fiat_display_currency']
        ) if result['starting_capital_fiat'] > 0 else ''
        output += (f" `, {starting_cap_fiat}`.\n"
                   ) if result['starting_capital_fiat'] > 0 else '.\n'

        total_dust_balance = 0
        total_dust_currencies = 0
        for curr in result['currencies']:
            curr_output = ''
            if (
                (curr['is_position'] or curr['est_stake'] > balance_dust_level)
                and (full_result or curr['is_bot_managed'])
            ):
                if curr['is_position']:
                    curr_output = (
                        f"*{curr['currency']}:*\n"
                        f"\t`{curr['side']}: {curr['position']:.8f}`\n"
                        f"\t`Leverage: {curr['leverage']:.1f}`\n"
                        f"\t`Est. {curr['stake']}: "
                        f"{fmt_coin(curr['est_stake'], curr['stake'], False)}`\n")
                else:
                    est_stake = fmt_coin(
                        curr['est_stake' if full_result else 'est_stake_bot'], curr['stake'], False)

                    curr_output = (
                        f"*{curr['currency']}:*\n"
                        f"\t`Available: {curr['free']:.8f}`\n"
                        f"\t`Balance: {curr['balance']:.8f}`\n"
                        f"\t`Pending: {curr['used']:.8f}`\n"
                        f"\t`Bot Owned: {curr['bot_owned']:.8f}`\n"
                        f"\t`Est. {curr['stake']}: {est_stake}`\n")

            elif curr['est_stake'] <= balance_dust_level:
                total_dust_balance += curr['est_stake']
                total_dust_currencies += 1

            # Handle overflowing message length
            if len(output + curr_output) >= MAX_MESSAGE_LENGTH:
                await self._send_msg(output)
                output = curr_output
            else:
                output += curr_output

        if total_dust_balance > 0:
            output += (
                f"*{total_dust_currencies} Other "
                f"{plural(total_dust_currencies, 'Currency', 'Currencies')} "
                f"(< {balance_dust_level} {result['stake']}):*\n"
                f"\t`Est. {result['stake']}: "
                f"{fmt_coin(total_dust_balance, result['stake'], False)}`\n")
        tc = result['trade_count'] > 0
        stake_improve = f" `({result['starting_capital_ratio']:.2%})`" if tc else ''
        fiat_val = f" `({result['starting_capital_fiat_ratio']:.2%})`" if tc else ''
        value = fmt_coin(
            result['value' if full_result else 'value_bot'], result['symbol'], False)
        total_stake = fmt_coin(
            result['total' if full_result else 'total_bot'], result['stake'], False)
        output += (
            f"\n*Estimated Value{' (Bot managed assets only)' if not full_result else ''}*:\n"
            f"\t`{result['stake']}: {total_stake}`{stake_improve}\n"
            f"\t`{result['symbol']}: {value}`{fiat_val}\n"
        )
        await self._send_msg(output, reload_able=True, callback_path="update_balance",
                             query=update.callback_query)

    @authorized_only
    async def _start(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /start.
        Starts TradeThread
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc._rpc_start()
        await self._send_msg(f"Status: `{msg['status']}`")

    @authorized_only
    async def _stop(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /stop.
        Stops TradeThread
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc._rpc_stop()
        await self._send_msg(f"Status: `{msg['status']}`")

    @authorized_only
    async def _reload_config(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /reload_config.
        Triggers a config file reload
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc._rpc_reload_config()
        await self._send_msg(f"Status: `{msg['status']}`")

    @authorized_only
    async def _stopentry(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /stop_buy.
        Sets max_open_trades to 0 and gracefully sells all open trades
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc._rpc_stopentry()
        await self._send_msg(f"Status: `{msg['status']}`")

    @authorized_only
    async def _reload_trade_from_exchange(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /reload_trade <tradeid>.
        """
        if not context.args or len(context.args) == 0:
            raise RPCException("Trade-id not set.")
        trade_id = int(context.args[0])
        msg = self._rpc._rpc_reload_trade_from_exchange(trade_id)
        await self._send_msg(f"Status: `{msg['status']}`")

    @authorized_only
    async def _force_exit(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /forceexit <id>.
        Sells the given trade at current price
        :param bot: telegram bot
        :param update: message update
        :return: None
        """

        if context.args:
            trade_id = context.args[0]
            await self._force_exit_action(trade_id)
        else:
            fiat_currency = self._config.get('fiat_display_currency', '')
            try:
                statlist, _, _ = self._rpc._rpc_status_table(
                    self._config['stake_currency'], fiat_currency)
            except RPCException:
                await self._send_msg(msg='No open trade found.')
                return
            trades = []
            for trade in statlist:
                trades.append((trade[0], f"{trade[0]} {trade[1]} {trade[2]} {trade[3]}"))

            trade_buttons = [
                InlineKeyboardButton(text=trade[1], callback_data=f"force_exit__{trade[0]}")
                for trade in trades]
            buttons_aligned = self._layout_inline_keyboard_onecol(trade_buttons)

            buttons_aligned.append([InlineKeyboardButton(
                text='Cancel', callback_data='force_exit__cancel')])
            await self._send_msg(msg="Which trade?", keyboard=buttons_aligned)

    async def _force_exit_action(self, trade_id):
        if trade_id != 'cancel':
            try:
                loop = asyncio.get_running_loop()
                # Workaround to avoid nested loops
                await loop.run_in_executor(None, safe_async_db(self._rpc._rpc_force_exit), trade_id)
            except RPCException as e:
                await self._send_msg(str(e))

    async def _force_exit_inline(self, update: Update, _: CallbackContext) -> None:
        if update.callback_query:
            query = update.callback_query
            if query.data and '__' in query.data:
                # Input data is "force_exit__<tradid|cancel>"
                trade_id = query.data.split("__")[1].split(' ')[0]
                if trade_id == 'cancel':
                    await query.answer()
                    await query.edit_message_text(text="Force exit canceled.")
                    return
                trade: Optional[Trade] = Trade.get_trades(trade_filter=Trade.id == trade_id).first()
                await query.answer()
                if trade:
                    await query.edit_message_text(
                        text=f"Manually exiting Trade #{trade_id}, {trade.pair}")
                    await self._force_exit_action(trade_id)
                else:
                    await query.edit_message_text(text=f"Trade {trade_id} not found.")

    async def _force_enter_action(self, pair, price: Optional[float], order_side: SignalDirection):
        if pair != 'cancel':
            try:
                @safe_async_db
                def _force_enter():
                    self._rpc._rpc_force_entry(pair, price, order_side=order_side)
                loop = asyncio.get_running_loop()
                # Workaround to avoid nested loops
                await loop.run_in_executor(None, _force_enter)
            except RPCException as e:
                logger.exception("Forcebuy error!")
                await self._send_msg(str(e), ParseMode.HTML)

    async def _force_enter_inline(self, update: Update, _: CallbackContext) -> None:
        if update.callback_query:
            query = update.callback_query
            if query.data and '__' in query.data:
                # Input data is "force_enter__<pair|cancel>_<side>"
                payload = query.data.split("__")[1]
                if payload == 'cancel':
                    await query.answer()
                    await query.edit_message_text(text="Force enter canceled.")
                    return
                if payload and '_||_' in payload:
                    pair, side = payload.split('_||_')
                    order_side = SignalDirection(side)
                    await query.answer()
                    await query.edit_message_text(text=f"Manually entering {order_side} for {pair}")
                    await self._force_enter_action(pair, None, order_side)

    @staticmethod
    def _layout_inline_keyboard(
            buttons: List[InlineKeyboardButton], cols=3) -> List[List[InlineKeyboardButton]]:
        return [buttons[i:i + cols] for i in range(0, len(buttons), cols)]

    @staticmethod
    def _layout_inline_keyboard_onecol(
            buttons: List[InlineKeyboardButton], cols=1) -> List[List[InlineKeyboardButton]]:
        return [buttons[i:i + cols] for i in range(0, len(buttons), cols)]

    @authorized_only
    async def _force_enter(
            self, update: Update, context: CallbackContext, order_side: SignalDirection) -> None:
        """
        Handler for /forcelong <asset> <price> and `/forceshort <asset> <price>
        Buys a pair trade at the given or current price
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        if context.args:
            pair = context.args[0]
            price = float(context.args[1]) if len(context.args) > 1 else None
            await self._force_enter_action(pair, price, order_side)
        else:
            whitelist = self._rpc._rpc_whitelist()['whitelist']
            pair_buttons = [
                InlineKeyboardButton(
                    text=pair, callback_data=f"force_enter__{pair}_||_{order_side}"
                ) for pair in sorted(whitelist)
            ]
            buttons_aligned = self._layout_inline_keyboard(pair_buttons)

            buttons_aligned.append([InlineKeyboardButton(text='Cancel',
                                                         callback_data='force_enter__cancel')])
            await self._send_msg(msg="Which pair?",
                                 keyboard=buttons_aligned,
                                 query=update.callback_query)

    @authorized_only
    async def _trades(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /trades <n>
        Returns last n recent trades.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        stake_cur = self._config['stake_currency']
        try:
            nrecent = int(context.args[0]) if context.args else 10
        except (TypeError, ValueError, IndexError):
            nrecent = 10
        trades = self._rpc._rpc_trade_history(
            nrecent
        )
        trades_tab = tabulate(
            [[dt_humanize(trade['close_date']),
                trade['pair'] + " (#" + str(trade['trade_id']) + ")",
                f"{(trade['close_profit']):.2%} ({trade['close_profit_abs']})"]
                for trade in trades['trades']],
            headers=[
                'Close Date',
                'Pair (ID)',
                f'Profit ({stake_cur})',
            ],
            tablefmt='simple')
        message = (f"<b>{min(trades['trades_count'], nrecent)} recent trades</b>:\n"
                   + (f"<pre>{trades_tab}</pre>" if trades['trades_count'] > 0 else ''))
        await self._send_msg(message, parse_mode=ParseMode.HTML)

    @authorized_only
    async def _delete_trade(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /delete <id>.
        Delete the given trade
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        if not context.args or len(context.args) == 0:
            raise RPCException("Trade-id not set.")
        trade_id = int(context.args[0])
        msg = self._rpc._rpc_delete(trade_id)
        await self._send_msg(
            f"`{msg['result_msg']}`\n"
            'Please make sure to take care of this asset on the exchange manually.'
        )

    @authorized_only
    async def _cancel_open_order(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /cancel_open_order <id>.
        Cancel open order for tradeid
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        if not context.args or len(context.args) == 0:
            raise RPCException("Trade-id not set.")
        trade_id = int(context.args[0])
        self._rpc._rpc_cancel_open_order(trade_id)
        await self._send_msg('Open order canceled.')

    @authorized_only
    async def _performance(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /performance.
        Shows a performance statistic from finished trades
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        trades = self._rpc._rpc_performance()
        output = "<b>Performance:</b>\n"
        for i, trade in enumerate(trades):
            stat_line = (
                f"{i + 1}.\t <code>{trade['pair']}\t"
                f"{fmt_coin(trade['profit_abs'], self._config['stake_currency'])} "
                f"({trade['profit_ratio']:.2%}) "
                f"({trade['count']})</code>\n")

            if len(output + stat_line) >= MAX_MESSAGE_LENGTH:
                await self._send_msg(output, parse_mode=ParseMode.HTML)
                output = stat_line
            else:
                output += stat_line

        await self._send_msg(output, parse_mode=ParseMode.HTML,
                             reload_able=True, callback_path="update_performance",
                             query=update.callback_query)

    @authorized_only
    async def _enter_tag_performance(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /entries PAIR .
        Shows a performance statistic from finished trades
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        pair = None
        if context.args and isinstance(context.args[0], str):
            pair = context.args[0]

        trades = self._rpc._rpc_enter_tag_performance(pair)
        output = "*Entry Tag Performance:*\n"
        for i, trade in enumerate(trades):
            stat_line = (
                f"{i + 1}.\t `{trade['enter_tag']}\t"
                f"{fmt_coin(trade['profit_abs'], self._config['stake_currency'])} "
                f"({trade['profit_ratio']:.2%}) "
                f"({trade['count']})`\n")

            if len(output + stat_line) >= MAX_MESSAGE_LENGTH:
                await self._send_msg(output, parse_mode=ParseMode.MARKDOWN)
                output = stat_line
            else:
                output += stat_line

        await self._send_msg(output, parse_mode=ParseMode.MARKDOWN,
                             reload_able=True, callback_path="update_enter_tag_performance",
                             query=update.callback_query)

    @authorized_only
    async def _exit_reason_performance(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /exits.
        Shows a performance statistic from finished trades
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        pair = None
        if context.args and isinstance(context.args[0], str):
            pair = context.args[0]

        trades = self._rpc._rpc_exit_reason_performance(pair)
        output = "*Exit Reason Performance:*\n"
        for i, trade in enumerate(trades):
            stat_line = (
                f"{i + 1}.\t `{trade['exit_reason']}\t"
                f"{fmt_coin(trade['profit_abs'], self._config['stake_currency'])} "
                f"({trade['profit_ratio']:.2%}) "
                f"({trade['count']})`\n")

            if len(output + stat_line) >= MAX_MESSAGE_LENGTH:
                await self._send_msg(output, parse_mode=ParseMode.MARKDOWN)
                output = stat_line
            else:
                output += stat_line

        await self._send_msg(output, parse_mode=ParseMode.MARKDOWN,
                             reload_able=True, callback_path="update_exit_reason_performance",
                             query=update.callback_query)

    @authorized_only
    async def _mix_tag_performance(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /mix_tags.
        Shows a performance statistic from finished trades
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        pair = None
        if context.args and isinstance(context.args[0], str):
            pair = context.args[0]

        trades = self._rpc._rpc_mix_tag_performance(pair)
        output = "*Mix Tag Performance:*\n"
        for i, trade in enumerate(trades):
            stat_line = (
                f"{i + 1}.\t `{trade['mix_tag']}\t"
                f"{fmt_coin(trade['profit_abs'], self._config['stake_currency'])} "
                f"({trade['profit_ratio']:.2%}) "
                f"({trade['count']})`\n")

            if len(output + stat_line) >= MAX_MESSAGE_LENGTH:
                await self._send_msg(output, parse_mode=ParseMode.MARKDOWN)
                output = stat_line
            else:
                output += stat_line

        await self._send_msg(output, parse_mode=ParseMode.MARKDOWN,
                             reload_able=True, callback_path="update_mix_tag_performance",
                             query=update.callback_query)

    @authorized_only
    async def _count(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /count.
        Returns the number of trades running
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        counts = self._rpc._rpc_count()
        message = tabulate({k: [v] for k, v in counts.items()},
                           headers=['current', 'max', 'total stake'],
                           tablefmt='simple')
        message = f"<pre>{message}</pre>"
        logger.debug(message)
        await self._send_msg(message, parse_mode=ParseMode.HTML,
                             reload_able=True, callback_path="update_count",
                             query=update.callback_query)

    @authorized_only
    async def _locks(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /locks.
        Returns the currently active locks
        """
        rpc_locks = self._rpc._rpc_locks()
        if not rpc_locks['locks']:
            await self._send_msg('No active locks.', parse_mode=ParseMode.HTML)

        for locks in chunks(rpc_locks['locks'], 25):
            message = tabulate([[
                lock['id'],
                lock['pair'],
                lock['lock_end_time'],
                lock['reason']] for lock in locks],
                headers=['ID', 'Pair', 'Until', 'Reason'],
                tablefmt='simple')
            message = f"<pre>{escape(message)}</pre>"
            logger.debug(message)
            await self._send_msg(message, parse_mode=ParseMode.HTML)

    @authorized_only
    async def _delete_locks(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /delete_locks.
        Returns the currently active locks
        """
        arg = context.args[0] if context.args and len(context.args) > 0 else None
        lockid = None
        pair = None
        if arg:
            try:
                lockid = int(arg)
            except ValueError:
                pair = arg

        self._rpc._rpc_delete_lock(lockid=lockid, pair=pair)
        await self._locks(update, context)

    @authorized_only
    async def _whitelist(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /whitelist
        Shows the currently active whitelist
        """
        whitelist = self._rpc._rpc_whitelist()

        if context.args:
            if "sorted" in context.args:
                whitelist['whitelist'] = sorted(whitelist['whitelist'])
            if "baseonly" in context.args:
                whitelist['whitelist'] = [pair.split("/")[0] for pair in whitelist['whitelist']]

        message = f"Using whitelist `{whitelist['method']}` with {whitelist['length']} pairs\n"
        message += f"`{', '.join(whitelist['whitelist'])}`"

        logger.debug(message)
        await self._send_msg(message)

    @authorized_only
    async def _blacklist(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /blacklist
        Shows the currently active blacklist
        """
        await self.send_blacklist_msg(self._rpc._rpc_blacklist(context.args))

    async def send_blacklist_msg(self, blacklist: Dict):
        errmsgs = []
        for pair, error in blacklist['errors'].items():
            errmsgs.append(f"Error: {error['error_msg']}")
        if errmsgs:
            await self._send_msg('\n'.join(errmsgs))

        message = f"Blacklist contains {blacklist['length']} pairs\n"
        message += f"`{', '.join(blacklist['blacklist'])}`"

        logger.debug(message)
        await self._send_msg(message)

    @authorized_only
    async def _blacklist_delete(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /bl_delete
        Deletes pair(s) from current blacklist
        """
        await self.send_blacklist_msg(self._rpc._rpc_blacklist_delete(context.args or []))

    @authorized_only
    async def _logs(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /logs
        Shows the latest logs
        """
        try:
            limit = int(context.args[0]) if context.args else 10
        except (TypeError, ValueError, IndexError):
            limit = 10
        logs = RPC._rpc_get_logs(limit)['logs']
        msgs = ''
        msg_template = "*{}* {}: {} \\- `{}`"
        for logrec in logs:
            msg = msg_template.format(escape_markdown(logrec[0], version=2),
                                      escape_markdown(logrec[2], version=2),
                                      escape_markdown(logrec[3], version=2),
                                      escape_markdown(logrec[4], version=2))
            if len(msgs + msg) + 10 >= MAX_MESSAGE_LENGTH:
                # Send message immediately if it would become too long
                await self._send_msg(msgs, parse_mode=ParseMode.MARKDOWN_V2)
                msgs = msg + '\n'
            else:
                # Append message to messages to send
                msgs += msg + '\n'

        if msgs:
            await self._send_msg(msgs, parse_mode=ParseMode.MARKDOWN_V2)

    @authorized_only
    async def _edge(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /edge
        Shows information related to Edge
        """
        edge_pairs = self._rpc._rpc_edge()
        if not edge_pairs:
            message = '<b>Edge only validated following pairs:</b>'
            await self._send_msg(message, parse_mode=ParseMode.HTML)

        for chunk in chunks(edge_pairs, 25):
            edge_pairs_tab = tabulate(chunk, headers='keys', tablefmt='simple')
            message = (f'<b>Edge only validated following pairs:</b>\n'
                       f'<pre>{edge_pairs_tab}</pre>')

            await self._send_msg(message, parse_mode=ParseMode.HTML)

    @authorized_only
    async def _help(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /help.
        Show commands of the bot
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        force_enter_text = ("*/forcelong <pair> [<rate>]:* `Instantly buys the given pair. "
                            "Optionally takes a rate at which to buy "
                            "(only applies to limit orders).` \n"
                            )
        if self._rpc._freqtrade.trading_mode != TradingMode.SPOT:
            force_enter_text += ("*/forceshort <pair> [<rate>]:* `Instantly shorts the given pair. "
                                 "Optionally takes a rate at which to sell "
                                 "(only applies to limit orders).` \n")
        message = (
            "_Bot Control_\n"
            "------------\n"
            "*/start:* `Starts the trader`\n"
            "*/stop:* Stops the trader\n"
            "*/stopentry:* `Stops entering, but handles open trades gracefully` \n"
            "*/forceexit <trade_id>|all:* `Instantly exits the given trade or all trades, "
            "regardless of profit`\n"
            "*/fx <trade_id>|all:* `Alias to /forceexit`\n"
            f"{force_enter_text if self._config.get('force_entry_enable', False) else ''}"
            "*/delete <trade_id>:* `Instantly delete the given trade in the database`\n"
            "*/reload_trade <trade_id>:* `Relade trade from exchange Orders`\n"
            "*/cancel_open_order <trade_id>:* `Cancels open orders for trade. "
            "Only valid when the trade has open orders.`\n"
            "*/coo <trade_id>|all:* `Alias to /cancel_open_order`\n"

            "*/whitelist [sorted] [baseonly]:* `Show current whitelist. Optionally in "
            "order and/or only displaying the base currency of each pairing.`\n"
            "*/blacklist [pair]:* `Show current blacklist, or adds one or more pairs "
            "to the blacklist.` \n"
            "*/blacklist_delete [pairs]| /bl_delete [pairs]:* "
            "`Delete pair / pattern from blacklist. Will reset on reload_conf.` \n"
            "*/reload_config:* `Reload configuration file` \n"
            "*/unlock <pair|id>:* `Unlock this Pair (or this lock id if it's numeric)`\n"

            "_Current state_\n"
            "------------\n"
            "*/show_config:* `Show running configuration` \n"
            "*/locks:* `Show currently locked pairs`\n"
            "*/balance:* `Show bot managed balance per currency`\n"
            "*/balance total:* `Show account balance per currency`\n"
            "*/logs [limit]:* `Show latest logs - defaults to 10` \n"
            "*/count:* `Show number of active trades compared to allowed number of trades`\n"
            "*/edge:* `Shows validated pairs by Edge if it is enabled` \n"
            "*/health* `Show latest process timestamp - defaults to 1970-01-01 00:00:00` \n"
            "*/marketdir [long | short | even | none]:* `Updates the user managed variable "
            "that represents the current market direction. If no direction is provided `"
            "`the currently set market direction will be output.` \n"
            "*/list_custom_data <trade_id> <key>:* `List custom_data for Trade ID & Key combo.`\n"
            "`If no Key is supplied it will list all key-value pairs found for that Trade ID.`"

            "_Statistics_\n"
            "------------\n"
            "*/status <trade_id>|[table]:* `Lists all open trades`\n"
            "         *<trade_id> :* `Lists one or more specific trades.`\n"
            "                        `Separate multiple <trade_id> with a blank space.`\n"
            "         *table :* `will display trades in a table`\n"
            "                `pending buy orders are marked with an asterisk (*)`\n"
            "                `pending sell orders are marked with a double asterisk (**)`\n"
            "*/entries <pair|none>:* `Shows the enter_tag performance`\n"
            "*/exits <pair|none>:* `Shows the exit reason performance`\n"
            "*/mix_tags <pair|none>:* `Shows combined entry tag + exit reason performance`\n"
            "*/trades [limit]:* `Lists last closed trades (limited to 10 by default)`\n"
            "*/profit [<n>]:* `Lists cumulative profit from all finished trades, "
            "over the last n days`\n"
            "*/performance:* `Show performance of each finished trade grouped by pair`\n"
            "*/daily <n>:* `Shows profit or loss per day, over the last n days`\n"
            "*/weekly <n>:* `Shows statistics per week, over the last n weeks`\n"
            "*/monthly <n>:* `Shows statistics per month, over the last n months`\n"
            "*/stats:* `Shows Wins / losses by Sell reason as well as "
            "Avg. holding durations for buys and sells.`\n"
            "*/help:* `This help message`\n"
            "*/version:* `Show version`\n"
            )

        await self._send_msg(message, parse_mode=ParseMode.MARKDOWN)

    @authorized_only
    async def _health(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /health
        Shows the last process timestamp
        """
        health = self._rpc.health()
        message = f"Last process: `{health['last_process_loc']}`\n"
        message += f"Initial bot start: `{health['bot_start_loc']}`\n"
        message += f"Last bot restart: `{health['bot_startup_loc']}`"
        await self._send_msg(message)

    @authorized_only
    async def _version(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /version.
        Show version information
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        strategy_version = self._rpc._freqtrade.strategy.version()
        version_string = f'*Version:* `{__version__}`'
        if strategy_version is not None:
            version_string += f'\n*Strategy version: * `{strategy_version}`'

        await self._send_msg(version_string)

    @authorized_only
    async def _show_config(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /show_config.
        Show config information information
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        val = RPC._rpc_show_config(self._config, self._rpc._freqtrade.state)

        if val['trailing_stop']:
            sl_info = (
                f"*Initial Stoploss:* `{val['stoploss']}`\n"
                f"*Trailing stop positive:* `{val['trailing_stop_positive']}`\n"
                f"*Trailing stop offset:* `{val['trailing_stop_positive_offset']}`\n"
                f"*Only trail above offset:* `{val['trailing_only_offset_is_reached']}`\n"
            )

        else:
            sl_info = f"*Stoploss:* `{val['stoploss']}`\n"

        if val['position_adjustment_enable']:
            pa_info = (
                f"*Position adjustment:* On\n"
                f"*Max enter position adjustment:* `{val['max_entry_position_adjustment']}`\n"
            )
        else:
            pa_info = "*Position adjustment:* Off\n"

        await self._send_msg(
            f"*Mode:* `{'Dry-run' if val['dry_run'] else 'Live'}`\n"
            f"*Exchange:* `{val['exchange']}`\n"
            f"*Market: * `{val['trading_mode']}`\n"
            f"*Stake per trade:* `{val['stake_amount']} {val['stake_currency']}`\n"
            f"*Max open Trades:* `{val['max_open_trades']}`\n"
            f"*Minimum ROI:* `{val['minimal_roi']}`\n"
            f"*Entry strategy:* ```\n{json.dumps(val['entry_pricing'])}```\n"
            f"*Exit strategy:* ```\n{json.dumps(val['exit_pricing'])}```\n"
            f"{sl_info}"
            f"{pa_info}"
            f"*Timeframe:* `{val['timeframe']}`\n"
            f"*Strategy:* `{val['strategy']}`\n"
            f"*Current state:* `{val['state']}`"
        )

    @authorized_only
    async def _list_custom_data(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /list_custom_data <id> <key>.
        List custom_data for specified trade (and key if supplied).
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        try:
            if not context.args or len(context.args) == 0:
                raise RPCException("Trade-id not set.")
            trade_id = int(context.args[0])
            key = None if len(context.args) < 2 else str(context.args[1])

            results = self._rpc._rpc_list_custom_data(trade_id, key)
            messages = []
            if len(results) > 0:
                messages.append(
                    'Found custom-data entr' + ('ies: ' if len(results) > 1 else 'y: ')
                )
                for result in results:
                    lines = [
                        f"*Key:* `{result['cd_key']}`",
                        f"*ID:* `{result['id']}`",
                        f"*Trade ID:* `{result['ft_trade_id']}`",
                        f"*Type:* `{result['cd_type']}`",
                        f"*Value:* `{result['cd_value']}`",
                        f"*Create Date:* `{format_date(result['created_at'])}`",
                        f"*Update Date:* `{format_date(result['updated_at'])}`"
                    ]
                    # Filter empty lines using list-comprehension
                    messages.append("\n".join([line for line in lines if line]))
                for msg in messages:
                    if len(msg) > MAX_MESSAGE_LENGTH:
                        msg = "Message dropped because length exceeds "
                        msg += f"maximum allowed characters: {MAX_MESSAGE_LENGTH}"
                        logger.warning(msg)
                    await self._send_msg(msg)
            else:
                message = f"Didn't find any custom-data entries for Trade ID: `{trade_id}`"
                message += f" and Key: `{key}`." if key is not None else ""
                await self._send_msg(message)

        except RPCException as e:
            await self._send_msg(str(e))

    async def _update_msg(self, query: CallbackQuery, msg: str, callback_path: str = "",
                          reload_able: bool = False, parse_mode: str = ParseMode.MARKDOWN) -> None:
        if reload_able:
            reply_markup = InlineKeyboardMarkup([
                [InlineKeyboardButton("Refresh", callback_data=callback_path)],
            ])
        else:
            reply_markup = InlineKeyboardMarkup([[]])
        msg += f"\nUpdated: {datetime.now().ctime()}"
        if not query.message:
            return

        try:
            await query.edit_message_text(
                text=msg,
                parse_mode=parse_mode,
                reply_markup=reply_markup
            )
        except BadRequest as e:
            if 'not modified' in e.message.lower():
                pass
            else:
                logger.warning('TelegramError: %s', e.message)
        except TelegramError as telegram_err:
            logger.warning('TelegramError: %s! Giving up on that message.', telegram_err.message)

    async def _send_msg(self, msg: str, parse_mode: str = ParseMode.MARKDOWN,
                        disable_notification: bool = False,
                        keyboard: Optional[List[List[InlineKeyboardButton]]] = None,
                        callback_path: str = "",
                        reload_able: bool = False,
                        query: Optional[CallbackQuery] = None) -> None:
        """
        Send given markdown message
        :param msg: message
        :param bot: alternative bot
        :param parse_mode: telegram parse mode
        :return: None
        """
        reply_markup: Union[InlineKeyboardMarkup, ReplyKeyboardMarkup]
        if query:
            await self._update_msg(query=query, msg=msg, parse_mode=parse_mode,
                                   callback_path=callback_path, reload_able=reload_able)
            return
        if reload_able and self._config['telegram'].get('reload', True):
            reply_markup = InlineKeyboardMarkup([
                [InlineKeyboardButton("Refresh", callback_data=callback_path)]])
        else:
            if keyboard is not None:
                reply_markup = InlineKeyboardMarkup(keyboard)
            else:
                reply_markup = ReplyKeyboardMarkup(self._keyboard, resize_keyboard=True)
        try:
            try:
                await self._app.bot.send_message(
                    self._config['telegram']['chat_id'],
                    text=msg,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                    disable_notification=disable_notification,
                )
            except NetworkError as network_err:
                # Sometimes the telegram server resets the current connection,
                # if this is the case we send the message again.
                logger.warning(
                    'Telegram NetworkError: %s! Trying one more time.',
                    network_err.message
                )
                await self._app.bot.send_message(
                    self._config['telegram']['chat_id'],
                    text=msg,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                    disable_notification=disable_notification,
                )
        except TelegramError as telegram_err:
            logger.warning(
                'TelegramError: %s! Giving up on that message.',
                telegram_err.message
            )

    @authorized_only
    async def _changemarketdir(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /marketdir.
        Updates the bot's market_direction
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        if context.args and len(context.args) == 1:
            new_market_dir_arg = context.args[0]
            old_market_dir = self._rpc._get_market_direction()
            new_market_dir = None
            if new_market_dir_arg == "long":
                new_market_dir = MarketDirection.LONG
            elif new_market_dir_arg == "short":
                new_market_dir = MarketDirection.SHORT
            elif new_market_dir_arg == "even":
                new_market_dir = MarketDirection.EVEN
            elif new_market_dir_arg == "none":
                new_market_dir = MarketDirection.NONE

            if new_market_dir is not None:
                self._rpc._update_market_direction(new_market_dir)
                await self._send_msg("Successfully updated market direction"
                                     f" from *{old_market_dir}* to *{new_market_dir}*.")
            else:
                raise RPCException("Invalid market direction provided. \n"
                                   "Valid market directions: *long, short, even, none*")
        elif context.args is not None and len(context.args) == 0:
            old_market_dir = self._rpc._get_market_direction()
            await self._send_msg(f"Currently set market direction: *{old_market_dir}*")
        else:
            raise RPCException("Invalid usage of command /marketdir. \n"
                               "Usage: */marketdir [short |  long | even | none]*")
