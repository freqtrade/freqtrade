# pragma pylint: disable=unused-argument, unused-variable, protected-access, invalid-name

"""
This module manage Telegram communication
"""
import json
import logging
from datetime import timedelta
from html import escape
from itertools import chain
from typing import Any, Callable, Dict, List, Union

import arrow
from tabulate import tabulate
from telegram import KeyboardButton, ParseMode, ReplyKeyboardMarkup, Update
from telegram.error import NetworkError, TelegramError
from telegram.ext import CallbackContext, CommandHandler, Updater
from telegram.utils.helpers import escape_markdown

from freqtrade.__init__ import __version__
from freqtrade.constants import DUST_PER_COIN
from freqtrade.exceptions import OperationalException
from freqtrade.misc import round_coin_value
from freqtrade.rpc import RPC, RPCException, RPCHandler, RPCMessageType


logger = logging.getLogger(__name__)

logger.debug('Included module rpc.telegram ...')

MAX_TELEGRAM_MESSAGE_LENGTH = 4096


def authorized_only(command_handler: Callable[..., None]) -> Callable[..., Any]:
    """
    Decorator to check if the message comes from the correct chat_id
    :param command_handler: Telegram CommandHandler
    :return: decorated function
    """

    def wrapper(self, *args, **kwargs):
        """ Decorator logic """
        update = kwargs.get('update') or args[0]

        # Reject unauthorized messages
        chat_id = int(self._config['telegram']['chat_id'])

        if int(update.message.chat_id) != chat_id:
            logger.info(
                'Rejected unauthorized message from: %s',
                update.message.chat_id
            )
            return wrapper

        logger.info(
            'Executing handler: %s for chat_id: %s',
            command_handler.__name__,
            chat_id
        )
        try:
            return command_handler(self, *args, **kwargs)
        except BaseException:
            logger.exception('Exception occurred within Telegram module')

    return wrapper


class Telegram(RPCHandler):
    """  This class handles all telegram communication """

    def __init__(self, rpc: RPC, config: Dict[str, Any]) -> None:

        """
        Init the Telegram call, and init the super class RPCHandler
        :param rpc: instance of RPC Helper class
        :param config: Configuration object
        :return: None
        """
        super().__init__(rpc, config)

        self._updater: Updater
        self._init_keyboard()
        self._init()

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
        # like /forcesell and /forcebuy
        # TODO: DRY! - its not good to list all valid cmds here. But otherwise
        #       this needs refacoring of the whole telegram module (same
        #       problem in _help()).
        valid_keys: List[str] = ['/start', '/stop', '/status', '/status table',
                                 '/trades', '/profit', '/performance', '/daily',
                                 '/stats', '/count', '/locks', '/balance',
                                 '/stopbuy', '/reload_config', '/show_config',
                                 '/logs', '/whitelist', '/blacklist', '/edge',
                                 '/help', '/version']

        # custom keyboard specified in config.json
        cust_keyboard = self._config['telegram'].get('keyboard', [])
        if cust_keyboard:
            # check for valid shortcuts
            invalid_keys = [b for b in chain.from_iterable(cust_keyboard)
                            if b not in valid_keys]
            if len(invalid_keys):
                err_msg = ('config.telegram.keyboard: Invalid commands for '
                           f'custom Telegram keyboard: {invalid_keys}'
                           f'\nvalid commands are: {valid_keys}')
                raise OperationalException(err_msg)
            else:
                self._keyboard = cust_keyboard
                logger.info('using custom keyboard from '
                            f'config.json: {self._keyboard}')

    def _init(self) -> None:
        """
        Initializes this module with the given config,
        registers all known command handlers
        and starts polling for message updates
        """
        self._updater = Updater(token=self._config['telegram']['token'], workers=0,
                                use_context=True)

        # Register command handler and start telegram message polling
        handles = [
            CommandHandler('status', self._status),
            CommandHandler('profit', self._profit),
            CommandHandler('balance', self._balance),
            CommandHandler('start', self._start),
            CommandHandler('stop', self._stop),
            CommandHandler('forcesell', self._forcesell),
            CommandHandler('forcebuy', self._forcebuy),
            CommandHandler('trades', self._trades),
            CommandHandler('delete', self._delete_trade),
            CommandHandler('performance', self._performance),
            CommandHandler('stats', self._stats),
            CommandHandler('daily', self._daily),
            CommandHandler('count', self._count),
            CommandHandler('locks', self._locks),
            CommandHandler(['unlock', 'delete_locks'], self._delete_locks),
            CommandHandler(['reload_config', 'reload_conf'], self._reload_config),
            CommandHandler(['show_config', 'show_conf'], self._show_config),
            CommandHandler('stopbuy', self._stopbuy),
            CommandHandler('whitelist', self._whitelist),
            CommandHandler('blacklist', self._blacklist),
            CommandHandler('logs', self._logs),
            CommandHandler('edge', self._edge),
            CommandHandler('help', self._help),
            CommandHandler('version', self._version),
        ]
        for handle in handles:
            self._updater.dispatcher.add_handler(handle)
        self._updater.start_polling(
            bootstrap_retries=-1,
            timeout=30,
            read_latency=60,
            drop_pending_updates=True,
        )
        logger.info(
            'rpc.telegram is listening for following commands: %s',
            [h.command for h in handles]
        )

    def cleanup(self) -> None:
        """
        Stops all running telegram threads.
        :return: None
        """
        self._updater.stop()

    def _format_buy_msg(self, msg: Dict[str, Any]) -> str:
        if self._rpc._fiat_converter:
            msg['stake_amount_fiat'] = self._rpc._fiat_converter.convert_amount(
                msg['stake_amount'], msg['stake_currency'], msg['fiat_currency'])
        else:
            msg['stake_amount_fiat'] = 0

        message = (f"\N{LARGE BLUE CIRCLE} *{msg['exchange']}:* Buying {msg['pair']}"
                   f" (#{msg['trade_id']})\n"
                   f"*Amount:* `{msg['amount']:.8f}`\n"
                   f"*Open Rate:* `{msg['limit']:.8f}`\n"
                   f"*Current Rate:* `{msg['current_rate']:.8f}`\n"
                   f"*Total:* `({round_coin_value(msg['stake_amount'], msg['stake_currency'])}")

        if msg.get('fiat_currency', None):
            message += f", {round_coin_value(msg['stake_amount_fiat'], msg['fiat_currency'])}"
        message += ")`"
        return message

    def _format_sell_msg(self, msg: Dict[str, Any]) -> str:
        msg['amount'] = round(msg['amount'], 8)
        msg['profit_percent'] = round(msg['profit_ratio'] * 100, 2)
        msg['duration'] = msg['close_date'].replace(
            microsecond=0) - msg['open_date'].replace(microsecond=0)
        msg['duration_min'] = msg['duration'].total_seconds() / 60

        msg['emoji'] = self._get_sell_emoji(msg)

        message = ("{emoji} *{exchange}:* Selling {pair} (#{trade_id})\n"
                   "*Amount:* `{amount:.8f}`\n"
                   "*Open Rate:* `{open_rate:.8f}`\n"
                   "*Current Rate:* `{current_rate:.8f}`\n"
                   "*Close Rate:* `{limit:.8f}`\n"
                   "*Sell Reason:* `{sell_reason}`\n"
                   "*Duration:* `{duration} ({duration_min:.1f} min)`\n"
                   "*Profit:* `{profit_percent:.2f}%`").format(**msg)

        # Check if all sell properties are available.
        # This might not be the case if the message origin is triggered by /forcesell
        if (all(prop in msg for prop in ['gain', 'fiat_currency', 'stake_currency'])
                and self._rpc._fiat_converter):
            msg['profit_fiat'] = self._rpc._fiat_converter.convert_amount(
                msg['profit_amount'], msg['stake_currency'], msg['fiat_currency'])
            message += (' `({gain}: {profit_amount:.8f} {stake_currency}'
                        ' / {profit_fiat:.3f} {fiat_currency})`').format(**msg)
        return message

    def send_msg(self, msg: Dict[str, Any]) -> None:
        """ Send a message to telegram channel """

        noti = self._config['telegram'].get('notification_settings', {}
                                            ).get(str(msg['type']), 'on')
        if noti == 'off':
            logger.info(f"Notification '{msg['type']}' not sent.")
            # Notification disabled
            return

        if msg['type'] == RPCMessageType.BUY:
            message = self._format_buy_msg(msg)

        elif msg['type'] in (RPCMessageType.BUY_CANCEL, RPCMessageType.SELL_CANCEL):
            msg['message_side'] = 'buy' if msg['type'] == RPCMessageType.BUY_CANCEL else 'sell'
            message = ("\N{WARNING SIGN} *{exchange}:* "
                       "Cancelling open {message_side} Order for {pair} (#{trade_id}). "
                       "Reason: {reason}.".format(**msg))

        elif msg['type'] == RPCMessageType.BUY_FILL:
            message = ("\N{LARGE CIRCLE} *{exchange}:* "
                       "Buy order for {pair} (#{trade_id}) filled "
                       "for {open_rate}.".format(**msg))
        elif msg['type'] == RPCMessageType.SELL_FILL:
            message = ("\N{LARGE CIRCLE} *{exchange}:* "
                       "Sell order for {pair} (#{trade_id}) filled "
                       "for {close_rate}.".format(**msg))
        elif msg['type'] == RPCMessageType.SELL:
            message = self._format_sell_msg(msg)

        elif msg['type'] == RPCMessageType.STATUS:
            message = '*Status:* `{status}`'.format(**msg)

        elif msg['type'] == RPCMessageType.WARNING:
            message = '\N{WARNING SIGN} *Warning:* `{status}`'.format(**msg)

        elif msg['type'] == RPCMessageType.STARTUP:
            message = '{status}'.format(**msg)

        else:
            raise NotImplementedError('Unknown message type: {}'.format(msg['type']))

        self._send_msg(message, disable_notification=(noti == 'silent'))

    def _get_sell_emoji(self, msg):
        """
        Get emoji for sell-side
        """

        if float(msg['profit_percent']) >= 5.0:
            return "\N{ROCKET}"
        elif float(msg['profit_percent']) >= 0.0:
            return "\N{EIGHT SPOKED ASTERISK}"
        elif msg['sell_reason'] == "stop_loss":
            return"\N{WARNING SIGN}"
        else:
            return "\N{CROSS MARK}"

    @authorized_only
    def _status(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /status.
        Returns the current TradeThread status
        :param bot: telegram bot
        :param update: message update
        :return: None
        """

        if context.args and 'table' in context.args:
            self._status_table(update, context)
            return

        try:

            # Check if there's at least one numerical ID provided.
            # If so, try to get only these trades.
            trade_ids = []
            if context.args and len(context.args) > 0:
                trade_ids = [int(i) for i in context.args if i.isnumeric()]

            results = self._rpc._rpc_trade_status(trade_ids=trade_ids)

            messages = []
            for r in results:
                r['open_date_hum'] = arrow.get(r['open_date']).humanize()
                lines = [
                    "*Trade ID:* `{trade_id}` `(since {open_date_hum})`",
                    "*Current Pair:* {pair}",
                    "*Amount:* `{amount} ({stake_amount} {base_currency})`",
                    "*Open Rate:* `{open_rate:.8f}`",
                    "*Close Rate:* `{close_rate}`" if r['close_rate'] else "",
                    "*Current Rate:* `{current_rate:.8f}`",
                    ("*Current Profit:* " if r['is_open'] else "*Close Profit: *")
                    + "`{profit_pct:.2f}%`",
                ]
                if (r['stop_loss_abs'] != r['initial_stop_loss_abs']
                        and r['initial_stop_loss_pct'] is not None):
                    # Adding initial stoploss only if it is different from stoploss
                    lines.append("*Initial Stoploss:* `{initial_stop_loss_abs:.8f}` "
                                 "`({initial_stop_loss_pct:.2f}%)`")

                # Adding stoploss and stoploss percentage only if it is not None
                lines.append("*Stoploss:* `{stop_loss_abs:.8f}` " +
                             ("`({stop_loss_pct:.2f}%)`" if r['stop_loss_pct'] else ""))
                lines.append("*Stoploss distance:* `{stoploss_current_dist:.8f}` "
                             "`({stoploss_current_dist_pct:.2f}%)`")
                if r['open_order']:
                    if r['sell_order_status']:
                        lines.append("*Open Order:* `{open_order}` - `{sell_order_status}`")
                    else:
                        lines.append("*Open Order:* `{open_order}`")

                # Filter empty lines using list-comprehension
                messages.append("\n".join([line for line in lines if line]).format(**r))

            for msg in messages:
                self._send_msg(msg)

        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _status_table(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /status table.
        Returns the current TradeThread status in table format
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        try:
            statlist, head = self._rpc._rpc_status_table(
                self._config['stake_currency'], self._config.get('fiat_display_currency', ''))

            max_trades_per_msg = 50
            """
            Calculate the number of messages of 50 trades per message
            0.99 is used to make sure that there are no extra (empty) messages
            As an example with 50 trades, there will be int(50/50 + 0.99) = 1 message
            """
            for i in range(0, max(int(len(statlist) / max_trades_per_msg + 0.99), 1)):
                message = tabulate(statlist[i * max_trades_per_msg:(i + 1) * max_trades_per_msg],
                                   headers=head,
                                   tablefmt='simple')
                self._send_msg(f"<pre>{message}</pre>", parse_mode=ParseMode.HTML)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _daily(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /daily <n>
        Returns a daily profit (in BTC) over the last n days.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        stake_cur = self._config['stake_currency']
        fiat_disp_cur = self._config.get('fiat_display_currency', '')
        try:
            timescale = int(context.args[0]) if context.args else 7
        except (TypeError, ValueError, IndexError):
            timescale = 7
        try:
            stats = self._rpc._rpc_daily_profit(
                timescale,
                stake_cur,
                fiat_disp_cur
            )
            stats_tab = tabulate(
                [[day['date'],
                  f"{round_coin_value(day['abs_profit'], stats['stake_currency'])}",
                  f"{day['fiat_value']:.3f} {stats['fiat_display_currency']}",
                  f"{day['trade_count']} trades"] for day in stats['data']],
                headers=[
                    'Day',
                    f'Profit {stake_cur}',
                    f'Profit {fiat_disp_cur}',
                    'Trades',
                ],
                tablefmt='simple')
            message = f'<b>Daily Profit over the last {timescale} days</b>:\n<pre>{stats_tab}</pre>'
            self._send_msg(message, parse_mode=ParseMode.HTML)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _profit(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /profit.
        Returns a cumulative profit statistics.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        stake_cur = self._config['stake_currency']
        fiat_disp_cur = self._config.get('fiat_display_currency', '')

        stats = self._rpc._rpc_trade_statistics(
            stake_cur,
            fiat_disp_cur)
        profit_closed_coin = stats['profit_closed_coin']
        profit_closed_percent_mean = stats['profit_closed_percent_mean']
        profit_closed_percent_sum = stats['profit_closed_percent_sum']
        profit_closed_fiat = stats['profit_closed_fiat']
        profit_all_coin = stats['profit_all_coin']
        profit_all_percent_mean = stats['profit_all_percent_mean']
        profit_all_percent_sum = stats['profit_all_percent_sum']
        profit_all_fiat = stats['profit_all_fiat']
        trade_count = stats['trade_count']
        first_trade_date = stats['first_trade_date']
        latest_trade_date = stats['latest_trade_date']
        avg_duration = stats['avg_duration']
        best_pair = stats['best_pair']
        best_rate = stats['best_rate']
        if stats['trade_count'] == 0:
            markdown_msg = 'No trades yet.'
        else:
            # Message to display
            if stats['closed_trade_count'] > 0:
                markdown_msg = ("*ROI:* Closed trades\n"
                                f"∙ `{round_coin_value(profit_closed_coin, stake_cur)} "
                                f"({profit_closed_percent_mean:.2f}%) "
                                f"({profit_closed_percent_sum} \N{GREEK CAPITAL LETTER SIGMA}%)`\n"
                                f"∙ `{round_coin_value(profit_closed_fiat, fiat_disp_cur)}`\n")
            else:
                markdown_msg = "`No closed trade` \n"

            markdown_msg += (f"*ROI:* All trades\n"
                             f"∙ `{round_coin_value(profit_all_coin, stake_cur)} "
                             f"({profit_all_percent_mean:.2f}%) "
                             f"({profit_all_percent_sum} \N{GREEK CAPITAL LETTER SIGMA}%)`\n"
                             f"∙ `{round_coin_value(profit_all_fiat, fiat_disp_cur)}`\n"
                             f"*Total Trade Count:* `{trade_count}`\n"
                             f"*First Trade opened:* `{first_trade_date}`\n"
                             f"*Latest Trade opened:* `{latest_trade_date}\n`"
                             f"*Win / Loss:* `{stats['winning_trades']} / {stats['losing_trades']}`"
                             )
            if stats['closed_trade_count'] > 0:
                markdown_msg += (f"\n*Avg. Duration:* `{avg_duration}`\n"
                                 f"*Best Performing:* `{best_pair}: {best_rate:.2f}%`")
        self._send_msg(markdown_msg)

    @authorized_only
    def _stats(self, update: Update, context: CallbackContext) -> None:
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
            'sell_signal': 'Sell Signal',
            'force_sell': 'Forcesell',
            'emergency_sell': 'Emergency Sell',
        }
        sell_reasons_tabulate = [
            [
                reason_map.get(reason, reason),
                sum(count.values()),
                count['wins'],
                count['losses']
            ] for reason, count in stats['sell_reasons'].items()
        ]
        sell_reasons_msg = tabulate(
            sell_reasons_tabulate,
            headers=['Sell Reason', 'Sells', 'Wins', 'Losses']
            )
        durations = stats['durations']
        duration_msg = tabulate([
            ['Wins', str(timedelta(seconds=durations['wins']))
             if durations['wins'] != 'N/A' else 'N/A'],
            ['Losses', str(timedelta(seconds=durations['losses']))
             if durations['losses'] != 'N/A' else 'N/A']
            ],
            headers=['', 'Avg. Duration']
        )
        msg = (f"""```\n{sell_reasons_msg}```\n```\n{duration_msg}```""")

        self._send_msg(msg, ParseMode.MARKDOWN)

    @authorized_only
    def _balance(self, update: Update, context: CallbackContext) -> None:
        """ Handler for /balance """
        try:
            result = self._rpc._rpc_balance(self._config['stake_currency'],
                                            self._config.get('fiat_display_currency', ''))

            balance_dust_level = self._config['telegram'].get('balance_dust_level', 0.0)
            if not balance_dust_level:
                balance_dust_level = DUST_PER_COIN.get(self._config['stake_currency'], 1.0)

            output = ''
            if self._config['dry_run']:
                output += (
                    f"*Warning:* Simulated balances in Dry Mode.\n"
                    "This mode is still experimental!\n"
                    "Starting capital: "
                    f"`{self._config['dry_run_wallet']}` {self._config['stake_currency']}.\n"
                )
            for curr in result['currencies']:
                if curr['est_stake'] > balance_dust_level:
                    curr_output = (
                        f"*{curr['currency']}:*\n"
                        f"\t`Available: {curr['free']:.8f}`\n"
                        f"\t`Balance: {curr['balance']:.8f}`\n"
                        f"\t`Pending: {curr['used']:.8f}`\n"
                        f"\t`Est. {curr['stake']}: "
                        f"{round_coin_value(curr['est_stake'], curr['stake'], False)}`\n")
                else:
                    curr_output = (f"*{curr['currency']}:* not showing <{balance_dust_level} "
                                   f"{curr['stake']} amount \n")

                # Handle overflowing messsage length
                if len(output + curr_output) >= MAX_TELEGRAM_MESSAGE_LENGTH:
                    self._send_msg(output)
                    output = curr_output
                else:
                    output += curr_output

            output += ("\n*Estimated Value*:\n"
                       f"\t`{result['stake']}: {result['total']: .8f}`\n"
                       f"\t`{result['symbol']}: "
                       f"{round_coin_value(result['value'], result['symbol'], False)}`\n")
            self._send_msg(output)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _start(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /start.
        Starts TradeThread
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc._rpc_start()
        self._send_msg('Status: `{status}`'.format(**msg))

    @authorized_only
    def _stop(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /stop.
        Stops TradeThread
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc._rpc_stop()
        self._send_msg('Status: `{status}`'.format(**msg))

    @authorized_only
    def _reload_config(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /reload_config.
        Triggers a config file reload
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc._rpc_reload_config()
        self._send_msg('Status: `{status}`'.format(**msg))

    @authorized_only
    def _stopbuy(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /stop_buy.
        Sets max_open_trades to 0 and gracefully sells all open trades
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc._rpc_stopbuy()
        self._send_msg('Status: `{status}`'.format(**msg))

    @authorized_only
    def _forcesell(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /forcesell <id>.
        Sells the given trade at current price
        :param bot: telegram bot
        :param update: message update
        :return: None
        """

        trade_id = context.args[0] if context.args and len(context.args) > 0 else None
        if not trade_id:
            self._send_msg("You must specify a trade-id or 'all'.")
            return
        try:
            msg = self._rpc._rpc_forcesell(trade_id)
            self._send_msg('Forcesell Result: `{result}`'.format(**msg))

        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _forcebuy(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /forcebuy <asset> <price>.
        Buys a pair trade at the given or current price
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        if context.args:
            pair = context.args[0]
            price = float(context.args[1]) if len(context.args) > 1 else None
            try:
                self._rpc._rpc_forcebuy(pair, price)
            except RPCException as e:
                self._send_msg(str(e))

    @authorized_only
    def _trades(self, update: Update, context: CallbackContext) -> None:
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
        try:
            trades = self._rpc._rpc_trade_history(
                nrecent
            )
            trades_tab = tabulate(
                [[arrow.get(trade['close_date']).humanize(),
                  trade['pair'] + " (#" + str(trade['trade_id']) + ")",
                  f"{(100 * trade['close_profit']):.2f}% ({trade['close_profit_abs']})"]
                 for trade in trades['trades']],
                headers=[
                    'Close Date',
                    'Pair (ID)',
                    f'Profit ({stake_cur})',
                ],
                tablefmt='simple')
            message = (f"<b>{min(trades['trades_count'], nrecent)} recent trades</b>:\n"
                       + (f"<pre>{trades_tab}</pre>" if trades['trades_count'] > 0 else ''))
            self._send_msg(message, parse_mode=ParseMode.HTML)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _delete_trade(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /delete <id>.
        Delete the given trade
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        try:
            if not context.args or len(context.args) == 0:
                raise RPCException("Trade-id not set.")
            trade_id = int(context.args[0])
            msg = self._rpc._rpc_delete(trade_id)
            self._send_msg((
                '`{result_msg}`\n'
                'Please make sure to take care of this asset on the exchange manually.'
            ).format(**msg))

        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _performance(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /performance.
        Shows a performance statistic from finished trades
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        try:
            trades = self._rpc._rpc_performance()
            output = "<b>Performance:</b>\n"
            for i, trade in enumerate(trades):
                stat_line = (f"{i+1}.\t <code>{trade['pair']}\t{trade['profit']:.2f}% "
                             f"({trade['count']})</code>\n")

                if len(output + stat_line) >= MAX_TELEGRAM_MESSAGE_LENGTH:
                    self._send_msg(output, parse_mode=ParseMode.HTML)
                    output = stat_line
                else:
                    output += stat_line

            self._send_msg(output, parse_mode=ParseMode.HTML)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _count(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /count.
        Returns the number of trades running
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        try:
            counts = self._rpc._rpc_count()
            message = tabulate({k: [v] for k, v in counts.items()},
                               headers=['current', 'max', 'total stake'],
                               tablefmt='simple')
            message = "<pre>{}</pre>".format(message)
            logger.debug(message)
            self._send_msg(message, parse_mode=ParseMode.HTML)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _locks(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /locks.
        Returns the currently active locks
        """
        locks = self._rpc._rpc_locks()
        message = tabulate([[
            lock['id'],
            lock['pair'],
            lock['lock_end_time'],
            lock['reason']] for lock in locks['locks']],
            headers=['ID', 'Pair', 'Until', 'Reason'],
            tablefmt='simple')
        message = f"<pre>{escape(message)}</pre>"
        logger.debug(message)
        self._send_msg(message, parse_mode=ParseMode.HTML)

    @authorized_only
    def _delete_locks(self, update: Update, context: CallbackContext) -> None:
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
        self._locks(update, context)

    @authorized_only
    def _whitelist(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /whitelist
        Shows the currently active whitelist
        """
        try:
            whitelist = self._rpc._rpc_whitelist()

            message = f"Using whitelist `{whitelist['method']}` with {whitelist['length']} pairs\n"
            message += f"`{', '.join(whitelist['whitelist'])}`"

            logger.debug(message)
            self._send_msg(message)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _blacklist(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /blacklist
        Shows the currently active blacklist
        """
        try:

            blacklist = self._rpc._rpc_blacklist(context.args)
            errmsgs = []
            for pair, error in blacklist['errors'].items():
                errmsgs.append(f"Error adding `{pair}` to blacklist: `{error['error_msg']}`")
            if errmsgs:
                self._send_msg('\n'.join(errmsgs))

            message = f"Blacklist contains {blacklist['length']} pairs\n"
            message += f"`{', '.join(blacklist['blacklist'])}`"

            logger.debug(message)
            self._send_msg(message)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _logs(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /logs
        Shows the latest logs
        """
        try:
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
                if len(msgs + msg) + 10 >= MAX_TELEGRAM_MESSAGE_LENGTH:
                    # Send message immediately if it would become too long
                    self._send_msg(msgs, parse_mode=ParseMode.MARKDOWN_V2)
                    msgs = msg + '\n'
                else:
                    # Append message to messages to send
                    msgs += msg + '\n'

            if msgs:
                self._send_msg(msgs, parse_mode=ParseMode.MARKDOWN_V2)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _edge(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /edge
        Shows information related to Edge
        """
        try:
            edge_pairs = self._rpc._rpc_edge()
            edge_pairs_tab = tabulate(edge_pairs, headers='keys', tablefmt='simple')
            message = f'<b>Edge only validated following pairs:</b>\n<pre>{edge_pairs_tab}</pre>'
            self._send_msg(message, parse_mode=ParseMode.HTML)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _help(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /help.
        Show commands of the bot
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        forcebuy_text = ("*/forcebuy <pair> [<rate>]:* `Instantly buys the given pair. "
                         "Optionally takes a rate at which to buy.` \n")
        message = ("*/start:* `Starts the trader`\n"
                   "*/stop:* `Stops the trader`\n"
                   "*/status <trade_id>|[table]:* `Lists all open trades`\n"
                   "         *<trade_id> :* `Lists one or more specific trades.`\n"
                   "                        `Separate multiple <trade_id> with a blank space.`\n"
                   "         *table :* `will display trades in a table`\n"
                   "                `pending buy orders are marked with an asterisk (*)`\n"
                   "                `pending sell orders are marked with a double asterisk (**)`\n"
                   "*/trades [limit]:* `Lists last closed trades (limited to 10 by default)`\n"
                   "*/profit:* `Lists cumulative profit from all finished trades`\n"
                   "*/forcesell <trade_id>|all:* `Instantly sells the given trade or all trades, "
                   "regardless of profit`\n"
                   f"{forcebuy_text if self._config.get('forcebuy_enable', False) else ''}"
                   "*/delete <trade_id>:* `Instantly delete the given trade in the database`\n"
                   "*/performance:* `Show performance of each finished trade grouped by pair`\n"
                   "*/daily <n>:* `Shows profit or loss per day, over the last n days`\n"
                   "*/stats:* `Shows Wins / losses by Sell reason as well as "
                   "Avg. holding durationsfor buys and sells.`\n"
                   "*/count:* `Show number of active trades compared to allowed number of trades`\n"
                   "*/locks:* `Show currently locked pairs`\n"
                   "*/unlock <pair|id>:* `Unlock this Pair (or this lock id if it's numeric)`\n"
                   "*/balance:* `Show account balance per currency`\n"
                   "*/stopbuy:* `Stops buying, but handles open trades gracefully` \n"
                   "*/reload_config:* `Reload configuration file` \n"
                   "*/show_config:* `Show running configuration` \n"
                   "*/logs [limit]:* `Show latest logs - defaults to 10` \n"
                   "*/whitelist:* `Show current whitelist` \n"
                   "*/blacklist [pair]:* `Show current blacklist, or adds one or more pairs "
                   "to the blacklist.` \n"
                   "*/edge:* `Shows validated pairs by Edge if it is enabled` \n"
                   "*/help:* `This help message`\n"
                   "*/version:* `Show version`")

        self._send_msg(message)

    @authorized_only
    def _version(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /version.
        Show version information
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        self._send_msg('*Version:* `{}`'.format(__version__))

    @authorized_only
    def _show_config(self, update: Update, context: CallbackContext) -> None:
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

        self._send_msg(
            f"*Mode:* `{'Dry-run' if val['dry_run'] else 'Live'}`\n"
            f"*Exchange:* `{val['exchange']}`\n"
            f"*Stake per trade:* `{val['stake_amount']} {val['stake_currency']}`\n"
            f"*Max open Trades:* `{val['max_open_trades']}`\n"
            f"*Minimum ROI:* `{val['minimal_roi']}`\n"
            f"*Ask strategy:* ```\n{json.dumps(val['ask_strategy'])}```\n"
            f"*Bid strategy:* ```\n{json.dumps(val['bid_strategy'])}```\n"
            f"{sl_info}"
            f"*Timeframe:* `{val['timeframe']}`\n"
            f"*Strategy:* `{val['strategy']}`\n"
            f"*Current state:* `{val['state']}`"
        )

    def _send_msg(self, msg: str, parse_mode: str = ParseMode.MARKDOWN,
                  disable_notification: bool = False) -> None:
        """
        Send given markdown message
        :param msg: message
        :param bot: alternative bot
        :param parse_mode: telegram parse mode
        :return: None
        """
        reply_markup = ReplyKeyboardMarkup(self._keyboard, resize_keyboard=True)
        try:
            try:
                self._updater.bot.send_message(
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
                self._updater.bot.send_message(
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
