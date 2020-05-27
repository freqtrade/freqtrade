# pragma pylint: disable=unused-argument, unused-variable, protected-access, invalid-name

"""
This module manage Telegram communication
"""
import logging
from typing import Any, Callable, Dict

from tabulate import tabulate
from telegram import ParseMode, ReplyKeyboardMarkup, Update
from telegram.error import NetworkError, TelegramError
from telegram.ext import CallbackContext, CommandHandler, Updater

from freqtrade.__init__ import __version__
from freqtrade.rpc import RPC, RPCException, RPCMessageType
from freqtrade.rpc.fiat_convert import CryptoToFiatConverter

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


class Telegram(RPC):
    """  This class handles all telegram communication """

    def __init__(self, freqtrade) -> None:
        """
        Init the Telegram call, and init the super class RPC
        :param freqtrade: Instance of a freqtrade bot
        :return: None
        """
        super().__init__(freqtrade)

        self._updater: Updater = None
        self._config = freqtrade.config
        self._init()
        if self._config.get('fiat_display_currency', None):
            self._fiat_converter = CryptoToFiatConverter()

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
            CommandHandler('performance', self._performance),
            CommandHandler('daily', self._daily),
            CommandHandler('count', self._count),
            CommandHandler('reload_conf', self._reload_conf),
            CommandHandler('show_config', self._show_config),
            CommandHandler('stopbuy', self._stopbuy),
            CommandHandler('whitelist', self._whitelist),
            CommandHandler('blacklist', self._blacklist),
            CommandHandler('edge', self._edge),
            CommandHandler('help', self._help),
            CommandHandler('version', self._version),
        ]
        for handle in handles:
            self._updater.dispatcher.add_handler(handle)
        self._updater.start_polling(
            clean=True,
            bootstrap_retries=-1,
            timeout=30,
            read_latency=60,
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

    def send_msg(self, msg: Dict[str, Any]) -> None:
        """ Send a message to telegram channel """

        if msg['type'] == RPCMessageType.BUY_NOTIFICATION:
            if self._fiat_converter:
                msg['stake_amount_fiat'] = self._fiat_converter.convert_amount(
                    msg['stake_amount'], msg['stake_currency'], msg['fiat_currency'])
            else:
                msg['stake_amount_fiat'] = 0

            message = ("*{exchange}:* Buying {pair}\n"
                       "*Amount:* `{amount:.8f}`\n"
                       "*Open Rate:* `{limit:.8f}`\n"
                       "*Current Rate:* `{current_rate:.8f}`\n"
                       "*Total:* `({stake_amount:.6f} {stake_currency}").format(**msg)

            if msg.get('fiat_currency', None):
                message += ", {stake_amount_fiat:.3f} {fiat_currency}".format(**msg)
            message += ")`"

        elif msg['type'] == RPCMessageType.BUY_CANCEL_NOTIFICATION:
            message = "*{exchange}:* Cancelling Open Buy Order for {pair}".format(**msg)

        elif msg['type'] == RPCMessageType.SELL_NOTIFICATION:
            msg['amount'] = round(msg['amount'], 8)
            msg['profit_percent'] = round(msg['profit_ratio'] * 100, 2)
            msg['duration'] = msg['close_date'].replace(
                microsecond=0) - msg['open_date'].replace(microsecond=0)
            msg['duration_min'] = msg['duration'].total_seconds() / 60

            message = ("*{exchange}:* Selling {pair}\n"
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
               and self._fiat_converter):
                msg['profit_fiat'] = self._fiat_converter.convert_amount(
                    msg['profit_amount'], msg['stake_currency'], msg['fiat_currency'])
                message += (' `({gain}: {profit_amount:.8f} {stake_currency}'
                            ' / {profit_fiat:.3f} {fiat_currency})`').format(**msg)

        elif msg['type'] == RPCMessageType.SELL_CANCEL_NOTIFICATION:
            message = ("*{exchange}:* Cancelling Open Sell Order "
                       "for {pair}. Reason: {reason}").format(**msg)

        elif msg['type'] == RPCMessageType.STATUS_NOTIFICATION:
            message = '*Status:* `{status}`'.format(**msg)

        elif msg['type'] == RPCMessageType.WARNING_NOTIFICATION:
            message = '*Warning:* `{status}`'.format(**msg)

        elif msg['type'] == RPCMessageType.CUSTOM_NOTIFICATION:
            message = '{status}'.format(**msg)

        else:
            raise NotImplementedError('Unknown message type: {}'.format(msg['type']))

        self._send_msg(message)

    @authorized_only
    def _status(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /status.
        Returns the current TradeThread status
        :param bot: telegram bot
        :param update: message update
        :return: None
        """

        if 'table' in context.args:
            self._status_table(update, context)
            return

        try:
            results = self._rpc_trade_status()

            messages = []
            for r in results:
                lines = [
                    "*Trade ID:* `{trade_id}` `(since {open_date_hum})`",
                    "*Current Pair:* {pair}",
                    "*Amount:* `{amount} ({stake_amount} {base_currency})`",
                    "*Open Rate:* `{open_rate:.8f}`",
                    "*Close Rate:* `{close_rate}`" if r['close_rate'] else "",
                    "*Current Rate:* `{current_rate:.8f}`",
                    ("*Close Profit:* `{close_profit_pct}`"
                     if r['close_profit_pct'] is not None else ""),
                    "*Current Profit:* `{current_profit_pct:.2f}%`",

                    # Adding initial stoploss only if it is different from stoploss
                    "*Initial Stoploss:* `{initial_stop_loss:.8f}` " +
                    ("`({initial_stop_loss_pct:.2f}%)`") if (
                        r['stop_loss'] != r['initial_stop_loss']
                        and r['initial_stop_loss_pct'] is not None) else "",

                    # Adding stoploss and stoploss percentage only if it is not None
                    "*Stoploss:* `{stop_loss:.8f}` " +
                    ("`({stop_loss_pct:.2f}%)`" if r['stop_loss_pct'] else ""),
                ]
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
            statlist, head = self._rpc_status_table(self._config['stake_currency'],
                                                    self._config.get('fiat_display_currency', ''))
            message = tabulate(statlist, headers=head, tablefmt='simple')
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
            timescale = int(context.args[0])
        except (TypeError, ValueError, IndexError):
            timescale = 7
        try:
            stats = self._rpc_daily_profit(
                timescale,
                stake_cur,
                fiat_disp_cur
            )
            stats_tab = tabulate(
                [[day['date'],
                  f"{day['abs_profit']} {stats['stake_currency']}",
                  f"{day['fiat_value']} {stats['fiat_display_currency']}",
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

        try:
            stats = self._rpc_trade_statistics(
                stake_cur,
                fiat_disp_cur)
            profit_closed_coin = stats['profit_closed_coin']
            profit_closed_percent = stats['profit_closed_percent']
            profit_closed_fiat = stats['profit_closed_fiat']
            profit_all_coin = stats['profit_all_coin']
            profit_all_percent = stats['profit_all_percent']
            profit_all_fiat = stats['profit_all_fiat']
            trade_count = stats['trade_count']
            first_trade_date = stats['first_trade_date']
            latest_trade_date = stats['latest_trade_date']
            avg_duration = stats['avg_duration']
            best_pair = stats['best_pair']
            best_rate = stats['best_rate']
            # Message to display
            markdown_msg = "*ROI:* Close trades\n" \
                           f"∙ `{profit_closed_coin:.8f} {stake_cur} "\
                           f"({profit_closed_percent:.2f}%)`\n" \
                           f"∙ `{profit_closed_fiat:.3f} {fiat_disp_cur}`\n" \
                           f"*ROI:* All trades\n" \
                           f"∙ `{profit_all_coin:.8f} {stake_cur} ({profit_all_percent:.2f}%)`\n" \
                           f"∙ `{profit_all_fiat:.3f} {fiat_disp_cur}`\n" \
                           f"*Total Trade Count:* `{trade_count}`\n" \
                           f"*First Trade opened:* `{first_trade_date}`\n" \
                           f"*Latest Trade opened:* `{latest_trade_date}`\n" \
                           f"*Avg. Duration:* `{avg_duration}`\n" \
                           f"*Best Performing:* `{best_pair}: {best_rate:.2f}%`"
            self._send_msg(markdown_msg)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _balance(self, update: Update, context: CallbackContext) -> None:
        """ Handler for /balance """
        try:
            result = self._rpc_balance(self._config['stake_currency'],
                                       self._config.get('fiat_display_currency', ''))

            output = ''
            if self._config['dry_run']:
                output += (
                    f"*Warning:* Simulated balances in Dry Mode.\n"
                    "This mode is still experimental!\n"
                    "Starting capital: "
                    f"`{self._config['dry_run_wallet']}` {self._config['stake_currency']}.\n"
                    )
            for currency in result['currencies']:
                if currency['est_stake'] > 0.0001:
                    curr_output = "*{currency}:*\n" \
                            "\t`Available: {free: .8f}`\n" \
                            "\t`Balance: {balance: .8f}`\n" \
                            "\t`Pending: {used: .8f}`\n" \
                            "\t`Est. {stake}: {est_stake: .8f}`\n".format(**currency)
                else:
                    curr_output = "*{currency}:* not showing <1$ amount \n".format(**currency)

                # Handle overflowing messsage length
                if len(output + curr_output) >= MAX_TELEGRAM_MESSAGE_LENGTH:
                    self._send_msg(output)
                    output = curr_output
                else:
                    output += curr_output

            output += "\n*Estimated Value*:\n" \
                      "\t`{stake}: {total: .8f}`\n" \
                      "\t`{symbol}: {value: .2f}`\n".format(**result)
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
        msg = self._rpc_start()
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
        msg = self._rpc_stop()
        self._send_msg('Status: `{status}`'.format(**msg))

    @authorized_only
    def _reload_conf(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /reload_conf.
        Triggers a config file reload
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc_reload_conf()
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
        msg = self._rpc_stopbuy()
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

        trade_id = context.args[0] if len(context.args) > 0 else None
        try:
            msg = self._rpc_forcesell(trade_id)
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

        pair = context.args[0]
        price = float(context.args[1]) if len(context.args) > 1 else None
        try:
            self._rpc_forcebuy(pair, price)
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
            trades = self._rpc_performance()
            stats = '\n'.join('{index}.\t<code>{pair}\t{profit:.2f}% ({count})</code>'.format(
                index=i + 1,
                pair=trade['pair'],
                profit=trade['profit'],
                count=trade['count']
            ) for i, trade in enumerate(trades))
            message = '<b>Performance:</b>\n{}'.format(stats)
            self._send_msg(message, parse_mode=ParseMode.HTML)
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
            counts = self._rpc_count()
            message = tabulate({k: [v] for k, v in counts.items()},
                               headers=['current', 'max', 'total stake'],
                               tablefmt='simple')
            message = "<pre>{}</pre>".format(message)
            logger.debug(message)
            self._send_msg(message, parse_mode=ParseMode.HTML)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _whitelist(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /whitelist
        Shows the currently active whitelist
        """
        try:
            whitelist = self._rpc_whitelist()

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

            blacklist = self._rpc_blacklist(context.args)

            message = f"Blacklist contains {blacklist['length']} pairs\n"
            message += f"`{', '.join(blacklist['blacklist'])}`"

            logger.debug(message)
            self._send_msg(message)
        except RPCException as e:
            self._send_msg(str(e))

    @authorized_only
    def _edge(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /edge
        Shows information related to Edge
        """
        try:
            edge_pairs = self._rpc_edge()
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
        forcebuy_text = "*/forcebuy <pair> [<rate>]:* `Instantly buys the given pair. " \
                        "Optionally takes a rate at which to buy.` \n"
        message = "*/start:* `Starts the trader`\n" \
                  "*/stop:* `Stops the trader`\n" \
                  "*/status [table]:* `Lists all open trades`\n" \
                  "         *table :* `will display trades in a table`\n" \
                  "                `pending buy orders are marked with an asterisk (*)`\n" \
                  "                `pending sell orders are marked with a double asterisk (**)`\n" \
                  "*/profit:* `Lists cumulative profit from all finished trades`\n" \
                  "*/forcesell <trade_id>|all:* `Instantly sells the given trade or all trades, " \
                  "regardless of profit`\n" \
                  f"{forcebuy_text if self._config.get('forcebuy_enable', False) else '' }" \
                  "*/performance:* `Show performance of each finished trade grouped by pair`\n" \
                  "*/daily <n>:* `Shows profit or loss per day, over the last n days`\n" \
                  "*/count:* `Show number of trades running compared to allowed number of trades`" \
                  "\n" \
                  "*/balance:* `Show account balance per currency`\n" \
                  "*/stopbuy:* `Stops buying, but handles open trades gracefully` \n" \
                  "*/reload_conf:* `Reload configuration file` \n" \
                  "*/show_config:* `Show running configuration` \n" \
                  "*/whitelist:* `Show current whitelist` \n" \
                  "*/blacklist [pair]:* `Show current blacklist, or adds one or more pairs " \
                  "to the blacklist.` \n" \
                  "*/edge:* `Shows validated pairs by Edge if it is enabled` \n" \
                  "*/help:* `This help message`\n" \
                  "*/version:* `Show version`"

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
        val = self._rpc_show_config()
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
            f"{sl_info}"
            f"*Ticker Interval:* `{val['ticker_interval']}`\n"
            f"*Strategy:* `{val['strategy']}`\n"
            f"*Current state:* `{val['state']}`"
        )

    def _send_msg(self, msg: str, parse_mode: ParseMode = ParseMode.MARKDOWN) -> None:
        """
        Send given markdown message
        :param msg: message
        :param bot: alternative bot
        :param parse_mode: telegram parse mode
        :return: None
        """

        keyboard = [['/daily', '/profit', '/balance'],
                    ['/status', '/status table', '/performance'],
                    ['/count', '/start', '/stop', '/help']]

        reply_markup = ReplyKeyboardMarkup(keyboard)

        try:
            try:
                self._updater.bot.send_message(
                    self._config['telegram']['chat_id'],
                    text=msg,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup
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
                    reply_markup=reply_markup
                )
        except TelegramError as telegram_err:
            logger.warning(
                'TelegramError: %s! Giving up on that message.',
                telegram_err.message
            )
