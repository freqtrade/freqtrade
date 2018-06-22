# pragma pylint: disable=unused-argument, unused-variable, protected-access, invalid-name

"""
This module manage Telegram communication
"""
import logging
from typing import Any, Callable

from tabulate import tabulate
from telegram import Bot, ParseMode, ReplyKeyboardMarkup, Update
from telegram.error import NetworkError, TelegramError
from telegram.ext import CommandHandler, Updater

from freqtrade.__init__ import __version__
from freqtrade.rpc.rpc import RPC, RPCException

logger = logging.getLogger(__name__)

logger.debug('Included module rpc.telegram ...')


def authorized_only(command_handler: Callable[[Any, Bot, Update], None]) -> Callable[..., Any]:
    """
    Decorator to check if the message comes from the correct chat_id
    :param command_handler: Telegram CommandHandler
    :return: decorated function
    """
    def wrapper(self, *args, **kwargs):
        """ Decorator logic """
        update = kwargs.get('update') or args[1]

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

    @property
    def name(self) -> str:
        return "telegram"

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

    def _init(self) -> None:
        """
        Initializes this module with the given config,
        registers all known command handlers
        and starts polling for message updates
        """
        self._updater = Updater(token=self._config['telegram']['token'], workers=0)

        # Register command handler and start telegram message polling
        handles = [
            CommandHandler('status', self._status),
            CommandHandler('profit', self._profit),
            CommandHandler('balance', self._balance),
            CommandHandler('start', self._start),
            CommandHandler('stop', self._stop),
            CommandHandler('forcesell', self._forcesell),
            CommandHandler('performance', self._performance),
            CommandHandler('daily', self._daily),
            CommandHandler('count', self._count),
            CommandHandler('reload_conf', self._reload_conf),
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

    def send_msg(self, msg: str) -> None:
        """ Send a message to telegram channel """
        self._send_msg(msg)

    @authorized_only
    def _status(self, bot: Bot, update: Update) -> None:
        """
        Handler for /status.
        Returns the current TradeThread status
        :param bot: telegram bot
        :param update: message update
        :return: None
        """

        # Check if additional parameters are passed
        params = update.message.text.replace('/status', '').split(' ') \
            if update.message.text else []
        if 'table' in params:
            self._status_table(bot, update)
            return

        try:
            results = self._rpc_trade_status()
            messages = [
                "*Trade ID:* `{trade_id}`\n"
                "*Current Pair:* [{pair}]({market_url})\n"
                "*Open Since:* `{date}`\n"
                "*Amount:* `{amount}`\n"
                "*Open Rate:* `{open_rate:.8f}`\n"
                "*Close Rate:* `{close_rate}`\n"
                "*Current Rate:* `{current_rate:.8f}`\n"
                "*Close Profit:* `{close_profit}`\n"
                "*Current Profit:* `{current_profit:.2f}%`\n"
                "*Open Order:* `{open_order}`".format(**result)
                for result in results
            ]
            for msg in messages:
                self._send_msg(msg, bot=bot)
        except RPCException as e:
            self._send_msg(str(e), bot=bot)

    @authorized_only
    def _status_table(self, bot: Bot, update: Update) -> None:
        """
        Handler for /status table.
        Returns the current TradeThread status in table format
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        try:
            df_statuses = self._rpc_status_table()
            message = tabulate(df_statuses, headers='keys', tablefmt='simple')
            self._send_msg(f"<pre>{message}</pre>", parse_mode=ParseMode.HTML)
        except RPCException as e:
            self._send_msg(str(e), bot=bot)

    @authorized_only
    def _daily(self, bot: Bot, update: Update) -> None:
        """
        Handler for /daily <n>
        Returns a daily profit (in BTC) over the last n days.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        stake_cur = self._config['stake_currency']
        fiat_disp_cur = self._config['fiat_display_currency']
        try:
            timescale = int(update.message.text.replace('/daily', '').strip())
        except (TypeError, ValueError):
            timescale = 7
        try:
            stats = self._rpc_daily_profit(
                timescale,
                stake_cur,
                fiat_disp_cur
            )
            stats = tabulate(stats,
                             headers=[
                                 'Day',
                                 f'Profit {stake_cur}',
                                 f'Profit {fiat_disp_cur}'
                             ],
                             tablefmt='simple')
            message = f'<b>Daily Profit over the last {timescale} days</b>:\n<pre>{stats}</pre>'
            self._send_msg(message, bot=bot, parse_mode=ParseMode.HTML)
        except RPCException as e:
            self._send_msg(str(e), bot=bot)

    @authorized_only
    def _profit(self, bot: Bot, update: Update) -> None:
        """
        Handler for /profit.
        Returns a cumulative profit statistics.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        stake_cur = self._config['stake_currency']
        fiat_disp_cur = self._config['fiat_display_currency']

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
            self._send_msg(markdown_msg, bot=bot)
        except RPCException as e:
            self._send_msg(str(e), bot=bot)

    @authorized_only
    def _balance(self, bot: Bot, update: Update) -> None:
        """ Handler for /balance """
        try:
            currencys, total, symbol, value = \
                self._rpc_balance(self._config['fiat_display_currency'])
            output = ''
            for currency in currencys:
                output += "*{currency}:*\n" \
                          "\t`Available: {available: .8f}`\n" \
                          "\t`Balance: {balance: .8f}`\n" \
                          "\t`Pending: {pending: .8f}`\n" \
                          "\t`Est. BTC: {est_btc: .8f}`\n".format(**currency)

            output += "\n*Estimated Value*:\n" \
                      "\t`BTC: {0: .8f}`\n" \
                      "\t`{1}: {2: .2f}`\n".format(total, symbol, value)
            self._send_msg(output, bot=bot)
        except RPCException as e:
            self._send_msg(str(e), bot=bot)

    @authorized_only
    def _start(self, bot: Bot, update: Update) -> None:
        """
        Handler for /start.
        Starts TradeThread
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc_start()
        self._send_msg('Status: `{status}`'.format(**msg), bot=bot)

    @authorized_only
    def _stop(self, bot: Bot, update: Update) -> None:
        """
        Handler for /stop.
        Stops TradeThread
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc_stop()
        self._send_msg('Status: `{status}`'.format(**msg), bot=bot)

    @authorized_only
    def _reload_conf(self, bot: Bot, update: Update) -> None:
        """
        Handler for /reload_conf.
        Triggers a config file reload
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc_reload_conf()
        self._send_msg('Status: `{status}`'.format(**msg), bot=bot)

    @authorized_only
    def _forcesell(self, bot: Bot, update: Update) -> None:
        """
        Handler for /forcesell <id>.
        Sells the given trade at current price
        :param bot: telegram bot
        :param update: message update
        :return: None
        """

        trade_id = update.message.text.replace('/forcesell', '').strip()
        try:
            self._rpc_forcesell(trade_id)
        except RPCException as e:
            self._send_msg(str(e), bot=bot)

    @authorized_only
    def _performance(self, bot: Bot, update: Update) -> None:
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
            self._send_msg(str(e), bot=bot)

    @authorized_only
    def _count(self, bot: Bot, update: Update) -> None:
        """
        Handler for /count.
        Returns the number of trades running
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        try:
            trades = self._rpc_count()
            message = tabulate({
                'current': [len(trades)],
                'max': [self._config['max_open_trades']],
                'total stake': [sum((trade.open_rate * trade.amount) for trade in trades)]
            }, headers=['current', 'max', 'total stake'], tablefmt='simple')
            message = "<pre>{}</pre>".format(message)
            logger.debug(message)
            self._send_msg(message, parse_mode=ParseMode.HTML)
        except RPCException as e:
            self._send_msg(str(e), bot=bot)

    @authorized_only
    def _help(self, bot: Bot, update: Update) -> None:
        """
        Handler for /help.
        Show commands of the bot
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        message = "*/start:* `Starts the trader`\n" \
                  "*/stop:* `Stops the trader`\n" \
                  "*/status [table]:* `Lists all open trades`\n" \
                  "         *table :* `will display trades in a table`\n" \
                  "*/profit:* `Lists cumulative profit from all finished trades`\n" \
                  "*/forcesell <trade_id>|all:* `Instantly sells the given trade or all trades, " \
                  "regardless of profit`\n" \
                  "*/performance:* `Show performance of each finished trade grouped by pair`\n" \
                  "*/daily <n>:* `Shows profit or loss per day, over the last n days`\n" \
                  "*/count:* `Show number of trades running compared to allowed number of trades`" \
                  "\n" \
                  "*/balance:* `Show account balance per currency`\n" \
                  "*/help:* `This help message`\n" \
                  "*/version:* `Show version`"

        self._send_msg(message, bot=bot)

    @authorized_only
    def _version(self, bot: Bot, update: Update) -> None:
        """
        Handler for /version.
        Show version information
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        self._send_msg('*Version:* `{}`'.format(__version__), bot=bot)

    def _send_msg(self, msg: str, bot: Bot = None,
                  parse_mode: ParseMode = ParseMode.MARKDOWN) -> None:
        """
        Send given markdown message
        :param msg: message
        :param bot: alternative bot
        :param parse_mode: telegram parse mode
        :return: None
        """
        bot = bot or self._updater.bot

        keyboard = [['/daily', '/profit', '/balance'],
                    ['/status', '/status table', '/performance'],
                    ['/count', '/start', '/stop', '/help']]

        reply_markup = ReplyKeyboardMarkup(keyboard)

        try:
            try:
                bot.send_message(
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
                bot.send_message(
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
