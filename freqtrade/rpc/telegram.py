"""
This module manage Telegram communication
"""

from typing import Any, Callable
from freqtrade.rpc.rpc import RPC
from tabulate import tabulate
from telegram import Bot, ParseMode, ReplyKeyboardMarkup, Update
from telegram.error import NetworkError, TelegramError
from telegram.ext import CommandHandler, Updater
from freqtrade.__init__ import __version__


def authorized_only(command_handler: Callable[[Bot, Update], None]) -> Callable[..., Any]:
    """
    Decorator to check if the message comes from the correct chat_id
    :param command_handler: Telegram CommandHandler
    :return: decorated function
    """

    #def wrapper(self, bot: Bot, update: Update):
    def wrapper(self, *args, **kwargs):

        update = kwargs.get('update') or args[1]

        # Reject unauthorized messages
        chat_id = int(self._config['telegram']['chat_id'])

        if int(update.message.chat_id) != chat_id:
            self.logger.info(
                'Rejected unauthorized message from: %s',
                update.message.chat_id
            )
            return wrapper

        self.logger.info(
            'Executing handler: %s for chat_id: %s',
            command_handler.__name__,
            chat_id
        )
        try:
            return command_handler(self, *args, **kwargs)
        except BaseException:
            self.logger.exception('Exception occurred within Telegram module')

    return wrapper

class Telegram(RPC):
    """
    Telegram, this class send messages to Telegram
    """
    def __init__(self, freqtrade) -> None:
        """
        Init the Telegram call, and init the super class RPC
        :param freqtrade: Instance of a freqtrade bot
        :return: None
        """
        super().__init__(freqtrade)

        self._updater = Updater = None
        self._config = freqtrade.config
        self._init()

    def _init(self) -> None:
        """
        Initializes this module with the given config,
        registers all known command handlers
        and starts polling for message updates
        :param config: config to use
        :return: None
        """
        if not self.is_enabled():
            return

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
        self.logger.info(
            'rpc.telegram is listening for following commands: %s',
            [h.command for h in handles]
        )

    def cleanup(self) -> None:
        """
        Stops all running telegram threads.
        :return: None
        """
        if not self.is_enabled():
            return

        self._updater.stop()

    def is_enabled(self) -> bool:
        """
        Returns True if the telegram module is activated, False otherwise
        """
        return bool(self._config.get('telegram', {}).get('enabled', False))

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

        # Fetch open trade
        (error, trades) = self.rpc_trade_status()
        if error:
            self.send_msg(trades, bot=bot)
        else:
            for trademsg in trades:
                self.send_msg(trademsg, bot=bot)

    @authorized_only
    def _status_table(self, bot: Bot, update: Update) -> None:
        """
        Handler for /status table.
        Returns the current TradeThread status in table format
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        # Fetch open trade
        (err, df_statuses) = self.rpc_status_table()
        if err:
            self.send_msg(df_statuses, bot=bot)
        else:
            message = tabulate(df_statuses, headers='keys', tablefmt='simple')
            message = "<pre>{}</pre>".format(message)

            self.send_msg(message, parse_mode=ParseMode.HTML)

    @authorized_only
    def _daily(self, bot: Bot, update: Update) -> None:
        """
        Handler for /daily <n>
        Returns a daily profit (in BTC) over the last n days.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        try:
            timescale = int(update.message.text.replace('/daily', '').strip())
        except (TypeError, ValueError):
            timescale = 7
        (error, stats) = self.rpc_daily_profit(
            timescale,
            self._config['stake_currency'],
            self._config['fiat_display_currency']
        )
        if error:
            self.send_msg(stats, bot=bot)
        else:
            stats = tabulate(stats,
                             headers=[
                                 'Day',
                                 'Profit {}'.format(self._config['stake_currency']),
                                 'Profit {}'.format(self._config['fiat_display_currency'])
                             ],
                             tablefmt='simple')
            message = '<b>Daily Profit over the last {} days</b>:\n<pre>{}</pre>'\
                    .format(
                        timescale,
                        stats
                    )
            self.send_msg(message, bot=bot, parse_mode=ParseMode.HTML)

    @authorized_only
    def _profit(self, bot: Bot, update: Update) -> None:
        """
        Handler for /profit.
        Returns a cumulative profit statistics.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        (error, stats) = self.rpc_trade_statistics(
            self._config['stake_currency'],
            self._config['fiat_display_currency']
        )
        if error:
            self.send_msg(stats, bot=bot)
            return

        # Message to display
        markdown_msg = "*ROI:* Close trades\n" \
                       "∙ `{profit_closed_coin:.8f} {coin} ({profit_closed_percent:.2f}%)`\n" \
                       "∙ `{profit_closed_fiat:.3f} {fiat}`\n" \
                       "*ROI:* All trades\n" \
                       "∙ `{profit_all_coin:.8f} {coin} ({profit_all_percent:.2f}%)`\n" \
                       "∙ `{profit_all_fiat:.3f} {fiat}`\n" \
                       "*Total Trade Count:* `{trade_count}`\n" \
                       "*First Trade opened:* `{first_trade_date}`\n" \
                       "*Latest Trade opened:* `{latest_trade_date}`\n" \
                       "*Avg. Duration:* `{avg_duration}`\n" \
                       "*Best Performing:* `{best_pair}: {best_rate:.2f}%`"\
                    .format(
                        coin=self._config['stake_currency'],
                        fiat=self._config['fiat_display_currency'],
                        profit_closed_coin=stats['profit_closed_coin'],
                        profit_closed_percent=stats['profit_closed_percent'],
                        profit_closed_fiat=stats['profit_closed_fiat'],
                        profit_all_coin=stats['profit_all_coin'],
                        profit_all_percent=stats['profit_all_percent'],
                        profit_all_fiat=stats['profit_all_fiat'],
                        trade_count=stats['trade_count'],
                        first_trade_date=stats['first_trade_date'],
                        latest_trade_date=stats['latest_trade_date'],
                        avg_duration=stats['avg_duration'],
                        best_pair=stats['best_pair'],
                        best_rate=stats['best_rate']
                    )
        self.send_msg(markdown_msg, bot=bot)

    @authorized_only
    def _balance(self, bot: Bot, update: Update) -> None:
        """
        Handler for /balance
        """
        (error, result) = self.rpc_balance(self._config['fiat_display_currency'])
        if error:
            self.send_msg('`All balances are zero.`')
            return

        (currencys, total, symbol, value) = result
        output = ''
        for currency in currencys:
            output += """*Currency*: {currency}
    *Available*: {available}
    *Balance*: {balance}
    *Pending*: {pending}
    *Est. BTC*: {est_btc: .8f}
    """.format(**currency)

        output += """*Estimated Value*:
    *BTC*: {0: .8f}
    *{1}*: {2: .2f}
    """.format(total, symbol, value)
        self.send_msg(output)

    @authorized_only
    def _start(self, bot: Bot, update: Update) -> None:
        """
        Handler for /start.
        Starts TradeThread
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        (error, msg) = self.rpc_start()
        if error:
            self.send_msg(msg, bot=bot)

    @authorized_only
    def _stop(self, bot: Bot, update: Update) -> None:
        """
        Handler for /stop.
        Stops TradeThread
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        (error, msg) = self.rpc_stop()
        self.send_msg(msg, bot=bot)

    # FIX: no test for this!!!!
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
        (error, message) = self.rpc_forcesell(trade_id)
        if error:
            self.send_msg(message, bot=bot)
            return

    @authorized_only
    def _performance(self, bot: Bot, update: Update) -> None:
        """
        Handler for /performance.
        Shows a performance statistic from finished trades
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        (error, trades) = self.rpc_performance()
        if error:
            self.send_msg(trades, bot=bot)
            return

        stats = '\n'.join('{index}.\t<code>{pair}\t{profit:.2f}% ({count})</code>'.format(
            index=i + 1,
            pair=trade['pair'],
            profit=trade['profit'],
            count=trade['count']
        ) for i, trade in enumerate(trades))
        message = '<b>Performance:</b>\n{}'.format(stats)
        self.send_msg(message, parse_mode=ParseMode.HTML)

    @authorized_only
    def _count(self, bot: Bot, update: Update) -> None:
        """
        Handler for /count.
        Returns the number of trades running
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        (error, trades) = self.rpc_count()
        if error:
            self.send_msg(trades, bot=bot)
            return

        message = tabulate({
            'current': [len(trades)],
            'max': [self._config['max_open_trades']]
        }, headers=['current', 'max'], tablefmt='simple')
        message = "<pre>{}</pre>".format(message)
        self.logger.debug(message)
        self.send_msg(message, parse_mode=ParseMode.HTML)

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
                  "*/forcesell <trade_id>|all:* `Instantly sells the given trade or all trades, regardless of profit`\n" \
                  "*/performance:* `Show performance of each finished trade grouped by pair`\n" \
                  "*/daily <n>:* `Shows profit or loss per day, over the last n days`\n" \
                  "*/count:* `Show number of trades running compared to allowed number of trades`\n" \
                  "*/balance:* `Show account balance per currency`\n" \
                  "*/help:* `This help message`\n" \
                  "*/version:* `Show version`"

        self.send_msg(message, bot=bot)

    @authorized_only
    def _version(self, bot: Bot, update: Update) -> None:
        """
        Handler for /version.
        Show version information
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        self.send_msg('*Version:* `{}`'.format(__version__), bot=bot)

    def send_msg(self, msg: str, bot: Bot = None, parse_mode: ParseMode = ParseMode.MARKDOWN) -> None:
        """
        Send given markdown message
        :param msg: message
        :param bot: alternative bot
        :param parse_mode: telegram parse mode
        :return: None
        """
        if not self.is_enabled():
            return

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
                self.logger.warning(
                    'Got Telegram NetworkError: %s! Trying one more time.',
                    network_err.message
                )
                bot.send_message(
                    self._config['telegram']['chat_id'],
                    text=msg,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup
                )
        except TelegramError as telegram_err:
            self.logger.warning(
                'Got TelegramError: %s! Giving up on that message.',
                telegram_err.message
            )
