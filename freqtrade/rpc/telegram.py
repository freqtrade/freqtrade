import logging
from typing import Any, Callable

from tabulate import tabulate
from telegram import Bot, ParseMode, ReplyKeyboardMarkup, Update
from telegram.error import NetworkError, TelegramError
from telegram.ext import CommandHandler, Updater

from freqtrade.rpc.__init__ import (rpc_status_table,
                                    rpc_trade_status,
                                    rpc_daily_profit,
                                    rpc_trade_statistics,
                                    rpc_balance,
                                    rpc_start,
                                    rpc_stop,
                                    rpc_forcesell,
                                    rpc_performance,
                                    rpc_count,
                                    )

from freqtrade import __version__


# Remove noisy log messages
logging.getLogger('requests.packages.urllib3').setLevel(logging.INFO)
logging.getLogger('telegram').setLevel(logging.INFO)
logger = logging.getLogger(__name__)

_UPDATER: Updater = None
_CONF = {}


def init(config: dict) -> None:
    """
    Initializes this module with the given config,
    registers all known command handlers
    and starts polling for message updates
    :param config: config to use
    :return: None
    """
    global _UPDATER

    _CONF.update(config)
    if not is_enabled():
        return

    _UPDATER = Updater(token=config['telegram']['token'], workers=0)

    # Register command handler and start telegram message polling
    handles = [
        CommandHandler('status', _status),
        CommandHandler('profit', _profit),
        CommandHandler('balance', _balance),
        CommandHandler('start', _start),
        CommandHandler('stop', _stop),
        CommandHandler('forcesell', _forcesell),
        CommandHandler('performance', _performance),
        CommandHandler('daily', _daily),
        CommandHandler('count', _count),
        CommandHandler('help', _help),
        CommandHandler('version', _version),
    ]
    for handle in handles:
        _UPDATER.dispatcher.add_handler(handle)
    _UPDATER.start_polling(
        clean=True,
        bootstrap_retries=-1,
        timeout=30,
        read_latency=60,
    )
    logger.info(
        'rpc.telegram is listening for following commands: %s',
        [h.command for h in handles]
    )


def cleanup() -> None:
    """
    Stops all running telegram threads.
    :return: None
    """
    if not is_enabled():
        return
    _UPDATER.stop()


def is_enabled() -> bool:
    """
    Returns True if the telegram module is activated, False otherwise
    """
    return bool(_CONF['telegram'].get('enabled', False))


def authorized_only(command_handler: Callable[[Bot, Update], None]) -> Callable[..., Any]:
    """
    Decorator to check if the message comes from the correct chat_id
    :param command_handler: Telegram CommandHandler
    :return: decorated function
    """
    def wrapper(*args, **kwargs):
        update = kwargs.get('update') or args[1]

        # Reject unauthorized messages
        chat_id = int(_CONF['telegram']['chat_id'])
        if int(update.message.chat_id) != chat_id:
            logger.info('Rejected unauthorized message from: %s', update.message.chat_id)
            return wrapper

        logger.info('Executing handler: %s for chat_id: %s', command_handler.__name__, chat_id)
        try:
            return command_handler(*args, **kwargs)
        except BaseException:
            logger.exception('Exception occurred within Telegram module')
    return wrapper


@authorized_only
def _status(bot: Bot, update: Update) -> None:
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
        _status_table(bot, update)
        return

    # Fetch open trade
    (error, trades) = rpc_trade_status()
    if error:
        send_msg(trades, bot=bot)
    else:
        for trademsg in trades:
            send_msg(trademsg, bot=bot)


@authorized_only
def _status_table(bot: Bot, update: Update) -> None:
    """
    Handler for /status table.
    Returns the current TradeThread status in table format
    :param bot: telegram bot
    :param update: message update
    :return: None
    """
    # Fetch open trade
    (err, df_statuses) = rpc_status_table()
    if err:
        send_msg(df_statuses, bot=bot)
    else:
        message = tabulate(df_statuses, headers='keys', tablefmt='simple')
        message = "<pre>{}</pre>".format(message)

        send_msg(message, parse_mode=ParseMode.HTML)


@authorized_only
def _daily(bot: Bot, update: Update) -> None:
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
    (error, stats) = rpc_daily_profit(timescale,
                                      _CONF['stake_currency'],
                                      _CONF['fiat_display_currency'])
    if error:
        send_msg(stats, bot=bot)
    else:
        stats = tabulate(stats,
                         headers=[
                             'Day',
                             'Profit {}'.format(_CONF['stake_currency']),
                             'Profit {}'.format(_CONF['fiat_display_currency'])
                         ],
                         tablefmt='simple')
        message = '<b>Daily Profit over the last {} days</b>:\n<pre>{}</pre>'.format(
                  timescale, stats)
        send_msg(message, bot=bot, parse_mode=ParseMode.HTML)


@authorized_only
def _profit(bot: Bot, update: Update) -> None:
    """
    Handler for /profit.
    Returns a cumulative profit statistics.
    :param bot: telegram bot
    :param update: message update
    :return: None
    """
    (error, stats) = rpc_trade_statistics(_CONF['stake_currency'],
                                          _CONF['fiat_display_currency'])
    if error:
        send_msg(stats, bot=bot)
        return

    # Message to display
    markdown_msg = """
*ROI:* Close trades
  ∙ `{profit_closed_coin:.8f} {coin} ({profit_closed_percent:.2f}%)`
  ∙ `{profit_closed_fiat:.3f} {fiat}`
*ROI:* All trades
  ∙ `{profit_all_coin:.8f} {coin} ({profit_all_percent:.2f}%)`
  ∙ `{profit_all_fiat:.3f} {fiat}`

*Total Trade Count:* `{trade_count}`
*First Trade opened:* `{first_trade_date}`
*Latest Trade opened:* `{latest_trade_date}`
*Avg. Duration:* `{avg_duration}`
*Best Performing:* `{best_pair}: {best_rate:.2f}%`
    """.format(
        coin=_CONF['stake_currency'],
        fiat=_CONF['fiat_display_currency'],
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
    send_msg(markdown_msg, bot=bot)


@authorized_only
def _balance(bot: Bot, update: Update) -> None:
    """
    Handler for /balance
    """
    (error, result) = rpc_balance(_CONF['fiat_display_currency'])
    if error:
        send_msg('`All balances are zero.`')
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
    send_msg(output)


@authorized_only
def _start(bot: Bot, update: Update) -> None:
    """
    Handler for /start.
    Starts TradeThread
    :param bot: telegram bot
    :param update: message update
    :return: None
    """
    (error, msg) = rpc_start()
    if error:
        send_msg(msg, bot=bot)


@authorized_only
def _stop(bot: Bot, update: Update) -> None:
    """
    Handler for /stop.
    Stops TradeThread
    :param bot: telegram bot
    :param update: message update
    :return: None
    """
    (error, msg) = rpc_stop()
    send_msg(msg, bot=bot)


# FIX: no test for this!!!!
@authorized_only
def _forcesell(bot: Bot, update: Update) -> None:
    """
    Handler for /forcesell <id>.
    Sells the given trade at current price
    :param bot: telegram bot
    :param update: message update
    :return: None
    """

    trade_id = update.message.text.replace('/forcesell', '').strip()
    (error, message) = rpc_forcesell(trade_id)
    if error:
        send_msg(message, bot=bot)
        return


@authorized_only
def _performance(bot: Bot, update: Update) -> None:
    """
    Handler for /performance.
    Shows a performance statistic from finished trades
    :param bot: telegram bot
    :param update: message update
    :return: None
    """
    (error, trades) = rpc_performance()
    if error:
        send_msg(trades, bot=bot)
        return

    stats = '\n'.join('{index}.\t<code>{pair}\t{profit:.2f}% ({count})</code>'.format(
        index=i + 1,
        pair=trade['pair'],
        profit=trade['profit'],
        count=trade['count']
    ) for i, trade in enumerate(trades))
    message = '<b>Performance:</b>\n{}'.format(stats)
    send_msg(message, parse_mode=ParseMode.HTML)


@authorized_only
def _count(bot: Bot, update: Update) -> None:
    """
    Handler for /count.
    Returns the number of trades running
    :param bot: telegram bot
    :param update: message update
    :return: None
    """
    (error, trades) = rpc_count()
    if error:
        send_msg(trades, bot=bot)
        return

    message = tabulate({
        'current': [len(trades)],
        'max': [_CONF['max_open_trades']]
    }, headers=['current', 'max'], tablefmt='simple')
    message = "<pre>{}</pre>".format(message)
    logger.debug(message)
    send_msg(message, parse_mode=ParseMode.HTML)


@authorized_only
def _help(bot: Bot, update: Update) -> None:
    """
    Handler for /help.
    Show commands of the bot
    :param bot: telegram bot
    :param update: message update
    :return: None
    """
    message = """
*/start:* `Starts the trader`
*/stop:* `Stops the trader`
*/status [table]:* `Lists all open trades`
            *table :* `will display trades in a table`
*/profit:* `Lists cumulative profit from all finished trades`
*/forcesell <trade_id>|all:* `Instantly sells the given trade or all trades, regardless of profit`
*/performance:* `Show performance of each finished trade grouped by pair`
*/daily <n>:* `Shows profit or loss per day, over the last n days`
*/count:* `Show number of trades running compared to allowed number of trades`
*/balance:* `Show account balance per currency`
*/help:* `This help message`
*/version:* `Show version`
    """
    send_msg(message, bot=bot)


@authorized_only
def _version(bot: Bot, update: Update) -> None:
    """
    Handler for /version.
    Show version information
    :param bot: telegram bot
    :param update: message update
    :return: None
    """
    send_msg('*Version:* `{}`'.format(__version__), bot=bot)


def send_msg(msg: str, bot: Bot = None, parse_mode: ParseMode = ParseMode.MARKDOWN) -> None:
    """
    Send given markdown message
    :param msg: message
    :param bot: alternative bot
    :param parse_mode: telegram parse mode
    :return: None
    """
    if not is_enabled():
        return

    bot = bot or _UPDATER.bot

    keyboard = [['/daily', '/profit', '/balance'],
                ['/status', '/status table', '/performance'],
                ['/count', '/start', '/stop', '/help']]

    reply_markup = ReplyKeyboardMarkup(keyboard)

    try:
        try:
            bot.send_message(
                _CONF['telegram']['chat_id'], msg,
                parse_mode=parse_mode, reply_markup=reply_markup
            )
        except NetworkError as network_err:
            # Sometimes the telegram server resets the current connection,
            # if this is the case we send the message again.
            logger.warning(
                'Got Telegram NetworkError: %s! Trying one more time.',
                network_err.message
            )
            bot.send_message(
                _CONF['telegram']['chat_id'], msg,
                parse_mode=parse_mode, reply_markup=reply_markup
            )
    except TelegramError as telegram_err:
        logger.warning('Got TelegramError: %s! Giving up on that message.', telegram_err.message)
