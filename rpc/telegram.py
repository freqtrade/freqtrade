import threading

import arrow
from datetime import timedelta

import logging
from telegram.ext import CommandHandler, Updater
from telegram import ParseMode

from persistence import Trade
from exchange import get_exchange_api
from utils import get_conf

logger = logging.getLogger(__name__)

_lock = threading.Condition()
_updater = None

conf = get_conf()
api_wrapper = get_exchange_api(conf)


class TelegramHandler(object):
    @staticmethod
    def get_updater(conf):
        """
        Returns the current telegram updater instantiates a new one
        :param conf:
        :return: telegram.ext.Updater
        """
        global _updater
        _lock.acquire()
        if not _updater:
            _updater = Updater(token=conf['telegram']['token'], workers=0)
        _lock.release()
        return _updater

    @staticmethod
    def listen():
        """
        Registers all known command handlers and starts polling for message updates
        :return: None
        """
        # Register command handler and start telegram message polling
        handles = [CommandHandler('status', TelegramHandler._status),
                   CommandHandler('profit', TelegramHandler._profit),
                   CommandHandler('start', TelegramHandler._start),
                   CommandHandler('stop', TelegramHandler._stop)]
        for handle in handles:
            TelegramHandler.get_updater(conf).dispatcher.add_handler(handle)
        TelegramHandler.get_updater(conf).start_polling()

    @staticmethod
    def _is_correct_scope(update):
        """
        Checks if it is save to process the given update
        :param update: 
        :return: True if valid else False
        """
        # Only answer to our chat
        return int(update.message.chat_id) == int(conf['telegram']['chat_id'])

    @staticmethod
    def send_msg(markdown_message, bot=None):
        """
        Send given markdown message
        :param markdown_message: message
        :param bot: alternative bot
        :return: None
        """
        if conf['telegram'].get('enabled', False):
            try:
                bot = bot or TelegramHandler.get_updater(conf).bot
                bot.send_message(
                    chat_id=conf['telegram']['chat_id'],
                    text=markdown_message,
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception:
                logger.exception('Exception occurred within telegram api')

    @staticmethod
    def _status(bot, update):
        """
        Handler for /status
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        if not TelegramHandler._is_correct_scope(update):
            return

        # Fetch open trade
        trade = Trade.query.filter(Trade.is_open.is_(True)).first()

        from main import TradeThread
        if not TradeThread.get_instance().is_alive():
            message = '*Status:* `trader stopped`'
        elif not trade:
            message = '*Status:* `no active order`'
        else:
            # calculate profit and send message to user
            current_rate = api_wrapper.get_ticker(trade.pair)['last']
            current_profit = 100 * ((current_rate - trade.open_rate) / trade.open_rate)
            open_orders = api_wrapper.get_open_orders(trade.pair)
            order = open_orders[0] if open_orders else None
            message = """
*Current Pair:* [{pair}](https://bittrex.com/Market/Index?MarketName={pair})
*Open Since:* `{date}`
*Amount:* `{amount}`
*Open Rate:* `{open_rate}`
*Close Rate:* `{close_rate}`
*Current Rate:* `{current_rate}`
*Close Profit:* `{close_profit}%`
*Current Profit:* `{current_profit}%`
*Open Order:* `{open_order}`
                    """.format(
                pair=trade.pair.replace('_', '-'),
                date=arrow.get(trade.open_date).humanize(),
                open_rate=trade.open_rate,
                close_rate=trade.close_rate,
                current_rate=current_rate,
                amount=round(trade.amount, 8),
                close_profit=round(trade.close_profit, 2) if trade.close_profit else 'None',
                current_profit=round(current_profit, 2),
                open_order='{} ({})'.format(
                    order['remaining'],
                    order['type']
                ) if order else None,
            )
        TelegramHandler.send_msg(message, bot=bot)

    @staticmethod
    def _profit(bot, update):
        """
        Handler for /profit
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        if not TelegramHandler._is_correct_scope(update):
            return
        trades = Trade.query.filter(Trade.is_open.is_(False)).all()
        trade_count = len(trades)
        profit_amount = sum((t.close_profit / 100) * t.btc_amount for t in trades)
        profit = sum(t.close_profit for t in trades)
        avg_stake_amount = sum(t.btc_amount for t in trades) / float(trade_count)
        durations_hours = [(t.close_date - t.open_date).total_seconds() / 3600.0 for t in trades]
        avg_duration = sum(durations_hours) / float(len(durations_hours))

        markdown_msg = """
*Total Balance:* `{total_amount} BTC`
*Total Profit:* `{profit_btc} BTC ({profit}%)`
*Trade Count:* `{trade_count}`
*First Action:* `{first_trade_date}`
*Latest Action:* `{latest_trade_date}`
*Avg. Stake Amount:* `{avg_open_amount} BTC`
*Avg. Duration:* `{avg_duration}` 
    """.format(
            total_amount=round(api_wrapper.get_balance('BTC'), 8),
            profit_btc=round(profit_amount, 8),
            profit=round(profit, 2),
            trade_count=trade_count,
            first_trade_date=arrow.get(trades[0].open_date).humanize(),
            latest_trade_date=arrow.get(trades[-1].open_date).humanize(),
            avg_open_amount=round(avg_stake_amount, 8),
            avg_duration=str(timedelta(hours=avg_duration)).split('.')[0],
        )
        TelegramHandler.send_msg(markdown_msg, bot=bot)

    @staticmethod
    def _start(bot, update):
        """
        Handler for /start
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        if not TelegramHandler._is_correct_scope(update):
            return
        from main import TradeThread
        if TradeThread.get_instance().is_alive():
            TelegramHandler.send_msg('*Status:* `already running`', bot=bot)
            return
        else:
            TradeThread.get_instance(recreate=True).start()

    @staticmethod
    def _stop(bot, update):
        """
        Handler for /stop
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        if not TelegramHandler._is_correct_scope(update):
            return
        from main import TradeThread
        if TradeThread.get_instance().is_alive():
            TradeThread.stop()
        else:
            TelegramHandler.send_msg('*Status:* `already stopped`', bot=bot)
