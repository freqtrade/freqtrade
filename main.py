#!/usr/bin/env python

import logging
import random
import threading
import time
import traceback
from datetime import datetime

from wrapt import synchronized

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from persistence import Trade, Session
from exchange import get_exchange_api
from rpc.telegram import TelegramHandler
from utils import get_conf

__author__ = "gcarq"
__copyright__ = "gcarq 2017"
__license__ = "custom"
__version__ = "0.5.1"



conf = get_conf()
api_wrapper = get_exchange_api(conf)


@synchronized
def get_instance(recreate=False):
    """
    Get the current instance of this thread. This is a singleton.
    :param recreate: Must be True if you want to start the instance
    :return: TradeThread instance
    """
    global _instance, _should_stop
    if recreate and not _instance.is_alive():
        logger.debug('Creating TradeThread instance')
        _should_stop = False
        _instance = TradeThread()
    return _instance


def stop_instance():
    global _should_stop
    _should_stop = True


class TradeThread(threading.Thread):
    def run(self):
        """
        Threaded main function
        :return: None
        """
        try:
            TelegramHandler.send_msg('*Status:* `trader started`')
            logger.info('Trader started')
            while not _should_stop:
                try:
                    self._process()
                except ValueError:
                    logger.exception('ValueError')
                finally:
                    Session.flush()
                    time.sleep(25)
        except RuntimeError:
            TelegramHandler.send_msg('*Status:* Got RuntimeError: ```\n{}\n```'.format(traceback.format_exc()))
            logger.exception('RuntimeError. Stopping trader ...')
        finally:
            TelegramHandler.send_msg('*Status:* `Trader has stopped`')

    @staticmethod
    def _process():
        """
        Queries the persistence layer for new trades and handles them
        :return: None
        """
        # Query trades from persistence layer
        trade = Trade.query.filter(Trade.is_open.is_(True)).first()
        if not trade:
            # Create entity and execute trade
            Session.add(create_trade(float(conf['stake_amount']), api_wrapper.exchange))
            return

        # Check if there is already an open order for this trade
        orders = api_wrapper.get_open_orders(trade.pair)
        orders = [o for o in orders if o['id'] == trade.open_order_id]
        if orders:
            msg = 'There exists an open order for this trade: (total: {}, remaining: {}, type: {}, id: {})' \
                .format(round(orders[0]['amount'], 8),
                        round(orders[0]['remaining'], 8),
                        orders[0]['type'],
                        orders[0]['id'])
            logger.info(msg)
            return

        # Update state
        trade.open_order_id = None
        # Check if this trade can be marked as closed
        if close_trade_if_fulfilled(trade):
            logger.info('No open orders found and trade is fulfilled. Marking as closed ...')
            return

        # Check if we can sell our current pair
        handle_trade(trade)

# Initial stopped TradeThread instance
_instance = TradeThread()
_should_stop = False


def close_trade_if_fulfilled(trade):
    """
    Checks if the trade is closable, and if so it is being closed.
    :param trade: Trade
    :return: True if trade has been closed else False
    """
    # If we don't have an open order and the close rate is already set,
    # we can close this trade.
    if trade.close_profit and trade.close_date and trade.close_rate and not trade.open_order_id:
        trade.is_open = False
        return True
    return False


def handle_trade(trade):
    """
    Sells the current pair if the threshold is reached and updates the trade record.
    :return: None
    """
    try:
        if not trade.is_open:
            raise ValueError('attempt to handle closed trade: {}'.format(trade))

        logger.debug('Handling open trade {} ...'.format(trade))
        # Get current rate
        current_rate = api_wrapper.get_ticker(trade.pair)['bid']
        current_profit = 100 * ((current_rate - trade.open_rate) / trade.open_rate)

        # Get available balance
        currency = trade.pair.split('_')[1]
        balance = api_wrapper.get_balance(currency)

        for duration, threshold in sorted(conf['trade_thresholds'].items()):
            duration, threshold = float(duration), float(threshold)
            # Check if time matches and current rate is above threshold
            time_diff = (datetime.utcnow() - trade.open_date).total_seconds() / 60
            if time_diff > duration and current_rate > (1 + threshold) * trade.open_rate:

                # Execute sell and update trade record
                order_id = api_wrapper.sell(trade.pair, current_rate, balance)
                trade.close_rate = current_rate
                trade.close_profit = current_profit
                trade.close_date = datetime.utcnow()
                trade.open_order_id = order_id

                message = '*{}:* Selling {} at rate `{:f} (profit: {}%)`'.format(
                    trade.exchange.name,
                    trade.pair.replace('_', '/'),
                    trade.close_rate,
                    round(current_profit, 2)
                )
                logger.info(message)
                TelegramHandler.send_msg(message)
                return
        else:
            logger.debug('Threshold not reached. (cur_profit: {}%)'.format(round(current_profit, 2)))
    except ValueError:
        logger.exception('Unable to handle open order')


def create_trade(stake_amount: float, exchange):
    """
    Creates a new trade record with a random pair
    :param stake_amount: amount of btc to spend
    :param exchange: exchange to use
    """
    whitelist = conf[exchange.name.lower()]['pair_whitelist']
    # Check if btc_amount is fulfilled
    if api_wrapper.get_balance('BTC') < stake_amount:
        raise ValueError('BTC amount is not fulfilled')

    # Remove latest trade pair from whitelist
    latest_trade = Trade.query.order_by(Trade.id.desc()).first()
    if latest_trade and latest_trade.pair in whitelist:
        whitelist.remove(latest_trade.pair)
        logger.debug('Ignoring {} in pair whitelist'.format(latest_trade.pair))
    if not whitelist:
        raise ValueError('No pair in whitelist')

    # Pick random pair and execute trade
    idx = random.randint(0, len(whitelist) - 1)
    pair = whitelist[idx]
    open_rate = api_wrapper.get_ticker(pair)['ask']
    amount = stake_amount / open_rate
    exchange = exchange
    order_id = api_wrapper.buy(pair, open_rate, amount)

    # Create trade entity and return
    message = '*{}:* Buying {} at rate `{:f}`'.format(exchange.name, pair.replace('_', '/'), open_rate)
    logger.info(message)
    TelegramHandler.send_msg(message)
    return Trade(pair=pair,
                 btc_amount=stake_amount,
                 open_rate=open_rate,
                 amount=amount,
                 exchange=exchange,
                 open_order_id=order_id)


if __name__ == '__main__':
    logger.info('Starting marginbot {}'.format(__version__))
    TelegramHandler.listen()
    while True:
        time.sleep(0.5)
