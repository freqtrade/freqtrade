#!/usr/bin/env python
import logging
import threading
import time
import traceback
from datetime import datetime
from json import JSONDecodeError
from typing import Optional

from requests import ConnectionError
from wrapt import synchronized
from analyze import get_buy_signal
from persistence import Trade, Session
from exchange import get_exchange_api, Exchange
from rpc.telegram import TelegramHandler
from utils import get_conf

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__author__ = "gcarq"
__copyright__ = "gcarq 2017"
__license__ = "GPLv3"
__version__ = "0.8.0"


CONFIG = get_conf()
api_wrapper = get_exchange_api(CONFIG)


class TradeThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self._should_stop = False

    def stop(self) -> None:
        """ stops the trader thread """
        self._should_stop = True

    def run(self) -> None:
        """
        Threaded main function
        :return: None
        """
        try:
            TelegramHandler.send_msg('*Status:* `trader started`')
            logger.info('Trader started')
            while not self._should_stop:
                try:
                    self._process()
                except (ConnectionError, JSONDecodeError, ValueError) as error:
                    msg = 'Got {} during _process()'.format(error.__class__.__name__)
                    logger.exception(msg)
                finally:
                    Session.flush()
                    time.sleep(25)
        except (RuntimeError, JSONDecodeError):
            TelegramHandler.send_msg('*Status:* Got RuntimeError: ```\n{}\n```'.format(traceback.format_exc()))
            logger.exception('RuntimeError. Stopping trader ...')
        finally:
            TelegramHandler.send_msg('*Status:* `Trader has stopped`')

    @staticmethod
    def _process() -> None:
        """
        Queries the persistence layer for open trades and handles them,
        otherwise a new trade is created.
        :return: None
        """
        # Query trades from persistence layer
        trades = Trade.query.filter(Trade.is_open.is_(True)).all()
        if len(trades) < CONFIG['max_open_trades']:
            try:
                # Create entity and execute trade
                trade = create_trade(float(CONFIG['stake_amount']), api_wrapper.exchange)
                if trade:
                    Session.add(trade)
                else:
                    logging.info('Got no buy signal...')
            except ValueError:
                logger.exception('Unable to create trade')

        for trade in trades:
            # Check if there is already an open order for this trade
            orders = api_wrapper.get_open_orders(trade.pair)
            orders = [o for o in orders if o['id'] == trade.open_order_id]
            if orders:
                msg = 'There exists an open order for {}: Order(total={}, remaining={}, type={}, id={})' \
                    .format(
                        trade,
                        round(orders[0]['amount'], 8),
                        round(orders[0]['remaining'], 8),
                        orders[0]['type'],
                        orders[0]['id'])
                logger.info(msg)
                continue

            # Update state
            trade.open_order_id = None
            # Check if this trade can be marked as closed
            if close_trade_if_fulfilled(trade):
                logger.info('No open orders found and trade is fulfilled. Marking %s as closed ...', trade)
                continue

            # Check if we can sell our current pair
            handle_trade(trade)

# Initial stopped TradeThread instance
_instance = TradeThread()


@synchronized
def get_instance(recreate: bool=False) -> TradeThread:
    """
    Get the current instance of this thread. This is a singleton.
    :param recreate: Must be True if you want to start the instance
    :return: TradeThread instance
    """
    global _instance
    if recreate and not _instance.is_alive():
        logger.debug('Creating thread instance...')
        _instance = TradeThread()
    return _instance


def close_trade_if_fulfilled(trade: Trade) -> bool:
    """
    Checks if the trade is closable, and if so it is being closed.
    :param trade: Trade
    :return: True if trade has been closed else False
    """
    # If we don't have an open order and the close rate is already set,
    # we can close this trade.
    if trade.close_profit and trade.close_date and trade.close_rate and not trade.open_order_id:
        trade.is_open = False
        Session.flush()
        return True
    return False


def handle_trade(trade: Trade) -> None:
    """
    Sells the current pair if the threshold is reached and updates the trade record.
    :return: None
    """
    try:
        if not trade.is_open:
            raise ValueError('attempt to handle closed trade: {}'.format(trade))

        logger.debug('Handling open trade %s ...', trade)
        # Get current rate
        current_rate = api_wrapper.get_ticker(trade.pair)['bid']
        current_profit = 100 * ((current_rate - trade.open_rate) / trade.open_rate)

        # Get available balance
        currency = trade.pair.split('_')[1]
        balance = api_wrapper.get_balance(currency)

        for duration, threshold in sorted(CONFIG['minimal_roi'].items()):
            duration, threshold = float(duration), float(threshold)
            # Check if time matches and current rate is above threshold
            time_diff = (datetime.utcnow() - trade.open_date).total_seconds() / 60
            if time_diff > duration and current_rate > (1 + threshold) * trade.open_rate:
                # Execute sell
                profit = trade.exec_sell_order(current_rate, balance)
                message = '*{}:* Selling [{}]({}) at rate `{:f} (profit: {}%)`'.format(
                    trade.exchange.name,
                    trade.pair.replace('_', '/'),
                    api_wrapper.get_pair_detail_url(trade.pair),
                    trade.close_rate,
                    round(profit, 2)
                )
                logger.info(message)
                TelegramHandler.send_msg(message)
                return
        else:
            logger.debug('Threshold not reached. (cur_profit: %1.2f%%)', current_profit)
    except ValueError:
        logger.exception('Unable to handle open order')


def create_trade(stake_amount: float, exchange: Exchange) -> Optional[Trade]:
    """
    Checks the implemented trading indicator(s) for a randomly picked pair,
    if one pair triggers the buy_signal a new trade record gets created
    :param stake_amount: amount of btc to spend
    :param exchange: exchange to use
    """
    logger.info('Creating new trade with stake_amount: %f ...', stake_amount)
    whitelist = CONFIG[exchange.name.lower()]['pair_whitelist']
    # Check if btc_amount is fulfilled
    if api_wrapper.get_balance(CONFIG['stake_currency']) < stake_amount:
        raise ValueError('stake amount is not fulfilled (currency={}'.format(CONFIG['stake_currency']))

    # Remove currently opened and latest pairs from whitelist
    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    latest_trade = Trade.query.filter(Trade.is_open.is_(False)).order_by(Trade.id.desc()).first()
    if latest_trade:
        trades.append(latest_trade)
    for trade in trades:
        if trade.pair in whitelist:
            whitelist.remove(trade.pair)
            logger.debug('Ignoring %s in pair whitelist', trade.pair)
    if not whitelist:
        raise ValueError('No pair in whitelist')

    # Pick pair based on StochRSI buy signals
    for p in whitelist:
        if get_buy_signal(p):
            pair = p
            break
    else:
        return None

    open_rate = api_wrapper.get_ticker(pair)['ask']
    amount = stake_amount / open_rate
    exchange = exchange
    order_id = api_wrapper.buy(pair, open_rate, amount)

    # Create trade entity and return
    message = '*{}:* Buying [{}]({}) at rate `{:f}`'.format(
        exchange.name,
        pair.replace('_', '/'),
        api_wrapper.get_pair_detail_url(pair),
        open_rate
    )
    logger.info(message)
    TelegramHandler.send_msg(message)
    return Trade(pair=pair,
                 btc_amount=stake_amount,
                 open_rate=open_rate,
                 amount=amount,
                 exchange=exchange,
                 open_order_id=order_id)


if __name__ == '__main__':
    logger.info('Starting freqtrade %s', __version__)
    TelegramHandler.listen()
    while True:
        time.sleep(0.5)
