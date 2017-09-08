#!/usr/bin/env python
import enum
import json
import logging
import time
import traceback
from datetime import datetime
from json import JSONDecodeError
from typing import Optional

from jsonschema import validate
from requests import ConnectionError
from wrapt import synchronized

import exchange
import persistence
from rpc import telegram
from analyze import get_buy_signal
from persistence import Trade
from misc import conf_schema

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__author__ = "gcarq"
__copyright__ = "gcarq 2017"
__license__ = "GPLv3"
__version__ = "0.8.0"


class State(enum.Enum):
    RUNNING = 0
    PAUSED = 1
    TERMINATE = 2


_conf = {}
_cur_state = State.RUNNING


@synchronized
def update_state(state: State) -> None:
    """
    Updates the application state
    :param state: new state
    :return: None
    """
    global _cur_state
    _cur_state = state


@synchronized
def get_state() -> State:
    """
    Gets the current application state
    :return:
    """
    return _cur_state


def _process() -> None:
    """
    Queries the persistence layer for open trades and handles them,
    otherwise a new trade is created.
    :return: None
    """
    # Query trades from persistence layer
    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    if len(trades) < _conf['max_open_trades']:
        try:
            # Create entity and execute trade
            trade = create_trade(float(_conf['stake_amount']), exchange.cur_exchange)
            if trade:
                Trade.session.add(trade)
            else:
                logging.info('Got no buy signal...')
        except ValueError:
            logger.exception('Unable to create trade')

    for trade in trades:
        if close_trade_if_fulfilled(trade):
            logger.info('No open orders found and trade is fulfilled. Marking %s as closed ...', trade)

    for trade in filter(lambda t: t.is_open, trades):
        # Check if there is already an open order for this trade
        orders = exchange.get_open_orders(trade.pair)
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
        else:
            # Update state
            trade.open_order_id = None
            # Check if we can sell our current pair
            handle_trade(trade)


def close_trade_if_fulfilled(trade: Trade) -> bool:
    """
    Checks if the trade is closable, and if so it is being closed.
    :param trade: Trade
    :return: True if trade has been closed else False
    """
    # If we don't have an open order and the close rate is already set,
    # we can close this trade.
    if trade.close_profit is not None \
            and trade.close_date is not None \
            and trade.close_rate is not None \
            and trade.open_order_id is None:
        trade.is_open = False
        return True
    return False


def execute_sell(trade: Trade, current_rate: float) -> None:
    """
    Executes a sell for the given trade and current rate
    :param trade: Trade instance
    :param current_rate: current rate
    :return: None
    """
    # Get available balance
    currency = trade.pair.split('_')[1]
    balance = exchange.get_balance(currency)

    profit = trade.exec_sell_order(current_rate, balance)
    message = '*{}:* Selling [{}]({}) at rate `{:f} (profit: {}%)`'.format(
        trade.exchange.name,
        trade.pair.replace('_', '/'),
        exchange.get_pair_detail_url(trade.pair),
        trade.close_rate,
        round(profit, 2)
    )
    logger.info(message)
    telegram.send_msg(message)


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
        current_rate = exchange.get_ticker(trade.pair)['bid']
        current_profit = 100.0 * ((current_rate - trade.open_rate) / trade.open_rate)

        if 'stoploss' in _conf and current_profit < float(_conf['stoploss']) * 100.0:
            logger.debug('Stop loss hit.')
            execute_sell(trade, current_rate)
            return

        for duration, threshold in sorted(_conf['minimal_roi'].items()):
            duration, threshold = float(duration), float(threshold)
            # Check if time matches and current rate is above threshold
            time_diff = (datetime.utcnow() - trade.open_date).total_seconds() / 60
            if time_diff > duration and current_rate > (1 + threshold) * trade.open_rate:
                execute_sell(trade, current_rate)
                return
        else:
            logger.debug('Threshold not reached. (cur_profit: %1.2f%%)', current_profit)
    except ValueError:
        logger.exception('Unable to handle open order')


def create_trade(stake_amount: float, _exchange: exchange.Exchange) -> Optional[Trade]:
    """
    Checks the implemented trading indicator(s) for a randomly picked pair,
    if one pair triggers the buy_signal a new trade record gets created
    :param stake_amount: amount of btc to spend
    :param _exchange: exchange to use
    """
    logger.info('Creating new trade with stake_amount: %f ...', stake_amount)
    whitelist = _conf[_exchange.name.lower()]['pair_whitelist']
    # Check if btc_amount is fulfilled
    if exchange.get_balance(_conf['stake_currency']) < stake_amount:
        raise ValueError('stake amount is not fulfilled (currency={}'.format(_conf['stake_currency']))

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

    open_rate = exchange.get_ticker(pair)['ask']
    amount = stake_amount / open_rate
    order_id = exchange.buy(pair, open_rate, amount)

    # Create trade entity and return
    message = '*{}:* Buying [{}]({}) at rate `{:f}`'.format(
        _exchange.name,
        pair.replace('_', '/'),
        exchange.get_pair_detail_url(pair),
        open_rate
    )
    logger.info(message)
    telegram.send_msg(message)
    return Trade(pair=pair,
                 btc_amount=stake_amount,
                 open_rate=open_rate,
                 open_date=datetime.utcnow(),
                 amount=amount,
                 exchange=_exchange,
                 open_order_id=order_id,
                 is_open=True)


def init(config: dict, db_url: Optional[str]=None) -> None:
    """
    Initializes all modules and updates the config
    :param config: config as dict
    :param db_url: database connector string for sqlalchemy (Optional)
    :return: None
    """
    global _conf

    # Initialize all modules
    telegram.init(config)
    persistence.init(config, db_url)
    exchange.init(config)
    _conf.update(config)


def app(config: dict) -> None:

    logger.info('Starting freqtrade %s', __version__)
    init(config)

    try:
        telegram.send_msg('*Status:* `trader started`')
        logger.info('Trader started')
        while True:
            state = get_state()
            if state == State.TERMINATE:
                return
            elif state == State.PAUSED:
                time.sleep(1)
            elif state == State.RUNNING:
                try:
                    _process()
                    Trade.session.flush()
                except (ConnectionError, JSONDecodeError, ValueError) as error:
                    msg = 'Got {} during _process()'.format(error.__class__.__name__)
                    logger.exception(msg)
                finally:
                    time.sleep(25)
    except (RuntimeError, JSONDecodeError):
        telegram.send_msg(
            '*Status:* Got RuntimeError: ```\n{}\n```'.format(traceback.format_exc())
        )
        logger.exception('RuntimeError. Stopping trader ...')
    finally:
        telegram.send_msg('*Status:* `Trader has stopped`')


if __name__ == '__main__':
    with open('config.json') as file:
        conf = json.load(file)
        validate(conf, conf_schema)
        app(conf)

