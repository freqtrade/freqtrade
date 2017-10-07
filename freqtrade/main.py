#!/usr/bin/env python
import copy
import json
import logging
import time
import traceback
from datetime import datetime
from typing import Dict, Optional

from jsonschema import validate

from freqtrade import __version__, exchange, persistence
from freqtrade.analyze import get_buy_signal
from freqtrade.misc import CONF_SCHEMA, State, get_state, update_state
from freqtrade.persistence import Trade
from freqtrade.rpc import telegram

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_CONF = {}


def _process() -> None:
    """
    Queries the persistence layer for open trades and handles them,
    otherwise a new trade is created.
    :return: None
    """
    try:
        # Query trades from persistence layer
        trades = Trade.query.filter(Trade.is_open.is_(True)).all()
        if len(trades) < _CONF['max_open_trades']:
            try:
                # Create entity and execute trade
                trade = create_trade(float(_CONF['stake_amount']))
                if trade:
                    Trade.session.add(trade)
                else:
                    logging.info('Got no buy signal...')
            except ValueError:
                logger.exception('Unable to create trade')

        for trade in trades:
            # Check if there is already an open order for this trade
            orders = exchange.get_open_orders(trade.pair)
            orders = [o for o in orders if o['id'] == trade.open_order_id]
            if orders:
                logger.info('There is an open order for: %s', orders[0])
            else:
                # Update state
                trade.open_order_id = None
                # Check if this trade can be closed
                if not close_trade_if_fulfilled(trade):
                    # Check if we can sell our current pair
                    handle_trade(trade)
        Trade.session.flush()
    except (ConnectionError, json.JSONDecodeError) as error:
        msg = 'Got {} in _process()'.format(error.__class__.__name__)
        logger.exception(msg)


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
        logger.info('No open orders found and trade is fulfilled. Marking %s as closed ...', trade)
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
        trade.exchange,
        trade.pair.replace('_', '/'),
        exchange.get_pair_detail_url(trade.pair),
        trade.close_rate,
        round(profit, 2)
    )
    logger.info(message)
    telegram.send_msg(message)


def should_sell(trade: Trade, current_rate: float, current_time: datetime) -> bool:
    """
    Based an earlier trade and current price and configuration, decides whether bot should sell
    :return True if bot should sell at current rate
    """
    current_profit = (current_rate - trade.open_rate) / trade.open_rate

    if 'stoploss' in _CONF and current_profit < float(_CONF['stoploss']):
        logger.debug('Stop loss hit.')
        return True

    for duration, threshold in sorted(_CONF['minimal_roi'].items()):
        duration, threshold = float(duration), float(threshold)
        # Check if time matches and current rate is above threshold
        time_diff = (current_time - trade.open_date).total_seconds() / 60
        if time_diff > duration and current_profit > threshold:
            return True

    logger.debug('Threshold not reached. (cur_profit: %1.2f%%)', current_profit * 100.0)
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

        current_rate = exchange.get_ticker(trade.pair)['bid']
        if should_sell(trade, current_rate, datetime.utcnow()):
            execute_sell(trade, current_rate)
            return

    except ValueError:
        logger.exception('Unable to handle open order')


def get_target_bid(ticker: Dict[str, float]) -> float:
    """ Calculates bid target between current ask price and last price """
    if ticker['ask'] < ticker['last']:
        return ticker['ask']
    balance = _CONF['bid_strategy']['ask_last_balance']
    return ticker['ask'] + balance * (ticker['last'] - ticker['ask'])


def create_trade(stake_amount: float) -> Optional[Trade]:
    """
    Checks the implemented trading indicator(s) for a randomly picked pair,
    if one pair triggers the buy_signal a new trade record gets created
    :param stake_amount: amount of btc to spend
    """
    logger.info('Creating new trade with stake_amount: %f ...', stake_amount)
    whitelist = copy.deepcopy(_CONF['exchange']['pair_whitelist'])
    # Check if stake_amount is fulfilled
    if exchange.get_balance(_CONF['stake_currency']) < stake_amount:
        raise ValueError(
            'stake amount is not fulfilled (currency={}'.format(_CONF['stake_currency'])
        )

    # Remove currently opened and latest pairs from whitelist
    for trade in Trade.query.filter(Trade.is_open.is_(True)).all():
        if trade.pair in whitelist:
            whitelist.remove(trade.pair)
            logger.debug('Ignoring %s in pair whitelist', trade.pair)
    if not whitelist:
        raise ValueError('No pair in whitelist')

    # Pick pair based on StochRSI buy signals
    for _pair in whitelist:
        if get_buy_signal(_pair):
            pair = _pair
            break
    else:
        return None

    open_rate = get_target_bid(exchange.get_ticker(pair))
    amount = stake_amount / open_rate
    order_id = exchange.buy(pair, open_rate, amount)

    # Create trade entity and return
    message = '*{}:* Buying [{}]({}) at rate `{:f}`'.format(
        exchange.EXCHANGE.name.upper(),
        pair.replace('_', '/'),
        exchange.get_pair_detail_url(pair),
        open_rate
    )
    logger.info(message)
    telegram.send_msg(message)
    return Trade(pair=pair,
                 stake_amount=stake_amount,
                 open_rate=open_rate,
                 open_date=datetime.utcnow(),
                 amount=amount,
                 exchange=exchange.EXCHANGE.name.upper(),
                 open_order_id=order_id,
                 is_open=True)


def init(config: dict, db_url: Optional[str] = None) -> None:
    """
    Initializes all modules and updates the config
    :param config: config as dict
    :param db_url: database connector string for sqlalchemy (Optional)
    :return: None
    """
    # Initialize all modules
    telegram.init(config)
    persistence.init(config, db_url)
    exchange.init(config)

    # Set initial application state
    initial_state = config.get('initial_state')
    if initial_state:
        update_state(State[initial_state.upper()])
    else:
        update_state(State.STOPPED)


def app(config: dict) -> None:
    """
    Main loop which handles the application state
    :param config: config as dict
    :return: None
    """
    logger.info('Starting freqtrade %s', __version__)
    init(config)
    try:
        old_state = get_state()
        logger.info('Initial State: %s', old_state)
        telegram.send_msg('*Status:* `{}`'.format(old_state.name.lower()))
        while True:
            new_state = get_state()
            # Log state transition
            if new_state != old_state:
                telegram.send_msg('*Status:* `{}`'.format(new_state.name.lower()))
                logging.info('Changing state to: %s', new_state.name)

            if new_state == State.STOPPED:
                time.sleep(1)
            elif new_state == State.RUNNING:
                _process()
                # We need to sleep here because otherwise we would run into bittrex rate limit
                time.sleep(exchange.EXCHANGE.sleep_time)
            old_state = new_state
    except RuntimeError:
        telegram.send_msg('*Status:* Got RuntimeError: ```\n{}\n```'.format(traceback.format_exc()))
        logger.exception('RuntimeError. Trader stopped!')
    finally:
        telegram.send_msg('*Status:* `Trader has stopped`')


def main():
    """
    Loads and validates the config and starts the main loop
    :return: None
    """
    global _CONF
    with open('config.json') as file:
        _CONF = json.load(file)
        validate(_CONF, CONF_SCHEMA)
        app(_CONF)


if __name__ == '__main__':
    main()
