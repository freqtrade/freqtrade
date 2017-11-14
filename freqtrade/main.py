#!/usr/bin/env python3
import copy
import json
import logging
import time
import traceback
from datetime import datetime
from signal import signal, SIGINT, SIGABRT, SIGTERM
from typing import Dict, Optional, List

import requests
from cachetools import cached, TTLCache
from jsonschema import validate

from freqtrade import __version__, exchange, persistence
from freqtrade.analyze import get_signal, SignalType
from freqtrade.misc import (
    CONF_SCHEMA, State, get_state, update_state, build_arg_parser, throttle, FreqtradeException
)
from freqtrade.persistence import Trade
from freqtrade.rpc import telegram

logger = logging.getLogger('freqtrade')

_CONF = {}


def refresh_whitelist(whitelist: Optional[List[str]] = None) -> None:
    """
    Check wallet health and remove pair from whitelist if necessary
    :param whitelist: a new whitelist (optional)
    :return: None
    """
    whitelist = whitelist or _CONF['exchange']['pair_whitelist']

    sanitized_whitelist = []
    health = exchange.get_wallet_health()
    for status in health:
        pair = '{}_{}'.format(_CONF['stake_currency'], status['Currency'])
        if pair not in whitelist:
            continue
        if status['IsActive']:
            sanitized_whitelist.append(pair)
        else:
            logger.info(
                'Ignoring %s from whitelist (reason: %s).',
                pair, status.get('Notice') or 'wallet is not active'
            )
    if _CONF['exchange']['pair_whitelist'] != sanitized_whitelist:
        logger.debug('Using refreshed pair whitelist: %s ...', sanitized_whitelist)
        _CONF['exchange']['pair_whitelist'] = sanitized_whitelist


def _process(dynamic_whitelist: Optional[bool] = False) -> bool:
    """
    Queries the persistence layer for open trades and handles them,
    otherwise a new trade is created.
    :param: dynamic_whitelist: True is a dynamic whitelist should be generated (optional)
    :return: True if a trade has been created or closed, False otherwise
    """
    state_changed = False
    try:
        # Refresh whitelist based on wallet maintenance
        refresh_whitelist(
            gen_pair_whitelist(_CONF['stake_currency']) if dynamic_whitelist else None
        )
        # Query trades from persistence layer
        trades = Trade.query.filter(Trade.is_open.is_(True)).all()
        if len(trades) < _CONF['max_open_trades']:
            try:
                # Create entity and execute trade
                trade = create_trade(float(_CONF['stake_amount']))
                if trade:
                    Trade.session.add(trade)
                    state_changed = True
                else:
                    logger.info(
                        'Checked all whitelisted currencies. '
                        'Found no suitable entry positions for buying. Will keep looking ...'
                    )
            except FreqtradeException as e:
                logger.warning('Unable to create trade: %s', e)

        for trade in trades:
            # Get order details for actual price per unit
            if trade.open_order_id:
                # Update trade with order values
                logger.info('Got open order for %s', trade)
                trade.update(exchange.get_order(trade.open_order_id))

            if not close_trade_if_fulfilled(trade):
                # Check if we can sell our current pair
                state_changed = handle_trade(trade) or state_changed

            Trade.session.flush()
    except (requests.exceptions.RequestException, json.JSONDecodeError) as error:
        logger.warning(
            'Got %s in _process(), retrying in 30 seconds...',
            error
        )
        time.sleep(30)
    except RuntimeError:
        telegram.send_msg('*Status:* Got RuntimeError:\n```\n{traceback}```{hint}'.format(
            traceback=traceback.format_exc(),
            hint='Issue `/start` if you think it is safe to restart.'
        ))
        logger.exception('Got RuntimeError. Stopping trader ...')
        update_state(State.STOPPED)
    return state_changed


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
        logger.info(
            'Marking %s as closed as the trade is fulfilled and found no open orders for it.',
            trade
        )
        return True
    return False


def execute_sell(trade: Trade, limit: float) -> None:
    """
    Executes a limit sell for the given trade and limit
    :param trade: Trade instance
    :param limit: limit rate for the sell order
    :return: None
    """
    # Execute sell and update trade record
    order_id = exchange.sell(str(trade.pair), limit, trade.amount)
    trade.open_order_id = order_id

    fmt_exp_profit = round(trade.calc_profit(limit) * 100, 2)
    message = '*{}:* Selling [{}]({}) with limit `{:.8f} (profit: ~{:.2f}%)`'.format(
        trade.exchange,
        trade.pair.replace('_', '/'),
        exchange.get_pair_detail_url(trade.pair),
        limit,
        fmt_exp_profit
    )
    logger.info(message)
    telegram.send_msg(message)


def should_sell(trade: Trade, current_rate: float, current_time: datetime) -> bool:
    """
    Based an earlier trade and current price and configuration, decides whether bot should sell
    :return True if bot should sell at current rate
    """
    current_profit = trade.calc_profit(current_rate)
    if 'stoploss' in _CONF and current_profit < float(_CONF['stoploss']):
        logger.debug('Stop loss hit.')
        return True

    for duration, threshold in sorted(_CONF['minimal_roi'].items()):
        # Check if time matches and current rate is above threshold
        time_diff = (current_time - trade.open_date).total_seconds() / 60
        if time_diff > float(duration) and current_profit > threshold:
            return True

    logger.debug('Threshold not reached. (cur_profit: %1.2f%%)', current_profit * 100.0)
    return False


def handle_trade(trade: Trade) -> bool:
    """
    Sells the current pair if the threshold is reached and updates the trade record.
    :return: True if trade has been sold, False otherwise
    """
    if not trade.is_open:
        raise ValueError('attempt to handle closed trade: {}'.format(trade))

    logger.debug('Handling %s ...', trade)
    current_rate = exchange.get_ticker(trade.pair)['bid']
    if should_sell(trade, current_rate, datetime.utcnow()) or get_signal(trade.pair, SignalType.SELL):
        execute_sell(trade, current_rate)
        return True
    return False


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
    logger.info(
        'Checking buy signals to create a new trade with stake_amount: %f ...',
        stake_amount
    )
    whitelist = copy.deepcopy(_CONF['exchange']['pair_whitelist'])
    # Check if stake_amount is fulfilled
    if exchange.get_balance(_CONF['stake_currency']) < stake_amount:
        raise FreqtradeException(
            'stake amount is not fulfilled (currency={})'.format(_CONF['stake_currency'])
        )

    # Remove currently opened and latest pairs from whitelist
    for trade in Trade.query.filter(Trade.is_open.is_(True)).all():
        if trade.pair in whitelist:
            whitelist.remove(trade.pair)
            logger.debug('Ignoring %s in pair whitelist', trade.pair)
    if not whitelist:
        raise FreqtradeException('No pair in whitelist')

    # Pick pair based on StochRSI buy signals
    for _pair in whitelist:
        if get_signal(_pair, SignalType.BUY):
            pair = _pair
            break
    else:
        return None

    # Calculate amount and subtract fee
    fee = exchange.get_fee()
    buy_limit = get_target_bid(exchange.get_ticker(pair))
    amount = (1 - fee) * stake_amount / buy_limit

    order_id = exchange.buy(pair, buy_limit, amount)
    # Create trade entity and return
    message = '*{}:* Buying [{}]({}) with limit `{:.8f}`'.format(
        exchange.get_name().upper(),
        pair.replace('_', '/'),
        exchange.get_pair_detail_url(pair),
        buy_limit
    )
    logger.info(message)
    telegram.send_msg(message)
    # Fee is applied twice because we make a LIMIT_BUY and LIMIT_SELL
    return Trade(pair=pair,
                 stake_amount=stake_amount,
                 amount=amount,
                 fee=fee * 2,
                 open_rate=buy_limit,
                 open_date=datetime.utcnow(),
                 exchange=exchange.get_name().upper(),
                 open_order_id=order_id)


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

    # Register signal handlers
    for sig in (SIGINT, SIGTERM, SIGABRT):
        signal(sig, cleanup)


@cached(TTLCache(maxsize=1, ttl=1800))
def gen_pair_whitelist(base_currency: str, topn: int = 20, key: str = 'BaseVolume') -> List[str]:
    """
    Updates the whitelist with with a dynamically generated list
    :param base_currency: base currency as str
    :param topn: maximum number of returned results
    :param key: sort key (defaults to 'BaseVolume')
    :return: List of pairs
    """
    summaries = sorted(
        (s for s in exchange.get_market_summaries() if s['MarketName'].startswith(base_currency)),
        key=lambda s: s.get(key) or 0.0,
        reverse=True
    )
    return [s['MarketName'].replace('-', '_') for s in summaries[:topn]]


def cleanup(*args, **kwargs) -> None:
    """
    Cleanup the application state und finish all pending tasks
    :return: None
    """
    telegram.send_msg('*Status:* `Stopping trader...`')
    logger.info('Stopping trader and cleaning up modules...')
    update_state(State.STOPPED)
    persistence.cleanup()
    telegram.cleanup()
    exit(0)


def main():
    """
    Loads and validates the config and handles the main loop
    :return: None
    """
    global _CONF
    args = build_arg_parser().parse_args()

    # Initialize logger
    logging.basicConfig(
        level=args.loglevel,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    logger.info(
        'Starting freqtrade %s (loglevel=%s)',
        __version__,
        logging.getLevelName(args.loglevel)
    )

    # Load and validate configuration
    with open(args.config) as file:
        _CONF = json.load(file)
    if 'internals' not in _CONF:
        _CONF['internals'] = {}
    logger.info('Validating configuration ...')
    validate(_CONF, CONF_SCHEMA)

    # Initialize all modules and start main loop
    if args.dynamic_whitelist:
        logger.info('Using dynamically generated whitelist. (--dynamic-whitelist detected)')
    init(_CONF)
    old_state = None
    while True:
        new_state = get_state()
        # Log state transition
        if new_state != old_state:
            telegram.send_msg('*Status:* `{}`'.format(new_state.name.lower()))
            logger.info('Changing state to: %s', new_state.name)

        if new_state == State.STOPPED:
            time.sleep(1)
        elif new_state == State.RUNNING:
            throttle(
                _process,
                min_secs=_CONF['internals'].get('process_throttle_secs', 10),
                dynamic_whitelist=args.dynamic_whitelist,
            )
        old_state = new_state


if __name__ == '__main__':
    main()
