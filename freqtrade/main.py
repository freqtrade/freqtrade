#!/usr/bin/env python3
import copy
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, Optional, List

import requests
from cachetools import cached, TTLCache

from freqtrade import __version__, exchange, persistence, rpc, DependencyException, \
    OperationalException
from freqtrade.analyze import get_signal, SignalType
from freqtrade.misc import State, get_state, update_state, parse_args, throttle, \
    load_config
from freqtrade.persistence import Trade
from freqtrade.fiat_convert import CryptoToFiatConverter

logger = logging.getLogger('freqtrade')

_CONF = {}


def refresh_whitelist(whitelist: List[str]) -> List[str]:
    """
    Check wallet health and remove pair from whitelist if necessary
    :param whitelist: the sorted list (based on BaseVolume) of pairs the user might want to trade
    :return: the list of pairs the user wants to trade without the one unavailable or black_listed
    """
    sanitized_whitelist = whitelist
    health = exchange.get_wallet_health()
    known_pairs = set()
    for status in health:
        pair = '{}_{}'.format(_CONF['stake_currency'], status['Currency'])
        # pair is not int the generated dynamic market, or in the blacklist ... ignore it
        if pair not in whitelist or pair in _CONF['exchange'].get('pair_blacklist', []):
            continue
        # else the pair is valid
        known_pairs.add(pair)
        # Market is not active
        if not status['IsActive']:
            sanitized_whitelist.remove(pair)
            logger.info(
                'Ignoring %s from whitelist (reason: %s).',
                pair, status.get('Notice') or 'wallet is not active'
            )

    # We need to remove pairs that are unknown
    final_list = [x for x in sanitized_whitelist if x in known_pairs]
    return final_list


def _process(nb_assets: Optional[int] = 0) -> bool:
    """
    Queries the persistence layer for open trades and handles them,
    otherwise a new trade is created.
    :param: nb_assets: the maximum number of pairs to be traded at the same time
    :return: True if a trade has been created or closed, False otherwise
    """
    state_changed = False
    try:
        # Refresh whitelist based on wallet maintenance
        sanitized_list = refresh_whitelist(
            gen_pair_whitelist(
                _CONF['stake_currency']
            ) if nb_assets else _CONF['exchange']['pair_whitelist']
        )

        # Keep only the subsets of pairs wanted (up to nb_assets)
        final_list = sanitized_list[:nb_assets] if nb_assets else sanitized_list
        _CONF['exchange']['pair_whitelist'] = final_list

        # Query trades from persistence layer
        trades = Trade.query.filter(Trade.is_open.is_(True)).all()
        if len(trades) < _CONF['max_open_trades']:
            try:
                # Create entity and execute trade
                state_changed = create_trade(float(_CONF['stake_amount']))
                if not state_changed:
                    logger.info(
                        'Checked all whitelisted currencies. '
                        'Found no suitable entry positions for buying. Will keep looking ...'
                    )
            except DependencyException as exception:
                logger.warning('Unable to create trade: %s', exception)

        for trade in trades:
            # Get order details for actual price per unit
            if trade.open_order_id:
                # Update trade with order values
                logger.info('Got open order for %s', trade)
                trade.update(exchange.get_order(trade.open_order_id))

            if trade.is_open and trade.open_order_id is None:
                # Check if we can sell our current pair
                state_changed = handle_trade(trade) or state_changed

            Trade.session.flush()
    except (requests.exceptions.RequestException, json.JSONDecodeError) as error:
        logger.warning(
            'Got %s in _process(), retrying in 30 seconds...',
            error
        )
        time.sleep(30)
    except OperationalException:
        rpc.send_msg('*Status:* Got OperationalException:\n```\n{traceback}```{hint}'.format(
            traceback=traceback.format_exc(),
            hint='Issue `/start` if you think it is safe to restart.'
        ))
        logger.exception('Got OperationalException. Stopping trader ...')
        update_state(State.STOPPED)
    return state_changed


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

    fmt_exp_profit = round(trade.calc_profit_percent(rate=limit) * 100, 2)
    profit_trade = trade.calc_profit(rate=limit)

    message = '*{exchange}:* Selling [{pair}]({pair_url}) with limit `{limit:.8f}`'.format(
                    exchange=trade.exchange,
                    pair=trade.pair.replace('_', '/'),
                    pair_url=exchange.get_pair_detail_url(trade.pair),
                    limit=limit
                )

    # For regular case, when the configuration exists
    if 'stake_currency' in _CONF and 'fiat_display_currency' in _CONF:
        fiat_converter = CryptoToFiatConverter()
        profit_fiat = fiat_converter.convert_amount(
            profit_trade,
            _CONF['stake_currency'],
            _CONF['fiat_display_currency']
        )
        message += '` ({gain}: {profit_percent:.2f}%, {profit_coin:.8f} {coin}`' \
                   '` / {profit_fiat:.3f} {fiat})`'.format(
                        gain="profit" if fmt_exp_profit > 0 else "loss",
                        profit_percent=fmt_exp_profit,
                        profit_coin=profit_trade,
                        coin=_CONF['stake_currency'],
                        profit_fiat=profit_fiat,
                        fiat=_CONF['fiat_display_currency'],
                   )
    # Because telegram._forcesell does not have the configuration
    # Ignore the FIAT value and does not show the stake_currency as well
    else:
        message += '` ({gain}: {profit_percent:.2f}%, {profit_coin:.8f})`'.format(
            gain="profit" if fmt_exp_profit > 0 else "loss",
            profit_percent=fmt_exp_profit,
            profit_coin=profit_trade
        )

    # Send the message
    rpc.send_msg(message)
    Trade.session.flush()


def min_roi_reached(trade: Trade, current_rate: float, current_time: datetime) -> bool:
    """
    Based an earlier trade and current price and ROI configuration, decides whether bot should sell
    :return True if bot should sell at current rate
    """
    current_profit = trade.calc_profit_percent(current_rate)
    if 'stoploss' in _CONF and current_profit < float(_CONF['stoploss']):
        logger.debug('Stop loss hit.')
        return True

    # Check if time matches and current rate is above threshold
    time_diff = (current_time - trade.open_date).total_seconds() / 60
    for duration, threshold in sorted(_CONF['minimal_roi'].items()):
        if time_diff > float(duration) and current_profit > threshold:
            return True

    logger.debug('Threshold not reached. (cur_profit: %1.2f%%)', float(current_profit) * 100.0)
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

    # Experimental: Check if the trade is profitable before selling it (avoid selling at loss)
    if _CONF.get('experimental', {}).get('sell_profit_only'):
        logger.debug('Checking if trade is profitable ...')
        if trade.calc_profit(rate=current_rate) <= 0:
            return False

    # Check if minimal roi has been reached
    if min_roi_reached(trade, current_rate, datetime.utcnow()):
        logger.debug('Executing sell due to ROI ...')
        execute_sell(trade, current_rate)
        return True

    # Experimental: Check if sell signal has been enabled and triggered
    if _CONF.get('experimental', {}).get('use_sell_signal'):
        logger.debug('Checking sell_signal ...')
        if get_signal(trade.pair, SignalType.SELL):
            logger.debug('Executing sell due to sell signal ...')
            execute_sell(trade, current_rate)
            return True

    return False


def get_target_bid(ticker: Dict[str, float]) -> float:
    """ Calculates bid target between current ask price and last price """
    if ticker['ask'] < ticker['last']:
        return ticker['ask']
    balance = _CONF['bid_strategy']['ask_last_balance']
    return ticker['ask'] + balance * (ticker['last'] - ticker['ask'])


def create_trade(stake_amount: float) -> bool:
    """
    Checks the implemented trading indicator(s) for a randomly picked pair,
    if one pair triggers the buy_signal a new trade record gets created
    :param stake_amount: amount of btc to spend
    :return: True if a trade object has been created and persisted, False otherwise
    """
    logger.info(
        'Checking buy signals to create a new trade with stake_amount: %f ...',
        stake_amount
    )
    whitelist = copy.deepcopy(_CONF['exchange']['pair_whitelist'])
    # Check if stake_amount is fulfilled
    if exchange.get_balance(_CONF['stake_currency']) < stake_amount:
        raise DependencyException(
            'stake amount is not fulfilled (currency={})'.format(_CONF['stake_currency'])
        )

    # Remove currently opened and latest pairs from whitelist
    for trade in Trade.query.filter(Trade.is_open.is_(True)).all():
        if trade.pair in whitelist:
            whitelist.remove(trade.pair)
            logger.debug('Ignoring %s in pair whitelist', trade.pair)
    if not whitelist:
        raise DependencyException('No pair in whitelist')

    # Pick pair based on StochRSI buy signals
    for _pair in whitelist:
        if get_signal(_pair, SignalType.BUY):
            pair = _pair
            break
    else:
        return False

    # Calculate amount
    buy_limit = get_target_bid(exchange.get_ticker(pair))
    amount = stake_amount / buy_limit

    order_id = exchange.buy(pair, buy_limit, amount)

    fiat_converter = CryptoToFiatConverter()
    stake_amount_fiat = fiat_converter.convert_amount(
        stake_amount,
        _CONF['stake_currency'],
        _CONF['fiat_display_currency']
    )

    # Create trade entity and return
    rpc.send_msg('*{}:* Buying [{}]({}) with limit `{:.8f} ({:.6f} {}, {:.3f} {})` '.format(
        exchange.get_name().upper(),
        pair.replace('_', '/'),
        exchange.get_pair_detail_url(pair),
        buy_limit, stake_amount, _CONF['stake_currency'],
        stake_amount_fiat, _CONF['fiat_display_currency']
    ))
    # Fee is applied twice because we make a LIMIT_BUY and LIMIT_SELL
    trade = Trade(
        pair=pair,
        stake_amount=stake_amount,
        amount=amount,
        fee=exchange.get_fee(),
        open_rate=buy_limit,
        open_date=datetime.utcnow(),
        exchange=exchange.get_name().upper(),
        open_order_id=order_id
    )
    Trade.session.add(trade)
    Trade.session.flush()
    return True


def init(config: dict, db_url: Optional[str] = None) -> None:
    """
    Initializes all modules and updates the config
    :param config: config as dict
    :param db_url: database connector string for sqlalchemy (Optional)
    :return: None
    """
    # Initialize all modules
    rpc.init(config)
    persistence.init(config, db_url)
    exchange.init(config)

    # Set initial application state
    initial_state = config.get('initial_state')
    if initial_state:
        update_state(State[initial_state.upper()])
    else:
        update_state(State.STOPPED)


@cached(TTLCache(maxsize=1, ttl=1800))
def gen_pair_whitelist(base_currency: str, key: str = 'BaseVolume') -> List[str]:
    """
    Updates the whitelist with with a dynamically generated list
    :param base_currency: base currency as str
    :param key: sort key (defaults to 'BaseVolume')
    :return: List of pairs
    """
    summaries = sorted(
        (s for s in exchange.get_market_summaries() if s['MarketName'].startswith(base_currency)),
        key=lambda s: s.get(key) or 0.0,
        reverse=True
    )

    return [s['MarketName'].replace('-', '_') for s in summaries]


def cleanup() -> None:
    """
    Cleanup the application state und finish all pending tasks
    :return: None
    """
    rpc.send_msg('*Status:* `Stopping trader...`')
    logger.info('Stopping trader and cleaning up modules...')
    update_state(State.STOPPED)
    persistence.cleanup()
    rpc.cleanup()
    exit(0)


def main() -> None:
    """
    Loads and validates the config and handles the main loop
    :return: None
    """
    global _CONF
    args = parse_args(sys.argv[1:])
    if not args:
        exit(0)

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
    _CONF = load_config(args.config)

    # Initialize all modules and start main loop
    if args.dynamic_whitelist:
        logger.info('Using dynamically generated whitelist. (--dynamic-whitelist detected)')

    # If the user ask for Dry run with a local DB instead of memory
    if args.dry_run_db:
        if _CONF.get('dry_run', False):
            _CONF.update({'dry_run_db': True})
            logger.info(
                'Dry_run will use the DB file: "tradesv3.dry_run.sqlite". (--dry_run_db detected)'
            )
        else:
            logger.info('Dry run is disabled. (--dry_run_db ignored)')

    try:
        init(_CONF)
        old_state = None
        while True:
            new_state = get_state()
            # Log state transition
            if new_state != old_state:
                rpc.send_msg('*Status:* `{}`'.format(new_state.name.lower()))
                logger.info('Changing state to: %s', new_state.name)

            if new_state == State.STOPPED:
                time.sleep(1)
            elif new_state == State.RUNNING:
                throttle(
                    _process,
                    min_secs=_CONF['internals'].get('process_throttle_secs', 10),
                    nb_assets=args.dynamic_whitelist,
                )
            old_state = new_state
    except KeyboardInterrupt:
        logger.info('Got SIGINT, aborting ...')
    except BaseException:
        logger.exception('Got fatal exception!')
    finally:
        cleanup()


if __name__ == '__main__':
    main()
