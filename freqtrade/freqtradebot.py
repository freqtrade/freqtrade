"""
Freqtrade is the main module of this bot. It contains the class Freqtrade()
"""

import copy
import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

import arrow
import requests
from cachetools import cached, TTLCache

from freqtrade import (DependencyException, OperationalException, exchange, persistence)
from freqtrade.analyze import Analyze
from freqtrade.constants import Constants
from freqtrade.fiat_convert import CryptoToFiatConverter
from freqtrade.logger import Logger
from freqtrade.persistence import Trade
from freqtrade.rpc.rpc_manager import RPCManager
from freqtrade.state import State


class FreqtradeBot(object):
    """
    Freqtrade is the main class of the bot.
    This is from here the bot start its logic.
    """

    def __init__(self, config: Dict[str, Any], db_url: Optional[str] = None):
        """
        Init all variables and object the bot need to work
        :param config: configuration dict, you can use the Configuration.get_config()
        method to get the config dict.
        :param db_url: database connector string for sqlalchemy (Optional)
        """

        # Init the logger
        self.logger = Logger(name=__name__, level=config.get('loglevel')).get_logger()

        # Init bot states
        self._state = State.STOPPED

        # Init objects
        self.config = config
        self.analyze = None
        self.fiat_converter = None
        self.rpc = None
        self.persistence = None
        self.exchange = None

        self._init_modules(db_url=db_url)

    def _init_modules(self, db_url: Optional[str] = None) -> None:
        """
        Initializes all modules and updates the config
        :param db_url: database connector string for sqlalchemy (Optional)
        :return: None
        """
        # Initialize all modules
        self.analyze = Analyze(self.config)
        self.fiat_converter = CryptoToFiatConverter()
        self.rpc = RPCManager(self)

        persistence.init(self.config, db_url)
        exchange.init(self.config)

        # Set initial application state
        initial_state = self.config.get('initial_state')

        if initial_state:
            self.update_state(State[initial_state.upper()])
        else:
            self.update_state(State.STOPPED)

    def clean(self) -> bool:
        """
        Cleanup the application state und finish all pending tasks
        :return: None
        """
        self.rpc.send_msg('*Status:* `Stopping trader...`')
        self.logger.info('Stopping trader and cleaning up modules...')
        self.update_state(State.STOPPED)
        self.rpc.cleanup()
        persistence.cleanup()
        return True

    def update_state(self, state: State) -> None:
        """
        Updates the application state
        :param state: new state
        :return: None
        """
        self._state = state

    def get_state(self) -> State:
        """
        Gets the current application state
        :return:
        """
        return self._state

    def worker(self, old_state: None) -> State:
        """
        Trading routine that must be run at each loop
        :param old_state: the previous service state from the previous call
        :return: current service state
        """
        new_state = self.get_state()
        # Log state transition
        if new_state != old_state:
            self.rpc.send_msg('*Status:* `{}`'.format(new_state.name.lower()))
            self.logger.info('Changing state to: %s', new_state.name)

        if new_state == State.STOPPED:
            time.sleep(1)
        elif new_state == State.RUNNING:
            min_secs = self.config.get('internals', {}).get(
                'process_throttle_secs',
                Constants.PROCESS_THROTTLE_SECS
            )

            nb_assets = self.config.get(
                'dynamic_whitelist',
                Constants.DYNAMIC_WHITELIST
            )

            self._throttle(func=self._process,
                           min_secs=min_secs,
                           nb_assets=nb_assets)
        return new_state

    def _throttle(self, func: Callable[..., Any], min_secs: float, *args, **kwargs) -> Any:
        """
        Throttles the given callable that it
        takes at least `min_secs` to finish execution.
        :param func: Any callable
        :param min_secs: minimum execution time in seconds
        :return: Any
        """
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = max(min_secs - (end - start), 0.0)
        self.logger.debug('Throttling %s for %.2f seconds', func.__name__, duration)
        time.sleep(duration)
        return result

    def _process(self, nb_assets: Optional[int] = 0) -> bool:
        """
        Queries the persistence layer for open trades and handles them,
        otherwise a new trade is created.
        :param: nb_assets: the maximum number of pairs to be traded at the same time
        :return: True if one or more trades has been created or closed, False otherwise
        """
        state_changed = False
        try:
            # Refresh whitelist based on wallet maintenance
            sanitized_list = self._refresh_whitelist(
                self._gen_pair_whitelist(
                    self.config['stake_currency']
                ) if nb_assets else self.config['exchange']['pair_whitelist']
            )

            # Keep only the subsets of pairs wanted (up to nb_assets)
            final_list = sanitized_list[:nb_assets] if nb_assets else sanitized_list
            self.config['exchange']['pair_whitelist'] = final_list

            # Query trades from persistence layer
            trades = Trade.query.filter(Trade.is_open.is_(True)).all()

            # First process current opened trades
            for trade in trades:
                state_changed |= self.process_maybe_execute_sell(trade)

            # Then looking for buy opportunities
            if len(trades) < self.config['max_open_trades']:
                state_changed = self.process_maybe_execute_buy()

            if 'unfilledtimeout' in self.config:
                # Check and handle any timed out open orders
                self.check_handle_timedout(self.config['unfilledtimeout'])
                Trade.session.flush()

        except (requests.exceptions.RequestException, json.JSONDecodeError) as error:
            self.logger.warning('%s, retrying in 30 seconds...', error)
            time.sleep(Constants.RETRY_TIMEOUT)
        except OperationalException:
            self.rpc.send_msg(
                '*Status:* OperationalException:\n```\n{traceback}```{hint}'
                .format(
                    traceback=traceback.format_exc(),
                    hint='Issue `/start` if you think it is safe to restart.'
                )
            )
            self.logger.exception('OperationalException. Stopping trader ...')
            self.update_state(State.STOPPED)
        return state_changed

    @cached(TTLCache(maxsize=1, ttl=1800))
    def _gen_pair_whitelist(self, base_currency: str, key: str = 'BaseVolume') -> List[str]:
        """
        Updates the whitelist with with a dynamically generated list
        :param base_currency: base currency as str
        :param key: sort key (defaults to 'BaseVolume')
        :return: List of pairs
        """
        summaries = sorted(
            (v for s, v in exchange.get_market_summaries().items() if v['symbol'].endswith(base_currency)),
            key=lambda v: v.get('info').get(key) or 0.0,
            reverse=True
        )

        return [s['symbol'] for s in summaries]

    def _refresh_whitelist(self, whitelist: List[str]) -> List[str]:
        """
        Check wallet health and remove pair from whitelist if necessary
        :param whitelist: the sorted list (based on BaseVolume) of pairs the user might want to
        trade
        :return: the list of pairs the user wants to trade without the one unavailable or
        black_listed
        """
        sanitized_whitelist = whitelist
        health = exchange.get_wallet_health()
        known_pairs = set()
        for symbol, status in health.items():
            pair = f"{status['base']}/{self.config['stake_currency']}"
            # pair is not int the generated dynamic market, or in the blacklist ... ignore it
            if pair not in whitelist or pair in self.config['exchange'].get('pair_blacklist', []):
                continue
            # else the pair is valid
            known_pairs.add(pair)
            # Market is not active
            if not status['active']:
                sanitized_whitelist.remove(pair)
                self.logger.info(
                    'Ignoring %s from whitelist (reason: %s).',
                    pair, status.get('Notice') or 'wallet is not active'
                )

        # We need to remove pairs that are unknown
        final_list = [x for x in sanitized_whitelist if x in known_pairs]
        return final_list

    def get_target_bid(self, ticker: Dict[str, float]) -> float:
        """
        Calculates bid target between current ask price and last price
        :param ticker: Ticker to use for getting Ask and Last Price
        :return: float: Price
        """
        if ticker['ask'] < ticker['last']:
            return ticker['ask']
        balance = self.config['bid_strategy']['ask_last_balance']
        return ticker['ask'] + balance * (ticker['last'] - ticker['ask'])

    def create_trade(self) -> bool:
        """
        Checks the implemented trading indicator(s) for a randomly picked pair,
        if one pair triggers the buy_signal a new trade record gets created
        :param stake_amount: amount of btc to spend
        :param interval: Ticker interval used for Analyze
        :return: True if a trade object has been created and persisted, False otherwise
        """
        stake_amount = self.config['stake_amount']
        interval = self.analyze.get_ticker_interval()

        self.logger.info(
            'Checking buy signals to create a new trade with stake_amount: %f ...',
            stake_amount
        )
        whitelist = copy.deepcopy(self.config['exchange']['pair_whitelist'])
        # Check if stake_amount is fulfilled
        if exchange.get_balance(self.config['stake_currency']) < stake_amount:
            raise DependencyException(
                'stake amount is not fulfilled (currency={})'.format(self.config['stake_currency'])
            )

        # Remove currently opened and latest pairs from whitelist
        for trade in Trade.query.filter(Trade.is_open.is_(True)).all():
            if trade.pair in whitelist:
                whitelist.remove(trade.pair)
                self.logger.debug('Ignoring %s in pair whitelist', trade.pair)

        if not whitelist:
            raise DependencyException('No currency pairs in whitelist')

        # Pick pair based on StochRSI buy signals
        for _pair in whitelist:
            (buy, sell) = self.analyze.get_signal(_pair, interval)
            if buy and not sell:
                pair = _pair
                break
        else:
            return False

        # Calculate amount
        buy_limit = self.get_target_bid(exchange.get_ticker(pair))
        amount = stake_amount / buy_limit

        order_id = exchange.buy(pair, buy_limit, amount)

        stake_amount_fiat = self.fiat_converter.convert_amount(
            stake_amount,
            self.config['stake_currency'],
            self.config['fiat_display_currency']
        )

        # Create trade entity and return
        self.rpc.send_msg(
            '*{}:* Buying [{}]({}) with limit `{:.8f} ({:.6f} {}, {:.3f} {})` '
            .format(
                exchange.get_name().upper(),
                pair.replace('_', '/'),
                exchange.get_pair_detail_url(pair),
                buy_limit,
                stake_amount,
                self.config['stake_currency'],
                stake_amount_fiat,
                self.config['fiat_display_currency']
            )
        )
        # Fee is applied twice because we make a LIMIT_BUY and LIMIT_SELL
        trade = Trade(
            pair=pair,
            stake_amount=stake_amount,
            amount=amount,
            fee=exchange.get_fee_maker(),
            open_rate=buy_limit,
            open_date=datetime.utcnow(),
            exchange=exchange.get_name().upper(),
            open_order_id=order_id
        )
        Trade.session.add(trade)
        Trade.session.flush()
        return True

    def process_maybe_execute_buy(self) -> bool:
        """
        Tries to execute a buy trade in a safe way
        :return: True if executed
        """
        try:
            # Create entity and execute trade
            if self.create_trade():
                return True

            self.logger.info('Found no buy signals for whitelisted currencies. Trying again..')
            return False
        except DependencyException as exception:
            self.logger.warning('Unable to create trade: %s', exception)
            return False

    def process_maybe_execute_sell(self, trade: Trade) -> bool:
        """
        Tries to execute a sell trade
        :return: True if executed
        """
        # Get order details for actual price per unit
        if trade.open_order_id:
            # Update trade with order values
            self.logger.info('Found open order for %s', trade)
            trade.update(exchange.get_order(trade.open_order_id))

        if trade.is_open and trade.open_order_id is None:
            # Check if we can sell our current pair
            return self.handle_trade(trade)
        return False

    def handle_trade(self, trade: Trade) -> bool:
        """
        Sells the current pair if the threshold is reached and updates the trade record.
        :return: True if trade has been sold, False otherwise
        """
        if not trade.is_open:
            raise ValueError('attempt to handle closed trade: {}'.format(trade))

        self.logger.debug('Handling %s ...', trade)
        current_rate = exchange.get_ticker(trade.pair)['bid']

        (buy, sell) = (False, False)

        if self.config.get('experimental', {}).get('use_sell_signal'):
            (buy, sell) = self.analyze.get_signal(trade.pair, self.analyze.get_ticker_interval())

        if self.analyze.should_sell(trade, current_rate, datetime.utcnow(), buy, sell):
            self.execute_sell(trade, current_rate)
            return True

        return False

    def check_handle_timedout(self, timeoutvalue: int) -> None:
        """
        Check if any orders are timed out and cancel if neccessary
        :param timeoutvalue: Number of minutes until order is considered timed out
        :return: None
        """
        timeoutthreashold = arrow.utcnow().shift(minutes=-timeoutvalue).datetime

        for trade in Trade.query.filter(Trade.open_order_id.isnot(None)).all():
            try:
                order = exchange.get_order(trade.open_order_id)
            except requests.exceptions.RequestException:
                self.logger.info(
                    'Cannot query order for %s due to %s',
                    trade,
                    traceback.format_exc())
                continue
            ordertime = arrow.get(order['opened'])

            # Check if trade is still actually open
            if int(order['remaining']) == 0:
                continue

            if order['type'] == "LIMIT_BUY" and ordertime < timeoutthreashold:
                self.handle_timedout_limit_buy(trade, order)
            elif order['type'] == "LIMIT_SELL" and ordertime < timeoutthreashold:
                self.handle_timedout_limit_sell(trade, order)

    # FIX: 20180110, why is cancel.order unconditionally here, whereas
    #                it is conditionally called in the
    #                handle_timedout_limit_sell()?
    def handle_timedout_limit_buy(self, trade: Trade, order: Dict) -> bool:
        """Buy timeout - cancel order
        :return: True if order was fully cancelled
        """
        exchange.cancel_order(trade.open_order_id)
        if order['remaining'] == order['amount']:
            # if trade is not partially completed, just delete the trade
            Trade.session.delete(trade)
            # FIX? do we really need to flush, caller of
            #      check_handle_timedout will flush afterwards
            Trade.session.flush()
            self.logger.info('Buy order timeout for %s.', trade)
            self.rpc.send_msg('*Timeout:* Unfilled buy order for {} cancelled'.format(
                trade.pair.replace('_', '/')))
            return True

        # if trade is partially complete, edit the stake details for the trade
        # and close the order
        trade.amount = order['amount'] - order['remaining']
        trade.stake_amount = trade.amount * trade.open_rate
        trade.open_order_id = None
        self.logger.info('Partial buy order timeout for %s.', trade)
        self.rpc.send_msg('*Timeout:* Remaining buy order for {} cancelled'.format(
            trade.pair.replace('_', '/')))
        return False

    # FIX: 20180110, should cancel_order() be cond. or unconditionally called?
    def handle_timedout_limit_sell(self, trade: Trade, order: Dict) -> bool:
        """
        Sell timeout - cancel order and update trade
        :return: True if order was fully cancelled
        """
        if order['remaining'] == order['amount']:
            # if trade is not partially completed, just cancel the trade
            exchange.cancel_order(trade.open_order_id)
            trade.close_rate = None
            trade.close_profit = None
            trade.close_date = None
            trade.is_open = True
            trade.open_order_id = None
            self.rpc.send_msg('*Timeout:* Unfilled sell order for {} cancelled'.format(
                trade.pair.replace('_', '/')))
            self.logger.info('Sell order timeout for %s.', trade)
            return True

        # TODO: figure out how to handle partially complete sell orders
        return False

    def execute_sell(self, trade: Trade, limit: float) -> None:
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
        current_rate = exchange.get_ticker(trade.pair, False)['bid']
        profit = trade.calc_profit_percent(current_rate)

        message = "*{exchange}:* Selling\n" \
                  "*Current Pair:* [{pair}]({pair_url})\n" \
                  "*Limit:* `{limit}`\n" \
                  "*Amount:* `{amount}`\n" \
                  "*Open Rate:* `{open_rate:.8f}`\n" \
                  "*Current Rate:* `{current_rate:.8f}`\n" \
                  "*Profit:* `{profit:.2f}%`" \
                  "".format(
                      exchange=trade.exchange,
                      pair=trade.pair,
                      pair_url=exchange.get_pair_detail_url(trade.pair),
                      limit=limit,
                      open_rate=trade.open_rate,
                      current_rate=current_rate,
                      amount=round(trade.amount, 8),
                      profit=round(profit * 100, 2),
                  )

        # For regular case, when the configuration exists
        if 'stake_currency' in self.config and 'fiat_display_currency' in self.config:
            fiat_converter = CryptoToFiatConverter()
            profit_fiat = fiat_converter.convert_amount(
                profit_trade,
                self.config['stake_currency'],
                self.config['fiat_display_currency']
            )
            message += '` ({gain}: {profit_percent:.2f}%, {profit_coin:.8f} {coin}`' \
                       '` / {profit_fiat:.3f} {fiat})`' \
                       ''.format(
                           gain="profit" if fmt_exp_profit > 0 else "loss",
                           profit_percent=fmt_exp_profit,
                           profit_coin=profit_trade,
                           coin=self.config['stake_currency'],
                           profit_fiat=profit_fiat,
                           fiat=self.config['fiat_display_currency'],
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
        self.rpc.send_msg(message)
        Trade.session.flush()
