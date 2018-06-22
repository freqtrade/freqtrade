"""
Freqtrade is the main module of this bot. It contains the class Freqtrade()
"""

import copy
import logging
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

import arrow
import requests
from cachetools import TTLCache, cached

from freqtrade import (
    DependencyException, OperationalException, TemporaryError, persistence, __version__,
)
from freqtrade import constants
from freqtrade.analyze import Analyze
from freqtrade.exchange import Exchange
from freqtrade.fiat_convert import CryptoToFiatConverter
from freqtrade.persistence import Trade
from freqtrade.rpc.rpc_manager import RPCManager
from freqtrade.state import State

logger = logging.getLogger(__name__)


class FreqtradeBot(object):
    """
    Freqtrade is the main class of the bot.
    This is from here the bot start its logic.
    """

    def __init__(self, config: Dict[str, Any])-> None:
        """
        Init all variables and object the bot need to work
        :param config: configuration dict, you can use the Configuration.get_config()
        method to get the config dict.
        """

        logger.info(
            'Starting freqtrade %s',
            __version__,
        )

        # Init bot states
        self.state = State.STOPPED

        # Init objects
        self.config = config
        self.analyze = Analyze(self.config)
        self.fiat_converter = CryptoToFiatConverter()
        self.rpc: RPCManager = RPCManager(self)
        self.persistence = None
        self.exchange = Exchange(self.config)

        self._init_modules()

    def _init_modules(self) -> None:
        """
        Initializes all modules and updates the config
        :return: None
        """
        # Initialize all modules

        persistence.init(self.config)

        # Set initial application state
        initial_state = self.config.get('initial_state')

        if initial_state:
            self.state = State[initial_state.upper()]
        else:
            self.state = State.STOPPED

    def cleanup(self) -> None:
        """
        Cleanup pending resources on an already stopped bot
        :return: None
        """
        logger.info('Cleaning up modules ...')
        self.rpc.cleanup()
        persistence.cleanup()

    def worker(self, old_state: State = None) -> State:
        """
        Trading routine that must be run at each loop
        :param old_state: the previous service state from the previous call
        :return: current service state
        """
        # Log state transition
        state = self.state
        if state != old_state:
            self.rpc.send_msg(f'*Status:* `{state.name.lower()}`')
            logger.info('Changing state to: %s', state.name)

        if state == State.STOPPED:
            time.sleep(1)
        elif state == State.RUNNING:
            min_secs = self.config.get('internals', {}).get(
                'process_throttle_secs',
                constants.PROCESS_THROTTLE_SECS
            )

            nb_assets = self.config.get('dynamic_whitelist', None)

            self._throttle(func=self._process,
                           min_secs=min_secs,
                           nb_assets=nb_assets)
        return state

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
        logger.debug('Throttling %s for %.2f seconds', func.__name__, duration)
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

        except TemporaryError as error:
            logger.warning('%s, retrying in 30 seconds...', error)
            time.sleep(constants.RETRY_TIMEOUT)
        except OperationalException:
            tb = traceback.format_exc()
            hint = 'Issue `/start` if you think it is safe to restart.'
            self.rpc.send_msg(
                f'*Status:* OperationalException:\n```\n{tb}```{hint}'
            )
            logger.exception('OperationalException. Stopping trader ...')
            self.state = State.STOPPED
        return state_changed

    @cached(TTLCache(maxsize=1, ttl=1800))
    def _gen_pair_whitelist(self, base_currency: str, key: str = 'quoteVolume') -> List[str]:
        """
        Updates the whitelist with with a dynamically generated list
        :param base_currency: base currency as str
        :param key: sort key (defaults to 'quoteVolume')
        :return: List of pairs
        """

        if not self.exchange.exchange_has('fetchTickers'):
            raise OperationalException(
                'Exchange does not support dynamic whitelist.'
                'Please edit your config and restart the bot'
            )

        tickers = self.exchange.get_tickers()
        # check length so that we make sure that '/' is actually in the string
        tickers = [v for k, v in tickers.items()
                   if len(k.split('/')) == 2 and k.split('/')[1] == base_currency]

        sorted_tickers = sorted(tickers, reverse=True, key=lambda t: t[key])
        pairs = [s['symbol'] for s in sorted_tickers]
        return pairs

    def _refresh_whitelist(self, whitelist: List[str]) -> List[str]:
        """
        Check available markets and remove pair from whitelist if necessary
        :param whitelist: the sorted list (based on BaseVolume) of pairs the user might want to
        trade
        :return: the list of pairs the user wants to trade without the one unavailable or
        black_listed
        """
        sanitized_whitelist = whitelist
        markets = self.exchange.get_markets()

        markets = [m for m in markets if m['quote'] == self.config['stake_currency']]
        known_pairs = set()
        for market in markets:
            pair = market['symbol']
            # pair is not int the generated dynamic market, or in the blacklist ... ignore it
            if pair not in whitelist or pair in self.config['exchange'].get('pair_blacklist', []):
                continue
            # else the pair is valid
            known_pairs.add(pair)
            # Market is not active
            if not market['active']:
                sanitized_whitelist.remove(pair)
                logger.info(
                    'Ignoring %s from whitelist. Market is not active.',
                    pair
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
        :return: True if a trade object has been created and persisted, False otherwise
        """
        stake_amount = self.config['stake_amount']
        interval = self.analyze.get_ticker_interval()
        stake_currency = self.config['stake_currency']
        fiat_currency = self.config['fiat_display_currency']
        exc_name = self.exchange.name

        logger.info(
            'Checking buy signals to create a new trade with stake_amount: %f ...',
            stake_amount
        )
        whitelist = copy.deepcopy(self.config['exchange']['pair_whitelist'])
        # Check if stake_amount is fulfilled
        if self.exchange.get_balance(stake_currency) < stake_amount:
            raise DependencyException(
                f'stake amount is not fulfilled (currency={stake_currency})')

        # Remove currently opened and latest pairs from whitelist
        for trade in Trade.query.filter(Trade.is_open.is_(True)).all():
            if trade.pair in whitelist:
                whitelist.remove(trade.pair)
                logger.debug('Ignoring %s in pair whitelist', trade.pair)

        if not whitelist:
            raise DependencyException('No currency pairs in whitelist')

        # Pick pair based on buy signals
        for _pair in whitelist:
            (buy, sell) = self.analyze.get_signal(self.exchange, _pair, interval)
            if buy and not sell:
                pair = _pair
                break
        else:
            return False
        pair_s = pair.replace('_', '/')
        pair_url = self.exchange.get_pair_detail_url(pair)
        # Calculate amount
        buy_limit = self.get_target_bid(self.exchange.get_ticker(pair))
        amount = stake_amount / buy_limit

        order_id = self.exchange.buy(pair, buy_limit, amount)['id']

        stake_amount_fiat = self.fiat_converter.convert_amount(
            stake_amount,
            stake_currency,
            fiat_currency
        )

        # Create trade entity and return
        self.rpc.send_msg(
            f"""*{exc_name}:* Buying [{pair_s}]({pair_url}) \
with limit `{buy_limit:.8f} ({stake_amount:.6f} \
{stake_currency}, {stake_amount_fiat:.3f} {fiat_currency})`"""
        )
        # Fee is applied twice because we make a LIMIT_BUY and LIMIT_SELL
        fee = self.exchange.get_fee(symbol=pair, taker_or_maker='maker')
        trade = Trade(
            pair=pair,
            stake_amount=stake_amount,
            amount=amount,
            fee_open=fee,
            fee_close=fee,
            open_rate=buy_limit,
            open_rate_requested=buy_limit,
            open_date=datetime.utcnow(),
            exchange=self.exchange.id,
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

            logger.info('Found no buy signals for whitelisted currencies. Trying again..')
            return False
        except DependencyException as exception:
            logger.warning('Unable to create trade: %s', exception)
            return False

    def process_maybe_execute_sell(self, trade: Trade) -> bool:
        """
        Tries to execute a sell trade
        :return: True if executed
        """
        try:
            # Get order details for actual price per unit
            if trade.open_order_id:
                # Update trade with order values
                logger.info('Found open order for %s', trade)
                order = self.exchange.get_order(trade.open_order_id, trade.pair)
                # Try update amount (binance-fix)
                try:
                    new_amount = self.get_real_amount(trade, order)
                    if order['amount'] != new_amount:
                        order['amount'] = new_amount
                        # Fee was applied, so set to 0
                        trade.fee_open = 0

                except OperationalException as exception:
                    logger.warning("could not update trade amount: %s", exception)

                trade.update(order)

            if trade.is_open and trade.open_order_id is None:
                # Check if we can sell our current pair
                return self.handle_trade(trade)
        except DependencyException as exception:
            logger.warning('Unable to sell trade: %s', exception)
        return False

    def get_real_amount(self, trade: Trade, order: Dict) -> float:
        """
        Get real amount for the trade
        Necessary for self.exchanges which charge fees in base currency (e.g. binance)
        """
        order_amount = order['amount']
        # Only run for closed orders
        if trade.fee_open == 0 or order['status'] == 'open':
            return order_amount

        # use fee from order-dict if possible
        if 'fee' in order and order['fee'] and (order['fee'].keys() >= {'currency', 'cost'}):
            if trade.pair.startswith(order['fee']['currency']):
                new_amount = order_amount - order['fee']['cost']
                logger.info("Applying fee on amount for %s (from %s to %s) from Order",
                            trade, order['amount'], new_amount)
                return new_amount

        # Fallback to Trades
        trades = self.exchange.get_trades_for_order(trade.open_order_id, trade.pair,
                                                    trade.open_date)

        if len(trades) == 0:
            logger.info("Applying fee on amount for %s failed: myTrade-Dict empty found", trade)
            return order_amount
        amount = 0
        fee_abs = 0
        for exectrade in trades:
            amount += exectrade['amount']
            if "fee" in exectrade and (exectrade['fee'].keys() >= {'currency', 'cost'}):
                # only applies if fee is in quote currency!
                if trade.pair.startswith(exectrade['fee']['currency']):
                    fee_abs += exectrade['fee']['cost']

        if amount != order_amount:
            logger.warning(f"amount {amount} does not match amount {trade.amount}")
            raise OperationalException("Half bought? Amounts don't match")
        real_amount = amount - fee_abs
        if fee_abs != 0:
            logger.info(f"""Applying fee on amount for {trade} \
(from {order_amount} to {real_amount}) from Trades""")
        return real_amount

    def handle_trade(self, trade: Trade) -> bool:
        """
        Sells the current pair if the threshold is reached and updates the trade record.
        :return: True if trade has been sold, False otherwise
        """
        if not trade.is_open:
            raise ValueError(f'attempt to handle closed trade: {trade}')

        logger.debug('Handling %s ...', trade)
        current_rate = self.exchange.get_ticker(trade.pair)['bid']

        (buy, sell) = (False, False)

        if (self.config.get('experimental', {}).get('use_sell_signal')
                or self.config.get('experimental', {}).get('ignore_roi_if_buy_signal')):
            (buy, sell) = self.analyze.get_signal(self.exchange,
                                                  trade.pair, self.analyze.get_ticker_interval())

        if self.analyze.should_sell(trade, current_rate, datetime.utcnow(), buy, sell):
            self.execute_sell(trade, current_rate)
            return True
        logger.info('Found no sell signals for whitelisted currencies. Trying again..')
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
                # FIXME: Somehow the query above returns results
                # where the open_order_id is in fact None.
                # This is probably because the record got
                # updated via /forcesell in a different thread.
                if not trade.open_order_id:
                    continue
                order = self.exchange.get_order(trade.open_order_id, trade.pair)
            except requests.exceptions.RequestException:
                logger.info(
                    'Cannot query order for %s due to %s',
                    trade,
                    traceback.format_exc())
                continue
            ordertime = arrow.get(order['datetime']).datetime

            # Check if trade is still actually open
            if int(order['remaining']) == 0:
                continue

            if order['side'] == 'buy' and ordertime < timeoutthreashold:
                self.handle_timedout_limit_buy(trade, order)
            elif order['side'] == 'sell' and ordertime < timeoutthreashold:
                self.handle_timedout_limit_sell(trade, order)

    # FIX: 20180110, why is cancel.order unconditionally here, whereas
    #                it is conditionally called in the
    #                handle_timedout_limit_sell()?
    def handle_timedout_limit_buy(self, trade: Trade, order: Dict) -> bool:
        """Buy timeout - cancel order
        :return: True if order was fully cancelled
        """
        pair_s = trade.pair.replace('_', '/')
        self.exchange.cancel_order(trade.open_order_id, trade.pair)
        if order['remaining'] == order['amount']:
            # if trade is not partially completed, just delete the trade
            Trade.session.delete(trade)
            Trade.session.flush()
            logger.info('Buy order timeout for %s.', trade)
            self.rpc.send_msg(f'*Timeout:* Unfilled buy order for {pair_s} cancelled')
            return True

        # if trade is partially complete, edit the stake details for the trade
        # and close the order
        trade.amount = order['amount'] - order['remaining']
        trade.stake_amount = trade.amount * trade.open_rate
        trade.open_order_id = None
        logger.info('Partial buy order timeout for %s.', trade)
        self.rpc.send_msg(f'*Timeout:* Remaining buy order for {pair_s} cancelled')
        return False

    # FIX: 20180110, should cancel_order() be cond. or unconditionally called?
    def handle_timedout_limit_sell(self, trade: Trade, order: Dict) -> bool:
        """
        Sell timeout - cancel order and update trade
        :return: True if order was fully cancelled
        """
        pair_s = trade.pair.replace('_', '/')
        if order['remaining'] == order['amount']:
            # if trade is not partially completed, just cancel the trade
            self.exchange.cancel_order(trade.open_order_id, trade.pair)
            trade.close_rate = None
            trade.close_profit = None
            trade.close_date = None
            trade.is_open = True
            trade.open_order_id = None
            self.rpc.send_msg(f'*Timeout:* Unfilled sell order for {pair_s} cancelled')
            logger.info('Sell order timeout for %s.', trade)
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
        exc = trade.exchange
        pair = trade.pair
        # Execute sell and update trade record
        order_id = self.exchange.sell(str(trade.pair), limit, trade.amount)['id']
        trade.open_order_id = order_id
        trade.close_rate_requested = limit

        fmt_exp_profit = round(trade.calc_profit_percent(rate=limit) * 100, 2)
        profit_trade = trade.calc_profit(rate=limit)
        current_rate = self.exchange.get_ticker(trade.pair)['bid']
        profit = trade.calc_profit_percent(limit)
        pair_url = self.exchange.get_pair_detail_url(trade.pair)
        gain = "profit" if fmt_exp_profit > 0 else "loss"

        message = f"*{exc}:* Selling\n" \
                  f"*Current Pair:* [{pair}]({pair_url})\n" \
                  f"*Limit:* `{limit}`\n" \
                  f"*Amount:* `{round(trade.amount, 8)}`\n" \
                  f"*Open Rate:* `{trade.open_rate:.8f}`\n" \
                  f"*Current Rate:* `{current_rate:.8f}`\n" \
                  f"*Profit:* `{round(profit * 100, 2):.2f}%`" \
                  ""

        # For regular case, when the configuration exists
        if 'stake_currency' in self.config and 'fiat_display_currency' in self.config:
            stake = self.config['stake_currency']
            fiat = self.config['fiat_display_currency']
            fiat_converter = CryptoToFiatConverter()
            profit_fiat = fiat_converter.convert_amount(
                profit_trade,
                stake,
                fiat
            )
            message += f'` ({gain}: {fmt_exp_profit:.2f}%, {profit_trade:.8f} {stake}`' \
                       f'` / {profit_fiat:.3f} {fiat})`'\
                       ''
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
