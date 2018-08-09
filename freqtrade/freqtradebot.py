"""
Freqtrade is the main module of this bot. It contains the class Freqtrade()
"""

import copy
import logging
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import asyncio
import arrow
import requests

from cachetools import TTLCache, cached


from freqtrade import (DependencyException, OperationalException,
                       TemporaryError, __version__, constants, persistence)
from freqtrade.exchange import Exchange
from freqtrade.persistence import Trade
from freqtrade.rpc import RPCManager, RPCMessageType
from freqtrade.state import State
from freqtrade.strategy.interface import SellType
from freqtrade.strategy.resolver import IStrategy, StrategyResolver

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
        self.strategy: IStrategy = StrategyResolver(self.config).strategy
        self.rpc: RPCManager = RPCManager(self)
        self.persistence = None
        self.exchange = Exchange(self.config)
        self._init_modules()
        self._klines: Dict[str, List[Dict]] = {}
        self._klines_last_fetched_time = 0

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
            self.rpc.send_msg({
                'type': RPCMessageType.STATUS_NOTIFICATION,
                'status': f'{state.name.lower()}'
            })
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

    def refresh_tickers(self, pair_list: List[str]) -> bool:
        """
        Refresh tickers asyncronously and return the result.
        """
        # TODO: maybe add since_ms to use async in the download-script?
        # TODO: Add tests for this and the async stuff above
        logger.debug("Refreshing klines for %d pairs", len(pair_list))
        datatups = asyncio.get_event_loop().run_until_complete(
            self.exchange.async_get_candles_history(pair_list, self.strategy.ticker_interval))

        # updating klines
        self._klines = {pair: data for (pair, data) in datatups}

        return True

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

            # Refreshing candles
            self.refresh_tickers(final_list)

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
                self.check_handle_timedout()
                Trade.session.flush()

        except TemporaryError as error:
            logger.warning('%s, retrying in 30 seconds...', error)
            time.sleep(constants.RETRY_TIMEOUT)
        except OperationalException:
            tb = traceback.format_exc()
            hint = 'Issue `/start` if you think it is safe to restart.'
            self.rpc.send_msg({
                'type': RPCMessageType.STATUS_NOTIFICATION,
                'status': f'OperationalException:\n```\n{tb}```{hint}'
            })
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

    def _get_trade_stake_amount(self) -> Optional[float]:
        """
        Check if stake amount can be fulfilled with the available balance
        for the stake currency
        :return: float: Stake Amount
        """
        stake_amount = self.config['stake_amount']
        avaliable_amount = self.exchange.get_balance(self.config['stake_currency'])

        if stake_amount == constants.UNLIMITED_STAKE_AMOUNT:
            open_trades = len(Trade.query.filter(Trade.is_open.is_(True)).all())
            if open_trades >= self.config['max_open_trades']:
                logger.warning('Can\'t open a new trade: max number of trades is reached')
                return None
            return avaliable_amount / (self.config['max_open_trades'] - open_trades)

        # Check if stake_amount is fulfilled
        if avaliable_amount < stake_amount:
            raise DependencyException(
                'Available balance(%f %s) is lower than stake amount(%f %s)' % (
                    avaliable_amount, self.config['stake_currency'],
                    stake_amount, self.config['stake_currency'])
            )

        return stake_amount

    def _get_min_pair_stake_amount(self, pair: str, price: float) -> Optional[float]:
        markets = self.exchange.get_markets()
        markets = [m for m in markets if m['symbol'] == pair]
        if not markets:
            raise ValueError(f'Can\'t get market information for symbol {pair}')

        market = markets[0]

        if 'limits' not in market:
            return None

        min_stake_amounts = []
        limits = market['limits']
        if ('cost' in limits and 'min' in limits['cost']
                and limits['cost']['min'] is not None):
            min_stake_amounts.append(limits['cost']['min'])

        if ('amount' in limits and 'min' in limits['amount']
                and limits['amount']['min'] is not None):
            min_stake_amounts.append(limits['amount']['min'] * price)

        if not min_stake_amounts:
            return None

        amount_reserve_percent = 1 - 0.05  # reserve 5% + stoploss
        if self.strategy.stoploss is not None:
            amount_reserve_percent += self.strategy.stoploss
        # it should not be more than 50%
        amount_reserve_percent = max(amount_reserve_percent, 0.5)
        return min(min_stake_amounts) / amount_reserve_percent

    def create_trade(self) -> bool:
        """
        Checks the implemented trading indicator(s) for a randomly picked pair,
        if one pair triggers the buy_signal a new trade record gets created
        :return: True if a trade object has been created and persisted, False otherwise
        """
        interval = self.strategy.ticker_interval
        stake_amount = self._get_trade_stake_amount()

        if not stake_amount:
            return False

        logger.info(
            'Checking buy signals to create a new trade with stake_amount: %f ...',
            stake_amount
        )
        whitelist = copy.deepcopy(self.config['exchange']['pair_whitelist'])

        # Remove currently opened and latest pairs from whitelist
        for trade in Trade.query.filter(Trade.is_open.is_(True)).all():
            if trade.pair in whitelist:
                whitelist.remove(trade.pair)
                logger.debug('Ignoring %s in pair whitelist', trade.pair)

        if not whitelist:
            raise DependencyException('No currency pairs in whitelist')

        # running get_signal on historical data fetched
        # to find buy signals
        for _pair in whitelist:
            (buy, sell) = self.strategy.get_signal(_pair, interval, self._klines.get(_pair))
            if buy and not sell:
                return self.execute_buy(_pair, stake_amount)

        return False

    def execute_buy(self, pair: str, stake_amount: float) -> bool:
        """
        Executes a limit buy for the given pair
        :param pair: pair for which we want to create a LIMIT_BUY
        :return: None
        """
        pair_s = pair.replace('_', '/')
        pair_url = self.exchange.get_pair_detail_url(pair)
        stake_currency = self.config['stake_currency']
        fiat_currency = self.config.get('fiat_display_currency', None)

        # Calculate amount
        buy_limit = self.get_target_bid(self.exchange.get_ticker(pair))

        min_stake_amount = self._get_min_pair_stake_amount(pair_s, buy_limit)
        if min_stake_amount is not None and min_stake_amount > stake_amount:
            logger.warning(
                f'Can\'t open a new trade for {pair_s}: stake amount'
                f' is too small ({stake_amount} < {min_stake_amount})'
            )
            return False

        amount = stake_amount / buy_limit

        order_id = self.exchange.buy(pair, buy_limit, amount)['id']

        self.rpc.send_msg({
            'type': RPCMessageType.BUY_NOTIFICATION,
            'exchange': self.exchange.name.capitalize(),
            'pair': pair_s,
            'market_url': pair_url,
            'limit': buy_limit,
            'stake_amount': stake_amount,
            'stake_currency': stake_currency,
            'fiat_currency': fiat_currency
        })
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
            open_order_id=order_id,
            strategy=self.strategy.get_strategy_name(),
            ticker_interval=constants.TICKER_INTERVAL_MINUTES[self.config['ticker_interval']]
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
        experimental = self.config.get('experimental', {})
        if experimental.get('use_sell_signal') or experimental.get('ignore_roi_if_buy_signal'):
            ticker = self._klines.get(trade.pair)
            (buy, sell) = self.strategy.get_signal(trade.pair, self.strategy.ticker_interval,
                                                   ticker)

        should_sell = self.strategy.should_sell(trade, current_rate, datetime.utcnow(), buy, sell)
        if should_sell.sell_flag:
            self.execute_sell(trade, current_rate, should_sell.sell_type)
            return True
        logger.info('Found no sell signals for whitelisted currencies. Trying again..')
        return False

    def check_handle_timedout(self) -> None:
        """
        Check if any orders are timed out and cancel if neccessary
        :param timeoutvalue: Number of minutes until order is considered timed out
        :return: None
        """
        buy_timeout = self.config['unfilledtimeout']['buy']
        sell_timeout = self.config['unfilledtimeout']['sell']
        buy_timeoutthreashold = arrow.utcnow().shift(minutes=-buy_timeout).datetime
        sell_timeoutthreashold = arrow.utcnow().shift(minutes=-sell_timeout).datetime

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

            # Check if trade is still actually open
            if order['status'] == 'open':
                if order['side'] == 'buy' and ordertime < buy_timeoutthreashold:
                    self.handle_timedout_limit_buy(trade, order)
                elif order['side'] == 'sell' and ordertime < sell_timeoutthreashold:
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
            self.rpc.send_msg({
                'type': RPCMessageType.STATUS_NOTIFICATION,
                'status': f'Unfilled buy order for {pair_s} cancelled due to timeout'
            })
            return True

        # if trade is partially complete, edit the stake details for the trade
        # and close the order
        trade.amount = order['amount'] - order['remaining']
        trade.stake_amount = trade.amount * trade.open_rate
        trade.open_order_id = None
        logger.info('Partial buy order timeout for %s.', trade)
        self.rpc.send_msg({
            'type': RPCMessageType.STATUS_NOTIFICATION,
            'status': f'Remaining buy order for {pair_s} cancelled due to timeout'
        })
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
            self.rpc.send_msg({
                'type': RPCMessageType.STATUS_NOTIFICATION,
                'status': f'Unfilled sell order for {pair_s} cancelled due to timeout'
            })
            logger.info('Sell order timeout for %s.', trade)
            return True

        # TODO: figure out how to handle partially complete sell orders
        return False

    def execute_sell(self, trade: Trade, limit: float, sell_reason: SellType) -> None:
        """
        Executes a limit sell for the given trade and limit
        :param trade: Trade instance
        :param limit: limit rate for the sell order
        :param sellreason: Reason the sell was triggered
        :return: None
        """
        # Execute sell and update trade record
        order_id = self.exchange.sell(str(trade.pair), limit, trade.amount)['id']
        trade.open_order_id = order_id
        trade.close_rate_requested = limit
        trade.sell_reason = sell_reason.value

        profit_trade = trade.calc_profit(rate=limit)
        current_rate = self.exchange.get_ticker(trade.pair)['bid']
        profit_percent = trade.calc_profit_percent(limit)
        pair_url = self.exchange.get_pair_detail_url(trade.pair)
        gain = "profit" if profit_percent > 0 else "loss"

        msg = {
            'type': RPCMessageType.SELL_NOTIFICATION,
            'exchange': trade.exchange.capitalize(),
            'pair': trade.pair,
            'gain': gain,
            'market_url': pair_url,
            'limit': limit,
            'amount': trade.amount,
            'open_rate': trade.open_rate,
            'current_rate': current_rate,
            'profit_amount': profit_trade,
            'profit_percent': profit_percent,
        }

        # For regular case, when the configuration exists
        if 'stake_currency' in self.config and 'fiat_display_currency' in self.config:
            stake_currency = self.config['stake_currency']
            fiat_currency = self.config['fiat_display_currency']
            msg.update({
                'stake_currency': stake_currency,
                'fiat_currency': fiat_currency,
            })

        # Send the message
        self.rpc.send_msg(msg)
        Trade.session.flush()
