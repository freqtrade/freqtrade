"""
Freqtrade is the main module of this bot. It contains the class Freqtrade()
"""
import copy
import logging
import traceback
from datetime import datetime
from math import isclose
from os import getpid
from typing import Any, Dict, List, Optional, Tuple

import arrow
from requests.exceptions import RequestException

from freqtrade import (DependencyException, InvalidOrderException, __version__,
                       constants, persistence)
from freqtrade.configuration import validate_config_consistency
from freqtrade.data.converter import order_book_to_dataframe
from freqtrade.data.dataprovider import DataProvider
from freqtrade.edge import Edge
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_next_date
from freqtrade.persistence import Trade
from freqtrade.resolvers import (ExchangeResolver, PairListResolver,
                                 StrategyResolver)
from freqtrade.rpc import RPCManager, RPCMessageType
from freqtrade.state import State
from freqtrade.strategy.interface import IStrategy, SellType
from freqtrade.wallets import Wallets

logger = logging.getLogger(__name__)


class FreqtradeBot:
    """
    Freqtrade is the main class of the bot.
    This is from here the bot start its logic.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Init all variables and objects the bot needs to work
        :param config: configuration dict, you can use Configuration.get_config()
        to get the config dict.
        """

        logger.info('Starting freqtrade %s', __version__)

        # Init bot state
        self.state = State.STOPPED

        # Init objects
        self.config = config

        self._heartbeat_msg = 0

        self.heartbeat_interval = self.config.get('internals', {}).get('heartbeat_interval', 60)

        self.strategy: IStrategy = StrategyResolver(self.config).strategy

        # Check config consistency here since strategies can set certain options
        validate_config_consistency(config)

        self.exchange = ExchangeResolver(self.config['exchange']['name'], self.config).exchange

        self.wallets = Wallets(self.config, self.exchange)
        self.dataprovider = DataProvider(self.config, self.exchange)

        # Attach Dataprovider to Strategy baseclass
        IStrategy.dp = self.dataprovider
        # Attach Wallets to Strategy baseclass
        IStrategy.wallets = self.wallets

        pairlistname = self.config.get('pairlist', {}).get('method', 'StaticPairList')
        self.pairlists = PairListResolver(pairlistname, self, self.config).pairlist

        # Initializing Edge only if enabled
        self.edge = Edge(self.config, self.exchange, self.strategy) if \
            self.config.get('edge', {}).get('enabled', False) else None

        self.active_pair_whitelist: List[str] = self.config['exchange']['pair_whitelist']

        persistence.init(self.config.get('db_url', None),
                         clean_open_orders=self.config.get('dry_run', False))

        # Set initial bot state from config
        initial_state = self.config.get('initial_state')
        self.state = State[initial_state.upper()] if initial_state else State.STOPPED

        # RPC runs in separate threads, can start handling external commands just after
        # initialization, even before Freqtradebot has a chance to start its throttling,
        # so anything in the Freqtradebot instance should be ready (initialized), including
        # the initial state of the bot.
        # Keep this at the end of this initialization method.
        self.rpc: RPCManager = RPCManager(self)

    def cleanup(self) -> None:
        """
        Cleanup pending resources on an already stopped bot
        :return: None
        """
        logger.info('Cleaning up modules ...')

        self.rpc.cleanup()
        persistence.cleanup()

    def startup(self) -> None:
        """
        Called on startup and after reloading the bot - triggers notifications and
        performs startup tasks
        """
        self.rpc.startup_messages(self.config, self.pairlists)
        if not self.edge:
            # Adjust stoploss if it was changed
            Trade.stoploss_reinitialization(self.strategy.stoploss)

    def process(self) -> None:
        """
        Queries the persistence layer for open trades and handles them,
        otherwise a new trade is created.
        :return: True if one or more trades has been created or closed, False otherwise
        """

        # Check whether markets have to be reloaded
        self.exchange._reload_markets()

        # Refresh whitelist
        self.pairlists.refresh_pairlist()
        self.active_pair_whitelist = self.pairlists.whitelist

        # Calculating Edge positioning
        if self.edge:
            self.edge.calculate()
            self.active_pair_whitelist = self.edge.adjust(self.active_pair_whitelist)

        # Query trades from persistence layer
        trades = Trade.get_open_trades()

        # Extend active-pair whitelist with pairs from open trades
        # It ensures that tickers are downloaded for open trades
        self._extend_whitelist_with_trades(self.active_pair_whitelist, trades)

        # Refreshing candles
        self.dataprovider.refresh(self._create_pair_whitelist(self.active_pair_whitelist),
                                  self.strategy.informative_pairs())

        # First process current opened trades
        self.process_maybe_execute_sells(trades)

        # Then looking for buy opportunities
        if len(trades) < self.config['max_open_trades']:
            self.process_maybe_execute_buys()

        if 'unfilledtimeout' in self.config:
            # Check and handle any timed out open orders
            self.check_handle_timedout()
            Trade.session.flush()

        if (self.heartbeat_interval
           and (arrow.utcnow().timestamp - self._heartbeat_msg > self.heartbeat_interval)):
            logger.info(f"Bot heartbeat. PID={getpid()}")
            self._heartbeat_msg = arrow.utcnow().timestamp

    def _extend_whitelist_with_trades(self, whitelist: List[str], trades: List[Any]):
        """
        Extend whitelist with pairs from open trades
        """
        whitelist.extend([trade.pair for trade in trades if trade.pair not in whitelist])

    def _create_pair_whitelist(self, pairs: List[str]) -> List[Tuple[str, str]]:
        """
        Create pair-whitelist tuple with (pair, ticker_interval)
        """
        return [(pair, self.config['ticker_interval']) for pair in pairs]

    def get_target_bid(self, pair: str, tick: Dict = None) -> float:
        """
        Calculates bid target between current ask price and last price
        :return: float: Price
        """
        config_bid_strategy = self.config.get('bid_strategy', {})
        if 'use_order_book' in config_bid_strategy and\
                config_bid_strategy.get('use_order_book', False):
            logger.info('Getting price from order book')
            order_book_top = config_bid_strategy.get('order_book_top', 1)
            order_book = self.exchange.get_order_book(pair, order_book_top)
            logger.debug('order_book %s', order_book)
            # top 1 = index 0
            order_book_rate = order_book['bids'][order_book_top - 1][0]
            logger.info('...top %s order book buy rate %0.8f', order_book_top, order_book_rate)
            used_rate = order_book_rate
        else:
            if not tick:
                logger.info('Using Last Ask / Last Price')
                ticker = self.exchange.get_ticker(pair)
            else:
                ticker = tick
            if ticker['ask'] < ticker['last']:
                ticker_rate = ticker['ask']
            else:
                balance = self.config['bid_strategy']['ask_last_balance']
                ticker_rate = ticker['ask'] + balance * (ticker['last'] - ticker['ask'])
            used_rate = ticker_rate

        return used_rate

    def _get_trade_stake_amount(self, pair) -> Optional[float]:
        """
        Check if stake amount can be fulfilled with the available balance
        for the stake currency
        :return: float: Stake Amount
        """
        if self.edge:
            return self.edge.stake_amount(
                pair,
                self.wallets.get_free(self.config['stake_currency']),
                self.wallets.get_total(self.config['stake_currency']),
                Trade.total_open_trades_stakes()
            )
        else:
            stake_amount = self.config['stake_amount']

        available_amount = self.wallets.get_free(self.config['stake_currency'])

        if stake_amount == constants.UNLIMITED_STAKE_AMOUNT:
            open_trades = len(Trade.get_open_trades())
            if open_trades >= self.config['max_open_trades']:
                logger.warning("Can't open a new trade: max number of trades is reached")
                return None
            return available_amount / (self.config['max_open_trades'] - open_trades)

        # Check if stake_amount is fulfilled
        if available_amount < stake_amount:
            raise DependencyException(
                f"Available balance({available_amount} {self.config['stake_currency']}) is "
                f"lower than stake amount({stake_amount} {self.config['stake_currency']})"
            )

        return stake_amount

    def _get_min_pair_stake_amount(self, pair: str, price: float) -> Optional[float]:
        try:
            market = self.exchange.markets[pair]
        except KeyError:
            raise ValueError(f"Can't get market information for symbol {pair}")

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

        # reserve some percent defined in config (5% default) + stoploss
        amount_reserve_percent = 1.0 - self.config.get('amount_reserve_percent',
                                                       constants.DEFAULT_AMOUNT_RESERVE_PERCENT)
        if self.strategy.stoploss is not None:
            amount_reserve_percent += self.strategy.stoploss
        # it should not be more than 50%
        amount_reserve_percent = max(amount_reserve_percent, 0.5)
        return min(min_stake_amounts) / amount_reserve_percent

    def create_trades(self) -> bool:
        """
        Checks the implemented trading strategy for buy-signals, using the active pair whitelist.
        If a pair triggers the buy_signal a new trade record gets created.
        Checks pairs as long as the open trade count is below `max_open_trades`.
        :return: True if at least one trade has been created.
        """
        whitelist = copy.deepcopy(self.active_pair_whitelist)

        if not whitelist:
            logger.info("Active pair whitelist is empty.")
            return False

        # Remove currently opened and latest pairs from whitelist
        for trade in Trade.get_open_trades():
            if trade.pair in whitelist:
                whitelist.remove(trade.pair)
                logger.debug('Ignoring %s in pair whitelist', trade.pair)

        if not whitelist:
            logger.info("No currency pair in active pair whitelist, "
                        "but checking to sell open trades.")
            return False

        buycount = 0
        # running get_signal on historical data fetched
        for _pair in whitelist:
            if self.strategy.is_pair_locked(_pair):
                logger.info(f"Pair {_pair} is currently locked.")
                continue

            (buy, sell) = self.strategy.get_signal(
                _pair, self.strategy.ticker_interval,
                self.dataprovider.ohlcv(_pair, self.strategy.ticker_interval))

            if buy and not sell and len(Trade.get_open_trades()) < self.config['max_open_trades']:
                stake_amount = self._get_trade_stake_amount(_pair)
                if not stake_amount:
                    continue

                logger.info(f"Buy signal found: about create a new trade with stake_amount: "
                            f"{stake_amount} ...")

                bidstrat_check_depth_of_market = self.config.get('bid_strategy', {}).\
                    get('check_depth_of_market', {})
                if (bidstrat_check_depth_of_market.get('enabled', False)) and\
                        (bidstrat_check_depth_of_market.get('bids_to_ask_delta', 0) > 0):
                    if self._check_depth_of_market_buy(_pair, bidstrat_check_depth_of_market):
                        buycount += self.execute_buy(_pair, stake_amount)
                    else:
                        continue

                buycount += self.execute_buy(_pair, stake_amount)

        return buycount > 0

    def _check_depth_of_market_buy(self, pair: str, conf: Dict) -> bool:
        """
        Checks depth of market before executing a buy
        """
        conf_bids_to_ask_delta = conf.get('bids_to_ask_delta', 0)
        logger.info('checking depth of market for %s', pair)
        order_book = self.exchange.get_order_book(pair, 1000)
        order_book_data_frame = order_book_to_dataframe(order_book['bids'], order_book['asks'])
        order_book_bids = order_book_data_frame['b_size'].sum()
        order_book_asks = order_book_data_frame['a_size'].sum()
        bids_ask_delta = order_book_bids / order_book_asks
        logger.info('bids: %s, asks: %s, delta: %s', order_book_bids,
                    order_book_asks, bids_ask_delta)
        if bids_ask_delta >= conf_bids_to_ask_delta:
            return True
        return False

    def execute_buy(self, pair: str, stake_amount: float, price: Optional[float] = None) -> bool:
        """
        Executes a limit buy for the given pair
        :param pair: pair for which we want to create a LIMIT_BUY
        :return: None
        """
        pair_s = pair.replace('_', '/')
        stake_currency = self.config['stake_currency']
        fiat_currency = self.config.get('fiat_display_currency', None)
        time_in_force = self.strategy.order_time_in_force['buy']

        if price:
            buy_limit_requested = price
        else:
            # Calculate amount
            buy_limit_requested = self.get_target_bid(pair)

        min_stake_amount = self._get_min_pair_stake_amount(pair_s, buy_limit_requested)
        if min_stake_amount is not None and min_stake_amount > stake_amount:
            logger.warning(
                f"Can't open a new trade for {pair_s}: stake amount "
                f"is too small ({stake_amount} < {min_stake_amount})"
            )
            return False

        amount = stake_amount / buy_limit_requested
        order_type = self.strategy.order_types['buy']
        order = self.exchange.buy(pair=pair, ordertype=order_type,
                                  amount=amount, rate=buy_limit_requested,
                                  time_in_force=time_in_force)
        order_id = order['id']
        order_status = order.get('status', None)

        # we assume the order is executed at the price requested
        buy_limit_filled_price = buy_limit_requested

        if order_status == 'expired' or order_status == 'rejected':
            order_tif = self.strategy.order_time_in_force['buy']

            # return false if the order is not filled
            if float(order['filled']) == 0:
                logger.warning('Buy %s order with time in force %s for %s is %s by %s.'
                               ' zero amount is fulfilled.',
                               order_tif, order_type, pair_s, order_status, self.exchange.name)
                return False
            else:
                # the order is partially fulfilled
                # in case of IOC orders we can check immediately
                # if the order is fulfilled fully or partially
                logger.warning('Buy %s order with time in force %s for %s is %s by %s.'
                               ' %s amount fulfilled out of %s (%s remaining which is canceled).',
                               order_tif, order_type, pair_s, order_status, self.exchange.name,
                               order['filled'], order['amount'], order['remaining']
                               )
                stake_amount = order['cost']
                amount = order['amount']
                buy_limit_filled_price = order['price']
                order_id = None

        # in case of FOK the order may be filled immediately and fully
        elif order_status == 'closed':
            stake_amount = order['cost']
            amount = order['amount']
            buy_limit_filled_price = order['price']

        self.rpc.send_msg({
            'type': RPCMessageType.BUY_NOTIFICATION,
            'exchange': self.exchange.name.capitalize(),
            'pair': pair_s,
            'limit': buy_limit_filled_price,
            'order_type': order_type,
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
            open_rate=buy_limit_filled_price,
            open_rate_requested=buy_limit_requested,
            open_date=datetime.utcnow(),
            exchange=self.exchange.id,
            open_order_id=order_id,
            strategy=self.strategy.get_strategy_name(),
            ticker_interval=timeframe_to_minutes(self.config['ticker_interval'])
        )

        # Update fees if order is closed
        if order_status == 'closed':
            self.update_trade_state(trade, order)

        Trade.session.add(trade)
        Trade.session.flush()

        # Updating wallets
        self.wallets.update()

        return True

    def process_maybe_execute_buys(self) -> None:
        """
        Tries to execute buy orders for trades in a safe way
        """
        try:
            # Create entity and execute trade
            if not self.create_trades():
                logger.debug('Found no buy signals for whitelisted currencies. Trying again...')
        except DependencyException as exception:
            logger.warning('Unable to create trade: %s', exception)

    def process_maybe_execute_sells(self, trades: List[Any]) -> None:
        """
        Tries to execute sell orders for trades in a safe way
        """
        result = False
        for trade in trades:
            try:
                self.update_trade_state(trade)

                if (self.strategy.order_types.get('stoploss_on_exchange') and
                        self.handle_stoploss_on_exchange(trade)):
                    result = True
                    continue
                # Check if we can sell our current pair
                if trade.open_order_id is None and self.handle_trade(trade):
                    result = True

            except DependencyException as exception:
                logger.warning('Unable to sell trade: %s', exception)

        # Updating wallets if any trade occured
        if result:
            self.wallets.update()

    def get_real_amount(self, trade: Trade, order: Dict, order_amount: float = None) -> float:
        """
        Get real amount for the trade
        Necessary for exchanges which charge fees in base currency (e.g. binance)
        """
        if order_amount is None:
            order_amount = order['amount']
        # Only run for closed orders
        if trade.fee_open == 0 or order['status'] == 'open':
            return order_amount

        # use fee from order-dict if possible
        if ('fee' in order and order['fee'] is not None and
                (order['fee'].keys() >= {'currency', 'cost'})):
            if (order['fee']['currency'] is not None and
                    order['fee']['cost'] is not None and
                    trade.pair.startswith(order['fee']['currency'])):
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
            if ("fee" in exectrade and exectrade['fee'] is not None and
                    (exectrade['fee'].keys() >= {'currency', 'cost'})):
                # only applies if fee is in quote currency!
                if (exectrade['fee']['currency'] is not None and
                        exectrade['fee']['cost'] is not None and
                        trade.pair.startswith(exectrade['fee']['currency'])):
                    fee_abs += exectrade['fee']['cost']

        if not isclose(amount, order_amount, abs_tol=constants.MATH_CLOSE_PREC):
            logger.warning(f"Amount {amount} does not match amount {trade.amount}")
            raise DependencyException("Half bought? Amounts don't match")
        real_amount = amount - fee_abs
        if fee_abs != 0:
            logger.info(f"Applying fee on amount for {trade} "
                        f"(from {order_amount} to {real_amount}) from Trades")
        return real_amount

    def update_trade_state(self, trade, action_order: dict = None):
        """
        Checks trades with open orders and updates the amount if necessary
        """
        # Get order details for actual price per unit
        if trade.open_order_id:
            # Update trade with order values
            logger.info('Found open order for %s', trade)
            try:
                order = action_order or self.exchange.get_order(trade.open_order_id, trade.pair)
            except InvalidOrderException as exception:
                logger.warning('Unable to fetch order %s: %s', trade.open_order_id, exception)
                return
            # Try update amount (binance-fix)
            try:
                new_amount = self.get_real_amount(trade, order)
                if not isclose(order['amount'], new_amount, abs_tol=constants.MATH_CLOSE_PREC):
                    order['amount'] = new_amount
                    # Fee was applied, so set to 0
                    trade.fee_open = 0

            except DependencyException as exception:
                logger.warning("Could not update trade amount: %s", exception)

            trade.update(order)

            # Updating wallets when order is closed
            if not trade.is_open:
                self.wallets.update()

    def get_sell_rate(self, pair: str, refresh: bool) -> float:
        """
        Get sell rate - either using get-ticker bid or first bid based on orderbook
        The orderbook portion is only used for rpc messaging, which would otherwise fail
        for BitMex (has no bid/ask in get_ticker)
        or remain static in any other case since it's not updating.
        :return: Bid rate
        """
        config_ask_strategy = self.config.get('ask_strategy', {})
        if config_ask_strategy.get('use_order_book', False):
            logger.debug('Using order book to get sell rate')

            order_book = self.exchange.get_order_book(pair, 1)
            rate = order_book['bids'][0][0]

        else:
            rate = self.exchange.get_ticker(pair, refresh)['bid']
        return rate

    def handle_trade(self, trade: Trade) -> bool:
        """
        Sells the current pair if the threshold is reached and updates the trade record.
        :return: True if trade has been sold, False otherwise
        """
        if not trade.is_open:
            raise DependencyException(f'Attempt to handle closed trade: {trade}')

        logger.debug('Handling %s ...', trade)

        (buy, sell) = (False, False)

        config_ask_strategy = self.config.get('ask_strategy', {})

        if (config_ask_strategy.get('use_sell_signal', True) or
                config_ask_strategy.get('ignore_roi_if_buy_signal')):
            (buy, sell) = self.strategy.get_signal(
                trade.pair, self.strategy.ticker_interval,
                self.dataprovider.ohlcv(trade.pair, self.strategy.ticker_interval))

        if config_ask_strategy.get('use_order_book', False):
            logger.info('Using order book for selling...')
            # logger.debug('Order book %s',orderBook)
            order_book_min = config_ask_strategy.get('order_book_min', 1)
            order_book_max = config_ask_strategy.get('order_book_max', 1)

            order_book = self.exchange.get_order_book(trade.pair, order_book_max)

            for i in range(order_book_min, order_book_max + 1):
                order_book_rate = order_book['asks'][i - 1][0]
                logger.info('  order book asks top %s: %0.8f', i, order_book_rate)
                sell_rate = order_book_rate

                if self._check_and_execute_sell(trade, sell_rate, buy, sell):
                    return True

        else:
            logger.debug('checking sell')
            sell_rate = self.get_sell_rate(trade.pair, True)
            if self._check_and_execute_sell(trade, sell_rate, buy, sell):
                return True

        logger.debug('Found no sell signal for %s.', trade)
        return False

    def create_stoploss_order(self, trade: Trade, stop_price: float, rate: float) -> bool:
        """
        Abstracts creating stoploss orders from the logic.
        Handles errors and updates the trade database object.
        Force-sells the pair (using EmergencySell reason) in case of Problems creating the order.
        :return: True if the order succeeded, and False in case of problems.
        """
        # Limit price threshold: As limit price should always be below price
        LIMIT_PRICE_PCT = 0.99

        try:
            stoploss_order = self.exchange.stoploss_limit(pair=trade.pair, amount=trade.amount,
                                                          stop_price=stop_price,
                                                          rate=rate * LIMIT_PRICE_PCT)
            trade.stoploss_order_id = str(stoploss_order['id'])
            return True
        except InvalidOrderException as e:
            trade.stoploss_order_id = None
            logger.error(f'Unable to place a stoploss order on exchange. {e}')
            logger.warning('Selling the trade forcefully')
            self.execute_sell(trade, trade.stop_loss, sell_reason=SellType.EMERGENCY_SELL)

        except DependencyException:
            trade.stoploss_order_id = None
            logger.exception('Unable to place a stoploss order on exchange.')
        return False

    def handle_stoploss_on_exchange(self, trade: Trade) -> bool:
        """
        Check if trade is fulfilled in which case the stoploss
        on exchange should be added immediately if stoploss on exchange
        is enabled.
        """

        logger.debug('Handling stoploss on exchange %s ...', trade)

        stoploss_order = None

        try:
            # First we check if there is already a stoploss on exchange
            stoploss_order = self.exchange.get_order(trade.stoploss_order_id, trade.pair) \
                if trade.stoploss_order_id else None
        except InvalidOrderException as exception:
            logger.warning('Unable to fetch stoploss order: %s', exception)

        # If buy order is fulfilled but there is no stoploss, we add a stoploss on exchange
        if (not trade.open_order_id and not stoploss_order):

            stoploss = self.edge.stoploss(pair=trade.pair) if self.edge else self.strategy.stoploss

            stop_price = trade.open_rate * (1 + stoploss)

            if self.create_stoploss_order(trade=trade, stop_price=stop_price, rate=stop_price):
                trade.stoploss_last_update = datetime.now()
                return False

        # If stoploss order is canceled for some reason we add it
        if stoploss_order and stoploss_order['status'] == 'canceled':
            if self.create_stoploss_order(trade=trade, stop_price=trade.stop_loss,
                                          rate=trade.stop_loss):
                return False
            else:
                trade.stoploss_order_id = None
                logger.warning('Stoploss order was cancelled, but unable to recreate one.')

        # We check if stoploss order is fulfilled
        if stoploss_order and stoploss_order['status'] == 'closed':
            trade.sell_reason = SellType.STOPLOSS_ON_EXCHANGE.value
            trade.update(stoploss_order)
            # Lock pair for one candle to prevent immediate rebuys
            self.strategy.lock_pair(trade.pair,
                                    timeframe_to_next_date(self.config['ticker_interval']))
            self._notify_sell(trade, "stoploss")
            return True

        # Finally we check if stoploss on exchange should be moved up because of trailing.
        if stoploss_order and self.config.get('trailing_stop', False):
            # if trailing stoploss is enabled we check if stoploss value has changed
            # in which case we cancel stoploss order and put another one with new
            # value immediately
            self.handle_trailing_stoploss_on_exchange(trade, stoploss_order)

        return False

    def handle_trailing_stoploss_on_exchange(self, trade: Trade, order):
        """
        Check to see if stoploss on exchange should be updated
        in case of trailing stoploss on exchange
        :param Trade: Corresponding Trade
        :param order: Current on exchange stoploss order
        :return: None
        """

        if trade.stop_loss > float(order['info']['stopPrice']):
            # we check if the update is neccesary
            update_beat = self.strategy.order_types.get('stoploss_on_exchange_interval', 60)
            if (datetime.utcnow() - trade.stoploss_last_update).total_seconds() >= update_beat:
                # cancelling the current stoploss on exchange first
                logger.info('Trailing stoploss: cancelling current stoploss on exchange (id:{%s})'
                            'in order to add another one ...', order['id'])
                try:
                    self.exchange.cancel_order(order['id'], trade.pair)
                except InvalidOrderException:
                    logger.exception(f"Could not cancel stoploss order {order['id']} "
                                     f"for pair {trade.pair}")

                # Create new stoploss order
                if self.create_stoploss_order(trade=trade, stop_price=trade.stop_loss,
                                              rate=trade.stop_loss):
                    return False
                else:
                    logger.warning(f"Could not create trailing stoploss order "
                                   f"for pair {trade.pair}.")

    def _check_and_execute_sell(self, trade: Trade, sell_rate: float,
                                buy: bool, sell: bool) -> bool:
        """
        Check and execute sell
        """
        should_sell = self.strategy.should_sell(
                trade, sell_rate, datetime.utcnow(), buy, sell,
                force_stoploss=self.edge.stoploss(trade.pair) if self.edge else 0
        )

        if should_sell.sell_flag:
            self.execute_sell(trade, sell_rate, should_sell.sell_type)
            logger.info('executed sell, reason: %s', should_sell.sell_type)
            return True
        return False

    def check_handle_timedout(self) -> None:
        """
        Check if any orders are timed out and cancel if neccessary
        :param timeoutvalue: Number of minutes until order is considered timed out
        :return: None
        """
        buy_timeout = self.config['unfilledtimeout']['buy']
        sell_timeout = self.config['unfilledtimeout']['sell']
        buy_timeout_threshold = arrow.utcnow().shift(minutes=-buy_timeout).datetime
        sell_timeout_threshold = arrow.utcnow().shift(minutes=-sell_timeout).datetime

        for trade in Trade.query.filter(Trade.open_order_id.isnot(None)).all():
            try:
                # FIXME: Somehow the query above returns results
                # where the open_order_id is in fact None.
                # This is probably because the record got
                # updated via /forcesell in a different thread.
                if not trade.open_order_id:
                    continue
                order = self.exchange.get_order(trade.open_order_id, trade.pair)
            except (RequestException, DependencyException, InvalidOrderException):
                logger.info(
                    'Cannot query order for %s due to %s',
                    trade,
                    traceback.format_exc())
                continue
            ordertime = arrow.get(order['datetime']).datetime

            # Check if trade is still actually open
            if float(order['remaining']) == 0.0:
                self.wallets.update()
                continue

            if ((order['side'] == 'buy' and order['status'] == 'canceled')
                or (order['status'] == 'open'
                    and order['side'] == 'buy' and ordertime < buy_timeout_threshold)):

                self.handle_timedout_limit_buy(trade, order)
                self.wallets.update()

            elif ((order['side'] == 'sell' and order['status'] == 'canceled')
                  or (order['status'] == 'open'
                      and order['side'] == 'sell' and ordertime < sell_timeout_threshold)):
                self.handle_timedout_limit_sell(trade, order)
                self.wallets.update()

    def handle_buy_order_full_cancel(self, trade: Trade, reason: str) -> None:
        """Close trade in database and send message"""
        Trade.session.delete(trade)
        Trade.session.flush()
        logger.info('Buy order %s for %s.', reason, trade)
        self.rpc.send_msg({
            'type': RPCMessageType.STATUS_NOTIFICATION,
            'status': f'Unfilled buy order for {trade.pair} {reason}'
        })

    def handle_timedout_limit_buy(self, trade: Trade, order: Dict) -> bool:
        """Buy timeout - cancel order
        :return: True if order was fully cancelled
        """
        reason = "cancelled due to timeout"
        if order['status'] != 'canceled':
            corder = self.exchange.cancel_order(trade.open_order_id, trade.pair)
        else:
            # Order was cancelled already, so we can reuse the existing dict
            corder = order
            reason = "canceled on Exchange"

        if corder['remaining'] == corder['amount']:
            # if trade is not partially completed, just delete the trade
            self.handle_buy_order_full_cancel(trade, reason)
            return True

        # if trade is partially complete, edit the stake details for the trade
        # and close the order
        trade.amount = corder['amount'] - corder['remaining']
        trade.stake_amount = trade.amount * trade.open_rate
        # verify if fees were taken from amount to avoid problems during selling
        try:
            new_amount = self.get_real_amount(trade, corder, trade.amount)
            if not isclose(order['amount'], new_amount, abs_tol=constants.MATH_CLOSE_PREC):
                trade.amount = new_amount
                # Fee was applied, so set to 0
                trade.fee_open = 0
        except DependencyException as e:
            logger.warning("Could not update trade amount: %s", e)

        trade.open_order_id = None
        logger.info('Partial buy order timeout for %s.', trade)
        self.rpc.send_msg({
            'type': RPCMessageType.STATUS_NOTIFICATION,
            'status': f'Remaining buy order for {trade.pair} cancelled due to timeout'
        })
        return False

    def handle_timedout_limit_sell(self, trade: Trade, order: Dict) -> bool:
        """
        Sell timeout - cancel order and update trade
        :return: True if order was fully cancelled
        """
        if order['remaining'] == order['amount']:
            # if trade is not partially completed, just cancel the trade
            if order["status"] != "canceled":
                reason = "due to timeout"
                self.exchange.cancel_order(trade.open_order_id, trade.pair)
                logger.info('Sell order timeout for %s.', trade)
            else:
                reason = "on exchange"
                logger.info('Sell order canceled on exchange for %s.', trade)
            trade.close_rate = None
            trade.close_profit = None
            trade.close_date = None
            trade.is_open = True
            trade.open_order_id = None
            self.rpc.send_msg({
                'type': RPCMessageType.STATUS_NOTIFICATION,
                'status': f'Unfilled sell order for {trade.pair} cancelled {reason}'
            })

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
        sell_type = 'sell'
        if sell_reason in (SellType.STOP_LOSS, SellType.TRAILING_STOP_LOSS):
            sell_type = 'stoploss'

        # if stoploss is on exchange and we are on dry_run mode,
        # we consider the sell price stop price
        if self.config.get('dry_run', False) and sell_type == 'stoploss' \
           and self.strategy.order_types['stoploss_on_exchange']:
            limit = trade.stop_loss

        # First cancelling stoploss on exchange ...
        if self.strategy.order_types.get('stoploss_on_exchange') and trade.stoploss_order_id:
            try:
                self.exchange.cancel_order(trade.stoploss_order_id, trade.pair)
            except InvalidOrderException:
                logger.exception(f"Could not cancel stoploss order {trade.stoploss_order_id}")

        ordertype = self.strategy.order_types[sell_type]
        if sell_reason == SellType.EMERGENCY_SELL:
            # Emergencysells (default to market!)
            ordertype = self.strategy.order_types.get("emergencysell", "market")

        # Execute sell and update trade record
        order = self.exchange.sell(pair=str(trade.pair),
                                   ordertype=ordertype,
                                   amount=trade.amount, rate=limit,
                                   time_in_force=self.strategy.order_time_in_force['sell']
                                   )

        trade.open_order_id = order['id']
        trade.close_rate_requested = limit
        trade.sell_reason = sell_reason.value
        # In case of market sell orders the order can be closed immediately
        if order.get('status', 'unknown') == 'closed':
            trade.update(order)
        Trade.session.flush()

        # Lock pair for one candle to prevent immediate rebuys
        self.strategy.lock_pair(trade.pair, timeframe_to_next_date(self.config['ticker_interval']))

        self._notify_sell(trade, ordertype)

    def _notify_sell(self, trade: Trade, order_type: str):
        """
        Sends rpc notification when a sell occured.
        """
        profit_rate = trade.close_rate if trade.close_rate else trade.close_rate_requested
        profit_trade = trade.calc_profit(rate=profit_rate)
        # Use cached ticker here - it was updated seconds ago.
        current_rate = self.get_sell_rate(trade.pair, False)
        profit_percent = trade.calc_profit_percent(profit_rate)
        gain = "profit" if profit_percent > 0 else "loss"

        msg = {
            'type': RPCMessageType.SELL_NOTIFICATION,
            'exchange': trade.exchange.capitalize(),
            'pair': trade.pair,
            'gain': gain,
            'limit': trade.close_rate_requested,
            'order_type': order_type,
            'amount': trade.amount,
            'open_rate': trade.open_rate,
            'current_rate': current_rate,
            'profit_amount': profit_trade,
            'profit_percent': profit_percent,
            'sell_reason': trade.sell_reason
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
