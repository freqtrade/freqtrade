"""
Freqtrade is the main module of this bot. It contains the class Freqtrade()
"""
import copy
import logging
import traceback
from datetime import datetime, time, timezone
from math import isclose
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from schedule import Scheduler

from freqtrade import __version__, constants
from freqtrade.configuration import validate_config_consistency
from freqtrade.data.converter import order_book_to_dataframe
from freqtrade.data.dataprovider import DataProvider
from freqtrade.edge import Edge
from freqtrade.enums import (Collateral, RPCMessageType, RunMode, SellType, SignalDirection, State,
                             TradingMode)
from freqtrade.exceptions import (DependencyException, ExchangeError, InsufficientFundsError,
                                  InvalidOrderException, PricingError)
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_seconds
from freqtrade.leverage import liquidation_price
from freqtrade.misc import safe_value_fallback, safe_value_fallback2
from freqtrade.mixins import LoggingMixin
from freqtrade.persistence import Order, PairLocks, Trade, cleanup_db, init_db
from freqtrade.plugins.pairlistmanager import PairListManager
from freqtrade.plugins.protectionmanager import ProtectionManager
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.rpc import RPCManager
from freqtrade.strategy.interface import IStrategy, SellCheckTuple
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.wallets import Wallets


logger = logging.getLogger(__name__)


class FreqtradeBot(LoggingMixin):
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
        self.active_pair_whitelist: List[str] = []

        logger.info('Starting freqtrade %s', __version__)

        # Init bot state
        self.state = State.STOPPED

        # Init objects
        self.config = config

        self.strategy: IStrategy = StrategyResolver.load_strategy(self.config)

        # Check config consistency here since strategies can set certain options
        validate_config_consistency(config)

        self.exchange = ExchangeResolver.load_exchange(self.config['exchange']['name'], self.config)

        init_db(self.config.get('db_url', None), clean_open_orders=self.config['dry_run'])

        self.wallets = Wallets(self.config, self.exchange)

        PairLocks.timeframe = self.config['timeframe']

        self.protections = ProtectionManager(self.config, self.strategy.protections)

        # RPC runs in separate threads, can start handling external commands just after
        # initialization, even before Freqtradebot has a chance to start its throttling,
        # so anything in the Freqtradebot instance should be ready (initialized), including
        # the initial state of the bot.
        # Keep this at the end of this initialization method.
        self.rpc: RPCManager = RPCManager(self)

        self.pairlists = PairListManager(self.exchange, self.config)

        self.dataprovider = DataProvider(self.config, self.exchange, self.pairlists)

        # Attach Dataprovider to strategy instance
        self.strategy.dp = self.dataprovider
        # Attach Wallets to strategy instance
        self.strategy.wallets = self.wallets

        # Initializing Edge only if enabled
        self.edge = Edge(self.config, self.exchange, self.strategy) if \
            self.config.get('edge', {}).get('enabled', False) else None

        self.active_pair_whitelist = self._refresh_active_whitelist()

        # Set initial bot state from config
        initial_state = self.config.get('initial_state')
        self.state = State[initial_state.upper()] if initial_state else State.STOPPED

        # Protect exit-logic from forcesell and vice versa
        self._exit_lock = Lock()
        LoggingMixin.__init__(self, logger, timeframe_to_seconds(self.strategy.timeframe))

        self.trading_mode = TradingMode(self.config.get('trading_mode', 'spot'))

        self.collateral_type: Optional[Collateral] = None
        if 'collateral_type' in self.config:
            self.collateral_type = Collateral(self.config['collateral_type'])

        self._schedule = Scheduler()

        if self.trading_mode == TradingMode.FUTURES:

            def update():
                self.update_funding_fees()
                self.wallets.update()

            # TODO: This would be more efficient if scheduled in utc time, and performed at each
            # TODO: funding interval, specified by funding_fee_times on the exchange classes
            for time_slot in range(0, 24):
                for minutes in [0, 15, 30, 45]:
                    t = str(time(time_slot, minutes, 2))
                    self._schedule.every().day.at(t).do(update)

    def notify_status(self, msg: str) -> None:
        """
        Public method for users of this class (worker, etc.) to send notifications
        via RPC about changes in the bot status.
        """
        self.rpc.send_msg({
            'type': RPCMessageType.STATUS,
            'status': msg
        })

    def cleanup(self) -> None:
        """
        Cleanup pending resources on an already stopped bot
        :return: None
        """
        logger.info('Cleaning up modules ...')

        if self.config['cancel_open_orders_on_exit']:
            self.cancel_all_open_orders()

        self.check_for_open_trades()

        self.rpc.cleanup()
        cleanup_db()
        self.exchange.close()

    def startup(self) -> None:
        """
        Called on startup and after reloading the bot - triggers notifications and
        performs startup tasks
        """
        self.rpc.startup_messages(self.config, self.pairlists, self.protections)
        if not self.edge:
            # Adjust stoploss if it was changed
            Trade.stoploss_reinitialization(self.strategy.stoploss)

        # Only update open orders on startup
        # This will update the database after the initial migration
        self.startup_update_open_orders()

    def process(self) -> None:
        """
        Queries the persistence layer for open trades and handles them,
        otherwise a new trade is created.
        :return: True if one or more trades has been created or closed, False otherwise
        """

        # Check whether markets have to be reloaded and reload them when it's needed
        self.exchange.reload_markets()

        self.update_closed_trades_without_assigned_fees()

        # Query trades from persistence layer
        trades = Trade.get_open_trades()

        self.active_pair_whitelist = self._refresh_active_whitelist(trades)

        # Refreshing candles
        self.dataprovider.refresh(self.pairlists.create_pair_list(self.active_pair_whitelist),
                                  self.strategy.gather_informative_pairs())

        strategy_safe_wrapper(self.strategy.bot_loop_start, supress_error=True)()

        self.strategy.analyze(self.active_pair_whitelist)

        with self._exit_lock:
            # Check and handle any timed out open orders
            self.check_handle_timedout()

        # Protect from collisions with forceexit.
        # Without this, freqtrade my try to recreate stoploss_on_exchange orders
        # while exiting is in process, since telegram messages arrive in an different thread.
        with self._exit_lock:
            trades = Trade.get_open_trades()
            # First process current opened trades (positions)
            self.exit_positions(trades)

        # Check if we need to adjust our current positions before attempting to buy new trades.
        if self.strategy.position_adjustment_enable:
            with self._exit_lock:
                self.process_open_trade_positions()

        # Then looking for buy opportunities
        if self.get_free_open_trades():
            self.enter_positions()
        if self.trading_mode == TradingMode.FUTURES:
            self._schedule.run_pending()
        Trade.commit()

    def process_stopped(self) -> None:
        """
        Close all orders that were left open
        """
        if self.config['cancel_open_orders_on_exit']:
            self.cancel_all_open_orders()

    def check_for_open_trades(self):
        """
        Notify the user when the bot is stopped (not reloaded)
        and there are still open trades active.
        """
        open_trades = Trade.get_trades([Trade.is_open.is_(True)]).all()

        if len(open_trades) != 0 and self.state != State.RELOAD_CONFIG:
            msg = {
                'type': RPCMessageType.WARNING,
                'status':
                    f"{len(open_trades)} open trades active.\n\n"
                    f"Handle these trades manually on {self.exchange.name}, "
                    f"or '/start' the bot again and use '/stopbuy' "
                    f"to handle open trades gracefully. \n"
                    f"{'Note: Trades are simulated (dry run).' if self.config['dry_run'] else ''}",
            }
            self.rpc.send_msg(msg)

    def _refresh_active_whitelist(self, trades: List[Trade] = []) -> List[str]:
        """
        Refresh active whitelist from pairlist or edge and extend it with
        pairs that have open trades.
        """
        # Refresh whitelist
        self.pairlists.refresh_pairlist()
        _whitelist = self.pairlists.whitelist

        # Calculating Edge positioning
        if self.edge:
            self.edge.calculate(_whitelist)
            _whitelist = self.edge.adjust(_whitelist)

        if trades:
            # Extend active-pair whitelist with pairs of open trades
            # It ensures that candle (OHLCV) data are downloaded for open trades as well
            _whitelist.extend([trade.pair for trade in trades if trade.pair not in _whitelist])
        return _whitelist

    def get_free_open_trades(self) -> int:
        """
        Return the number of free open trades slots or 0 if
        max number of open trades reached
        """
        open_trades = len(Trade.get_open_trades())
        return max(0, self.config['max_open_trades'] - open_trades)

    def update_funding_fees(self):
        if self.trading_mode == TradingMode.FUTURES:
            trades = Trade.get_open_trades()
            for trade in trades:
                funding_fees = self.exchange.get_funding_fees(
                    pair=trade.pair,
                    amount=trade.amount,
                    is_short=trade.is_short,
                    open_date=trade.open_date
                )
                trade.funding_fees = funding_fees
        else:
            return 0.0

    def startup_update_open_orders(self):
        """
        Updates open orders based on order list kept in the database.
        Mainly updates the state of orders - but may also close trades
        """
        if self.config['dry_run'] or self.config['exchange'].get('skip_open_order_update', False):
            # Updating open orders in dry-run does not make sense and will fail.
            return

        orders = Order.get_open_orders()
        logger.info(f"Updating {len(orders)} open orders.")
        for order in orders:
            try:
                fo = self.exchange.fetch_order_or_stoploss_order(order.order_id, order.ft_pair,
                                                                 order.ft_order_side == 'stoploss')

                self.update_trade_state(order.trade, order.order_id, fo)

            except ExchangeError as e:

                logger.warning(f"Error updating Order {order.order_id} due to {e}")

        if self.trading_mode == TradingMode.FUTURES:
            self._schedule.run_pending()

    def update_closed_trades_without_assigned_fees(self):
        """
        Update closed trades without close fees assigned.
        Only acts when Orders are in the database, otherwise the last order-id is unknown.
        """
        if self.config['dry_run']:
            # Updating open orders in dry-run does not make sense and will fail.
            return

        trades: List[Trade] = Trade.get_closed_trades_without_assigned_fees()
        for trade in trades:
            if not trade.is_open and not trade.fee_updated(trade.exit_side):
                # Get sell fee
                order = trade.select_order(trade.exit_side, False)
                if order:
                    logger.info(
                        f"Updating {trade.exit_side}-fee on trade {trade}"
                        f"for order {order.order_id}."
                    )
                    self.update_trade_state(trade, order.order_id,
                                            stoploss_order=order.ft_order_side == 'stoploss',
                                            send_msg=False)

        trades: List[Trade] = Trade.get_open_trades_without_assigned_fees()
        for trade in trades:
            if trade.is_open and not trade.fee_updated(trade.enter_side):
                order = trade.select_order(trade.enter_side, False)
                open_order = trade.select_order(trade.enter_side, True)
                if order and open_order is None:
                    logger.info(
                        f"Updating {trade.enter_side}-fee on trade {trade}"
                        f"for order {order.order_id}."
                    )
                    self.update_trade_state(trade, order.order_id, send_msg=False)

    def handle_insufficient_funds(self, trade: Trade):
        """
        Determine if we ever opened a exiting order for this trade.
        If not, try update entering fees - otherwise "refind" the open order we obviously lost.
        """
        exit_order = trade.select_order(trade.exit_side, None)
        if exit_order:
            self.refind_lost_order(trade)
        else:
            self.reupdate_enter_order_fees(trade)

    def reupdate_enter_order_fees(self, trade: Trade):
        """
        Get buy order from database, and try to reupdate.
        Handles trades where the initial fee-update did not work.
        """
        logger.info(f"Trying to reupdate {trade.enter_side} fees for {trade}")
        order = trade.select_order(trade.enter_side, False)
        if order:
            logger.info(
                f"Updating {trade.enter_side}-fee on trade {trade} for order {order.order_id}.")
            self.update_trade_state(trade, order.order_id, send_msg=False)

    def refind_lost_order(self, trade):
        """
        Try refinding a lost trade.
        Only used when InsufficientFunds appears on exit orders (stoploss or long sell/short buy).
        Tries to walk the stored orders and sell them off eventually.
        """
        logger.info(f"Trying to refind lost order for {trade}")
        for order in trade.orders:
            logger.info(f"Trying to refind {order}")
            fo = None
            if not order.ft_is_open:
                logger.debug(f"Order {order} is no longer open.")
                continue
            if order.ft_order_side == trade.enter_side:
                # Skip buy side - this is handled by reupdate_enter_order_fees
                continue
            try:
                fo = self.exchange.fetch_order_or_stoploss_order(order.order_id, order.ft_pair,
                                                                 order.ft_order_side == 'stoploss')
                if order.ft_order_side == 'stoploss':
                    if fo and fo['status'] == 'open':
                        # Assume this as the open stoploss order
                        trade.stoploss_order_id = order.order_id
                elif order.ft_order_side == trade.exit_side:
                    if fo and fo['status'] == 'open':
                        # Assume this as the open order
                        trade.open_order_id = order.order_id
                if fo:
                    logger.info(f"Found {order} for trade {trade}.")
                    self.update_trade_state(trade, order.order_id, fo,
                                            stoploss_order=order.ft_order_side == 'stoploss')

            except ExchangeError:
                logger.warning(f"Error updating {order.order_id}.")

#
# BUY / enter positions / open trades logic and methods
#

    def enter_positions(self) -> int:
        """
        Tries to execute entry orders for new trades (positions)
        """
        trades_created = 0

        whitelist = copy.deepcopy(self.active_pair_whitelist)
        if not whitelist:
            logger.info("Active pair whitelist is empty.")
            return trades_created
        # Remove pairs for currently opened trades from the whitelist
        for trade in Trade.get_open_trades():
            if trade.pair in whitelist:
                whitelist.remove(trade.pair)
                logger.debug('Ignoring %s in pair whitelist', trade.pair)

        if not whitelist:
            logger.info("No currency pair in active pair whitelist, "
                        "but checking to exit open trades.")
            return trades_created
        if PairLocks.is_global_lock():
            lock = PairLocks.get_pair_longest_lock('*')
            if lock:
                self.log_once(f"Global pairlock active until "
                              f"{lock.lock_end_time.strftime(constants.DATETIME_PRINT_FORMAT)}. "
                              f"Not creating new trades, reason: {lock.reason}.", logger.info)
            else:
                self.log_once("Global pairlock active. Not creating new trades.", logger.info)
            return trades_created
        # Create entity and execute trade for each pair from whitelist
        for pair in whitelist:
            try:
                trades_created += self.create_trade(pair)
            except DependencyException as exception:
                logger.warning('Unable to create trade for %s: %s', pair, exception)

        if not trades_created:
            logger.debug("Found no enter signals for whitelisted currencies. Trying again...")

        return trades_created

    def create_trade(self, pair: str) -> bool:
        """
        Check the implemented trading strategy for buy signals.

        If the pair triggers the buy signal a new trade record gets created
        and the buy-order opening the trade gets issued towards the exchange.

        :return: True if a trade has been created.
        """
        logger.debug(f"create_trade for pair {pair}")

        analyzed_df, _ = self.dataprovider.get_analyzed_dataframe(pair, self.strategy.timeframe)
        nowtime = analyzed_df.iloc[-1]['date'] if len(analyzed_df) > 0 else None
        if self.strategy.is_pair_locked(pair, nowtime):
            lock = PairLocks.get_pair_longest_lock(pair, nowtime)
            if lock:
                self.log_once(f"Pair {pair} is still locked until "
                              f"{lock.lock_end_time.strftime(constants.DATETIME_PRINT_FORMAT)} "
                              f"due to {lock.reason}.",
                              logger.info)
            else:
                self.log_once(f"Pair {pair} is still locked.", logger.info)
            return False

        # get_free_open_trades is checked before create_trade is called
        # but it is still used here to prevent opening too many trades within one iteration
        if not self.get_free_open_trades():
            logger.debug(f"Can't open a new trade for {pair}: max number of trades is reached.")
            return False

        # running get_signal on historical data fetched
        (signal, enter_tag) = self.strategy.get_entry_signal(
            pair,
            self.strategy.timeframe,
            analyzed_df
        )

        if signal:
            stake_amount = self.wallets.get_trade_stake_amount(pair, self.edge)

            bid_check_dom = self.config.get('bid_strategy', {}).get('check_depth_of_market', {})
            if ((bid_check_dom.get('enabled', False)) and
                    (bid_check_dom.get('bids_to_ask_delta', 0) > 0)):
                if self._check_depth_of_market(pair, bid_check_dom, side=signal):
                    return self.execute_entry(
                        pair,
                        stake_amount,
                        enter_tag=enter_tag,
                        is_short=(signal == SignalDirection.SHORT)
                    )
                else:
                    return False

            return self.execute_entry(
                pair,
                stake_amount,
                enter_tag=enter_tag,
                is_short=(signal == SignalDirection.SHORT)
            )
        else:
            return False

#
# BUY / increase positions / DCA logic and methods
#
    def process_open_trade_positions(self):
        """
        Tries to execute additional buy or sell orders for open trades (positions)
        """
        # Walk through each pair and check if it needs changes
        for trade in Trade.get_open_trades():
            # If there is any open orders, wait for them to finish.
            if trade.open_order_id is None:
                try:
                    self.check_and_call_adjust_trade_position(trade)
                except DependencyException as exception:
                    logger.warning(
                        f"Unable to adjust position of trade for {trade.pair}: {exception}")

    def check_and_call_adjust_trade_position(self, trade: Trade):
        """
        Check the implemented trading strategy for adjustment command.
        If the strategy triggers the adjustment, a new order gets issued.
        Once that completes, the existing trade is modified to match new data.
        """
        # TODO-lev: Check what changes are necessary for DCA in relation to shorts.
        if self.strategy.max_entry_position_adjustment > -1:
            count_of_buys = trade.nr_of_successful_buys
            if count_of_buys > self.strategy.max_entry_position_adjustment:
                logger.debug(f"Max adjustment entries for {trade.pair} has been reached.")
                return
        else:
            logger.debug("Max adjustment entries is set to unlimited.")
        current_rate = self.exchange.get_rate(trade.pair, refresh=True, side="buy")
        current_profit = trade.calc_profit_ratio(current_rate)

        min_stake_amount = self.exchange.get_min_pair_stake_amount(trade.pair,
                                                                   current_rate,
                                                                   self.strategy.stoploss)
        max_stake_amount = self.wallets.get_available_stake_amount()
        logger.debug(f"Calling adjust_trade_position for pair {trade.pair}")
        stake_amount = strategy_safe_wrapper(self.strategy.adjust_trade_position,
                                             default_retval=None)(
            trade=trade, current_time=datetime.now(timezone.utc), current_rate=current_rate,
            current_profit=current_profit, min_stake=min_stake_amount, max_stake=max_stake_amount)

        if stake_amount is not None and stake_amount > 0.0:
            # We should increase our position
            self.execute_entry(trade.pair, stake_amount, trade=trade)

        if stake_amount is not None and stake_amount < 0.0:
            # We should decrease our position
            # TODO: Selling part of the trade not implemented yet.
            logger.error(f"Unable to decrease trade position / sell partially"
                         f" for pair {trade.pair}, feature not implemented.")

    def _check_depth_of_market(
        self,
        pair: str,
        conf: Dict,
        side: SignalDirection
    ) -> bool:
        """
        Checks depth of market before executing a buy
        """
        conf_bids_to_ask_delta = conf.get('bids_to_ask_delta', 0)
        logger.info(f"Checking depth of market for {pair} ...")
        order_book = self.exchange.fetch_l2_order_book(pair, 1000)
        order_book_data_frame = order_book_to_dataframe(order_book['bids'], order_book['asks'])
        order_book_bids = order_book_data_frame['b_size'].sum()
        order_book_asks = order_book_data_frame['a_size'].sum()

        enter_side = order_book_bids if side == SignalDirection.LONG else order_book_asks
        exit_side = order_book_asks if side == SignalDirection.LONG else order_book_bids
        bids_ask_delta = enter_side / exit_side

        bids = f"Bids: {order_book_bids}"
        asks = f"Asks: {order_book_asks}"
        delta = f"Delta: {bids_ask_delta}"

        logger.info(
            f"{bids}, {asks}, {delta}, Direction: {side.value}"
            f"Bid Price: {order_book['bids'][0][0]}, Ask Price: {order_book['asks'][0][0]}, "
            f"Immediate Bid Quantity: {order_book['bids'][0][1]}, "
            f"Immediate Ask Quantity: {order_book['asks'][0][1]}."
        )
        if bids_ask_delta >= conf_bids_to_ask_delta:
            logger.info(f"Bids to asks delta for {pair} DOES satisfy condition.")
            return True
        else:
            logger.info(f"Bids to asks delta for {pair} does not satisfy condition.")
            return False

    def leverage_prep(
        self,
        pair: str,
        open_rate: float,
        amount: float,
        leverage: float,
        is_short: bool
    ) -> Tuple[float, Optional[float]]:

        interest_rate = 0.0
        isolated_liq = None

        # if TradingMode == TradingMode.MARGIN:
        #     interest_rate = self.exchange.get_interest_rate(
        #         pair=pair,
        #         open_rate=open_rate,
        #         is_short=is_short
        #     )
        maintenance_amt, mm_rate = self.exchange.get_mm_amt_rate(pair, amount)

        if self.collateral_type == Collateral.ISOLATED:
            if self.config['dry_run']:
                isolated_liq = liquidation_price(
                    exchange_name=self.exchange.name,
                    open_rate=open_rate,
                    is_short=is_short,
                    leverage=leverage,
                    trading_mode=self.trading_mode,
                    collateral=Collateral.ISOLATED,
                    mm_ex_1=0.0,
                    upnl_ex_1=0.0,
                    position=amount * open_rate,
                    wallet_balance=amount/leverage,  # TODO-lev: Is this correct?
                    maintenance_amt=maintenance_amt,
                    mm_rate=mm_rate,
                )
            else:
                isolated_liq = self.exchange.get_liquidation_price(pair)

        return interest_rate, isolated_liq

    def execute_entry(
        self,
        pair: str,
        stake_amount: float,
        price: Optional[float] = None,
        *,
        is_short: bool = False,
        ordertype: Optional[str] = None,
        enter_tag: Optional[str] = None,
        trade: Optional[Trade] = None,
    ) -> bool:
        """
        Executes a limit buy for the given pair
        :param pair: pair for which we want to create a LIMIT_BUY
        :param stake_amount: amount of stake-currency for the pair
        :param leverage: amount of leverage applied to this trade
        :return: True if a buy order is created, false if it fails.
        """
        time_in_force = self.strategy.order_time_in_force['buy']

        [side, name] = ['sell', 'Short'] if is_short else ['buy', 'Long']
        trade_side = 'short' if is_short else 'long'
        pos_adjust = trade is not None

        enter_limit_requested, stake_amount = self.get_valid_enter_price_and_stake(
            pair, price, stake_amount, side, trade_side, enter_tag, trade)

        if not stake_amount:
            return False

        max_leverage = self.exchange.get_max_leverage(pair, stake_amount)
        leverage = strategy_safe_wrapper(self.strategy.leverage, default_retval=1.0)(
            pair=pair,
            current_time=datetime.now(timezone.utc),
            current_rate=enter_limit_requested,
            proposed_leverage=1.0,
            max_leverage=max_leverage,
            side=trade_side,
        ) if self.trading_mode != TradingMode.SPOT else 1.0
        # Cap leverage between 1.0 and max_leverage.
        leverage = min(max(leverage, 1.0), max_leverage)
        if pos_adjust:
            logger.info(f"Position adjust: about to create a new order for {pair} with stake: "
                        f"{stake_amount} for {trade}")
        else:
            logger.info(
                f"{name} signal found: about create a new trade for {pair} with stake_amount: "
                f"{stake_amount} ...")

        amount = (stake_amount / enter_limit_requested) * leverage
        order_type = ordertype or self.strategy.order_types['buy']

        if not pos_adjust and not strategy_safe_wrapper(
                self.strategy.confirm_trade_entry, default_retval=True)(
                pair=pair, order_type=order_type, amount=amount, rate=enter_limit_requested,
                time_in_force=time_in_force, current_time=datetime.now(timezone.utc),
                entry_tag=enter_tag, side=trade_side):
            logger.info(f"User requested abortion of buying {pair}")
            return False
        amount = self.exchange.amount_to_precision(pair, amount)
        order = self.exchange.create_order(
            pair=pair,
            ordertype=order_type,
            side=side,
            amount=amount,
            rate=enter_limit_requested,
            time_in_force=time_in_force,
            leverage=leverage
        )
        order_obj = Order.parse_from_ccxt_object(order, pair, side)
        order_id = order['id']
        order_status = order.get('status', None)
        logger.info(f"Order #{order_id} was created for {pair} and status is {order_status}.")

        # we assume the order is executed at the price requested
        enter_limit_filled_price = enter_limit_requested
        amount_requested = amount

        if order_status == 'expired' or order_status == 'rejected':
            order_tif = self.strategy.order_time_in_force['buy']

            # return false if the order is not filled
            if float(order['filled']) == 0:
                logger.warning('%s %s order with time in force %s for %s is %s by %s.'
                               ' zero amount is fulfilled.',
                               name, order_tif, order_type, pair, order_status, self.exchange.name)
                return False
            else:
                # the order is partially fulfilled
                # in case of IOC orders we can check immediately
                # if the order is fulfilled fully or partially
                logger.warning('%s %s order with time in force %s for %s is %s by %s.'
                               ' %s amount fulfilled out of %s (%s remaining which is canceled).',
                               name, order_tif, order_type, pair, order_status, self.exchange.name,
                               order['filled'], order['amount'], order['remaining']
                               )
                stake_amount = order['cost']
                amount = safe_value_fallback(order, 'filled', 'amount')
                enter_limit_filled_price = safe_value_fallback(order, 'average', 'price')

        # in case of FOK the order may be filled immediately and fully
        elif order_status == 'closed':
            stake_amount = order['cost']
            amount = safe_value_fallback(order, 'filled', 'amount')
            enter_limit_filled_price = safe_value_fallback(order, 'average', 'price')

        interest_rate, isolated_liq = self.leverage_prep(
            leverage=leverage,
            pair=pair,
            amount=amount,
            open_rate=enter_limit_filled_price,
            is_short=is_short
        )

        # Fee is applied twice because we make a LIMIT_BUY and LIMIT_SELL
        fee = self.exchange.get_fee(symbol=pair, taker_or_maker='maker')
        open_date = datetime.now(timezone.utc)
        funding_fees = self.exchange.get_funding_fees(
            pair=pair, amount=amount, is_short=is_short, open_date=open_date)
        # This is a new trade
        if trade is None:
            trade = Trade(
                pair=pair,
                stake_amount=stake_amount,
                amount=amount,
                is_open=True,
                amount_requested=amount_requested,
                fee_open=fee,
                fee_close=fee,
                open_rate=enter_limit_filled_price,
                open_rate_requested=enter_limit_requested,
                open_date=open_date,
                exchange=self.exchange.id,
                open_order_id=order_id,
                strategy=self.strategy.get_strategy_name(),
                enter_tag=enter_tag,
                timeframe=timeframe_to_minutes(self.config['timeframe']),
                leverage=leverage,
                is_short=is_short,
                interest_rate=interest_rate,
                isolated_liq=isolated_liq,
                trading_mode=self.trading_mode,
                funding_fees=funding_fees
            )
        else:
            # This is additional buy, we reset fee_open_currency so timeout checking can work
            trade.is_open = True
            trade.fee_open_currency = None
            trade.open_rate_requested = enter_limit_requested
            trade.open_order_id = order_id

        trade.orders.append(order_obj)
        trade.recalc_trade_from_orders()
        Trade.query.session.add(trade)
        Trade.commit()

        # Updating wallets
        self.wallets.update()

        self._notify_enter(trade, order, order_type)

        if pos_adjust:
            if order_status == 'closed':
                logger.info(f"DCA order closed, trade should be up to date: {trade}")
                trade = self.cancel_stoploss_on_exchange(trade)
            else:
                logger.info(f"DCA order {order_status}, will wait for resolution: {trade}")

        # Update fees if order is closed
        if order_status == 'closed':
            self.update_trade_state(trade, order_id, order)

        return True

    def cancel_stoploss_on_exchange(self, trade: Trade) -> Trade:
        # First cancelling stoploss on exchange ...
        if self.strategy.order_types.get('stoploss_on_exchange') and trade.stoploss_order_id:
            try:
                logger.info(f"Canceling stoploss on exchange for {trade}")
                co = self.exchange.cancel_stoploss_order_with_result(
                    trade.stoploss_order_id, trade.pair, trade.amount)
                trade.update_order(co)
            except InvalidOrderException:
                logger.exception(f"Could not cancel stoploss order {trade.stoploss_order_id}")
        return trade

    def get_valid_enter_price_and_stake(
            self, pair: str, price: Optional[float], stake_amount: float,
            side: str, trade_side: str,
            entry_tag: Optional[str],
            trade: Optional[Trade]) -> Tuple[float, float]:
        if price:
            enter_limit_requested = price
        else:
            # Calculate price
            proposed_enter_rate = self.exchange.get_rate(pair, refresh=True, side=side)
            custom_entry_price = strategy_safe_wrapper(self.strategy.custom_entry_price,
                                                       default_retval=proposed_enter_rate)(
                pair=pair, current_time=datetime.now(timezone.utc),
                proposed_rate=proposed_enter_rate, entry_tag=entry_tag)

            enter_limit_requested = self.get_valid_price(custom_entry_price, proposed_enter_rate)

        if not enter_limit_requested:
            raise PricingError(f'Could not determine {side} price.')

        # Min-stake-amount should actually include Leverage - this way our "minimal"
        # stake- amount might be higher than necessary.
        # We do however also need min-stake to determine leverage, therefore this is ignored as
        # edge-case for now.
        min_stake_amount = self.exchange.get_min_pair_stake_amount(
            pair, enter_limit_requested, self.strategy.stoploss,)

        if not self.edge and trade is None:
            max_stake_amount = self.wallets.get_available_stake_amount()
            stake_amount = strategy_safe_wrapper(self.strategy.custom_stake_amount,
                                                 default_retval=stake_amount)(
                pair=pair, current_time=datetime.now(timezone.utc),
                current_rate=enter_limit_requested, proposed_stake=stake_amount,
                min_stake=min_stake_amount, max_stake=max_stake_amount,
                entry_tag=entry_tag, side=trade_side
            )

        stake_amount = self.wallets.validate_stake_amount(pair, stake_amount, min_stake_amount)

        return enter_limit_requested, stake_amount

    def _notify_enter(self, trade: Trade, order: Dict, order_type: Optional[str] = None,
                      fill: bool = False) -> None:
        """
        Sends rpc notification when a entry order occurred.
        """
        if fill:
            msg_type = RPCMessageType.SHORT_FILL if trade.is_short else RPCMessageType.BUY_FILL
        else:
            msg_type = RPCMessageType.SHORT if trade.is_short else RPCMessageType.BUY
        open_rate = safe_value_fallback(order, 'average', 'price')
        if open_rate is None:
            open_rate = trade.open_rate

        current_rate = trade.open_rate_requested
        if self.dataprovider.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
            current_rate = self.exchange.get_rate(trade.pair, refresh=False, side=trade.enter_side)

        msg = {
            'trade_id': trade.id,
            'type': msg_type,
            'buy_tag': trade.enter_tag,
            'enter_tag': trade.enter_tag,
            'exchange': self.exchange.name.capitalize(),
            'pair': trade.pair,
            'leverage': trade.leverage if trade.leverage else None,
            'direction': 'Short' if trade.is_short else 'Long',
            'limit': open_rate,  # Deprecated (?)
            'open_rate': open_rate,
            'order_type': order_type,
            'stake_amount': trade.stake_amount,
            'stake_currency': self.config['stake_currency'],
            'fiat_currency': self.config.get('fiat_display_currency', None),
            'amount': safe_value_fallback(order, 'filled', 'amount') or trade.amount,
            'open_date': trade.open_date or datetime.utcnow(),
            'current_rate': current_rate,
        }

        # Send the message
        self.rpc.send_msg(msg)

    def _notify_enter_cancel(self, trade: Trade, order_type: str, reason: str) -> None:
        """
        Sends rpc notification when a entry order cancel occurred.
        """
        current_rate = self.exchange.get_rate(trade.pair, refresh=False, side=trade.enter_side)
        msg_type = RPCMessageType.SHORT_CANCEL if trade.is_short else RPCMessageType.BUY_CANCEL
        msg = {
            'trade_id': trade.id,
            'type': msg_type,
            'buy_tag': trade.enter_tag,
            'enter_tag': trade.enter_tag,
            'exchange': self.exchange.name.capitalize(),
            'pair': trade.pair,
            'leverage': trade.leverage,
            'direction': 'Short' if trade.is_short else 'Long',
            'limit': trade.open_rate,
            'order_type': order_type,
            'stake_amount': trade.stake_amount,
            'stake_currency': self.config['stake_currency'],
            'fiat_currency': self.config.get('fiat_display_currency', None),
            'amount': trade.amount,
            'open_date': trade.open_date,
            'current_rate': current_rate,
            'reason': reason,
        }

        # Send the message
        self.rpc.send_msg(msg)

#
# SELL / exit positions / close trades logic and methods
#

    def exit_positions(self, trades: List[Any]) -> int:
        """
        Tries to execute exit orders for open trades (positions)
        """
        trades_closed = 0
        for trade in trades:
            try:

                if (self.strategy.order_types.get('stoploss_on_exchange') and
                        self.handle_stoploss_on_exchange(trade)):
                    trades_closed += 1
                    Trade.commit()
                    continue
                # Check if we can sell our current pair
                if trade.open_order_id is None and trade.is_open and self.handle_trade(trade):
                    trades_closed += 1

            except DependencyException as exception:
                logger.warning(f'Unable to exit trade {trade.pair}: {exception}')

        # Updating wallets if any trade occurred
        if trades_closed:
            self.wallets.update()

        return trades_closed

    def handle_trade(self, trade: Trade) -> bool:
        """
        Sells/exits_short the current pair if the threshold is reached and updates the trade record.
        :return: True if trade has been sold/exited_short, False otherwise
        """
        if not trade.is_open:
            raise DependencyException(f'Attempt to handle closed trade: {trade}')

        logger.debug('Handling %s ...', trade)

        (enter, exit_) = (False, False)
        exit_tag = None
        exit_signal_type = "exit_short" if trade.is_short else "exit_long"

        if (self.config.get('use_sell_signal', True) or
                self.config.get('ignore_roi_if_buy_signal', False)):
            analyzed_df, _ = self.dataprovider.get_analyzed_dataframe(trade.pair,
                                                                      self.strategy.timeframe)

            (enter, exit_, exit_tag) = self.strategy.get_exit_signal(
                trade.pair,
                self.strategy.timeframe,
                analyzed_df,
                is_short=trade.is_short
            )

        logger.debug('checking exit')
        exit_rate = self.exchange.get_rate(trade.pair, refresh=True, side=trade.exit_side)
        if self._check_and_execute_exit(trade, exit_rate, enter, exit_, exit_tag):
            return True

        logger.debug(f'Found no {exit_signal_type} signal for %s.', trade)
        return False

    def create_stoploss_order(self, trade: Trade, stop_price: float) -> bool:
        """
        Abstracts creating stoploss orders from the logic.
        Handles errors and updates the trade database object.
        Force-sells the pair (using EmergencySell reason) in case of Problems creating the order.
        :return: True if the order succeeded, and False in case of problems.
        """
        try:
            stoploss_order = self.exchange.stoploss(
                pair=trade.pair,
                amount=trade.amount,
                stop_price=stop_price,
                order_types=self.strategy.order_types,
                side=trade.exit_side,
                leverage=trade.leverage
            )

            order_obj = Order.parse_from_ccxt_object(stoploss_order, trade.pair, 'stoploss')
            trade.orders.append(order_obj)
            trade.stoploss_order_id = str(stoploss_order['id'])
            return True
        except InsufficientFundsError as e:
            logger.warning(f"Unable to place stoploss order {e}.")
            # Try to figure out what went wrong
            self.handle_insufficient_funds(trade)

        except InvalidOrderException as e:
            trade.stoploss_order_id = None
            logger.error(f'Unable to place a stoploss order on exchange. {e}')
            logger.warning('Exiting the trade forcefully')
            self.execute_trade_exit(trade, trade.stop_loss, sell_reason=SellCheckTuple(
                sell_type=SellType.EMERGENCY_SELL))

        except ExchangeError:
            trade.stoploss_order_id = None
            logger.exception('Unable to place a stoploss order on exchange.')
        return False

    def handle_stoploss_on_exchange(self, trade: Trade) -> bool:
        """
        Check if trade is fulfilled in which case the stoploss
        on exchange should be added immediately if stoploss on exchange
        is enabled.
        # TODO-lev: liquidation price always on exchange, even without stoploss_on_exchange
        """

        logger.debug('Handling stoploss on exchange %s ...', trade)

        stoploss_order = None

        try:
            # First we check if there is already a stoploss on exchange
            stoploss_order = self.exchange.fetch_stoploss_order(
                trade.stoploss_order_id, trade.pair) if trade.stoploss_order_id else None
        except InvalidOrderException as exception:
            logger.warning('Unable to fetch stoploss order: %s', exception)

        if stoploss_order:
            trade.update_order(stoploss_order)

        # We check if stoploss order is fulfilled
        if stoploss_order and stoploss_order['status'] in ('closed', 'triggered'):
            trade.sell_reason = SellType.STOPLOSS_ON_EXCHANGE.value
            self.update_trade_state(trade, trade.stoploss_order_id, stoploss_order,
                                    stoploss_order=True)
            # Lock pair for one candle to prevent immediate rebuys
            self.strategy.lock_pair(trade.pair, datetime.now(timezone.utc),
                                    reason='Auto lock')
            self._notify_exit(trade, "stoploss")
            return True

        if trade.open_order_id or not trade.is_open:
            # Trade has an open Buy or Sell order, Stoploss-handling can't happen in this case
            # as the Amount on the exchange is tied up in another trade.
            # The trade can be closed already (sell-order fill confirmation came in this iteration)
            return False

        # If enter order is fulfilled but there is no stoploss, we add a stoploss on exchange
        if not stoploss_order:
            stoploss = self.edge.stoploss(pair=trade.pair) if self.edge else self.strategy.stoploss
            if trade.is_short:
                stop_price = trade.open_rate * (1 - stoploss)
            else:
                stop_price = trade.open_rate * (1 + stoploss)

            if self.create_stoploss_order(trade=trade, stop_price=stop_price):
                trade.stoploss_last_update = datetime.utcnow()
                return False

        # If stoploss order is canceled for some reason we add it
        if stoploss_order and stoploss_order['status'] in ('canceled', 'cancelled'):
            if self.create_stoploss_order(trade=trade, stop_price=trade.stop_loss):
                return False
            else:
                trade.stoploss_order_id = None
                logger.warning('Stoploss order was cancelled, but unable to recreate one.')

        # Finally we check if stoploss on exchange should be moved up because of trailing.
        # Triggered Orders are now real orders - so don't replace stoploss anymore
        if (
            stoploss_order
            and stoploss_order.get('status_stop') != 'triggered'
            and (self.config.get('trailing_stop', False)
                 or self.config.get('use_custom_stoploss', False))
        ):
            # if trailing stoploss is enabled we check if stoploss value has changed
            # in which case we cancel stoploss order and put another one with new
            # value immediately
            self.handle_trailing_stoploss_on_exchange(trade, stoploss_order)

        return False

    def handle_trailing_stoploss_on_exchange(self, trade: Trade, order: dict) -> None:
        """
        Check to see if stoploss on exchange should be updated
        in case of trailing stoploss on exchange
        :param trade: Corresponding Trade
        :param order: Current on exchange stoploss order
        :return: None
        """
        if self.exchange.stoploss_adjust(trade.stop_loss, order, side=trade.exit_side):
            # we check if the update is necessary
            update_beat = self.strategy.order_types.get('stoploss_on_exchange_interval', 60)
            if (datetime.utcnow() - trade.stoploss_last_update).total_seconds() >= update_beat:
                # cancelling the current stoploss on exchange first
                logger.info(f"Cancelling current stoploss on exchange for pair {trade.pair} "
                            f"(orderid:{order['id']}) in order to add another one ...")
                try:
                    co = self.exchange.cancel_stoploss_order_with_result(order['id'], trade.pair,
                                                                         trade.amount)
                    trade.update_order(co)
                except InvalidOrderException:
                    logger.exception(f"Could not cancel stoploss order {order['id']} "
                                     f"for pair {trade.pair}")

                # Create new stoploss order
                if not self.create_stoploss_order(trade=trade, stop_price=trade.stop_loss):
                    logger.warning(f"Could not create trailing stoploss order "
                                   f"for pair {trade.pair}.")

    def _check_and_execute_exit(self, trade: Trade, exit_rate: float,
                                enter: bool, exit_: bool, exit_tag: Optional[str]) -> bool:
        """
        Check and execute trade exit
        """
        should_exit: SellCheckTuple = self.strategy.should_exit(
            trade,
            exit_rate,
            datetime.now(timezone.utc),
            enter=enter,
            exit_=exit_,
            force_stoploss=self.edge.stoploss(trade.pair) if self.edge else 0
        )

        if should_exit.sell_flag:
            logger.info(f'Exit for {trade.pair} detected. Reason: {should_exit.sell_type}'
                        f'Tag: {exit_tag if exit_tag is not None else "None"}')
            self.execute_trade_exit(trade, exit_rate, should_exit, exit_tag=exit_tag)
            return True
        return False

    def check_handle_timedout(self) -> None:
        """
        Check if any orders are timed out and cancel if necessary
        :param timeoutvalue: Number of minutes until order is considered timed out
        :return: None
        """

        for trade in Trade.get_open_order_trades():
            try:
                if not trade.open_order_id:
                    continue
                order = self.exchange.fetch_order(trade.open_order_id, trade.pair)
            except (ExchangeError):
                logger.info('Cannot query order for %s due to %s', trade, traceback.format_exc())
                continue

            fully_cancelled = self.update_trade_state(trade, trade.open_order_id, order)
            is_entering = order['side'] == trade.enter_side
            not_closed = order['status'] == 'open' or fully_cancelled
            time_method = 'sell' if order['side'] == 'sell' else 'buy'
            max_timeouts = self.config.get('unfilledtimeout', {}).get('exit_timeout_count', 0)

            if not_closed and (fully_cancelled or self.strategy.ft_check_timed_out(
                        time_method, trade, order, datetime.now(timezone.utc))
                    ):
                if is_entering:
                    self.handle_cancel_enter(trade, order, constants.CANCEL_REASON['TIMEOUT'])
                else:
                    self.handle_cancel_exit(trade, order, constants.CANCEL_REASON['TIMEOUT'])
                    canceled_count = trade.get_exit_order_count()
                    if max_timeouts > 0 and canceled_count >= max_timeouts:
                        logger.warning(f'Emergencyselling trade {trade}, as the sell order '
                                       f'timed out {max_timeouts} times.')
                        try:
                            self.execute_trade_exit(
                                trade, order.get('price'),
                                sell_reason=SellCheckTuple(sell_type=SellType.EMERGENCY_SELL))
                        except DependencyException as exception:
                            logger.warning(
                                f'Unable to emergency sell trade {trade.pair}: {exception}')

    def cancel_all_open_orders(self) -> None:
        """
        Cancel all orders that are currently open
        :return: None
        """

        for trade in Trade.get_open_order_trades():
            try:
                order = self.exchange.fetch_order(trade.open_order_id, trade.pair)
            except (ExchangeError):
                logger.info('Cannot query order for %s due to %s', trade, traceback.format_exc())
                continue

            if order['side'] == trade.enter_side:
                self.handle_cancel_enter(trade, order, constants.CANCEL_REASON['ALL_CANCELLED'])

            elif order['side'] == trade.exit_side:
                self.handle_cancel_exit(trade, order, constants.CANCEL_REASON['ALL_CANCELLED'])
        Trade.commit()

    def handle_cancel_enter(self, trade: Trade, order: Dict, reason: str) -> bool:
        """
        Buy cancel - cancel order
        :return: True if order was fully cancelled
        """
        was_trade_fully_canceled = False

        # Cancelled orders may have the status of 'canceled' or 'closed'
        if order['status'] not in constants.NON_OPEN_EXCHANGE_STATES:
            filled_val = order.get('filled', 0.0) or 0.0
            filled_stake = filled_val * trade.open_rate
            minstake = self.exchange.get_min_pair_stake_amount(
                trade.pair, trade.open_rate, self.strategy.stoploss)

            if filled_val > 0 and filled_stake < minstake:
                logger.warning(
                    f"Order {trade.open_order_id} for {trade.pair} not cancelled, "
                    f"as the filled amount of {filled_val} would result in an unexitable trade.")
                return False
            corder = self.exchange.cancel_order_with_result(trade.open_order_id, trade.pair,
                                                            trade.amount)
            # Avoid race condition where the order could not be cancelled coz its already filled.
            # Simply bailing here is the only safe way - as this order will then be
            # handled in the next iteration.
            if corder.get('status') not in constants.NON_OPEN_EXCHANGE_STATES:
                logger.warning(f"Order {trade.open_order_id} for {trade.pair} not cancelled.")
                return False
        else:
            # Order was cancelled already, so we can reuse the existing dict
            corder = order
            reason = constants.CANCEL_REASON['CANCELLED_ON_EXCHANGE']

        side = trade.enter_side.capitalize()
        logger.info('%s order %s for %s.', side, reason, trade)

        # Using filled to determine the filled amount
        filled_amount = safe_value_fallback2(corder, order, 'filled', 'filled')
        if isclose(filled_amount, 0.0, abs_tol=constants.MATH_CLOSE_PREC):
            logger.info(f'{side} order fully cancelled. Removing {trade} from database.')
            # if trade is not partially completed and it's the only order, just delete the trade
            if len(trade.orders) <= 1:
                trade.delete()
                was_trade_fully_canceled = True
                reason += f", {constants.CANCEL_REASON['FULLY_CANCELLED']}"
            else:
                # FIXME TODO: This could possibly reworked to not duplicate the code 15 lines below.
                self.update_trade_state(trade, trade.open_order_id, corder)
                trade.open_order_id = None
                logger.info(f'Partial {side} order timeout for {trade}.')
        else:
            # if trade is partially complete, edit the stake details for the trade
            # and close the order
            # cancel_order may not contain the full order dict, so we need to fallback
            # to the order dict acquired before cancelling.
            # we need to fall back to the values from order if corder does not contain these keys.
            trade.amount = filled_amount
            # TODO-lev: Check edge cases, we don't want to make leverage > 1.0 if we don't have to

            trade.stake_amount = trade.amount * trade.open_rate
            self.update_trade_state(trade, trade.open_order_id, corder)

            trade.open_order_id = None
            logger.info('Partial %s order timeout for %s.', trade.enter_side, trade)
            reason += f", {constants.CANCEL_REASON['PARTIALLY_FILLED']}"

        self.wallets.update()
        self._notify_enter_cancel(trade, order_type=self.strategy.order_types[trade.enter_side],
                                  reason=reason)
        return was_trade_fully_canceled

    def handle_cancel_exit(self, trade: Trade, order: Dict, reason: str) -> str:
        """
        exit order cancel - cancel order and update trade
        :return: Reason for cancel
        """
        # if trade is not partially completed, just cancel the order
        if order['remaining'] == order['amount'] or order.get('filled') == 0.0:
            if not self.exchange.check_order_canceled_empty(order):
                try:
                    # if trade is not partially completed, just delete the order
                    co = self.exchange.cancel_order_with_result(trade.open_order_id, trade.pair,
                                                                trade.amount)
                    trade.update_order(co)
                except InvalidOrderException:
                    logger.exception(
                        f"Could not cancel {trade.exit_side} order {trade.open_order_id}")
                    return 'error cancelling order'
                logger.info('%s order %s for %s.', trade.exit_side.capitalize(), reason, trade)
            else:
                reason = constants.CANCEL_REASON['CANCELLED_ON_EXCHANGE']
                logger.info('%s order %s for %s.', trade.exit_side.capitalize(), reason, trade)
                trade.update_order(order)

            trade.close_rate = None
            trade.close_rate_requested = None
            trade.close_profit = None
            trade.close_profit_abs = None
            trade.close_date = None
            trade.is_open = True
            trade.open_order_id = None
        else:
            # TODO: figure out how to handle partially complete sell orders
            reason = constants.CANCEL_REASON['PARTIALLY_FILLED_KEEP_OPEN']

        self.wallets.update()
        self._notify_exit_cancel(
            trade,
            order_type=self.strategy.order_types[trade.exit_side],
            reason=reason
        )
        return reason

    def _safe_exit_amount(self, pair: str, amount: float) -> float:
        """
        Get sellable amount.
        Should be trade.amount - but will fall back to the available amount if necessary.
        This should cover cases where get_real_amount() was not able to update the amount
        for whatever reason.
        :param pair: Pair we're trying to sell
        :param amount: amount we expect to be available
        :return: amount to sell
        :raise: DependencyException: if available balance is not within 2% of the available amount.
        """
        # TODO-lev Maybe update?
        # Update wallets to ensure amounts tied up in a stoploss is now free!
        self.wallets.update()
        trade_base_currency = self.exchange.get_pair_base_currency(pair)
        wallet_amount = self.wallets.get_free(trade_base_currency)
        logger.debug(f"{pair} - Wallet: {wallet_amount} - Trade-amount: {amount}")
        if wallet_amount >= amount:
            return amount
        elif wallet_amount > amount * 0.98:
            logger.info(f"{pair} - Falling back to wallet-amount {wallet_amount} -> {amount}.")
            return wallet_amount
        else:
            raise DependencyException(
                f"Not enough amount to exit trade. Trade-amount: {amount}, Wallet: {wallet_amount}")

    def execute_trade_exit(
            self,
            trade: Trade,
            limit: float,
            sell_reason: SellCheckTuple,
            *,
            exit_tag: Optional[str] = None,
            ordertype: Optional[str] = None,
    ) -> bool:
        """
        Executes a trade exit for the given trade and limit
        :param trade: Trade instance
        :param limit: limit rate for the sell order
        :param sell_reason: Reason the sell was triggered
        :return: True if it succeeds (supported) False (not supported)
        """
        trade.funding_fees = self.exchange.get_funding_fees(
            pair=trade.pair,
            amount=trade.amount,
            is_short=trade.is_short,
            open_date=trade.open_date,
        )
        exit_type = 'sell'
        if sell_reason.sell_type in (SellType.STOP_LOSS, SellType.TRAILING_STOP_LOSS):
            exit_type = 'stoploss'

        # if stoploss is on exchange and we are on dry_run mode,
        # we consider the sell price stop price
        if self.config['dry_run'] and exit_type == 'stoploss' \
           and self.strategy.order_types['stoploss_on_exchange']:
            limit = trade.stop_loss

        # set custom_exit_price if available
        proposed_limit_rate = limit
        current_profit = trade.calc_profit_ratio(limit)
        custom_exit_price = strategy_safe_wrapper(self.strategy.custom_exit_price,
                                                  default_retval=proposed_limit_rate)(
            pair=trade.pair, trade=trade,
            current_time=datetime.now(timezone.utc),
            proposed_rate=proposed_limit_rate, current_profit=current_profit)

        limit = self.get_valid_price(custom_exit_price, proposed_limit_rate)

        # First cancelling stoploss on exchange ...
        trade = self.cancel_stoploss_on_exchange(trade)

        order_type = ordertype or self.strategy.order_types[exit_type]
        if sell_reason.sell_type == SellType.EMERGENCY_SELL:
            # Emergency sells (default to market!)
            order_type = self.strategy.order_types.get("emergencysell", "market")

        amount = self._safe_exit_amount(trade.pair, trade.amount)
        time_in_force = self.strategy.order_time_in_force['sell']

        if not strategy_safe_wrapper(self.strategy.confirm_trade_exit, default_retval=True)(
                pair=trade.pair, trade=trade, order_type=order_type, amount=amount, rate=limit,
                time_in_force=time_in_force, sell_reason=sell_reason.sell_reason,
                current_time=datetime.now(timezone.utc)):
            logger.info(f"User requested abortion of exiting {trade.pair}")
            return False

        try:
            # Execute sell and update trade record
            order = self.exchange.create_order(
                pair=trade.pair,
                ordertype=order_type,
                side=trade.exit_side,
                amount=amount,
                rate=limit,
                time_in_force=time_in_force
            )
        except InsufficientFundsError as e:
            logger.warning(f"Unable to place order {e}.")
            # Try to figure out what went wrong
            self.handle_insufficient_funds(trade)
            return False

        order_obj = Order.parse_from_ccxt_object(order, trade.pair, trade.exit_side)
        trade.orders.append(order_obj)

        trade.open_order_id = order['id']
        trade.sell_order_status = ''
        trade.close_rate_requested = limit
        trade.sell_reason = exit_tag or sell_reason.sell_reason

        # Lock pair for one candle to prevent immediate re-trading
        self.strategy.lock_pair(trade.pair, datetime.now(timezone.utc),
                                reason='Auto lock')

        self._notify_exit(trade, order_type)
        # In case of market sell orders the order can be closed immediately
        if order.get('status', 'unknown') in ('closed', 'expired'):
            self.update_trade_state(trade, trade.open_order_id, order)
        Trade.commit()

        return True

    def _notify_exit(self, trade: Trade, order_type: str, fill: bool = False) -> None:
        """
        Sends rpc notification when a sell occurred.
        """
        profit_rate = trade.close_rate if trade.close_rate else trade.close_rate_requested
        profit_trade = trade.calc_profit(rate=profit_rate)
        # Use cached rates here - it was updated seconds ago.
        current_rate = self.exchange.get_rate(
            trade.pair, refresh=False, side=trade.exit_side) if not fill else None
        profit_ratio = trade.calc_profit_ratio(profit_rate)
        gain = "profit" if profit_ratio > 0 else "loss"

        msg = {
            'type': (RPCMessageType.SELL_FILL if fill
                     else RPCMessageType.SELL),
            'trade_id': trade.id,
            'exchange': trade.exchange.capitalize(),
            'pair': trade.pair,
            'leverage': trade.leverage,
            'direction': 'Short' if trade.is_short else 'Long',
            'gain': gain,
            'limit': profit_rate,
            'order_type': order_type,
            'amount': trade.amount,
            'open_rate': trade.open_rate,
            'close_rate': trade.close_rate,
            'current_rate': current_rate,
            'profit_amount': profit_trade,
            'profit_ratio': profit_ratio,
            'buy_tag': trade.enter_tag,
            'enter_tag': trade.enter_tag,
            'sell_reason': trade.sell_reason,
            'open_date': trade.open_date,
            'close_date': trade.close_date or datetime.utcnow(),
            'stake_currency': self.config['stake_currency'],
            'fiat_currency': self.config.get('fiat_display_currency', None),
        }

        if 'fiat_display_currency' in self.config:
            msg.update({
                'fiat_currency': self.config['fiat_display_currency'],
            })

        # Send the message
        self.rpc.send_msg(msg)

    def _notify_exit_cancel(self, trade: Trade, order_type: str, reason: str) -> None:
        """
        Sends rpc notification when a sell cancel occurred.
        """
        if trade.sell_order_status == reason:
            return
        else:
            trade.sell_order_status = reason

        profit_rate = trade.close_rate if trade.close_rate else trade.close_rate_requested
        profit_trade = trade.calc_profit(rate=profit_rate)
        current_rate = self.exchange.get_rate(trade.pair, refresh=False, side=trade.exit_side)
        profit_ratio = trade.calc_profit_ratio(profit_rate)
        gain = "profit" if profit_ratio > 0 else "loss"

        msg = {
            'type': RPCMessageType.SELL_CANCEL,
            'trade_id': trade.id,
            'exchange': trade.exchange.capitalize(),
            'pair': trade.pair,
            'leverage': trade.leverage,
            'direction': 'Short' if trade.is_short else 'Long',
            'gain': gain,
            'limit': profit_rate or 0,
            'order_type': order_type,
            'amount': trade.amount,
            'open_rate': trade.open_rate,
            'current_rate': current_rate,
            'profit_amount': profit_trade,
            'profit_ratio': profit_ratio,
            'buy_tag': trade.enter_tag,
            'enter_tag': trade.enter_tag,
            'sell_reason': trade.sell_reason,
            'open_date': trade.open_date,
            'close_date': trade.close_date or datetime.now(timezone.utc),
            'stake_currency': self.config['stake_currency'],
            'fiat_currency': self.config.get('fiat_display_currency', None),
            'reason': reason,
        }

        if 'fiat_display_currency' in self.config:
            msg.update({
                'fiat_currency': self.config['fiat_display_currency'],
            })

        # Send the message
        self.rpc.send_msg(msg)

#
# Common update trade state methods
#

    def update_trade_state(self, trade: Trade, order_id: str, action_order: Dict[str, Any] = None,
                           stoploss_order: bool = False, send_msg: bool = True) -> bool:
        """
        Checks trades with open orders and updates the amount if necessary
        Handles closing both buy and sell orders.
        :param trade: Trade object of the trade we're analyzing
        :param order_id: Order-id of the order we're analyzing
        :param action_order: Already acquired order object
        :param send_msg: Send notification - should always be True except in "recovery" methods
        :return: True if order has been cancelled without being filled partially, False otherwise
        """
        if not order_id:
            logger.warning(f'Orderid for trade {trade} is empty.')
            return False

        # Update trade with order values
        logger.info(f'Found open order for {trade}')
        try:
            order = action_order or self.exchange.fetch_order_or_stoploss_order(order_id,
                                                                                trade.pair,
                                                                                stoploss_order)
        except InvalidOrderException as exception:
            logger.warning('Unable to fetch order %s: %s', order_id, exception)
            return False

        trade.update_order(order)

        if self.exchange.check_order_canceled_empty(order):
            # Trade has been cancelled on exchange
            # Handling of this will happen in check_handle_timedout.
            return True

        order = self.handle_order_fee(trade, order)

        trade.update(order)
        trade.recalc_trade_from_orders()
        Trade.commit()

        if order['status'] in constants.NON_OPEN_EXCHANGE_STATES:
            # If a buy order was closed, force update on stoploss on exchange
            if order.get('side', None) == 'buy':
                trade = self.cancel_stoploss_on_exchange(trade)
            # Updating wallets when order is closed
            self.wallets.update()

        if not trade.is_open:
            if send_msg and not stoploss_order and not trade.open_order_id:
                self._notify_exit(trade, '', True)
            self.handle_protections(trade.pair)
        elif send_msg and not trade.open_order_id:
            # Buy fill
            self._notify_enter(trade, order, fill=True)

        return False

    def handle_protections(self, pair: str) -> None:
        prot_trig = self.protections.stop_per_pair(pair)
        if prot_trig:
            msg = {'type': RPCMessageType.PROTECTION_TRIGGER, }
            msg.update(prot_trig.to_json())
            self.rpc.send_msg(msg)

        prot_trig_glb = self.protections.global_stop()
        if prot_trig_glb:
            msg = {'type': RPCMessageType.PROTECTION_TRIGGER_GLOBAL, }
            msg.update(prot_trig_glb.to_json())
            self.rpc.send_msg(msg)

    def apply_fee_conditional(self, trade: Trade, trade_base_currency: str,
                              amount: float, fee_abs: float) -> float:
        """
        Applies the fee to amount (either from Order or from Trades).
        Can eat into dust if more than the required asset is available.
        Can't happen in Futures mode - where Fees are always in settlement currency,
        never in base currency.
        """
        self.wallets.update()
        if fee_abs != 0 and self.wallets.get_free(trade_base_currency) >= amount:
            # Eat into dust if we own more than base currency
            logger.info(f"Fee amount for {trade} was in base currency - "
                        f"Eating Fee {fee_abs} into dust.")
        elif fee_abs != 0:
            real_amount = self.exchange.amount_to_precision(trade.pair, amount - fee_abs)
            logger.info(f"Applying fee on amount for {trade} "
                        f"(from {amount} to {real_amount}).")
            return real_amount
        return amount

    def handle_order_fee(self, trade: Trade, order: Dict[str, Any]) -> Dict[str, Any]:
        # Try update amount (binance-fix)
        try:
            new_amount = self.get_real_amount(trade, order)
            if not isclose(safe_value_fallback(order, 'filled', 'amount'), new_amount,
                           abs_tol=constants.MATH_CLOSE_PREC):
                order['amount'] = new_amount
                order.pop('filled', None)
        except DependencyException as exception:
            logger.warning("Could not update trade amount: %s", exception)
        return order

    def get_real_amount(self, trade: Trade, order: Dict) -> float:
        """
        Detect and update trade fee.
        Calls trade.update_fee() upon correct detection.
        Returns modified amount if the fee was taken from the destination currency.
        Necessary for exchanges which charge fees in base currency (e.g. binance)
        :return: identical (or new) amount for the trade
        """
        # Init variables
        order_amount = safe_value_fallback(order, 'filled', 'amount')
        # Only run for closed orders
        if trade.fee_updated(order.get('side', '')) or order['status'] == 'open':
            return order_amount

        trade_base_currency = self.exchange.get_pair_base_currency(trade.pair)
        # use fee from order-dict if possible
        if self.exchange.order_has_fee(order):
            fee_cost, fee_currency, fee_rate = self.exchange.extract_cost_curr_rate(order)
            logger.info(f"Fee for Trade {trade} [{order.get('side')}]: "
                        f"{fee_cost:.8g} {fee_currency} - rate: {fee_rate}")
            if fee_rate is None or fee_rate < 0.02:
                # Reject all fees that report as > 2%.
                # These are most likely caused by a parsing bug in ccxt
                # due to multiple trades (https://github.com/ccxt/ccxt/issues/8025)
                trade.update_fee(fee_cost, fee_currency, fee_rate, order.get('side', ''))
                if trade_base_currency == fee_currency:
                    # Apply fee to amount
                    return self.apply_fee_conditional(trade, trade_base_currency,
                                                      amount=order_amount, fee_abs=fee_cost)
                return order_amount
        return self.fee_detection_from_trades(trade, order, order_amount, order.get('trades', []))

    def fee_detection_from_trades(self, trade: Trade, order: Dict, order_amount: float,
                                  trades: List) -> float:
        """
        fee-detection fallback to Trades.
        Either uses provided trades list or the result of fetch_my_trades to get correct fee.
        """
        if not trades:
            trades = self.exchange.get_trades_for_order(
                self.exchange.get_order_id_conditional(order), trade.pair, trade.open_date)

        if len(trades) == 0:
            logger.info("Applying fee on amount for %s failed: myTrade-Dict empty found", trade)
            return order_amount
        fee_currency = None
        amount = 0
        fee_abs = 0.0
        fee_cost = 0.0
        trade_base_currency = self.exchange.get_pair_base_currency(trade.pair)
        fee_rate_array: List[float] = []
        for exectrade in trades:
            amount += exectrade['amount']
            if self.exchange.order_has_fee(exectrade):
                fee_cost_, fee_currency, fee_rate_ = self.exchange.extract_cost_curr_rate(exectrade)
                fee_cost += fee_cost_
                if fee_rate_ is not None:
                    fee_rate_array.append(fee_rate_)
                # only applies if fee is in quote currency!
                if trade_base_currency == fee_currency:
                    fee_abs += fee_cost_
        # Ensure at least one trade was found:
        if fee_currency:
            # fee_rate should use mean
            fee_rate = sum(fee_rate_array) / float(len(fee_rate_array)) if fee_rate_array else None
            if fee_rate is not None and fee_rate < 0.02:
                # Only update if fee-rate is < 2%
                trade.update_fee(fee_cost, fee_currency, fee_rate, order.get('side', ''))

        if not isclose(amount, order_amount, abs_tol=constants.MATH_CLOSE_PREC):
            # TODO-lev: leverage?
            logger.warning(f"Amount {amount} does not match amount {trade.amount}")
            raise DependencyException("Half bought? Amounts don't match")

        if fee_abs != 0:
            return self.apply_fee_conditional(trade, trade_base_currency,
                                              amount=amount, fee_abs=fee_abs)
        else:
            return amount

    def get_valid_price(self, custom_price: float, proposed_price: float) -> float:
        """
        Return the valid price.
        Check if the custom price is of the good type if not return proposed_price
        :return: valid price for the order
        """
        if custom_price:
            try:
                valid_custom_price = float(custom_price)
            except ValueError:
                valid_custom_price = proposed_price
        else:
            valid_custom_price = proposed_price

        cust_p_max_dist_r = self.config.get('custom_price_max_distance_ratio', 0.02)
        min_custom_price_allowed = proposed_price - (proposed_price * cust_p_max_dist_r)
        max_custom_price_allowed = proposed_price + (proposed_price * cust_p_max_dist_r)

        # Bracket between min_custom_price_allowed and max_custom_price_allowed
        return max(
            min(valid_custom_price, max_custom_price_allowed),
            min_custom_price_allowed)
