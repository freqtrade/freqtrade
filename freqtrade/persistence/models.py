"""
This module contains the class to persist trades into SQLite
"""
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import arrow
from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer, String,
                        create_engine, desc, func, inspect)
from sqlalchemy.exc import NoSuchModuleError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Query, relationship
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.sql.schema import UniqueConstraint

from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.misc import safe_value_fallback
from freqtrade.persistence.migrations import check_migrate


logger = logging.getLogger(__name__)


_DECL_BASE: Any = declarative_base()
_SQL_DOCS_URL = 'http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls'


def init_db(db_url: str, clean_open_orders: bool = False) -> None:
    """
    Initializes this module with the given config,
    registers all known command handlers
    and starts polling for message updates
    :param db_url: Database to use
    :param clean_open_orders: Remove open orders from the database.
        Useful for dry-run or if all orders have been reset on the exchange.
    :return: None
    """
    kwargs = {}

    # Take care of thread ownership if in-memory db
    if db_url == 'sqlite://':
        kwargs.update({
            'connect_args': {'check_same_thread': False},
            'poolclass': StaticPool,
            'echo': False,
        })

    try:
        engine = create_engine(db_url, **kwargs)
    except NoSuchModuleError:
        raise OperationalException(f"Given value for db_url: '{db_url}' "
                                   f"is no valid database URL! (See {_SQL_DOCS_URL})")

    # https://docs.sqlalchemy.org/en/13/orm/contextual.html#thread-local-scope
    # Scoped sessions proxy requests to the appropriate thread-local session.
    # We should use the scoped_session object - not a seperately initialized version
    Trade.session = scoped_session(sessionmaker(bind=engine, autoflush=True, autocommit=True))
    Trade.query = Trade.session.query_property()
    # Copy session attributes to order object too
    Order.session = Trade.session
    Order.query = Order.session.query_property()
    PairLock.session = Trade.session
    PairLock.query = PairLock.session.query_property()

    previous_tables = inspect(engine).get_table_names()
    _DECL_BASE.metadata.create_all(engine)
    check_migrate(engine, decl_base=_DECL_BASE, previous_tables=previous_tables)

    # Clean dry_run DB if the db is not in-memory
    if clean_open_orders and db_url != 'sqlite://':
        clean_dry_run_db()


def cleanup_db() -> None:
    """
    Flushes all pending operations to disk.
    :return: None
    """
    Trade.session.flush()


def clean_dry_run_db() -> None:
    """
    Remove open_order_id from a Dry_run DB
    :return: None
    """
    for trade in Trade.query.filter(Trade.open_order_id.isnot(None)).all():
        # Check we are updating only a dry_run order not a prod one
        if 'dry_run' in trade.open_order_id:
            trade.open_order_id = None


class Order(_DECL_BASE):
    """
    Order database model
    Keeps a record of all orders placed on the exchange

    One to many relationship with Trades:
      - One trade can have many orders
      - One Order can only be associated with one Trade

    Mirrors CCXT Order structure
    """
    __tablename__ = 'orders'
    # Uniqueness should be ensured over pair, order_id
    # its likely that order_id is unique per Pair on some exchanges.
    __table_args__ = (UniqueConstraint('ft_pair', 'order_id', name="_order_pair_order_id"),)

    id = Column(Integer, primary_key=True)
    ft_trade_id = Column(Integer, ForeignKey('trades.id'), index=True)

    trade = relationship("Trade", back_populates="orders")

    ft_order_side = Column(String, nullable=False)
    ft_pair = Column(String, nullable=False)
    ft_is_open = Column(Boolean, nullable=False, default=True, index=True)

    order_id = Column(String, nullable=False, index=True)
    status = Column(String, nullable=True)
    symbol = Column(String, nullable=True)
    order_type = Column(String, nullable=True)
    side = Column(String, nullable=True)
    price = Column(Float, nullable=True)
    amount = Column(Float, nullable=True)
    filled = Column(Float, nullable=True)
    remaining = Column(Float, nullable=True)
    cost = Column(Float, nullable=True)
    order_date = Column(DateTime, nullable=True, default=datetime.utcnow)
    order_filled_date = Column(DateTime, nullable=True)
    order_update_date = Column(DateTime, nullable=True)

    def __repr__(self):

        return (f'Order(id={self.id}, order_id={self.order_id}, trade_id={self.ft_trade_id}, '
                f'side={self.side}, order_type={self.order_type}, status={self.status})')

    def update_from_ccxt_object(self, order):
        """
        Update Order from ccxt response
        Only updates if fields are available from ccxt -
        """
        if self.order_id != str(order['id']):
            raise DependencyException("Order-id's don't match")

        self.status = order.get('status', self.status)
        self.symbol = order.get('symbol', self.symbol)
        self.order_type = order.get('type', self.order_type)
        self.side = order.get('side', self.side)
        self.price = order.get('price', self.price)
        self.amount = order.get('amount', self.amount)
        self.filled = order.get('filled', self.filled)
        self.remaining = order.get('remaining', self.remaining)
        self.cost = order.get('cost', self.cost)
        if 'timestamp' in order and order['timestamp'] is not None:
            self.order_date = datetime.fromtimestamp(order['timestamp'] / 1000, tz=timezone.utc)

        self.ft_is_open = True
        if self.status in ('closed', 'canceled', 'cancelled'):
            self.ft_is_open = False
            if order.get('filled', 0) > 0:
                self.order_filled_date = arrow.utcnow().datetime
        self.order_update_date = arrow.utcnow().datetime

    @staticmethod
    def update_orders(orders: List['Order'], order: Dict[str, Any]):
        """
        Get all non-closed orders - useful when trying to batch-update orders
        """
        if not isinstance(order, dict):
            logger.warning(f"{order} is not a valid response object.")
            return

        filtered_orders = [o for o in orders if o.order_id == order.get('id')]
        if filtered_orders:
            oobj = filtered_orders[0]
            oobj.update_from_ccxt_object(order)
        else:
            logger.warning(f"Did not find order for {order}.")

    @staticmethod
    def parse_from_ccxt_object(order: Dict[str, Any], pair: str, side: str) -> 'Order':
        """
        Parse an order from a ccxt object and return a new order Object.
        """
        o = Order(order_id=str(order['id']), ft_order_side=side, ft_pair=pair)

        o.update_from_ccxt_object(order)
        return o

    @staticmethod
    def get_open_orders() -> List['Order']:
        """
        """
        return Order.query.filter(Order.ft_is_open.is_(True)).all()


class Trade(_DECL_BASE):
    """
    Trade database model.
    Also handles updating and querying trades
    """
    __tablename__ = 'trades'

    use_db: bool = True
    # Trades container for backtesting
    trades: List['Trade'] = []

    id = Column(Integer, primary_key=True)

    orders = relationship("Order", order_by="Order.id", cascade="all, delete-orphan")

    exchange = Column(String, nullable=False)
    pair = Column(String, nullable=False, index=True)
    is_open = Column(Boolean, nullable=False, default=True, index=True)
    fee_open = Column(Float, nullable=False, default=0.0)
    fee_open_cost = Column(Float, nullable=True)
    fee_open_currency = Column(String, nullable=True)
    fee_close = Column(Float, nullable=False, default=0.0)
    fee_close_cost = Column(Float, nullable=True)
    fee_close_currency = Column(String, nullable=True)
    open_rate = Column(Float)
    open_rate_requested = Column(Float)
    # open_trade_value - calculated via _calc_open_trade_value
    open_trade_value = Column(Float)
    close_rate = Column(Float)
    close_rate_requested = Column(Float)
    close_profit = Column(Float)
    close_profit_abs = Column(Float)
    stake_amount = Column(Float, nullable=False)
    amount = Column(Float)
    amount_requested = Column(Float)
    open_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    close_date = Column(DateTime)
    open_order_id = Column(String)
    # absolute value of the stop loss
    stop_loss = Column(Float, nullable=True, default=0.0)
    # percentage value of the stop loss
    stop_loss_pct = Column(Float, nullable=True)
    # absolute value of the initial stop loss
    initial_stop_loss = Column(Float, nullable=True, default=0.0)
    # percentage value of the initial stop loss
    initial_stop_loss_pct = Column(Float, nullable=True)
    # stoploss order id which is on exchange
    stoploss_order_id = Column(String, nullable=True, index=True)
    # last update time of the stoploss order on exchange
    stoploss_last_update = Column(DateTime, nullable=True)
    # absolute value of the highest reached price
    max_rate = Column(Float, nullable=True, default=0.0)
    # Lowest price reached
    min_rate = Column(Float, nullable=True)
    sell_reason = Column(String, nullable=True)
    sell_order_status = Column(String, nullable=True)
    strategy = Column(String, nullable=True)
    timeframe = Column(Integer, nullable=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recalc_open_trade_value()

    def __repr__(self):
        open_since = self.open_date.strftime(DATETIME_PRINT_FORMAT) if self.is_open else 'closed'

        return (f'Trade(id={self.id}, pair={self.pair}, amount={self.amount:.8f}, '
                f'open_rate={self.open_rate:.8f}, open_since={open_since})')

    def to_json(self) -> Dict[str, Any]:
        return {
            'trade_id': self.id,
            'pair': self.pair,
            'is_open': self.is_open,
            'exchange': self.exchange,
            'amount': round(self.amount, 8),
            'amount_requested': round(self.amount_requested, 8) if self.amount_requested else None,
            'stake_amount': round(self.stake_amount, 8),
            'strategy': self.strategy,
            'timeframe': self.timeframe,

            'fee_open': self.fee_open,
            'fee_open_cost': self.fee_open_cost,
            'fee_open_currency': self.fee_open_currency,
            'fee_close': self.fee_close,
            'fee_close_cost': self.fee_close_cost,
            'fee_close_currency': self.fee_close_currency,

            'open_date_hum': arrow.get(self.open_date).humanize(),
            'open_date': self.open_date.strftime(DATETIME_PRINT_FORMAT),
            'open_timestamp': int(self.open_date.replace(tzinfo=timezone.utc).timestamp() * 1000),
            'open_rate': self.open_rate,
            'open_rate_requested': self.open_rate_requested,
            'open_trade_value': round(self.open_trade_value, 8),

            'close_date_hum': (arrow.get(self.close_date).humanize()
                               if self.close_date else None),
            'close_date': (self.close_date.strftime(DATETIME_PRINT_FORMAT)
                           if self.close_date else None),
            'close_timestamp': int(self.close_date.replace(
                tzinfo=timezone.utc).timestamp() * 1000) if self.close_date else None,
            'close_rate': self.close_rate,
            'close_rate_requested': self.close_rate_requested,
            'close_profit': self.close_profit,  # Deprecated
            'close_profit_pct': round(self.close_profit * 100, 2) if self.close_profit else None,
            'close_profit_abs': self.close_profit_abs,  # Deprecated

            'trade_duration_s': (int((self.close_date - self.open_date).total_seconds())
                                 if self.close_date else None),
            'trade_duration': (int((self.close_date - self.open_date).total_seconds() // 60)
                               if self.close_date else None),

            'profit_ratio': self.close_profit,
            'profit_pct': round(self.close_profit * 100, 2) if self.close_profit else None,
            'profit_abs': self.close_profit_abs,

            'sell_reason': self.sell_reason,
            'sell_order_status': self.sell_order_status,
            'stop_loss_abs': self.stop_loss,
            'stop_loss_ratio': self.stop_loss_pct if self.stop_loss_pct else None,
            'stop_loss_pct': (self.stop_loss_pct * 100) if self.stop_loss_pct else None,
            'stoploss_order_id': self.stoploss_order_id,
            'stoploss_last_update': (self.stoploss_last_update.strftime(DATETIME_PRINT_FORMAT)
                                     if self.stoploss_last_update else None),
            'stoploss_last_update_timestamp': int(self.stoploss_last_update.replace(
                tzinfo=timezone.utc).timestamp() * 1000) if self.stoploss_last_update else None,
            'initial_stop_loss_abs': self.initial_stop_loss,
            'initial_stop_loss_ratio': (self.initial_stop_loss_pct
                                        if self.initial_stop_loss_pct else None),
            'initial_stop_loss_pct': (self.initial_stop_loss_pct * 100
                                      if self.initial_stop_loss_pct else None),
            'min_rate': self.min_rate,
            'max_rate': self.max_rate,

            'open_order_id': self.open_order_id,
        }

    @staticmethod
    def reset_trades() -> None:
        """
        Resets all trades. Only active for backtesting mode.
        """
        if not Trade.use_db:
            Trade.trades = []

    def adjust_min_max_rates(self, current_price: float) -> None:
        """
        Adjust the max_rate and min_rate.
        """
        self.max_rate = max(current_price, self.max_rate or self.open_rate)
        self.min_rate = min(current_price, self.min_rate or self.open_rate)

    def _set_new_stoploss(self, new_loss: float, stoploss: float):
        """Assign new stop value"""
        self.stop_loss = new_loss
        self.stop_loss_pct = -1 * abs(stoploss)
        self.stoploss_last_update = datetime.utcnow()

    def adjust_stop_loss(self, current_price: float, stoploss: float,
                         initial: bool = False) -> None:
        """
        This adjusts the stop loss to it's most recently observed setting
        :param current_price: Current rate the asset is traded
        :param stoploss: Stoploss as factor (sample -0.05 -> -5% below current price).
        :param initial: Called to initiate stop_loss.
            Skips everything if self.stop_loss is already set.
        """
        if initial and not (self.stop_loss is None or self.stop_loss == 0):
            # Don't modify if called with initial and nothing to do
            return

        new_loss = float(current_price * (1 - abs(stoploss)))

        # no stop loss assigned yet
        if not self.stop_loss:
            logger.debug(f"{self.pair} - Assigning new stoploss...")
            self._set_new_stoploss(new_loss, stoploss)
            self.initial_stop_loss = new_loss
            self.initial_stop_loss_pct = -1 * abs(stoploss)

        # evaluate if the stop loss needs to be updated
        else:
            if new_loss > self.stop_loss:  # stop losses only walk up, never down!
                logger.debug(f"{self.pair} - Adjusting stoploss...")
                self._set_new_stoploss(new_loss, stoploss)
            else:
                logger.debug(f"{self.pair} - Keeping current stoploss...")

        logger.debug(
            f"{self.pair} - Stoploss adjusted. current_price={current_price:.8f}, "
            f"open_rate={self.open_rate:.8f}, max_rate={self.max_rate:.8f}, "
            f"initial_stop_loss={self.initial_stop_loss:.8f}, "
            f"stop_loss={self.stop_loss:.8f}. "
            f"Trailing stoploss saved us: "
            f"{float(self.stop_loss) - float(self.initial_stop_loss):.8f}.")

    def update(self, order: Dict) -> None:
        """
        Updates this entity with amount and actual open/close rates.
        :param order: order retrieved by exchange.fetch_order()
        :return: None
        """
        order_type = order['type']
        # Ignore open and cancelled orders
        if order['status'] == 'open' or safe_value_fallback(order, 'average', 'price') is None:
            return

        logger.info('Updating trade (id=%s) ...', self.id)

        if order_type in ('market', 'limit') and order['side'] == 'buy':
            # Update open rate and actual amount
            self.open_rate = Decimal(safe_value_fallback(order, 'average', 'price'))
            self.amount = Decimal(safe_value_fallback(order, 'filled', 'amount'))
            self.recalc_open_trade_value()
            if self.is_open:
                logger.info(f'{order_type.upper()}_BUY has been fulfilled for {self}.')
            self.open_order_id = None
        elif order_type in ('market', 'limit') and order['side'] == 'sell':
            if self.is_open:
                logger.info(f'{order_type.upper()}_SELL has been fulfilled for {self}.')
            self.close(safe_value_fallback(order, 'average', 'price'))
        elif order_type in ('stop_loss_limit', 'stop-loss', 'stop-loss-limit', 'stop'):
            self.stoploss_order_id = None
            self.close_rate_requested = self.stop_loss
            if self.is_open:
                logger.info(f'{order_type.upper()} is hit for {self}.')
            self.close(order['average'])
        else:
            raise ValueError(f'Unknown order type: {order_type}')
        cleanup_db()

    def close(self, rate: float, *, show_msg: bool = True) -> None:
        """
        Sets close_rate to the given rate, calculates total profit
        and marks trade as closed
        """
        self.close_rate = Decimal(rate)
        self.close_profit = self.calc_profit_ratio()
        self.close_profit_abs = self.calc_profit()
        self.close_date = self.close_date or datetime.utcnow()
        self.is_open = False
        self.sell_order_status = 'closed'
        self.open_order_id = None
        if show_msg:
            logger.info(
                'Marking %s as closed as the trade is fulfilled and found no open orders for it.',
                self
            )

    def update_fee(self, fee_cost: float, fee_currency: Optional[str], fee_rate: Optional[float],
                   side: str) -> None:
        """
        Update Fee parameters. Only acts once per side
        """
        if side == 'buy' and self.fee_open_currency is None:
            self.fee_open_cost = fee_cost
            self.fee_open_currency = fee_currency
            if fee_rate is not None:
                self.fee_open = fee_rate
                # Assume close-fee will fall into the same fee category and take an educated guess
                self.fee_close = fee_rate
        elif side == 'sell' and self.fee_close_currency is None:
            self.fee_close_cost = fee_cost
            self.fee_close_currency = fee_currency
            if fee_rate is not None:
                self.fee_close = fee_rate

    def fee_updated(self, side: str) -> bool:
        """
        Verify if this side (buy / sell) has already been updated
        """
        if side == 'buy':
            return self.fee_open_currency is not None
        elif side == 'sell':
            return self.fee_close_currency is not None
        else:
            return False

    def update_order(self, order: Dict) -> None:
        Order.update_orders(self.orders, order)

    def delete(self) -> None:

        for order in self.orders:
            Order.session.delete(order)

        Trade.session.delete(self)
        Trade.session.flush()

    def _calc_open_trade_value(self) -> float:
        """
        Calculate the open_rate including open_fee.
        :return: Price in of the open trade incl. Fees
        """
        buy_trade = Decimal(self.amount) * Decimal(self.open_rate)
        fees = buy_trade * Decimal(self.fee_open)
        return float(buy_trade + fees)

    def recalc_open_trade_value(self) -> None:
        """
        Recalculate open_trade_value.
        Must be called whenever open_rate or fee_open is changed.
        """
        self.open_trade_value = self._calc_open_trade_value()

    def calc_close_trade_value(self, rate: Optional[float] = None,
                               fee: Optional[float] = None) -> float:
        """
        Calculate the close_rate including fee
        :param fee: fee to use on the close rate (optional).
            If rate is not set self.fee will be used
        :param rate: rate to compare with (optional).
            If rate is not set self.close_rate will be used
        :return: Price in BTC of the open trade
        """
        if rate is None and not self.close_rate:
            return 0.0

        sell_trade = Decimal(self.amount) * Decimal(rate or self.close_rate)
        fees = sell_trade * Decimal(fee or self.fee_close)
        return float(sell_trade - fees)

    def calc_profit(self, rate: Optional[float] = None,
                    fee: Optional[float] = None) -> float:
        """
        Calculate the absolute profit in stake currency between Close and Open trade
        :param fee: fee to use on the close rate (optional).
            If rate is not set self.fee will be used
        :param rate: close rate to compare with (optional).
            If rate is not set self.close_rate will be used
        :return:  profit in stake currency as float
        """
        close_trade_value = self.calc_close_trade_value(
            rate=(rate or self.close_rate),
            fee=(fee or self.fee_close)
        )
        profit = close_trade_value - self.open_trade_value
        return float(f"{profit:.8f}")

    def calc_profit_ratio(self, rate: Optional[float] = None,
                          fee: Optional[float] = None) -> float:
        """
        Calculates the profit as ratio (including fee).
        :param rate: rate to compare with (optional).
            If rate is not set self.close_rate will be used
        :param fee: fee to use on the close rate (optional).
        :return: profit ratio as float
        """
        close_trade_value = self.calc_close_trade_value(
            rate=(rate or self.close_rate),
            fee=(fee or self.fee_close)
        )
        profit_ratio = (close_trade_value / self.open_trade_value) - 1
        return float(f"{profit_ratio:.8f}")

    def select_order(self, order_side: str, is_open: Optional[bool]) -> Optional[Order]:
        """
        Finds latest order for this orderside and status
        :param order_side: Side of the order (either 'buy' or 'sell')
        :param is_open: Only search for open orders?
        :return: latest Order object if it exists, else None
        """
        orders = [o for o in self.orders if o.side == order_side]
        if is_open is not None:
            orders = [o for o in orders if o.ft_is_open == is_open]
        if len(orders) > 0:
            return orders[-1]
        else:
            return None

    @staticmethod
    def get_trades(trade_filter=None) -> Query:
        """
        Helper function to query Trades using filters.
        :param trade_filter: Optional filter to apply to trades
                             Can be either a Filter object, or a List of filters
                             e.g. `(trade_filter=[Trade.id == trade_id, Trade.is_open.is_(True),])`
                             e.g. `(trade_filter=Trade.id == trade_id)`
        :return: unsorted query object
        """
        if trade_filter is not None:
            if not isinstance(trade_filter, list):
                trade_filter = [trade_filter]
            return Trade.query.filter(*trade_filter)
        else:
            return Trade.query

    @staticmethod
    def get_trades_proxy(*, pair: str = None, is_open: bool = None,
                         open_date: datetime = None, close_date: datetime = None,
                         ) -> List['Trade']:
        """
        Helper function to query Trades.
        Returns a List of trades, filtered on the parameters given.
        In live mode, converts the filter to a database query and returns all rows
        In Backtest mode, uses filters on Trade.trades to get the result.

        :return: unsorted List[Trade]
        """
        if Trade.use_db:
            trade_filter = []
            if pair:
                trade_filter.append(Trade.pair == pair)
            if open_date:
                trade_filter.append(Trade.open_date > open_date)
            if close_date:
                trade_filter.append(Trade.close_date > close_date)
            if is_open is not None:
                trade_filter.append(Trade.is_open.is_(is_open))
            return Trade.get_trades(trade_filter).all()
        else:
            # Offline mode - without database
            sel_trades = [trade for trade in Trade.trades]
            if pair:
                sel_trades = [trade for trade in sel_trades if trade.pair == pair]
            if open_date:
                sel_trades = [trade for trade in sel_trades if trade.open_date > open_date]
            if close_date:
                sel_trades = [trade for trade in sel_trades if trade.close_date
                              and trade.close_date > close_date]
            if is_open is not None:
                sel_trades = [trade for trade in sel_trades if trade.is_open == is_open]
            return sel_trades

    @staticmethod
    def get_open_trades() -> List[Any]:
        """
        Query trades from persistence layer
        """
        return Trade.get_trades(Trade.is_open.is_(True)).all()

    @staticmethod
    def get_open_order_trades():
        """
        Returns all open trades
        """
        return Trade.get_trades(Trade.open_order_id.isnot(None)).all()

    @staticmethod
    def get_open_trades_without_assigned_fees():
        """
        Returns all open trades which don't have open fees set correctly
        """
        return Trade.get_trades([Trade.fee_open_currency.is_(None),
                                 Trade.orders.any(),
                                 Trade.is_open.is_(True),
                                 ]).all()

    @staticmethod
    def get_sold_trades_without_assigned_fees():
        """
        Returns all closed trades which don't have fees set correctly
        """
        return Trade.get_trades([Trade.fee_close_currency.is_(None),
                                 Trade.orders.any(),
                                 Trade.is_open.is_(False),
                                 ]).all()

    @staticmethod
    def total_open_trades_stakes() -> float:
        """
        Calculates total invested amount in open trades
        in stake currency
        """
        total_open_stake_amount = Trade.session.query(func.sum(Trade.stake_amount))\
            .filter(Trade.is_open.is_(True))\
            .scalar()
        return total_open_stake_amount or 0

    @staticmethod
    def get_overall_performance() -> List[Dict[str, Any]]:
        """
        Returns List of dicts containing all Trades, including profit and trade count
        """
        pair_rates = Trade.session.query(
            Trade.pair,
            func.sum(Trade.close_profit).label('profit_sum'),
            func.count(Trade.pair).label('count')
        ).filter(Trade.is_open.is_(False))\
            .group_by(Trade.pair) \
            .order_by(desc('profit_sum')) \
            .all()
        return [
            {
                'pair': pair,
                'profit': rate,
                'count': count
            }
            for pair, rate, count in pair_rates
        ]

    @staticmethod
    def get_best_pair():
        """
        Get best pair with closed trade.
        :returns: Tuple containing (pair, profit_sum)
        """
        best_pair = Trade.session.query(
            Trade.pair, func.sum(Trade.close_profit).label('profit_sum')
        ).filter(Trade.is_open.is_(False)) \
            .group_by(Trade.pair) \
            .order_by(desc('profit_sum')).first()
        return best_pair

    @staticmethod
    def stoploss_reinitialization(desired_stoploss):
        """
        Adjust initial Stoploss to desired stoploss for all open trades.
        """
        for trade in Trade.get_open_trades():
            logger.info("Found open trade: %s", trade)

            # skip case if trailing-stop changed the stoploss already.
            if (trade.stop_loss == trade.initial_stop_loss
               and trade.initial_stop_loss_pct != desired_stoploss):
                # Stoploss value got changed

                logger.info(f"Stoploss for {trade} needs adjustment...")
                # Force reset of stoploss
                trade.stop_loss = None
                trade.adjust_stop_loss(trade.open_rate, desired_stoploss)
                logger.info(f"New stoploss: {trade.stop_loss}.")


class PairLock(_DECL_BASE):
    """
    Pair Locks database model.
    """
    __tablename__ = 'pairlocks'

    id = Column(Integer, primary_key=True)

    pair = Column(String, nullable=False, index=True)
    reason = Column(String, nullable=True)
    # Time the pair was locked (start time)
    lock_time = Column(DateTime, nullable=False)
    # Time until the pair is locked (end time)
    lock_end_time = Column(DateTime, nullable=False, index=True)

    active = Column(Boolean, nullable=False, default=True, index=True)

    def __repr__(self):
        lock_time = self.lock_time.strftime(DATETIME_PRINT_FORMAT)
        lock_end_time = self.lock_end_time.strftime(DATETIME_PRINT_FORMAT)
        return (f'PairLock(id={self.id}, pair={self.pair}, lock_time={lock_time}, '
                f'lock_end_time={lock_end_time})')

    @staticmethod
    def query_pair_locks(pair: Optional[str], now: datetime) -> Query:
        """
        Get all currently active locks for this pair
        :param pair: Pair to check for. Returns all current locks if pair is empty
        :param now: Datetime object (generated via datetime.now(timezone.utc)).
        """

        filters = [PairLock.lock_end_time > now,
                   # Only active locks
                   PairLock.active.is_(True), ]
        if pair:
            filters.append(PairLock.pair == pair)
        return PairLock.query.filter(
            *filters
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'pair': self.pair,
            'lock_time': self.lock_time.strftime(DATETIME_PRINT_FORMAT),
            'lock_timestamp': int(self.lock_time.replace(tzinfo=timezone.utc).timestamp() * 1000),
            'lock_end_time': self.lock_end_time.strftime(DATETIME_PRINT_FORMAT),
            'lock_end_timestamp': int(self.lock_end_time.replace(tzinfo=timezone.utc
                                                                 ).timestamp() * 1000),
            'reason': self.reason,
            'active': self.active,
        }
