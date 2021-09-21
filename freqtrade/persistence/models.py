"""
This module contains the class to persist trades into SQLite
"""
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import (Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer, String,
                        create_engine, desc, func, inspect)
from sqlalchemy.exc import NoSuchModuleError
from sqlalchemy.orm import Query, declarative_base, relationship, scoped_session, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.sql.schema import UniqueConstraint

from freqtrade.constants import DATETIME_PRINT_FORMAT, NON_OPEN_EXCHANGE_STATES
from freqtrade.enums import Collateral, SellType, TradingMode
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.leverage import interest
from freqtrade.leverage import liquidation_price
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

    if db_url == 'sqlite://':
        kwargs.update({
            'poolclass': StaticPool,
        })
    # Take care of thread ownership
    if db_url.startswith('sqlite://'):
        kwargs.update({
            'connect_args': {'check_same_thread': False},
        })

    try:
        engine = create_engine(db_url, future=True, **kwargs)
    except NoSuchModuleError:
        raise OperationalException(f"Given value for db_url: '{db_url}' "
                                   f"is no valid database URL! (See {_SQL_DOCS_URL})")

    # https://docs.sqlalchemy.org/en/13/orm/contextual.html#thread-local-scope
    # Scoped sessions proxy requests to the appropriate thread-local session.
    # We should use the scoped_session object - not a seperately initialized version
    Trade._session = scoped_session(sessionmaker(bind=engine, autoflush=True))
    Trade.query = Trade._session.query_property()
    Order.query = Trade._session.query_property()
    PairLock.query = Trade._session.query_property()

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
    Trade.commit()


def clean_dry_run_db() -> None:
    """
    Remove open_order_id from a Dry_run DB
    :return: None
    """
    for trade in Trade.query.filter(Trade.open_order_id.isnot(None)).all():
        # Check we are updating only a dry_run order not a prod one
        if 'dry_run' in trade.open_order_id:
            trade.open_order_id = None
    Trade.commit()


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

    ft_order_side = Column(String(25), nullable=False)
    ft_pair = Column(String(25), nullable=False)
    ft_is_open = Column(Boolean, nullable=False, default=True, index=True)

    order_id = Column(String(255), nullable=False, index=True)
    status = Column(String(255), nullable=True)
    symbol = Column(String(25), nullable=True)
    order_type = Column(String(50), nullable=True)
    side = Column(String(25), nullable=True)
    price = Column(Float, nullable=True)
    average = Column(Float, nullable=True)
    amount = Column(Float, nullable=True)
    filled = Column(Float, nullable=True)
    remaining = Column(Float, nullable=True)
    cost = Column(Float, nullable=True)
    order_date = Column(DateTime, nullable=True, default=datetime.utcnow)
    order_filled_date = Column(DateTime, nullable=True)
    order_update_date = Column(DateTime, nullable=True)

    leverage = Column(Float, nullable=True, default=1.0)

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
        self.average = order.get('average', self.average)
        self.remaining = order.get('remaining', self.remaining)
        self.cost = order.get('cost', self.cost)
        self.leverage = order.get('leverage', self.leverage)

        if 'timestamp' in order and order['timestamp'] is not None:
            self.order_date = datetime.fromtimestamp(order['timestamp'] / 1000, tz=timezone.utc)

        self.ft_is_open = True
        if self.status in NON_OPEN_EXCHANGE_STATES:
            self.ft_is_open = False
            if (order.get('filled', 0.0) or 0.0) > 0:
                self.order_filled_date = datetime.now(timezone.utc)
        self.order_update_date = datetime.now(timezone.utc)

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
            Order.query.session.commit()
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
        Retrieve open orders from the database
        :return: List of open orders
        """
        return Order.query.filter(Order.ft_is_open.is_(True)).all()


class LocalTrade():
    """
    Trade database model.
    Used in backtesting - must be aligned to Trade model!

    """
    use_db: bool = False
    # Trades container for backtesting
    trades: List['LocalTrade'] = []
    trades_open: List['LocalTrade'] = []
    total_profit: float = 0

    id: int = 0

    orders: List[Order] = []

    exchange: str = ''
    pair: str = ''
    is_open: bool = True
    fee_open: float = 0.0
    fee_open_cost: Optional[float] = None
    fee_open_currency: str = ''
    fee_close: float = 0.0
    fee_close_cost: Optional[float] = None
    fee_close_currency: str = ''
    open_rate: float = 0.0
    open_rate_requested: Optional[float] = None
    # open_trade_value - calculated via _calc_open_trade_value
    open_trade_value: float = 0.0
    close_rate: Optional[float] = None
    close_rate_requested: Optional[float] = None
    close_profit: Optional[float] = None
    close_profit_abs: Optional[float] = None
    stake_amount: float = 0.0
    amount: float = 0.0
    amount_requested: Optional[float] = None
    open_date: datetime
    close_date: Optional[datetime] = None
    open_order_id: Optional[str] = None
    # absolute value of the stop loss
    stop_loss: float = 0.0
    # percentage value of the stop loss
    stop_loss_pct: float = 0.0
    # absolute value of the initial stop loss
    initial_stop_loss: float = 0.0
    # percentage value of the initial stop loss
    initial_stop_loss_pct: float = 0.0
    # stoploss order id which is on exchange
    stoploss_order_id: Optional[str] = None
    # last update time of the stoploss order on exchange
    stoploss_last_update: Optional[datetime] = None
    # absolute value of the highest reached price
    max_rate: float = 0.0
    # Lowest price reached
    min_rate: float = 0.0
    sell_reason: str = ''
    sell_order_status: str = ''
    strategy: str = ''
    enter_tag: Optional[str] = None
    timeframe: Optional[int] = None

    trading_mode: TradingMode = TradingMode.SPOT

    # Leverage trading properties
    isolated_liq: Optional[float] = None
    is_short: bool = False
    leverage: float = 1.0

    # Margin trading properties
    interest_rate: float = 0.0

    # Futures properties
    funding_fees: Optional[float] = None

    @property
    def buy_tag(self) -> Optional[str]:
        """
        Compatibility between buy_tag (old) and enter_tag (new)
        Consider buy_tag deprecated
        """
        return self.enter_tag

    @property
    def has_no_leverage(self) -> bool:
        """Returns true if this is a non-leverage, non-short trade"""
        return ((self.leverage == 1.0 or self.leverage is None) and not self.is_short)

    @property
    def borrowed(self) -> float:
        """
            The amount of currency borrowed from the exchange for leverage trades
            If a long trade, the amount is in base currency
            If a short trade, the amount is in the other currency being traded
        """
        if self.has_no_leverage:
            return 0.0
        elif not self.is_short:
            return (self.amount * self.open_rate) * ((self.leverage-1)/self.leverage)
        else:
            return self.amount

    @property
    def open_date_utc(self):
        return self.open_date.replace(tzinfo=timezone.utc)

    @property
    def close_date_utc(self):
        return self.close_date.replace(tzinfo=timezone.utc)

    @property
    def enter_side(self) -> str:
        if self.is_short:
            return "sell"
        else:
            return "buy"

    @property
    def exit_side(self) -> str:
        if self.is_short:
            return "buy"
        else:
            return "sell"

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if self.isolated_liq:
            self.set_isolated_liq(isolated_liq=self.isolated_liq)
        self.recalc_open_trade_value()
        if self.trading_mode == TradingMode.MARGIN and self.interest_rate is None:
            raise OperationalException(
                f"{self.trading_mode.value} trading requires param interest_rate on trades")

    def _set_stop_loss(self, stop_loss: float, percent: float):
        """
        Method you should use to set self.stop_loss.
        Assures stop_loss is not passed the liquidation price
        """
        if self.isolated_liq is not None:
            if self.is_short:
                sl = min(stop_loss, self.isolated_liq)
            else:
                sl = max(stop_loss, self.isolated_liq)
        else:
            sl = stop_loss

        if not self.stop_loss:
            self.initial_stop_loss = sl
        self.stop_loss = sl

        if self.is_short:
            self.stop_loss_pct = abs(percent)
        else:
            self.stop_loss_pct = -1 * abs(percent)
        self.stoploss_last_update = datetime.utcnow()

    def set_isolated_liq(
        self,
        isolated_liq: Optional[float],
        wallet_balance: Optional[float],
        current_price: Optional[float]
    ):
        """
        Method you should use to set self.liquidation price.
        Assures stop_loss is not passed the liquidation price
        """
        if not isolated_liq:
            if not wallet_balance or not current_price:
                raise OperationalException(
                    "wallet balance must be passed to LocalTrade.set_isolated_liq when param"
                    "isolated_liq is None"
                )
            isolated_liq = liquidation_price(
                exchange_name=self.exchange,
                open_rate=self.open_rate,
                is_short=self.is_short,
                leverage=self.leverage,
                trading_mode=self.trading_mode,
                collateral=Collateral.ISOLATED,
                mm_ex_1=0.0,
                upnl_ex_1=0.0,
                position=self.amount * current_price,
                wallet_balance=wallet_balance,
                # TODO-lev: maintenance_amt,
                # TODO-lev: mm_rate,

            )
        if isolated_liq is None:
            raise OperationalException(
                "leverage/isolated_liq returned None. This exception should never happen"
            )

        if self.stop_loss is not None:
            if self.is_short:
                self.stop_loss = min(self.stop_loss, isolated_liq)
            else:
                self.stop_loss = max(self.stop_loss, isolated_liq)
        else:
            self.initial_stop_loss = isolated_liq
            self.stop_loss = isolated_liq

        self.isolated_liq = isolated_liq

    def __repr__(self):
        open_since = self.open_date.strftime(DATETIME_PRINT_FORMAT) if self.is_open else 'closed'
        leverage = self.leverage or 1.0
        is_short = self.is_short or False

        return (
            f'Trade(id={self.id}, pair={self.pair}, amount={self.amount:.8f}, '
            f'is_short={is_short}, leverage={leverage}, '
            f'open_rate={self.open_rate:.8f}, open_since={open_since})'
        )

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
            'buy_tag': self.enter_tag,
            'enter_tag': self.enter_tag,
            'timeframe': self.timeframe,

            'fee_open': self.fee_open,
            'fee_open_cost': self.fee_open_cost,
            'fee_open_currency': self.fee_open_currency,
            'fee_close': self.fee_close,
            'fee_close_cost': self.fee_close_cost,
            'fee_close_currency': self.fee_close_currency,

            'open_date': self.open_date.strftime(DATETIME_PRINT_FORMAT),
            'open_timestamp': int(self.open_date.replace(tzinfo=timezone.utc).timestamp() * 1000),
            'open_rate': self.open_rate,
            'open_rate_requested': self.open_rate_requested,
            'open_trade_value': round(self.open_trade_value, 8),

            'close_date': (self.close_date.strftime(DATETIME_PRINT_FORMAT)
                           if self.close_date else None),
            'close_timestamp': int(self.close_date.replace(
                tzinfo=timezone.utc).timestamp() * 1000) if self.close_date else None,
            'close_rate': self.close_rate,
            'close_rate_requested': self.close_rate_requested,
            'close_profit': self.close_profit,  # Deprecated
            'close_profit_pct': round(self.close_profit * 100, 2) if self.close_profit else None,
            'close_profit_abs': self.close_profit_abs,  # Deprecated

            'trade_duration_s': (int((self.close_date_utc - self.open_date_utc).total_seconds())
                                 if self.close_date else None),
            'trade_duration': (int((self.close_date_utc - self.open_date_utc).total_seconds() // 60)
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

            'leverage': self.leverage,
            'interest_rate': self.interest_rate,
            'isolated_liq': self.isolated_liq,
            'is_short': self.is_short,
            'trading_mode': self.trading_mode,
            'funding_fees': self.funding_fees,
            'open_order_id': self.open_order_id,
        }

    @staticmethod
    def reset_trades() -> None:
        """
        Resets all trades. Only active for backtesting mode.
        """
        LocalTrade.trades = []
        LocalTrade.trades_open = []
        LocalTrade.total_profit = 0

    def adjust_min_max_rates(self, current_price: float, current_price_low: float) -> None:
        """
        Adjust the max_rate and min_rate.
        """
        self.max_rate = max(current_price, self.max_rate or self.open_rate)
        self.min_rate = min(current_price_low, self.min_rate or self.open_rate)

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

        if self.is_short:
            new_loss = float(current_price * (1 + abs(stoploss)))
            # If trading with leverage, don't set the stoploss below the liquidation price
            if self.isolated_liq:
                new_loss = min(self.isolated_liq, new_loss)
        else:
            new_loss = float(current_price * (1 - abs(stoploss)))
            # If trading with leverage, don't set the stoploss below the liquidation price
            if self.isolated_liq:
                new_loss = max(self.isolated_liq, new_loss)

        # no stop loss assigned yet
        if not self.stop_loss:
            logger.debug(f"{self.pair} - Assigning new stoploss...")
            self._set_stop_loss(new_loss, stoploss)
            self.initial_stop_loss = new_loss
            if self.is_short:
                self.initial_stop_loss_pct = abs(stoploss)
            else:
                self.initial_stop_loss_pct = -1 * abs(stoploss)

        # evaluate if the stop loss needs to be updated
        else:

            higher_stop = new_loss > self.stop_loss
            lower_stop = new_loss < self.stop_loss

            # stop losses only walk up, never down!,
            #   ? But adding more to a leveraged trade would create a lower liquidation price,
            #   ? decreasing the minimum stoploss
            if (higher_stop and not self.is_short) or (lower_stop and self.is_short):
                logger.debug(f"{self.pair} - Adjusting stoploss...")
                self._set_stop_loss(new_loss, stoploss)
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

        if 'is_short' in order and order['side'] == 'sell':
            # Only set's is_short on opening trades, ignores non-shorts
            self.is_short = order['is_short']

        # Ignore open and cancelled orders
        if order['status'] == 'open' or safe_value_fallback(order, 'average', 'price') is None:
            return

        logger.info('Updating trade (id=%s) ...', self.id)

        if order_type in ('market', 'limit') and self.enter_side == order['side']:
            # Update open rate and actual amount
            self.open_rate = float(safe_value_fallback(order, 'average', 'price'))
            self.amount = float(safe_value_fallback(order, 'filled', 'amount'))
            if 'leverage' in order:
                self.leverage = order['leverage']
            if self.is_open:
                payment = "SELL" if self.is_short else "BUY"
                logger.info(f'{order_type.upper()}_{payment} has been fulfilled for {self}.')
            self.open_order_id = None
            self.recalc_trade_from_orders()
        elif order_type in ('market', 'limit') and self.exit_side == order['side']:
            if self.is_open:
                payment = "BUY" if self.is_short else "SELL"
                # * On margin shorts, you buy a little bit more than the amount (amount + interest)
                logger.info(f'{order_type.upper()}_{payment} has been fulfilled for {self}.')
            # TODO-lev: Double check this
            self.close(safe_value_fallback(order, 'average', 'price'))
        elif order_type in ('stop_loss_limit', 'stop-loss', 'stop-loss-limit', 'stop'):
            self.stoploss_order_id = None
            self.close_rate_requested = self.stop_loss
            self.sell_reason = SellType.STOPLOSS_ON_EXCHANGE.value
            if self.is_open:
                logger.info(f'{order_type.upper()} is hit for {self}.')
            self.close(safe_value_fallback(order, 'average', 'price'))
        else:
            raise ValueError(f'Unknown order type: {order_type}')
        Trade.commit()

    def close(self, rate: float, *, show_msg: bool = True) -> None:
        """
        Sets close_rate to the given rate, calculates total profit
        and marks trade as closed
        """
        self.close_rate = rate
        self.close_date = self.close_date or datetime.utcnow()
        self.close_profit = self.calc_profit_ratio()
        self.close_profit_abs = self.calc_profit()
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
        if self.enter_side == side and self.fee_open_currency is None:
            self.fee_open_cost = fee_cost
            self.fee_open_currency = fee_currency
            if fee_rate is not None:
                self.fee_open = fee_rate
                # Assume close-fee will fall into the same fee category and take an educated guess
                self.fee_close = fee_rate
        elif self.exit_side == side and self.fee_close_currency is None:
            self.fee_close_cost = fee_cost
            self.fee_close_currency = fee_currency
            if fee_rate is not None:
                self.fee_close = fee_rate

    def fee_updated(self, side: str) -> bool:
        """
        Verify if this side (buy / sell) has already been updated
        """
        if self.enter_side == side:
            return self.fee_open_currency is not None
        elif self.exit_side == side:
            return self.fee_close_currency is not None
        else:
            return False

    def update_order(self, order: Dict) -> None:
        Order.update_orders(self.orders, order)

    def get_exit_order_count(self) -> int:
        """
        Get amount of failed exiting orders
        assumes full exits.
        """
        return len([o for o in self.orders if o.ft_order_side == 'sell'])

    def _calc_open_trade_value(self) -> float:
        """
        Calculate the open_rate including open_fee.
        :return: Price in of the open trade incl. Fees
        """
        open_trade = Decimal(self.amount) * Decimal(self.open_rate)
        fees = open_trade * Decimal(self.fee_open)
        if self.is_short:
            return float(open_trade - fees)
        else:
            return float(open_trade + fees)

    def recalc_open_trade_value(self) -> None:
        """
        Recalculate open_trade_value.
        Must be called whenever open_rate, fee_open or is_short is changed.

        """
        self.open_trade_value = self._calc_open_trade_value()

    def calculate_interest(self, interest_rate: Optional[float] = None) -> Decimal:
        """
        : param interest_rate: interest_charge for borrowing this coin(optional).
        If interest_rate is not set self.interest_rate will be used
        """

        zero = Decimal(0.0)
        # If nothing was borrowed
        if self.has_no_leverage or self.trading_mode != TradingMode.MARGIN:
            return zero

        open_date = self.open_date.replace(tzinfo=None)
        now = (self.close_date or datetime.now(timezone.utc)).replace(tzinfo=None)
        sec_per_hour = Decimal(3600)
        total_seconds = Decimal((now - open_date).total_seconds())
        hours = total_seconds/sec_per_hour or zero

        rate = Decimal(interest_rate or self.interest_rate)
        borrowed = Decimal(self.borrowed)

        return interest(exchange_name=self.exchange, borrowed=borrowed, rate=rate, hours=hours)

    def _calc_base_close(self, amount: Decimal, rate: Optional[float] = None,
                         fee: Optional[float] = None) -> Decimal:

        close_trade = Decimal(amount) * Decimal(rate or self.close_rate)  # type: ignore
        fees = close_trade * Decimal(fee or self.fee_close)

        if self.is_short:
            return close_trade + fees
        else:
            return close_trade - fees

    def calc_close_trade_value(self, rate: Optional[float] = None,
                               fee: Optional[float] = None,
                               interest_rate: Optional[float] = None) -> float:
        """
        Calculate the close_rate including fee
        :param fee: fee to use on the close rate (optional).
            If rate is not set self.fee will be used
        :param rate: rate to compare with (optional).
            If rate is not set self.close_rate will be used
        :param interest_rate: interest_charge for borrowing this coin (optional).
            If interest_rate is not set self.interest_rate will be used
        :return: Price in BTC of the open trade
        """
        if rate is None and not self.close_rate:
            return 0.0

        amount = Decimal(self.amount)
        trading_mode = self.trading_mode or TradingMode.SPOT

        if trading_mode == TradingMode.SPOT:
            return float(self._calc_base_close(amount, rate, fee))

        elif (trading_mode == TradingMode.MARGIN):

            total_interest = self.calculate_interest(interest_rate)

            if self.is_short:
                amount = amount + total_interest
                return float(self._calc_base_close(amount, rate, fee))
            else:
                # Currency already owned for longs, no need to purchase
                return float(self._calc_base_close(amount, rate, fee) - total_interest)

        elif (trading_mode == TradingMode.FUTURES):
            funding_fees = self.funding_fees or 0.0
            # Positive funding_fees -> Trade has gained from fees.
            # Negative funding_fees -> Trade had to pay the fees.
            if self.is_short:
                return float(self._calc_base_close(amount, rate, fee)) - funding_fees
            else:
                return float(self._calc_base_close(amount, rate, fee)) + funding_fees
        else:
            raise OperationalException(
                f"{self.trading_mode.value} trading is not yet available using freqtrade")

    def calc_profit(self, rate: Optional[float] = None,
                    fee: Optional[float] = None,
                    interest_rate: Optional[float] = None) -> float:
        """
        Calculate the absolute profit in stake currency between Close and Open trade
        :param fee: fee to use on the close rate (optional).
            If fee is not set self.fee will be used
        :param rate: close rate to compare with (optional).
            If rate is not set self.close_rate will be used
        :param interest_rate: interest_charge for borrowing this coin (optional).
            If interest_rate is not set self.interest_rate will be used
        :return:  profit in stake currency as float
        """
        close_trade_value = self.calc_close_trade_value(
            rate=(rate or self.close_rate),
            fee=(fee or self.fee_close),
            interest_rate=(interest_rate or self.interest_rate)
        )

        if self.is_short:
            profit = self.open_trade_value - close_trade_value
        else:
            profit = close_trade_value - self.open_trade_value
        return float(f"{profit:.8f}")

    def calc_profit_ratio(self, rate: Optional[float] = None,
                          fee: Optional[float] = None,
                          interest_rate: Optional[float] = None) -> float:
        """
        Calculates the profit as ratio (including fee).
        :param rate: rate to compare with (optional).
            If rate is not set self.close_rate will be used
        :param fee: fee to use on the close rate (optional).
        :param interest_rate: interest_charge for borrowing this coin (optional).
            If interest_rate is not set self.interest_rate will be used
        :return: profit ratio as float
        """
        close_trade_value = self.calc_close_trade_value(
            rate=(rate or self.close_rate),
            fee=(fee or self.fee_close),
            interest_rate=(interest_rate or self.interest_rate)
        )

        short_close_zero = (self.is_short and close_trade_value == 0.0)
        long_close_zero = (not self.is_short and self.open_trade_value == 0.0)
        leverage = self.leverage or 1.0

        if (short_close_zero or long_close_zero):
            return 0.0
        else:
            if self.is_short:
                profit_ratio = (1 - (close_trade_value/self.open_trade_value)) * leverage
            else:
                profit_ratio = ((close_trade_value/self.open_trade_value) - 1) * leverage

        return float(f"{profit_ratio:.8f}")

    def recalc_trade_from_orders(self):
        # We need at least 2 orders for averaging amounts and rates.
        if len(self.orders) < 2:
            # Just in case, still recalc open trade value
            self.recalc_open_trade_value()
            return

        total_amount = 0.0
        total_stake = 0.0
        for o in self.orders:
            if (o.ft_is_open or
                    (o.ft_order_side != self.enter_side) or
                    (o.status not in NON_OPEN_EXCHANGE_STATES)):
                continue

            tmp_amount = o.amount
            tmp_price = o.average or o.price
            if o.filled is not None:
                tmp_amount = o.filled
            if tmp_amount > 0.0 and tmp_price is not None:
                total_amount += tmp_amount
                total_stake += tmp_price * tmp_amount

        if total_amount > 0:
            self.open_rate = total_stake / total_amount
            self.stake_amount = total_stake
            self.amount = total_amount
            self.fee_open_cost = self.fee_open * self.stake_amount
            self.recalc_open_trade_value()
            if self.stop_loss_pct is not None and self.open_rate is not None:
                self.adjust_stop_loss(self.open_rate, self.stop_loss_pct)

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

    def select_filled_orders(self, order_side: str) -> List['Order']:
        """
        Finds filled orders for this orderside.
        :param order_side: Side of the order (either 'buy' or 'sell')
        :return: array of Order objects
        """
        return [o for o in self.orders if o.ft_order_side == order_side and
                o.ft_is_open is False and
                (o.filled or 0) > 0 and
                o.status in NON_OPEN_EXCHANGE_STATES]

    @property
    def nr_of_successful_buys(self) -> int:
        """
        Helper function to count the number of buy orders that have been filled.
        :return: int count of buy orders that have been filled for this trade.
        """

        return len(self.select_filled_orders('buy'))

    @property
    def nr_of_successful_sells(self) -> int:
        """
        Helper function to count the number of sell orders that have been filled.
        :return: int count of sell orders that have been filled for this trade.
        """
        return len(self.select_filled_orders('sell'))

    @staticmethod
    def get_trades_proxy(*, pair: str = None, is_open: bool = None,
                         open_date: datetime = None, close_date: datetime = None,
                         ) -> List['LocalTrade']:
        """
        Helper function to query Trades.
        Returns a List of trades, filtered on the parameters given.
        In live mode, converts the filter to a database query and returns all rows
        In Backtest mode, uses filters on Trade.trades to get the result.

        :return: unsorted List[Trade]
        """

        # Offline mode - without database
        if is_open is not None:
            if is_open:
                sel_trades = LocalTrade.trades_open
            else:
                sel_trades = LocalTrade.trades

        else:
            # Not used during backtesting, but might be used by a strategy
            sel_trades = list(LocalTrade.trades + LocalTrade.trades_open)

        if pair:
            sel_trades = [trade for trade in sel_trades if trade.pair == pair]
        if open_date:
            sel_trades = [trade for trade in sel_trades if trade.open_date > open_date]
        if close_date:
            sel_trades = [trade for trade in sel_trades if trade.close_date
                          and trade.close_date > close_date]

        return sel_trades

    @staticmethod
    def close_bt_trade(trade):
        LocalTrade.trades_open.remove(trade)
        LocalTrade.trades.append(trade)
        LocalTrade.total_profit += trade.close_profit_abs

    @staticmethod
    def add_bt_trade(trade):
        if trade.is_open:
            LocalTrade.trades_open.append(trade)
        else:
            LocalTrade.trades.append(trade)

    @staticmethod
    def get_open_trades() -> List[Any]:
        """
        Query trades from persistence layer
        """
        return Trade.get_trades_proxy(is_open=True)

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


class Trade(_DECL_BASE, LocalTrade):
    """
    Trade database model.
    Also handles updating and querying trades

    Note: Fields must be aligned with LocalTrade class
    """
    __tablename__ = 'trades'

    use_db: bool = True

    id = Column(Integer, primary_key=True)

    orders = relationship("Order", order_by="Order.id", cascade="all, delete-orphan", lazy="joined")

    exchange = Column(String(25), nullable=False)
    pair = Column(String(25), nullable=False, index=True)
    is_open = Column(Boolean, nullable=False, default=True, index=True)
    fee_open = Column(Float, nullable=False, default=0.0)
    fee_open_cost = Column(Float, nullable=True)
    fee_open_currency = Column(String(25), nullable=True)
    fee_close = Column(Float, nullable=False, default=0.0)
    fee_close_cost = Column(Float, nullable=True)
    fee_close_currency = Column(String(25), nullable=True)
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
    open_order_id = Column(String(255))
    # absolute value of the stop loss
    stop_loss = Column(Float, nullable=True, default=0.0)
    # percentage value of the stop loss
    stop_loss_pct = Column(Float, nullable=True)
    # absolute value of the initial stop loss
    initial_stop_loss = Column(Float, nullable=True, default=0.0)
    # percentage value of the initial stop loss
    initial_stop_loss_pct = Column(Float, nullable=True)
    # stoploss order id which is on exchange
    stoploss_order_id = Column(String(255), nullable=True, index=True)
    # last update time of the stoploss order on exchange
    stoploss_last_update = Column(DateTime, nullable=True)
    # absolute value of the highest reached price
    max_rate = Column(Float, nullable=True, default=0.0)
    # Lowest price reached
    min_rate = Column(Float, nullable=True)
    sell_reason = Column(String(100), nullable=True)
    sell_order_status = Column(String(100), nullable=True)
    strategy = Column(String(100), nullable=True)
    enter_tag = Column(String(100), nullable=True)
    timeframe = Column(Integer, nullable=True)

    trading_mode = Column(Enum(TradingMode), nullable=True)

    # Leverage trading properties
    leverage = Column(Float, nullable=True, default=1.0)
    is_short = Column(Boolean, nullable=False, default=False)
    isolated_liq = Column(Float, nullable=True)

    # Margin Trading Properties
    interest_rate = Column(Float, nullable=False, default=0.0)

    # Futures properties
    funding_fees = Column(Float, nullable=True, default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recalc_open_trade_value()

    def delete(self) -> None:

        for order in self.orders:
            Order.query.session.delete(order)

        Trade.query.session.delete(self)
        Trade.commit()

    @staticmethod
    def commit():
        Trade.query.session.commit()

    @staticmethod
    def get_trades_proxy(*, pair: str = None, is_open: bool = None,
                         open_date: datetime = None, close_date: datetime = None,
                         ) -> List['LocalTrade']:
        """
        Helper function to query Trades.j
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
            return LocalTrade.get_trades_proxy(
                pair=pair, is_open=is_open,
                open_date=open_date,
                close_date=close_date
            )

    @staticmethod
    def get_trades(trade_filter=None) -> Query:
        """
        Helper function to query Trades using filters.
        NOTE: Not supported in Backtesting.
        :param trade_filter: Optional filter to apply to trades
                             Can be either a Filter object, or a List of filters
                             e.g. `(trade_filter=[Trade.id == trade_id, Trade.is_open.is_(True),])`
                             e.g. `(trade_filter=Trade.id == trade_id)`
        :return: unsorted query object
        """
        if not Trade.use_db:
            raise NotImplementedError('`Trade.get_trades()` not supported in backtesting mode.')
        if trade_filter is not None:
            if not isinstance(trade_filter, list):
                trade_filter = [trade_filter]
            return Trade.query.filter(*trade_filter)
        else:
            return Trade.query

    @staticmethod
    def get_open_order_trades() -> List['Trade']:
        """
        Returns all open trades
        NOTE: Not supported in Backtesting.
        """
        return Trade.get_trades(Trade.open_order_id.isnot(None)).all()

    @staticmethod
    def get_open_trades_without_assigned_fees():
        """
        Returns all open trades which don't have open fees set correctly
        NOTE: Not supported in Backtesting.
        """
        return Trade.get_trades([Trade.fee_open_currency.is_(None),
                                 Trade.orders.any(),
                                 Trade.is_open.is_(True),
                                 ]).all()

    @staticmethod
    def get_closed_trades_without_assigned_fees():
        """
        Returns all closed trades which don't have fees set correctly
        NOTE: Not supported in Backtesting.
        """
        return Trade.get_trades([Trade.fee_close_currency.is_(None),
                                 Trade.orders.any(),
                                 Trade.is_open.is_(False),
                                 ]).all()

    @staticmethod
    def get_total_closed_profit() -> float:
        """
        Retrieves total realized profit
        """
        if Trade.use_db:
            total_profit = Trade.query.with_entities(
                func.sum(Trade.close_profit_abs)).filter(Trade.is_open.is_(False)).scalar()
        else:
            total_profit = sum(
                t.close_profit_abs for t in LocalTrade.get_trades_proxy(is_open=False))
        return total_profit or 0

    @staticmethod
    def total_open_trades_stakes() -> float:
        """
        Calculates total invested amount in open trades
        in stake currency
        """
        if Trade.use_db:
            total_open_stake_amount = Trade.query.with_entities(
                func.sum(Trade.stake_amount)).filter(Trade.is_open.is_(True)).scalar()
        else:
            total_open_stake_amount = sum(
                t.stake_amount for t in LocalTrade.get_trades_proxy(is_open=True))
        return total_open_stake_amount or 0

    @staticmethod
    def get_overall_performance(minutes=None) -> List[Dict[str, Any]]:
        """
        Returns List of dicts containing all Trades, including profit and trade count
        NOTE: Not supported in Backtesting.
        """
        filters = [Trade.is_open.is_(False)]
        if minutes:
            start_date = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            filters.append(Trade.close_date >= start_date)
        pair_rates = Trade.query.with_entities(
            Trade.pair,
            func.sum(Trade.close_profit).label('profit_sum'),
            func.sum(Trade.close_profit_abs).label('profit_sum_abs'),
            func.count(Trade.pair).label('count')
        ).filter(*filters)\
            .group_by(Trade.pair) \
            .order_by(desc('profit_sum_abs')) \
            .all()
        return [
            {
                'pair': pair,
                'profit_ratio': profit,
                'profit': round(profit * 100, 2),  # Compatibility mode
                'profit_pct': round(profit * 100, 2),
                'profit_abs': profit_abs,
                'count': count
            }
            for pair, profit, profit_abs, count in pair_rates
        ]

    @staticmethod
    def get_enter_tag_performance(pair: Optional[str]) -> List[Dict[str, Any]]:
        """
        Returns List of dicts containing all Trades, based on buy tag performance
        Can either be average for all pairs or a specific pair provided
        NOTE: Not supported in Backtesting.
        """

        filters = [Trade.is_open.is_(False)]
        if(pair is not None):
            filters.append(Trade.pair == pair)

        enter_tag_perf = Trade.query.with_entities(
            Trade.enter_tag,
            func.sum(Trade.close_profit).label('profit_sum'),
            func.sum(Trade.close_profit_abs).label('profit_sum_abs'),
            func.count(Trade.pair).label('count')
        ).filter(*filters)\
            .group_by(Trade.enter_tag) \
            .order_by(desc('profit_sum_abs')) \
            .all()

        return [
            {
                'enter_tag': enter_tag if enter_tag is not None else "Other",
                'profit_ratio': profit,
                'profit_pct': round(profit * 100, 2),
                'profit_abs': profit_abs,
                'count': count
            }
            for enter_tag, profit, profit_abs, count in enter_tag_perf
        ]

    @staticmethod
    def get_sell_reason_performance(pair: Optional[str]) -> List[Dict[str, Any]]:
        """
        Returns List of dicts containing all Trades, based on sell reason performance
        Can either be average for all pairs or a specific pair provided
        NOTE: Not supported in Backtesting.
        """

        filters = [Trade.is_open.is_(False)]
        if(pair is not None):
            filters.append(Trade.pair == pair)

        sell_tag_perf = Trade.query.with_entities(
            Trade.sell_reason,
            func.sum(Trade.close_profit).label('profit_sum'),
            func.sum(Trade.close_profit_abs).label('profit_sum_abs'),
            func.count(Trade.pair).label('count')
        ).filter(*filters)\
            .group_by(Trade.sell_reason) \
            .order_by(desc('profit_sum_abs')) \
            .all()

        return [
            {
                'sell_reason': sell_reason if sell_reason is not None else "Other",
                'profit_ratio': profit,
                'profit_pct': round(profit * 100, 2),
                'profit_abs': profit_abs,
                'count': count
            }
            for sell_reason, profit, profit_abs, count in sell_tag_perf
        ]

    @staticmethod
    def get_mix_tag_performance(pair: Optional[str]) -> List[Dict[str, Any]]:
        """
        Returns List of dicts containing all Trades, based on buy_tag + sell_reason performance
        Can either be average for all pairs or a specific pair provided
        NOTE: Not supported in Backtesting.
        """

        filters = [Trade.is_open.is_(False)]
        if(pair is not None):
            filters.append(Trade.pair == pair)

        mix_tag_perf = Trade.query.with_entities(
            Trade.id,
            Trade.enter_tag,
            Trade.sell_reason,
            func.sum(Trade.close_profit).label('profit_sum'),
            func.sum(Trade.close_profit_abs).label('profit_sum_abs'),
            func.count(Trade.pair).label('count')
        ).filter(*filters)\
            .group_by(Trade.id) \
            .order_by(desc('profit_sum_abs')) \
            .all()

        return_list: List[Dict] = []
        for id, enter_tag, sell_reason, profit, profit_abs, count in mix_tag_perf:
            enter_tag = enter_tag if enter_tag is not None else "Other"
            sell_reason = sell_reason if sell_reason is not None else "Other"

            if(sell_reason is not None and enter_tag is not None):
                mix_tag = enter_tag + " " + sell_reason
                i = 0
                if not any(item["mix_tag"] == mix_tag for item in return_list):
                    return_list.append({'mix_tag': mix_tag,
                                        'profit': profit,
                                        'profit_pct': round(profit * 100, 2),
                                        'profit_abs': profit_abs,
                                        'count': count})
                else:
                    while i < len(return_list):
                        if return_list[i]["mix_tag"] == mix_tag:
                            return_list[i] = {
                                'mix_tag': mix_tag,
                                'profit': profit + return_list[i]["profit"],
                                'profit_pct': round(profit + return_list[i]["profit"] * 100, 2),
                                'profit_abs': profit_abs + return_list[i]["profit_abs"],
                                'count': 1 + return_list[i]["count"]}
                        i += 1

        return return_list

    @staticmethod
    def get_best_pair(start_date: datetime = datetime.fromtimestamp(0)):
        """
        Get best pair with closed trade.
        NOTE: Not supported in Backtesting.
        :returns: Tuple containing (pair, profit_sum)
        """
        best_pair = Trade.query.with_entities(
            Trade.pair, func.sum(Trade.close_profit).label('profit_sum')
        ).filter(Trade.is_open.is_(False) & (Trade.close_date >= start_date)) \
            .group_by(Trade.pair) \
            .order_by(desc('profit_sum')).first()
        return best_pair


class PairLock(_DECL_BASE):
    """
    Pair Locks database model.
    """
    __tablename__ = 'pairlocks'

    id = Column(Integer, primary_key=True)

    pair = Column(String(25), nullable=False, index=True)
    reason = Column(String(255), nullable=True)
    # Time the pair was locked (start time)
    lock_time = Column(DateTime, nullable=False)
    # Time until the pair is locked (end time)
    lock_end_time = Column(DateTime, nullable=False, index=True)

    active = Column(Boolean, nullable=False, default=True, index=True)

    def __repr__(self):
        lock_time = self.lock_time.strftime(DATETIME_PRINT_FORMAT)
        lock_end_time = self.lock_end_time.strftime(DATETIME_PRINT_FORMAT)
        return (f'PairLock(id={self.id}, pair={self.pair}, lock_time={lock_time}, '
                f'lock_end_time={lock_end_time}, reason={self.reason}, active={self.active})')

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
