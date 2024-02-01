"""
This module contains the class to persist trades into SQLite
"""
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isclose
from typing import Any, ClassVar, Dict, List, Optional, Sequence, cast

from sqlalchemy import (Enum, Float, ForeignKey, Integer, ScalarResult, Select, String,
                        UniqueConstraint, desc, func, select)
from sqlalchemy.orm import Mapped, lazyload, mapped_column, relationship, validates
from typing_extensions import Self

from freqtrade.constants import (CANCELED_EXCHANGE_STATES, CUSTOM_TAG_MAX_LENGTH,
                                 DATETIME_PRINT_FORMAT, MATH_CLOSE_PREC, NON_OPEN_EXCHANGE_STATES,
                                 BuySell, LongShort)
from freqtrade.enums import ExitType, TradingMode
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.exchange import (ROUND_DOWN, ROUND_UP, amount_to_contract_precision,
                                price_to_precision)
from freqtrade.leverage import interest
from freqtrade.misc import safe_value_fallback
from freqtrade.persistence.base import ModelBase, SessionType
from freqtrade.util import FtPrecise, dt_from_ts, dt_now, dt_ts


logger = logging.getLogger(__name__)


@dataclass
class ProfitStruct:
    profit_abs: float
    profit_ratio: float
    total_profit: float
    total_profit_ratio: float


class Order(ModelBase):
    """
    Order database model
    Keeps a record of all orders placed on the exchange

    One to many relationship with Trades:
      - One trade can have many orders
      - One Order can only be associated with one Trade

    Mirrors CCXT Order structure
    """
    __tablename__ = 'orders'
    __allow_unmapped__ = True
    session: ClassVar[SessionType]

    # Uniqueness should be ensured over pair, order_id
    # its likely that order_id is unique per Pair on some exchanges.
    __table_args__ = (UniqueConstraint('ft_pair', 'order_id', name="_order_pair_order_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ft_trade_id: Mapped[int] = mapped_column(Integer, ForeignKey('trades.id'), index=True)

    _trade_live: Mapped["Trade"] = relationship("Trade", back_populates="orders", lazy="immediate")
    _trade_bt: "LocalTrade" = None  # type: ignore

    # order_side can only be 'buy', 'sell' or 'stoploss'
    ft_order_side: Mapped[str] = mapped_column(String(25), nullable=False)
    ft_pair: Mapped[str] = mapped_column(String(25), nullable=False)
    ft_is_open: Mapped[bool] = mapped_column(nullable=False, default=True, index=True)
    ft_amount: Mapped[float] = mapped_column(Float(), nullable=False)
    ft_price: Mapped[float] = mapped_column(Float(), nullable=False)
    ft_cancel_reason: Mapped[str] = mapped_column(String(CUSTOM_TAG_MAX_LENGTH), nullable=True)

    order_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    symbol: Mapped[Optional[str]] = mapped_column(String(25), nullable=True)
    # TODO: type: order_type type is Optional[str]
    order_type: Mapped[str] = mapped_column(String(50), nullable=True)
    side: Mapped[str] = mapped_column(String(25), nullable=True)
    price: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    average: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    amount: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    filled: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    remaining: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    cost: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    stop_price: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    order_date: Mapped[datetime] = mapped_column(nullable=True, default=dt_now)
    order_filled_date: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    order_update_date: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    funding_fee: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)

    ft_fee_base: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)

    @property
    def order_date_utc(self) -> datetime:
        """ Order-date with UTC timezoneinfo"""
        return self.order_date.replace(tzinfo=timezone.utc)

    @property
    def order_filled_utc(self) -> Optional[datetime]:
        """ last order-date with UTC timezoneinfo"""
        return (
            self.order_filled_date.replace(tzinfo=timezone.utc) if self.order_filled_date else None
        )

    @property
    def safe_amount(self) -> float:
        return self.amount or self.ft_amount

    @property
    def safe_placement_price(self) -> float:
        """Price at which the order was placed"""
        return self.price or self.stop_price or self.ft_price

    @property
    def safe_price(self) -> float:
        return self.average or self.price or self.stop_price or self.ft_price

    @property
    def safe_filled(self) -> float:
        return self.filled if self.filled is not None else 0.0

    @property
    def safe_cost(self) -> float:
        return self.cost or 0.0

    @property
    def safe_remaining(self) -> float:
        return (
            self.remaining if self.remaining is not None else
            self.safe_amount - (self.filled or 0.0)
        )

    @property
    def safe_fee_base(self) -> float:
        return self.ft_fee_base or 0.0

    @property
    def safe_amount_after_fee(self) -> float:
        return self.safe_filled - self.safe_fee_base

    @property
    def trade(self) -> "LocalTrade":
        return self._trade_bt or self._trade_live

    @property
    def stake_amount(self) -> float:
        """ Amount in stake currency used for this order"""
        return self.safe_amount * self.safe_price / self.trade.leverage

    def __repr__(self):

        return (f"Order(id={self.id}, trade={self.ft_trade_id}, order_id={self.order_id}, "
                f"side={self.side}, filled={self.safe_filled}, price={self.safe_price}, "
                f"status={self.status}, date={self.order_date_utc:{DATETIME_PRINT_FORMAT}})")

    def update_from_ccxt_object(self, order):
        """
        Update Order from ccxt response
        Only updates if fields are available from ccxt -
        """
        if self.order_id != str(order['id']):
            raise DependencyException("Order-id's don't match")

        self.status = safe_value_fallback(order, 'status', default_value=self.status)
        self.symbol = safe_value_fallback(order, 'symbol', default_value=self.symbol)
        self.order_type = safe_value_fallback(order, 'type', default_value=self.order_type)
        self.side = safe_value_fallback(order, 'side', default_value=self.side)
        self.price = safe_value_fallback(order, 'price', default_value=self.price)
        self.amount = safe_value_fallback(order, 'amount', default_value=self.amount)
        self.filled = safe_value_fallback(order, 'filled', default_value=self.filled)
        self.average = safe_value_fallback(order, 'average', default_value=self.average)
        self.remaining = safe_value_fallback(order, 'remaining', default_value=self.remaining)
        self.cost = safe_value_fallback(order, 'cost', default_value=self.cost)
        self.stop_price = safe_value_fallback(order, 'stopPrice', default_value=self.stop_price)
        order_date = safe_value_fallback(order, 'timestamp')
        if order_date:
            self.order_date = datetime.fromtimestamp(order_date / 1000, tz=timezone.utc)

        self.ft_is_open = True
        if self.status in NON_OPEN_EXCHANGE_STATES:
            self.ft_is_open = False
            if (order.get('filled', 0.0) or 0.0) > 0 and not self.order_filled_date:
                self.order_filled_date = dt_from_ts(
                    safe_value_fallback(order, 'lastTradeTimestamp', default_value=dt_ts())
                )
        self.order_update_date = datetime.now(timezone.utc)

    def to_ccxt_object(self, stopPriceName: str = 'stopPrice') -> Dict[str, Any]:
        order: Dict[str, Any] = {
            'id': self.order_id,
            'symbol': self.ft_pair,
            'price': self.price,
            'average': self.average,
            'amount': self.amount,
            'cost': self.cost,
            'type': self.order_type,
            'side': self.ft_order_side,
            'filled': self.filled,
            'remaining': self.remaining,
            'datetime': self.order_date_utc.strftime('%Y-%m-%dT%H:%M:%S.%f'),
            'timestamp': int(self.order_date_utc.timestamp() * 1000),
            'status': self.status,
            'fee': None,
            'info': {},
        }
        if self.ft_order_side == 'stoploss':
            order.update({
                stopPriceName: self.stop_price,
                'ft_order_type': 'stoploss',
            })

        return order

    def to_json(self, entry_side: str, minified: bool = False) -> Dict[str, Any]:
        resp = {
            'amount': self.safe_amount,
            'safe_price': self.safe_price,
            'ft_order_side': self.ft_order_side,
            'order_filled_timestamp': int(self.order_filled_date.replace(
                tzinfo=timezone.utc).timestamp() * 1000) if self.order_filled_date else None,
            'ft_is_entry': self.ft_order_side == entry_side,
        }
        if not minified:
            resp.update({
                'pair': self.ft_pair,
                'order_id': self.order_id,
                'status': self.status,
                'average': round(self.average, 8) if self.average else 0,
                'cost': self.cost if self.cost else 0,
                'filled': self.filled,
                'is_open': self.ft_is_open,
                'order_date': self.order_date.strftime(DATETIME_PRINT_FORMAT)
                if self.order_date else None,
                'order_timestamp': int(self.order_date.replace(
                    tzinfo=timezone.utc).timestamp() * 1000) if self.order_date else None,
                'order_filled_date': self.order_filled_date.strftime(DATETIME_PRINT_FORMAT)
                if self.order_filled_date else None,
                'order_type': self.order_type,
                'price': self.price,
                'remaining': self.remaining,
                'ft_fee_base': self.ft_fee_base,
                'funding_fee': self.funding_fee,
            })
        return resp

    def close_bt_order(self, close_date: datetime, trade: 'LocalTrade'):
        self.order_filled_date = close_date
        self.filled = self.amount
        self.remaining = 0
        self.status = 'closed'
        self.ft_is_open = False
        # Assign funding fees to Order.
        # Assumes backtesting will use date_last_filled_utc to calculate future funding fees.
        self.funding_fee = trade.funding_fee_running
        trade.funding_fee_running = 0.0

        if (self.ft_order_side == trade.entry_side and self.price):
            trade.open_rate = self.price
            trade.recalc_trade_from_orders()
            if trade.nr_of_successful_entries == 1:
                trade.initial_stop_loss_pct = None
                trade.is_stop_loss_trailing = False
            trade.adjust_stop_loss(trade.open_rate, trade.stop_loss_pct)

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
            Trade.commit()
        else:
            logger.warning(f"Did not find order for {order}.")

    @classmethod
    def parse_from_ccxt_object(
            cls, order: Dict[str, Any], pair: str, side: str,
            amount: Optional[float] = None, price: Optional[float] = None) -> Self:
        """
        Parse an order from a ccxt object and return a new order Object.
        Optional support for overriding amount and price is only used for test simplification.
        """
        o = cls(
            order_id=str(order['id']),
            ft_order_side=side,
            ft_pair=pair,
            ft_amount=amount if amount else order['amount'],
            ft_price=price if price else order['price'],
            )

        o.update_from_ccxt_object(order)
        return o

    @staticmethod
    def get_open_orders() -> Sequence['Order']:
        """
        Retrieve open orders from the database
        :return: List of open orders
        """
        return Order.session.scalars(select(Order).filter(Order.ft_is_open.is_(True))).all()

    @staticmethod
    def order_by_id(order_id: str) -> Optional['Order']:
        """
        Retrieve order based on order_id
        :return: Order or None
        """
        return Order.session.scalars(select(Order).filter(Order.order_id == order_id)).first()


class LocalTrade:
    """
    Trade database model.
    Used in backtesting - must be aligned to Trade model!

    """
    use_db: bool = False
    # Trades container for backtesting
    trades: List['LocalTrade'] = []
    trades_open: List['LocalTrade'] = []
    # Copy of trades_open - but indexed by pair
    bt_trades_open_pp: Dict[str, List['LocalTrade']] = defaultdict(list)
    bt_open_open_trade_count: int = 0
    total_profit: float = 0
    realized_profit: float = 0

    id: int = 0

    orders: List[Order] = []

    exchange: str = ''
    pair: str = ''
    base_currency: Optional[str] = ''
    stake_currency: Optional[str] = ''
    is_open: bool = True
    fee_open: float = 0.0
    fee_open_cost: Optional[float] = None
    fee_open_currency: Optional[str] = ''
    fee_close: Optional[float] = 0.0
    fee_close_cost: Optional[float] = None
    fee_close_currency: Optional[str] = ''
    open_rate: float = 0.0
    open_rate_requested: Optional[float] = None
    # open_trade_value - calculated via _calc_open_trade_value
    open_trade_value: float = 0.0
    close_rate: Optional[float] = None
    close_rate_requested: Optional[float] = None
    close_profit: Optional[float] = None
    close_profit_abs: Optional[float] = None
    stake_amount: float = 0.0
    max_stake_amount: Optional[float] = 0.0
    amount: float = 0.0
    amount_requested: Optional[float] = None
    open_date: datetime
    close_date: Optional[datetime] = None
    # absolute value of the stop loss
    stop_loss: float = 0.0
    # percentage value of the stop loss
    stop_loss_pct: Optional[float] = 0.0
    # absolute value of the initial stop loss
    initial_stop_loss: Optional[float] = 0.0
    # percentage value of the initial stop loss
    initial_stop_loss_pct: Optional[float] = None
    is_stop_loss_trailing: bool = False
    # stoploss order id which is on exchange
    stoploss_order_id: Optional[str] = None
    # last update time of the stoploss order on exchange
    stoploss_last_update: Optional[datetime] = None
    # absolute value of the highest reached price
    max_rate: Optional[float] = None
    # Lowest price reached
    min_rate: Optional[float] = None
    exit_reason: Optional[str] = ''
    exit_order_status: Optional[str] = ''
    strategy: Optional[str] = ''
    enter_tag: Optional[str] = None
    timeframe: Optional[int] = None

    trading_mode: TradingMode = TradingMode.SPOT
    amount_precision: Optional[float] = None
    price_precision: Optional[float] = None
    precision_mode: Optional[int] = None
    contract_size: Optional[float] = None

    # Leverage trading properties
    liquidation_price: Optional[float] = None
    is_short: bool = False
    leverage: float = 1.0

    # Margin trading properties
    interest_rate: float = 0.0

    # Futures properties
    funding_fees: Optional[float] = None
    # Used to keep running funding fees - between the last filled order and now
    # Shall not be used for calculations!
    funding_fee_running: Optional[float] = None

    @property
    def stoploss_or_liquidation(self) -> float:
        if self.liquidation_price:
            if self.is_short:
                return min(self.stop_loss, self.liquidation_price)
            else:
                return max(self.stop_loss, self.liquidation_price)

        return self.stop_loss

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
            return (self.amount * self.open_rate) * ((self.leverage - 1) / self.leverage)
        else:
            return self.amount

    @property
    def _date_last_filled_utc(self) -> Optional[datetime]:
        """ Date of the last filled order"""
        orders = self.select_filled_orders()
        if orders:
            return max(o.order_filled_utc for o in orders if o.order_filled_utc)
        return None

    @property
    def date_last_filled_utc(self) -> datetime:
        """ Date of the last filled order - or open_date if no orders are filled"""
        dt_last_filled = self._date_last_filled_utc
        if not dt_last_filled:
            return self.open_date_utc
        return max([self.open_date_utc, dt_last_filled])

    @property
    def open_date_utc(self):
        return self.open_date.replace(tzinfo=timezone.utc)

    @property
    def stoploss_last_update_utc(self):
        if self.stoploss_last_update:
            return self.stoploss_last_update.replace(tzinfo=timezone.utc)
        return None

    @property
    def close_date_utc(self):
        return self.close_date.replace(tzinfo=timezone.utc) if self.close_date else None

    @property
    def entry_side(self) -> str:
        if self.is_short:
            return "sell"
        else:
            return "buy"

    @property
    def exit_side(self) -> BuySell:
        if self.is_short:
            return "buy"
        else:
            return "sell"

    @property
    def trade_direction(self) -> LongShort:
        if self.is_short:
            return "short"
        else:
            return "long"

    @property
    def safe_base_currency(self) -> str:
        """
        Compatibility layer for asset - which can be empty for old trades.
        """
        try:
            return self.base_currency or self.pair.split('/')[0]
        except IndexError:
            return ''

    @property
    def safe_quote_currency(self) -> str:
        """
        Compatibility layer for asset - which can be empty for old trades.
        """
        try:
            return self.stake_currency or self.pair.split('/')[1].split(':')[0]
        except IndexError:
            return ''

    @property
    def open_orders(self) -> List[Order]:
        """
        All open orders for this trade excluding stoploss orders
        """
        return [o for o in self.orders if o.ft_is_open and o.ft_order_side != 'stoploss']

    @property
    def has_open_orders(self) -> int:
        """
        True if there are open orders for this trade excluding stoploss orders
        """
        open_orders_wo_sl = [
            o for o in self.orders
            if o.ft_order_side not in ['stoploss'] and o.ft_is_open
        ]
        return len(open_orders_wo_sl) > 0

    @property
    def open_orders_ids(self) -> List[str]:
        open_orders_ids_wo_sl = [
            oo.order_id for oo in self.open_orders
            if oo.ft_order_side not in ['stoploss']
        ]
        return open_orders_ids_wo_sl

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.recalc_open_trade_value()
        self.orders = []
        if self.trading_mode == TradingMode.MARGIN and self.interest_rate is None:
            raise OperationalException(
                f"{self.trading_mode.value} trading requires param interest_rate on trades")

    def __repr__(self):
        open_since = (
            self.open_date_utc.strftime(DATETIME_PRINT_FORMAT) if self.is_open else 'closed'
        )

        return (
            f'Trade(id={self.id}, pair={self.pair}, amount={self.amount:.8f}, '
            f'is_short={self.is_short or False}, leverage={self.leverage or 1.0}, '
            f'open_rate={self.open_rate:.8f}, open_since={open_since})'
        )

    def to_json(self, minified: bool = False) -> Dict[str, Any]:
        """
        :param minified: If True, only return a subset of the data is returned.
                         Only used for backtesting.
        :return: Dictionary with trade data
        """
        filled_or_open_orders = self.select_filled_or_open_orders()
        orders_json = [order.to_json(self.entry_side, minified) for order in filled_or_open_orders]

        return {
            'trade_id': self.id,
            'pair': self.pair,
            'base_currency': self.safe_base_currency,
            'quote_currency': self.safe_quote_currency,
            'is_open': self.is_open,
            'exchange': self.exchange,
            'amount': round(self.amount, 8),
            'amount_requested': round(self.amount_requested, 8) if self.amount_requested else None,
            'stake_amount': round(self.stake_amount, 8),
            'max_stake_amount': round(self.max_stake_amount, 8) if self.max_stake_amount else None,
            'strategy': self.strategy,
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
            'realized_profit': self.realized_profit or 0.0,
            # Close-profit corresponds to relative realized_profit ratio
            'realized_profit_ratio': self.close_profit or None,
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

            'exit_reason': self.exit_reason,
            'exit_order_status': self.exit_order_status,
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
            'liquidation_price': self.liquidation_price,
            'is_short': self.is_short,
            'trading_mode': self.trading_mode,
            'funding_fees': self.funding_fees,
            'amount_precision': self.amount_precision,
            'price_precision': self.price_precision,
            'precision_mode': self.precision_mode,
            'contract_size': self.contract_size,
            'has_open_orders': self.has_open_orders,
            'orders': orders_json,
        }

    @staticmethod
    def reset_trades() -> None:
        """
        Resets all trades. Only active for backtesting mode.
        """
        LocalTrade.trades = []
        LocalTrade.trades_open = []
        LocalTrade.bt_trades_open_pp = defaultdict(list)
        LocalTrade.bt_open_open_trade_count = 0
        LocalTrade.total_profit = 0

    def adjust_min_max_rates(self, current_price: float, current_price_low: float) -> None:
        """
        Adjust the max_rate and min_rate.
        """
        self.max_rate = max(current_price, self.max_rate or self.open_rate)
        self.min_rate = min(current_price_low, self.min_rate or self.open_rate)

    def set_liquidation_price(self, liquidation_price: Optional[float]):
        """
        Method you should use to set self.liquidation price.
        Assures stop_loss is not passed the liquidation price
        """
        if not liquidation_price:
            return
        self.liquidation_price = liquidation_price

    def set_funding_fees(self, funding_fee: float) -> None:
        """
        Assign funding fees to Trade.
        """
        if funding_fee is None:
            return
        self.funding_fee_running = funding_fee
        prior_funding_fees = sum([o.funding_fee for o in self.orders if o.funding_fee])
        self.funding_fees = prior_funding_fees + funding_fee

    def __set_stop_loss(self, stop_loss: float, percent: float):
        """
        Method used internally to set self.stop_loss.
        """
        if not self.stop_loss:
            self.initial_stop_loss = stop_loss
        self.stop_loss = stop_loss

        self.stop_loss_pct = -1 * abs(percent)

    def adjust_stop_loss(self, current_price: float, stoploss: Optional[float],
                         initial: bool = False, allow_refresh: bool = False) -> None:
        """
        This adjusts the stop loss to it's most recently observed setting
        :param current_price: Current rate the asset is traded
        :param stoploss: Stoploss as factor (sample -0.05 -> -5% below current price).
        :param initial: Called to initiate stop_loss.
            Skips everything if self.stop_loss is already set.
        :param refresh: Called to refresh stop_loss, allows adjustment in both directions
        """
        if stoploss is None or (initial and not (self.stop_loss is None or self.stop_loss == 0)):
            # Don't modify if called with initial and nothing to do
            return

        leverage = self.leverage or 1.0
        if self.is_short:
            new_loss = float(current_price * (1 + abs(stoploss / leverage)))
        else:
            new_loss = float(current_price * (1 - abs(stoploss / leverage)))

        stop_loss_norm = price_to_precision(new_loss, self.price_precision, self.precision_mode,
                                            rounding_mode=ROUND_DOWN if self.is_short else ROUND_UP)
        # no stop loss assigned yet
        if self.initial_stop_loss_pct is None:
            self.__set_stop_loss(stop_loss_norm, stoploss)
            self.initial_stop_loss = price_to_precision(
                stop_loss_norm, self.price_precision, self.precision_mode,
                rounding_mode=ROUND_DOWN if self.is_short else ROUND_UP)
            self.initial_stop_loss_pct = -1 * abs(stoploss)

        # evaluate if the stop loss needs to be updated
        else:
            higher_stop = stop_loss_norm > self.stop_loss
            lower_stop = stop_loss_norm < self.stop_loss

            # stop losses only walk up, never down!,
            #   ? But adding more to a leveraged trade would create a lower liquidation price,
            #   ? decreasing the minimum stoploss
            if (
                allow_refresh
                or (higher_stop and not self.is_short)
                or (lower_stop and self.is_short)
            ):
                logger.debug(f"{self.pair} - Adjusting stoploss...")
                if not allow_refresh:
                    self.is_stop_loss_trailing = True
                self.__set_stop_loss(stop_loss_norm, stoploss)
            else:
                logger.debug(f"{self.pair} - Keeping current stoploss...")

        logger.debug(
            f"{self.pair} - Stoploss adjusted. current_price={current_price:.8f}, "
            f"open_rate={self.open_rate:.8f}, max_rate={self.max_rate or self.open_rate:.8f}, "
            f"initial_stop_loss={self.initial_stop_loss:.8f}, "
            f"stop_loss={self.stop_loss:.8f}. "
            f"Trailing stoploss saved us: "
            f"{float(self.stop_loss) - float(self.initial_stop_loss or 0.0):.8f}.")

    def update_trade(self, order: Order, recalculating: bool = False) -> None:
        """
        Updates this entity with amount and actual open/close rates.
        :param order: order retrieved by exchange.fetch_order()
        :return: None
        """

        # Ignore open and cancelled orders
        if order.status == 'open' or order.safe_price is None:
            return

        logger.info(f'Updating trade (id={self.id}) ...')
        if order.ft_order_side != 'stoploss':
            order.funding_fee = self.funding_fee_running
            # Reset running funding fees
            self.funding_fee_running = 0.0

        if order.ft_order_side == self.entry_side:
            # Update open rate and actual amount
            self.open_rate = order.safe_price
            self.amount = order.safe_amount_after_fee
            if self.is_open:
                payment = "SELL" if self.is_short else "BUY"
                logger.info(f'{order.order_type.upper()}_{payment} has been fulfilled for {self}.')

            self.recalc_trade_from_orders()
        elif order.ft_order_side == self.exit_side:
            if self.is_open:
                payment = "BUY" if self.is_short else "SELL"
                # * On margin shorts, you buy a little bit more than the amount (amount + interest)
                logger.info(f'{order.order_type.upper()}_{payment} has been fulfilled for {self}.')

        elif order.ft_order_side == 'stoploss' and order.status not in ('open', ):
            self.stoploss_order_id = None
            self.close_rate_requested = self.stop_loss
            self.exit_reason = ExitType.STOPLOSS_ON_EXCHANGE.value
            if self.is_open and order.safe_filled > 0:
                logger.info(f'{order.order_type.upper()} is hit for {self}.')
        else:
            raise ValueError(f'Unknown order type: {order.order_type}')

        if order.ft_order_side != self.entry_side:
            amount_tr = amount_to_contract_precision(self.amount, self.amount_precision,
                                                     self.precision_mode, self.contract_size)
            if (
                isclose(order.safe_amount_after_fee, amount_tr, abs_tol=MATH_CLOSE_PREC)
                or (not recalculating and order.safe_amount_after_fee > amount_tr)
            ):
                # When recalculating a trade, only comming out to 0 can force a close
                self.close(order.safe_price)
            else:
                self.recalc_trade_from_orders()

        Trade.commit()

    def close(self, rate: float, *, show_msg: bool = True) -> None:
        """
        Sets close_rate to the given rate, calculates total profit
        and marks trade as closed
        """
        self.close_rate = rate
        self.close_date = self.close_date or self._date_last_filled_utc or dt_now()
        self.is_open = False
        self.exit_order_status = 'closed'
        self.recalc_trade_from_orders(is_closing=True)
        if show_msg:
            logger.info(f"Marking {self} as closed as the trade is fulfilled "
                        "and found no open orders for it.")

    def update_fee(self, fee_cost: float, fee_currency: Optional[str], fee_rate: Optional[float],
                   side: str) -> None:
        """
        Update Fee parameters. Only acts once per side
        """
        if self.entry_side == side and self.fee_open_currency is None:
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
        if self.entry_side == side:
            return self.fee_open_currency is not None
        elif self.exit_side == side:
            return self.fee_close_currency is not None
        else:
            return False

    def update_order(self, order: Dict) -> None:
        Order.update_orders(self.orders, order)

    def get_canceled_exit_order_count(self) -> int:
        """
        Get amount of failed exiting orders
        assumes full exits.
        """
        return len([o for o in self.orders if o.ft_order_side == self.exit_side
                    and o.status in CANCELED_EXCHANGE_STATES])

    def _calc_open_trade_value(self, amount: float, open_rate: float) -> float:
        """
        Calculate the open_rate including open_fee.
        :return: Price in of the open trade incl. Fees
        """
        open_trade = FtPrecise(amount) * FtPrecise(open_rate)
        fees = open_trade * FtPrecise(self.fee_open)
        if self.is_short:
            return float(open_trade - fees)
        else:
            return float(open_trade + fees)

    def recalc_open_trade_value(self) -> None:
        """
        Recalculate open_trade_value.
        Must be called whenever open_rate, fee_open is changed.
        """
        self.open_trade_value = self._calc_open_trade_value(self.amount, self.open_rate)

    def calculate_interest(self) -> FtPrecise:
        """
        Calculate interest for this trade. Only applicable for Margin trading.
        """
        zero = FtPrecise(0.0)
        # If nothing was borrowed
        if self.trading_mode != TradingMode.MARGIN or self.has_no_leverage:
            return zero

        open_date = self.open_date.replace(tzinfo=None)
        now = (self.close_date or datetime.now(timezone.utc)).replace(tzinfo=None)
        sec_per_hour = FtPrecise(3600)
        total_seconds = FtPrecise((now - open_date).total_seconds())
        hours = total_seconds / sec_per_hour or zero

        rate = FtPrecise(self.interest_rate)
        borrowed = FtPrecise(self.borrowed)

        return interest(exchange_name=self.exchange, borrowed=borrowed, rate=rate, hours=hours)

    def _calc_base_close(self, amount: FtPrecise, rate: float, fee: Optional[float]) -> FtPrecise:

        close_trade = amount * FtPrecise(rate)
        fees = close_trade * FtPrecise(fee or 0.0)

        if self.is_short:
            return close_trade + fees
        else:
            return close_trade - fees

    def calc_close_trade_value(self, rate: float, amount: Optional[float] = None) -> float:
        """
        Calculate the Trade's close value including fees
        :param rate: rate to compare with.
        :return: value in stake currency of the open trade
        """
        if rate is None and not self.close_rate:
            return 0.0

        amount1 = FtPrecise(amount or self.amount)
        trading_mode = self.trading_mode or TradingMode.SPOT

        if trading_mode == TradingMode.SPOT:
            return float(self._calc_base_close(amount1, rate, self.fee_close))

        elif (trading_mode == TradingMode.MARGIN):

            total_interest = self.calculate_interest()

            if self.is_short:
                amount1 = amount1 + total_interest
                return float(self._calc_base_close(amount1, rate, self.fee_close))
            else:
                # Currency already owned for longs, no need to purchase
                return float(self._calc_base_close(amount1, rate, self.fee_close) - total_interest)

        elif (trading_mode == TradingMode.FUTURES):
            funding_fees = self.funding_fees or 0.0
            # Positive funding_fees -> Trade has gained from fees.
            # Negative funding_fees -> Trade had to pay the fees.
            if self.is_short:
                return float(self._calc_base_close(amount1, rate, self.fee_close)) - funding_fees
            else:
                return float(self._calc_base_close(amount1, rate, self.fee_close)) + funding_fees
        else:
            raise OperationalException(
                f"{self.trading_mode.value} trading is not yet available using freqtrade")

    def calc_profit(self, rate: float, amount: Optional[float] = None,
                    open_rate: Optional[float] = None) -> float:
        """
        Calculate the absolute profit in stake currency between Close and Open trade
        Deprecated - only available for backwards compatibility
        :param rate: close rate to compare with.
        :param amount: Amount to use for the calculation. Falls back to trade.amount if not set.
        :param open_rate: open_rate to use. Defaults to self.open_rate if not provided.
        :return: profit in stake currency as float
        """
        prof = self.calculate_profit(rate, amount, open_rate)
        return prof.profit_abs

    def calculate_profit(self, rate: float, amount: Optional[float] = None,
                         open_rate: Optional[float] = None) -> ProfitStruct:
        """
        Calculate profit metrics (absolute, ratio, total, total ratio).
        All calculations include fees.
        :param rate: close rate to compare with.
        :param amount: Amount to use for the calculation. Falls back to trade.amount if not set.
        :param open_rate: open_rate to use. Defaults to self.open_rate if not provided.
        :return: Profit structure, containing absolute and relative profits.
        """

        close_trade_value = self.calc_close_trade_value(rate, amount)
        if amount is None or open_rate is None:
            open_trade_value = self.open_trade_value
        else:
            open_trade_value = self._calc_open_trade_value(amount, open_rate)

        if self.is_short:
            profit_abs = open_trade_value - close_trade_value
        else:
            profit_abs = close_trade_value - open_trade_value

        try:
            if self.is_short:
                profit_ratio = (1 - (close_trade_value / open_trade_value)) * self.leverage
            else:
                profit_ratio = ((close_trade_value / open_trade_value) - 1) * self.leverage
            profit_ratio = float(f"{profit_ratio:.8f}")
        except ZeroDivisionError:
            profit_ratio = 0.0

        total_profit_abs = profit_abs + self.realized_profit
        total_profit_ratio = (
            (total_profit_abs / self.max_stake_amount) * self.leverage
            if self.max_stake_amount else 0.0
        )
        total_profit_ratio = float(f"{total_profit_ratio:.8f}")
        profit_abs = float(f"{profit_abs:.8f}")

        return ProfitStruct(
            profit_abs=profit_abs,
            profit_ratio=profit_ratio,
            total_profit=profit_abs + self.realized_profit,
            total_profit_ratio=total_profit_ratio,
        )

    def calc_profit_ratio(
            self, rate: float, amount: Optional[float] = None,
            open_rate: Optional[float] = None) -> float:
        """
        Calculates the profit as ratio (including fee).
        :param rate: rate to compare with.
        :param amount: Amount to use for the calculation. Falls back to trade.amount if not set.
        :param open_rate: open_rate to use. Defaults to self.open_rate if not provided.
        :return: profit ratio as float
        """
        close_trade_value = self.calc_close_trade_value(rate, amount)

        if amount is None or open_rate is None:
            open_trade_value = self.open_trade_value
        else:
            open_trade_value = self._calc_open_trade_value(amount, open_rate)

        short_close_zero = (self.is_short and close_trade_value == 0.0)
        long_close_zero = (not self.is_short and open_trade_value == 0.0)

        if (short_close_zero or long_close_zero):
            return 0.0
        else:
            if self.is_short:
                profit_ratio = (1 - (close_trade_value / open_trade_value)) * self.leverage
            else:
                profit_ratio = ((close_trade_value / open_trade_value) - 1) * self.leverage

        return float(f"{profit_ratio:.8f}")

    def recalc_trade_from_orders(self, *, is_closing: bool = False):
        ZERO = FtPrecise(0.0)
        current_amount = FtPrecise(0.0)
        current_stake = FtPrecise(0.0)
        max_stake_amount = FtPrecise(0.0)
        total_stake = 0.0  # Total stake after all buy orders (does not subtract!)
        avg_price = FtPrecise(0.0)
        close_profit = 0.0
        close_profit_abs = 0.0
        # Reset funding fees
        self.funding_fees = 0.0
        funding_fees = 0.0
        ordercount = len(self.orders) - 1
        for i, o in enumerate(self.orders):
            if o.ft_is_open or not o.filled:
                continue
            funding_fees += (o.funding_fee or 0.0)
            tmp_amount = FtPrecise(o.safe_amount_after_fee)
            tmp_price = FtPrecise(o.safe_price)

            is_exit = o.ft_order_side != self.entry_side
            side = FtPrecise(-1 if is_exit else 1)
            if tmp_amount > ZERO and tmp_price is not None:
                current_amount += tmp_amount * side
                price = avg_price if is_exit else tmp_price
                current_stake += price * tmp_amount * side

                if current_amount > ZERO and not is_exit:
                    avg_price = current_stake / current_amount

            if is_exit:
                # Process exits
                if i == ordercount and is_closing:
                    # Apply funding fees only to the last closing order
                    self.funding_fees = funding_fees

                exit_rate = o.safe_price
                exit_amount = o.safe_amount_after_fee
                prof = self.calculate_profit(exit_rate, exit_amount, float(avg_price))
                close_profit_abs += prof.profit_abs
                if total_stake > 0:
                    # This needs to be calculated based on the last occuring exit to be aligned
                    # with realized_profit.
                    close_profit = (close_profit_abs / total_stake) * self.leverage
            else:
                total_stake = total_stake + self._calc_open_trade_value(tmp_amount, price)
                max_stake_amount += (tmp_amount * price)
        self.funding_fees = funding_fees
        self.max_stake_amount = float(max_stake_amount)

        if close_profit:
            self.close_profit = close_profit
            self.realized_profit = close_profit_abs
            self.close_profit_abs = prof.profit_abs

        current_amount_tr = amount_to_contract_precision(
            float(current_amount), self.amount_precision, self.precision_mode, self.contract_size)
        if current_amount_tr > 0.0:
            # Trade is still open
            # Leverage not updated, as we don't allow changing leverage through DCA at the moment.
            self.open_rate = float(current_stake / current_amount)
            self.amount = current_amount_tr
            self.stake_amount = float(current_stake) / (self.leverage or 1.0)
            self.fee_open_cost = self.fee_open * float(current_stake)
            self.recalc_open_trade_value()
            if self.stop_loss_pct is not None and self.open_rate is not None:
                self.adjust_stop_loss(self.open_rate, self.stop_loss_pct)
        elif is_closing and total_stake > 0:
            # Close profit abs / maximum owned
            # Fees are considered as they are part of close_profit_abs
            self.close_profit = (close_profit_abs / total_stake) * self.leverage
            self.close_profit_abs = close_profit_abs

    def select_order_by_order_id(self, order_id: str) -> Optional[Order]:
        """
        Finds order object by Order id.
        :param order_id: Exchange order id
        """
        for o in self.orders:
            if o.order_id == order_id:
                return o
        return None

    def select_order(self, order_side: Optional[str] = None,
                     is_open: Optional[bool] = None, only_filled: bool = False) -> Optional[Order]:
        """
        Finds latest order for this orderside and status
        :param order_side: ft_order_side of the order (either 'buy', 'sell' or 'stoploss')
        :param is_open: Only search for open orders?
        :param only_filled: Only search for Filled orders (only valid with is_open=False).
        :return: latest Order object if it exists, else None
        """
        orders = self.orders
        if order_side:
            orders = [o for o in orders if o.ft_order_side == order_side]
        if is_open is not None:
            orders = [o for o in orders if o.ft_is_open == is_open]
        if is_open is False and only_filled:
            orders = [o for o in orders if o.filled and o.status in NON_OPEN_EXCHANGE_STATES]
        if len(orders) > 0:
            return orders[-1]
        else:
            return None

    def select_filled_orders(self, order_side: Optional[str] = None) -> List['Order']:
        """
        Finds filled orders for this order side.
        Will not return open orders which already partially filled.
        :param order_side: Side of the order (either 'buy', 'sell', or None)
        :return: array of Order objects
        """
        return [o for o in self.orders if ((o.ft_order_side == order_side) or (order_side is None))
                and o.ft_is_open is False
                and o.filled
                and o.status in NON_OPEN_EXCHANGE_STATES]

    def select_filled_or_open_orders(self) -> List['Order']:
        """
        Finds filled or open orders
        :param order_side: Side of the order (either 'buy', 'sell', or None)
        :return: array of Order objects
        """
        return [o for o in self.orders if
                (
                    o.ft_is_open is False
                    and (o.filled or 0) > 0
                    and o.status in NON_OPEN_EXCHANGE_STATES
                    )
                or (o.ft_is_open is True and o.status is not None)
                ]

    @property
    def nr_of_successful_entries(self) -> int:
        """
        Helper function to count the number of entry orders that have been filled.
        :return: int count of entry orders that have been filled for this trade.
        """

        return len(self.select_filled_orders(self.entry_side))

    @property
    def nr_of_successful_exits(self) -> int:
        """
        Helper function to count the number of exit orders that have been filled.
        :return: int count of exit orders that have been filled for this trade.
        """
        return len(self.select_filled_orders(self.exit_side))

    @property
    def nr_of_successful_buys(self) -> int:
        """
        Helper function to count the number of buy orders that have been filled.
        WARNING: Please use nr_of_successful_entries for short support.
        :return: int count of buy orders that have been filled for this trade.
        """

        return len(self.select_filled_orders('buy'))

    @property
    def nr_of_successful_sells(self) -> int:
        """
        Helper function to count the number of sell orders that have been filled.
        WARNING: Please use nr_of_successful_exits for short support.
        :return: int count of sell orders that have been filled for this trade.
        """
        return len(self.select_filled_orders('sell'))

    @property
    def sell_reason(self) -> Optional[str]:
        """ DEPRECATED! Please use exit_reason instead."""
        return self.exit_reason

    @property
    def safe_close_rate(self) -> float:
        return self.close_rate or self.close_rate_requested or 0.0

    @staticmethod
    def get_trades_proxy(*, pair: Optional[str] = None, is_open: Optional[bool] = None,
                         open_date: Optional[datetime] = None,
                         close_date: Optional[datetime] = None,
                         ) -> List['LocalTrade']:
        """
        Helper function to query Trades.
        Returns a List of trades, filtered on the parameters given.
        In live mode, converts the filter to a database query and returns all rows
        In Backtest mode, uses filters on Trade.trades to get the result.

        :param pair: Filter by pair
        :param is_open: Filter by open/closed status
        :param open_date: Filter by open_date (filters via trade.open_date > input)
        :param close_date: Filter by close_date (filters via trade.close_date > input)
                           Will implicitly only return closed trades.
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
        LocalTrade.bt_trades_open_pp[trade.pair].remove(trade)
        LocalTrade.bt_open_open_trade_count -= 1
        LocalTrade.trades.append(trade)
        LocalTrade.total_profit += trade.close_profit_abs

    @staticmethod
    def add_bt_trade(trade):
        if trade.is_open:
            LocalTrade.trades_open.append(trade)
            LocalTrade.bt_trades_open_pp[trade.pair].append(trade)
            LocalTrade.bt_open_open_trade_count += 1
        else:
            LocalTrade.trades.append(trade)

    @staticmethod
    def remove_bt_trade(trade):
        LocalTrade.trades_open.remove(trade)
        LocalTrade.bt_trades_open_pp[trade.pair].remove(trade)
        LocalTrade.bt_open_open_trade_count -= 1

    @staticmethod
    def get_open_trades() -> List[Any]:
        """
        Retrieve open trades
        """
        return Trade.get_trades_proxy(is_open=True)

    @staticmethod
    def get_open_trade_count() -> int:
        """
        get open trade count
        """
        if Trade.use_db:
            return Trade.session.execute(
                select(func.count(Trade.id)).filter(Trade.is_open.is_(True))
            ).scalar_one()
        else:
            return LocalTrade.bt_open_open_trade_count

    @staticmethod
    def stoploss_reinitialization(desired_stoploss: float):
        """
        Adjust initial Stoploss to desired stoploss for all open trades.
        """
        trade: Trade
        for trade in Trade.get_open_trades():
            logger.info(f"Found open trade: {trade}")

            # skip case if trailing-stop changed the stoploss already.
            if (not trade.is_stop_loss_trailing
                    and trade.initial_stop_loss_pct != desired_stoploss):
                # Stoploss value got changed

                logger.info(f"Stoploss for {trade} needs adjustment...")
                # Force reset of stoploss
                trade.stop_loss = 0.0
                trade.initial_stop_loss_pct = None
                trade.adjust_stop_loss(trade.open_rate, desired_stoploss)
                logger.info(f"New stoploss: {trade.stop_loss}.")

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """
        Create a Trade instance from a json string.

        Used for debugging purposes - please keep.
        :param json_str: json string to parse
        :return: Trade instance
        """
        import rapidjson
        data = rapidjson.loads(json_str)
        trade = cls(
            __FROM_JSON=True,
            id=data["trade_id"],
            pair=data["pair"],
            base_currency=data["base_currency"],
            stake_currency=data["quote_currency"],
            is_open=data["is_open"],
            exchange=data["exchange"],
            amount=data["amount"],
            amount_requested=data["amount_requested"],
            stake_amount=data["stake_amount"],
            strategy=data["strategy"],
            enter_tag=data["enter_tag"],
            timeframe=data["timeframe"],
            fee_open=data["fee_open"],
            fee_open_cost=data["fee_open_cost"],
            fee_open_currency=data["fee_open_currency"],
            fee_close=data["fee_close"],
            fee_close_cost=data["fee_close_cost"],
            fee_close_currency=data["fee_close_currency"],
            open_date=datetime.fromtimestamp(data["open_timestamp"] // 1000, tz=timezone.utc),
            open_rate=data["open_rate"],
            open_rate_requested=data["open_rate_requested"],
            open_trade_value=data["open_trade_value"],
            close_date=(datetime.fromtimestamp(data["close_timestamp"] // 1000, tz=timezone.utc)
                        if data["close_timestamp"] else None),
            realized_profit=data["realized_profit"],
            close_rate=data["close_rate"],
            close_rate_requested=data["close_rate_requested"],
            close_profit=data["close_profit"],
            close_profit_abs=data["close_profit_abs"],
            exit_reason=data["exit_reason"],
            exit_order_status=data["exit_order_status"],
            stop_loss=data["stop_loss_abs"],
            stop_loss_pct=data["stop_loss_ratio"],
            stoploss_order_id=data["stoploss_order_id"],
            stoploss_last_update=(
                datetime.fromtimestamp(data["stoploss_last_update_timestamp"] // 1000,
                                       tz=timezone.utc)
                if data["stoploss_last_update_timestamp"] else None),
            initial_stop_loss=data["initial_stop_loss_abs"],
            initial_stop_loss_pct=data["initial_stop_loss_ratio"],
            min_rate=data["min_rate"],
            max_rate=data["max_rate"],
            leverage=data["leverage"],
            interest_rate=data["interest_rate"],
            liquidation_price=data["liquidation_price"],
            is_short=data["is_short"],
            trading_mode=data["trading_mode"],
            funding_fees=data["funding_fees"],
            amount_precision=data.get('amount_precision', None),
            price_precision=data.get('price_precision', None),
            precision_mode=data.get('precision_mode', None),
            contract_size=data.get('contract_size', None),
        )
        for order in data["orders"]:

            order_obj = Order(
                amount=order["amount"],
                ft_amount=order["amount"],
                ft_order_side=order["ft_order_side"],
                ft_pair=order["pair"],
                ft_is_open=order["is_open"],
                order_id=order["order_id"],
                status=order["status"],
                average=order["average"],
                cost=order["cost"],
                filled=order["filled"],
                order_date=datetime.strptime(order["order_date"], DATETIME_PRINT_FORMAT),
                order_filled_date=(datetime.fromtimestamp(
                    order["order_filled_timestamp"] // 1000, tz=timezone.utc)
                    if order["order_filled_timestamp"] else None),
                order_type=order["order_type"],
                price=order["price"],
                ft_price=order["price"],
                remaining=order["remaining"],
                funding_fee=order.get("funding_fee", None),
            )
            trade.orders.append(order_obj)

        return trade


class Trade(ModelBase, LocalTrade):
    """
    Trade database model.
    Also handles updating and querying trades

    Note: Fields must be aligned with LocalTrade class
    """
    __tablename__ = 'trades'
    session: ClassVar[SessionType]

    use_db: bool = True

    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # type: ignore

    orders: Mapped[List[Order]] = relationship(
        "Order", order_by="Order.id", cascade="all, delete-orphan", lazy="selectin",
        innerjoin=True)  # type: ignore

    exchange: Mapped[str] = mapped_column(String(25), nullable=False)  # type: ignore
    pair: Mapped[str] = mapped_column(String(25), nullable=False, index=True)  # type: ignore
    base_currency: Mapped[Optional[str]] = mapped_column(String(25), nullable=True)  # type: ignore
    stake_currency: Mapped[Optional[str]] = mapped_column(String(25), nullable=True)  # type: ignore
    is_open: Mapped[bool] = mapped_column(nullable=False, default=True, index=True)  # type: ignore
    fee_open: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)  # type: ignore
    fee_open_cost: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)  # type: ignore
    fee_open_currency: Mapped[Optional[str]] = mapped_column(
        String(25), nullable=True)  # type: ignore
    fee_close: Mapped[Optional[float]] = mapped_column(
        Float(), nullable=False, default=0.0)  # type: ignore
    fee_close_cost: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)  # type: ignore
    fee_close_currency: Mapped[Optional[str]] = mapped_column(
        String(25), nullable=True)  # type: ignore
    open_rate: Mapped[float] = mapped_column(Float())  # type: ignore
    open_rate_requested: Mapped[Optional[float]] = mapped_column(
        Float(), nullable=True)  # type: ignore
    # open_trade_value - calculated via _calc_open_trade_value
    open_trade_value: Mapped[float] = mapped_column(Float(), nullable=True)  # type: ignore
    close_rate: Mapped[Optional[float]] = mapped_column(Float())  # type: ignore
    close_rate_requested: Mapped[Optional[float]] = mapped_column(Float())  # type: ignore
    realized_profit: Mapped[float] = mapped_column(
        Float(), default=0.0, nullable=True)  # type: ignore
    close_profit: Mapped[Optional[float]] = mapped_column(Float())  # type: ignore
    close_profit_abs: Mapped[Optional[float]] = mapped_column(Float())  # type: ignore
    stake_amount: Mapped[float] = mapped_column(Float(), nullable=False)  # type: ignore
    max_stake_amount: Mapped[Optional[float]] = mapped_column(Float())  # type: ignore
    amount: Mapped[float] = mapped_column(Float())  # type: ignore
    amount_requested: Mapped[Optional[float]] = mapped_column(Float())  # type: ignore
    open_date: Mapped[datetime] = mapped_column(
        nullable=False, default=datetime.utcnow)  # type: ignore
    close_date: Mapped[Optional[datetime]] = mapped_column()  # type: ignore
    # absolute value of the stop loss
    stop_loss: Mapped[float] = mapped_column(Float(), nullable=True, default=0.0)  # type: ignore
    # percentage value of the stop loss
    stop_loss_pct: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)  # type: ignore
    # absolute value of the initial stop loss
    initial_stop_loss: Mapped[Optional[float]] = mapped_column(
        Float(), nullable=True, default=0.0)  # type: ignore
    # percentage value of the initial stop loss
    initial_stop_loss_pct: Mapped[Optional[float]] = mapped_column(
        Float(), nullable=True)  # type: ignore
    is_stop_loss_trailing: Mapped[bool] = mapped_column(
        nullable=False, default=False)  # type: ignore
    # stoploss order id which is on exchange
    stoploss_order_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True)  # type: ignore
    # last update time of the stoploss order on exchange
    stoploss_last_update: Mapped[Optional[datetime]] = mapped_column(nullable=True)  # type: ignore
    # absolute value of the highest reached price
    max_rate: Mapped[Optional[float]] = mapped_column(
        Float(), nullable=True, default=0.0)  # type: ignore
    # Lowest price reached
    min_rate: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)  # type: ignore
    exit_reason: Mapped[Optional[str]] = mapped_column(
        String(CUSTOM_TAG_MAX_LENGTH), nullable=True)  # type: ignore
    exit_order_status: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True)  # type: ignore
    strategy: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # type: ignore
    enter_tag: Mapped[Optional[str]] = mapped_column(
        String(CUSTOM_TAG_MAX_LENGTH), nullable=True)  # type: ignore
    timeframe: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # type: ignore

    trading_mode: Mapped[TradingMode] = mapped_column(
        Enum(TradingMode), nullable=True)  # type: ignore
    amount_precision: Mapped[Optional[float]] = mapped_column(
        Float(), nullable=True)  # type: ignore
    price_precision: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)  # type: ignore
    precision_mode: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # type: ignore
    contract_size: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)  # type: ignore

    # Leverage trading properties
    leverage: Mapped[float] = mapped_column(Float(), nullable=True, default=1.0)  # type: ignore
    is_short: Mapped[bool] = mapped_column(nullable=False, default=False)  # type: ignore
    liquidation_price: Mapped[Optional[float]] = mapped_column(
        Float(), nullable=True)  # type: ignore

    # Margin Trading Properties
    interest_rate: Mapped[float] = mapped_column(
        Float(), nullable=False, default=0.0)  # type: ignore

    # Futures properties
    funding_fees: Mapped[Optional[float]] = mapped_column(
        Float(), nullable=True, default=None)  # type: ignore
    funding_fee_running: Mapped[Optional[float]] = mapped_column(
        Float(), nullable=True, default=None)  # type: ignore

    def __init__(self, **kwargs):
        from_json = kwargs.pop('__FROM_JSON', None)
        super().__init__(**kwargs)
        if not from_json:
            # Skip recalculation when loading from json
            self.realized_profit = 0
            self.recalc_open_trade_value()

    @validates('enter_tag', 'exit_reason')
    def validate_string_len(self, key, value):
        max_len = getattr(self.__class__, key).prop.columns[0].type.length
        if value and len(value) > max_len:
            return value[:max_len]
        return value

    def delete(self) -> None:

        for order in self.orders:
            Order.session.delete(order)

        Trade.session.delete(self)
        Trade.commit()

    @staticmethod
    def commit():
        Trade.session.commit()

    @staticmethod
    def rollback():
        Trade.session.rollback()

    @staticmethod
    def get_trades_proxy(*, pair: Optional[str] = None, is_open: Optional[bool] = None,
                         open_date: Optional[datetime] = None,
                         close_date: Optional[datetime] = None,
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
            return cast(List[LocalTrade], Trade.get_trades(trade_filter).all())
        else:
            return LocalTrade.get_trades_proxy(
                pair=pair, is_open=is_open,
                open_date=open_date,
                close_date=close_date
            )

    @staticmethod
    def get_trades_query(trade_filter=None, include_orders: bool = True) -> Select:
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
            this_query = select(Trade).filter(*trade_filter)
        else:
            this_query = select(Trade)
        if not include_orders:
            # Don't load order relations
            # Consider using noload or raiseload instead of lazyload
            this_query = this_query.options(lazyload(Trade.orders))
        return this_query

    @staticmethod
    def get_trades(trade_filter=None, include_orders: bool = True) -> ScalarResult['Trade']:
        """
        Helper function to query Trades using filters.
        NOTE: Not supported in Backtesting.
        :param trade_filter: Optional filter to apply to trades
                             Can be either a Filter object, or a List of filters
                             e.g. `(trade_filter=[Trade.id == trade_id, Trade.is_open.is_(True),])`
                             e.g. `(trade_filter=Trade.id == trade_id)`
        :return: unsorted query object
        """
        query = Trade.get_trades_query(trade_filter, include_orders)
        # this should remain split. if use_db is False, session is not available and the above will
        # raise an exception.
        return Trade.session.scalars(query)

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
            total_profit = Trade.session.execute(
                select(func.sum(Trade.close_profit_abs)).filter(Trade.is_open.is_(False))
            ).scalar_one()
        else:
            total_profit = sum(t.close_profit_abs  # type: ignore
                               for t in LocalTrade.get_trades_proxy(is_open=False))
        return total_profit or 0

    @staticmethod
    def total_open_trades_stakes() -> float:
        """
        Calculates total invested amount in open trades
        in stake currency
        """
        if Trade.use_db:
            total_open_stake_amount = Trade.session.scalar(
                select(func.sum(Trade.stake_amount)).filter(Trade.is_open.is_(True))
            )
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
        filters: List = [Trade.is_open.is_(False)]
        if minutes:
            start_date = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            filters.append(Trade.close_date >= start_date)

        pair_rates = Trade.session.execute(
            select(
                Trade.pair,
                func.sum(Trade.close_profit).label('profit_sum'),
                func.sum(Trade.close_profit_abs).label('profit_sum_abs'),
                func.count(Trade.pair).label('count')
            ).filter(*filters)
            .group_by(Trade.pair)
            .order_by(desc('profit_sum_abs'))
            ).all()

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

        filters: List = [Trade.is_open.is_(False)]
        if (pair is not None):
            filters.append(Trade.pair == pair)

        enter_tag_perf = Trade.session.execute(
            select(
                Trade.enter_tag,
                func.sum(Trade.close_profit).label('profit_sum'),
                func.sum(Trade.close_profit_abs).label('profit_sum_abs'),
                func.count(Trade.pair).label('count')
            ).filter(*filters)
            .group_by(Trade.enter_tag)
            .order_by(desc('profit_sum_abs'))
        ).all()

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
    def get_exit_reason_performance(pair: Optional[str]) -> List[Dict[str, Any]]:
        """
        Returns List of dicts containing all Trades, based on exit reason performance
        Can either be average for all pairs or a specific pair provided
        NOTE: Not supported in Backtesting.
        """

        filters: List = [Trade.is_open.is_(False)]
        if (pair is not None):
            filters.append(Trade.pair == pair)
        sell_tag_perf = Trade.session.execute(
            select(
                Trade.exit_reason,
                func.sum(Trade.close_profit).label('profit_sum'),
                func.sum(Trade.close_profit_abs).label('profit_sum_abs'),
                func.count(Trade.pair).label('count')
            ).filter(*filters)
            .group_by(Trade.exit_reason)
            .order_by(desc('profit_sum_abs'))
        ).all()

        return [
            {
                'exit_reason': exit_reason if exit_reason is not None else "Other",
                'profit_ratio': profit,
                'profit_pct': round(profit * 100, 2),
                'profit_abs': profit_abs,
                'count': count
            }
            for exit_reason, profit, profit_abs, count in sell_tag_perf
        ]

    @staticmethod
    def get_mix_tag_performance(pair: Optional[str]) -> List[Dict[str, Any]]:
        """
        Returns List of dicts containing all Trades, based on entry_tag + exit_reason performance
        Can either be average for all pairs or a specific pair provided
        NOTE: Not supported in Backtesting.
        """

        filters: List = [Trade.is_open.is_(False)]
        if (pair is not None):
            filters.append(Trade.pair == pair)
        mix_tag_perf = Trade.session.execute(
            select(
                Trade.id,
                Trade.enter_tag,
                Trade.exit_reason,
                func.sum(Trade.close_profit).label('profit_sum'),
                func.sum(Trade.close_profit_abs).label('profit_sum_abs'),
                func.count(Trade.pair).label('count')
            ).filter(*filters)
            .group_by(Trade.id)
            .order_by(desc('profit_sum_abs'))
        ).all()

        resp: List[Dict] = []
        for id, enter_tag, exit_reason, profit, profit_abs, count in mix_tag_perf:
            enter_tag = enter_tag if enter_tag is not None else "Other"
            exit_reason = exit_reason if exit_reason is not None else "Other"

            if (exit_reason is not None and enter_tag is not None):
                mix_tag = enter_tag + " " + exit_reason
                i = 0
                if not any(item["mix_tag"] == mix_tag for item in resp):
                    resp.append({'mix_tag': mix_tag,
                                 'profit_ratio': profit,
                                 'profit_pct': round(profit * 100, 2),
                                 'profit_abs': profit_abs,
                                 'count': count})
                else:
                    while i < len(resp):
                        if resp[i]["mix_tag"] == mix_tag:
                            resp[i] = {
                                'mix_tag': mix_tag,
                                'profit_ratio': profit + resp[i]["profit_ratio"],
                                'profit_pct': round(profit + resp[i]["profit_ratio"] * 100, 2),
                                'profit_abs': profit_abs + resp[i]["profit_abs"],
                                'count': 1 + resp[i]["count"]
                            }
                        i += 1

        return resp

    @staticmethod
    def get_best_pair(start_date: datetime = datetime.fromtimestamp(0)):
        """
        Get best pair with closed trade.
        NOTE: Not supported in Backtesting.
        :returns: Tuple containing (pair, profit_sum)
        """
        best_pair = Trade.session.execute(
            select(
                Trade.pair,
                func.sum(Trade.close_profit).label('profit_sum')
            ).filter(Trade.is_open.is_(False) & (Trade.close_date >= start_date))
            .group_by(Trade.pair)
            .order_by(desc('profit_sum'))
        ).first()

        return best_pair

    @staticmethod
    def get_trading_volume(start_date: datetime = datetime.fromtimestamp(0)) -> float:
        """
        Get Trade volume based on Orders
        NOTE: Not supported in Backtesting.
        :returns: Tuple containing (pair, profit_sum)
        """
        trading_volume = Trade.session.execute(
            select(
                func.sum(Order.cost).label('volume')
            ).filter(
                Order.order_filled_date >= start_date,
                Order.status == 'closed'
            )).scalar_one()
        return trading_volume or 0.0
