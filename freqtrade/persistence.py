"""
This module contains the class to persist trades into SQLite
"""
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import arrow
from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, String,
                        create_engine, desc, func, inspect)
from sqlalchemy.exc import NoSuchModuleError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Query
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.pool import StaticPool

from freqtrade.exceptions import OperationalException

logger = logging.getLogger(__name__)


_DECL_BASE: Any = declarative_base()
_SQL_DOCS_URL = 'http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls'


def init(db_url: str, clean_open_orders: bool = False) -> None:
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
    _DECL_BASE.metadata.create_all(engine)
    check_migrate(engine)

    # Clean dry_run DB if the db is not in-memory
    if clean_open_orders and db_url != 'sqlite://':
        clean_dry_run_db()


def has_column(columns: List, searchname: str) -> bool:
    return len(list(filter(lambda x: x["name"] == searchname, columns))) == 1


def get_column_def(columns: List, column: str, default: str) -> str:
    return default if not has_column(columns, column) else column


def check_migrate(engine) -> None:
    """
    Checks if migration is necessary and migrates if necessary
    """
    inspector = inspect(engine)

    cols = inspector.get_columns('trades')
    tabs = inspector.get_table_names()
    table_back_name = 'trades_bak'
    for i, table_back_name in enumerate(tabs):
        table_back_name = f'trades_bak{i}'
        logger.debug(f'trying {table_back_name}')

    # Check for latest column
    if not has_column(cols, 'sell_order_status'):
        logger.info(f'Running database migration - backup available as {table_back_name}')

        fee_open = get_column_def(cols, 'fee_open', 'fee')
        fee_open_cost = get_column_def(cols, 'fee_open_cost', 'null')
        fee_open_currency = get_column_def(cols, 'fee_open_currency', 'null')
        fee_close = get_column_def(cols, 'fee_close', 'fee')
        fee_close_cost = get_column_def(cols, 'fee_close_cost', 'null')
        fee_close_currency = get_column_def(cols, 'fee_close_currency', 'null')
        open_rate_requested = get_column_def(cols, 'open_rate_requested', 'null')
        close_rate_requested = get_column_def(cols, 'close_rate_requested', 'null')
        stop_loss = get_column_def(cols, 'stop_loss', '0.0')
        stop_loss_pct = get_column_def(cols, 'stop_loss_pct', 'null')
        initial_stop_loss = get_column_def(cols, 'initial_stop_loss', '0.0')
        initial_stop_loss_pct = get_column_def(cols, 'initial_stop_loss_pct', 'null')
        stoploss_order_id = get_column_def(cols, 'stoploss_order_id', 'null')
        stoploss_last_update = get_column_def(cols, 'stoploss_last_update', 'null')
        max_rate = get_column_def(cols, 'max_rate', '0.0')
        min_rate = get_column_def(cols, 'min_rate', 'null')
        sell_reason = get_column_def(cols, 'sell_reason', 'null')
        strategy = get_column_def(cols, 'strategy', 'null')
        ticker_interval = get_column_def(cols, 'ticker_interval', 'null')
        open_trade_price = get_column_def(cols, 'open_trade_price',
                                          f'amount * open_rate * (1 + {fee_open})')
        close_profit_abs = get_column_def(
            cols, 'close_profit_abs',
            f"(amount * close_rate * (1 - {fee_close})) - {open_trade_price}")
        sell_order_status = get_column_def(cols, 'sell_order_status', 'null')

        # Schema migration necessary
        engine.execute(f"alter table trades rename to {table_back_name}")
        # drop indexes on backup table
        for index in inspector.get_indexes(table_back_name):
            engine.execute(f"drop index {index['name']}")
        # let SQLAlchemy create the schema as required
        _DECL_BASE.metadata.create_all(engine)

        # Copy data back - following the correct schema
        engine.execute(f"""insert into trades
                (id, exchange, pair, is_open,
                fee_open, fee_open_cost, fee_open_currency,
                fee_close, fee_close_cost, fee_open_currency, open_rate,
                open_rate_requested, close_rate, close_rate_requested, close_profit,
                stake_amount, amount, open_date, close_date, open_order_id,
                stop_loss, stop_loss_pct, initial_stop_loss, initial_stop_loss_pct,
                stoploss_order_id, stoploss_last_update,
                max_rate, min_rate, sell_reason, sell_order_status, strategy,
                ticker_interval, open_trade_price, close_profit_abs
                )
            select id, lower(exchange),
                case
                    when instr(pair, '_') != 0 then
                    substr(pair,    instr(pair, '_') + 1) || '/' ||
                    substr(pair, 1, instr(pair, '_') - 1)
                    else pair
                    end
                pair,
                is_open, {fee_open} fee_open, {fee_open_cost} fee_open_cost,
                {fee_open_currency} fee_open_currency, {fee_close} fee_close,
                {fee_close_cost} fee_close_cost, {fee_close_currency} fee_close_currency,
                open_rate, {open_rate_requested} open_rate_requested, close_rate,
                {close_rate_requested} close_rate_requested, close_profit,
                stake_amount, amount, open_date, close_date, open_order_id,
                {stop_loss} stop_loss, {stop_loss_pct} stop_loss_pct,
                {initial_stop_loss} initial_stop_loss,
                {initial_stop_loss_pct} initial_stop_loss_pct,
                {stoploss_order_id} stoploss_order_id, {stoploss_last_update} stoploss_last_update,
                {max_rate} max_rate, {min_rate} min_rate, {sell_reason} sell_reason,
                {sell_order_status} sell_order_status,
                {strategy} strategy, {ticker_interval} ticker_interval,
                {open_trade_price} open_trade_price, {close_profit_abs} close_profit_abs
                from {table_back_name}
             """)

        # Reread columns - the above recreated the table!
        inspector = inspect(engine)
        cols = inspector.get_columns('trades')


def cleanup() -> None:
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


class Trade(_DECL_BASE):
    """
    Class used to define a trade structure
    """
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
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
    # open_trade_price - calculated via _calc_open_trade_price
    open_trade_price = Column(Float)
    close_rate = Column(Float)
    close_rate_requested = Column(Float)
    close_profit = Column(Float)
    close_profit_abs = Column(Float)
    stake_amount = Column(Float, nullable=False)
    amount = Column(Float)
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
    ticker_interval = Column(Integer, nullable=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recalc_open_trade_price()

    def __repr__(self):
        open_since = self.open_date.strftime('%Y-%m-%d %H:%M:%S') if self.is_open else 'closed'

        return (f'Trade(id={self.id}, pair={self.pair}, amount={self.amount:.8f}, '
                f'open_rate={self.open_rate:.8f}, open_since={open_since})')

    def to_json(self) -> Dict[str, Any]:
        return {
            'trade_id': self.id,
            'pair': self.pair,
            'is_open': self.is_open,
            'fee_open': self.fee_open,
            'fee_open_cost': self.fee_open_cost,
            'fee_open_currency': self.fee_open_currency,
            'fee_close': self.fee_close,
            'fee_close_cost': self.fee_close_cost,
            'fee_close_currency': self.fee_close_currency,
            'open_date_hum': arrow.get(self.open_date).humanize(),
            'open_date': self.open_date.strftime("%Y-%m-%d %H:%M:%S"),
            'open_timestamp': int(self.open_date.timestamp() * 1000),
            'close_date_hum': (arrow.get(self.close_date).humanize()
                               if self.close_date else None),
            'close_date': (self.close_date.strftime("%Y-%m-%d %H:%M:%S")
                           if self.close_date else None),
            'close_timestamp': int(self.close_date.timestamp() * 1000) if self.close_date else None,
            'open_rate': self.open_rate,
            'open_rate_requested': self.open_rate_requested,
            'open_trade_price': self.open_trade_price,
            'close_rate': self.close_rate,
            'close_rate_requested': self.close_rate_requested,
            'amount': round(self.amount, 8),
            'stake_amount': round(self.stake_amount, 8),
            'close_profit': self.close_profit,
            'sell_reason': self.sell_reason,
            'sell_order_status': self.sell_order_status,
            'stop_loss': self.stop_loss,
            'stop_loss_pct': (self.stop_loss_pct * 100) if self.stop_loss_pct else None,
            'initial_stop_loss': self.initial_stop_loss,
            'initial_stop_loss_pct': (self.initial_stop_loss_pct * 100
                                      if self.initial_stop_loss_pct else None),
            'min_rate': self.min_rate,
            'max_rate': self.max_rate,
            'strategy': self.strategy,
            'ticker_interval': self.ticker_interval,
            'open_order_id': self.open_order_id,
        }

    def adjust_min_max_rates(self, current_price: float) -> None:
        """
        Adjust the max_rate and min_rate.
        """
        self.max_rate = max(current_price, self.max_rate or self.open_rate)
        self.min_rate = min(current_price, self.min_rate or self.open_rate)

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
            self.stop_loss = new_loss
            self.stop_loss_pct = -1 * abs(stoploss)
            self.initial_stop_loss = new_loss
            self.initial_stop_loss_pct = -1 * abs(stoploss)
            self.stoploss_last_update = datetime.utcnow()

        # evaluate if the stop loss needs to be updated
        else:
            if new_loss > self.stop_loss:  # stop losses only walk up, never down!
                logger.debug(f"{self.pair} - Adjusting stoploss...")
                self.stop_loss = new_loss
                self.stop_loss_pct = -1 * abs(stoploss)
                self.stoploss_last_update = datetime.utcnow()
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
        :param order: order retrieved by exchange.get_order()
        :return: None
        """
        order_type = order['type']
        # Ignore open and cancelled orders
        if order['status'] == 'open' or order['price'] is None:
            return

        logger.info('Updating trade (id=%s) ...', self.id)

        if order_type in ('market', 'limit') and order['side'] == 'buy':
            # Update open rate and actual amount
            self.open_rate = Decimal(order['price'])
            self.amount = Decimal(order.get('filled', order['amount']))
            self.recalc_open_trade_price()
            logger.info('%s_BUY has been fulfilled for %s.', order_type.upper(), self)
            self.open_order_id = None
        elif order_type in ('market', 'limit') and order['side'] == 'sell':
            self.close(order['price'])
            logger.info('%s_SELL has been fulfilled for %s.', order_type.upper(), self)
        elif order_type in ('stop_loss_limit', 'stop-loss'):
            self.stoploss_order_id = None
            self.close_rate_requested = self.stop_loss
            logger.info('%s is hit for %s.', order_type.upper(), self)
            self.close(order['average'])
        else:
            raise ValueError(f'Unknown order type: {order_type}')
        cleanup()

    def close(self, rate: float) -> None:
        """
        Sets close_rate to the given rate, calculates total profit
        and marks trade as closed
        """
        self.close_rate = Decimal(rate)
        self.close_profit = self.calc_profit_ratio()
        self.close_profit_abs = self.calc_profit()
        self.close_date = datetime.utcnow()
        self.is_open = False
        self.sell_order_status = 'closed'
        self.open_order_id = None
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

    def _calc_open_trade_price(self) -> float:
        """
        Calculate the open_rate including open_fee.
        :return: Price in of the open trade incl. Fees
        """
        buy_trade = Decimal(self.amount) * Decimal(self.open_rate)
        fees = buy_trade * Decimal(self.fee_open)
        return float(buy_trade + fees)

    def recalc_open_trade_price(self) -> None:
        """
        Recalculate open_trade_price.
        Must be called whenever open_rate or fee_open is changed.
        """
        self.open_trade_price = self._calc_open_trade_price()

    def calc_close_trade_price(self, rate: Optional[float] = None,
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
        close_trade_price = self.calc_close_trade_price(
            rate=(rate or self.close_rate),
            fee=(fee or self.fee_close)
        )
        profit = close_trade_price - self.open_trade_price
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
        close_trade_price = self.calc_close_trade_price(
            rate=(rate or self.close_rate),
            fee=(fee or self.fee_close)
        )
        profit_ratio = (close_trade_price / self.open_trade_price) - 1
        return float(f"{profit_ratio:.8f}")

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
