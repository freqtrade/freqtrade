"""
This module contains the class to persist trades into SQLite
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

import arrow
from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, String,
                        create_engine, inspect)
from sqlalchemy.exc import NoSuchModuleError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.pool import StaticPool

from freqtrade import OperationalException

logger = logging.getLogger(__name__)

_DECL_BASE: Any = declarative_base()
_SQL_DOCS_URL = 'http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls'


def init(config: Dict) -> None:
    """
    Initializes this module with the given config,
    registers all known command handlers
    and starts polling for message updates
    :param config: config to use
    :return: None
    """
    db_url = config.get('db_url', None)
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
        raise OperationalException(f'Given value for db_url: \'{db_url}\' '
                                   f'is no valid database URL! (See {_SQL_DOCS_URL})')

    session = scoped_session(sessionmaker(bind=engine, autoflush=True, autocommit=True))
    Trade.session = session()
    Trade.query = session.query_property()
    _DECL_BASE.metadata.create_all(engine)
    check_migrate(engine)

    # Clean dry_run DB if the db is not in-memory
    if config.get('dry_run', False) and db_url != 'sqlite://':
        clean_dry_run_db()


def has_column(columns, searchname: str) -> bool:
    return len(list(filter(lambda x: x["name"] == searchname, columns))) == 1


def get_column_def(columns, column: str, default: str) -> str:
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
    if not has_column(cols, 'ticker_interval'):
        logger.info(f'Running database migration - backup available as {table_back_name}')

        fee_open = get_column_def(cols, 'fee_open', 'fee')
        fee_close = get_column_def(cols, 'fee_close', 'fee')
        open_rate_requested = get_column_def(cols, 'open_rate_requested', 'null')
        close_rate_requested = get_column_def(cols, 'close_rate_requested', 'null')
        stop_loss = get_column_def(cols, 'stop_loss', '0.0')
        initial_stop_loss = get_column_def(cols, 'initial_stop_loss', '0.0')
        max_rate = get_column_def(cols, 'max_rate', '0.0')
        sell_reason = get_column_def(cols, 'sell_reason', 'null')
        strategy = get_column_def(cols, 'strategy', 'null')
        ticker_interval = get_column_def(cols, 'ticker_interval', 'null')

        # Schema migration necessary
        engine.execute(f"alter table trades rename to {table_back_name}")
        # let SQLAlchemy create the schema as required
        _DECL_BASE.metadata.create_all(engine)

        # Copy data back - following the correct schema
        engine.execute(f"""insert into trades
                (id, exchange, pair, is_open, fee_open, fee_close, open_rate,
                open_rate_requested, close_rate, close_rate_requested, close_profit,
                stake_amount, amount, open_date, close_date, open_order_id,
                stop_loss, initial_stop_loss, max_rate, sell_reason, strategy,
                ticker_interval
                )
            select id, lower(exchange),
                case
                    when instr(pair, '_') != 0 then
                    substr(pair,    instr(pair, '_') + 1) || '/' ||
                    substr(pair, 1, instr(pair, '_') - 1)
                    else pair
                    end
                pair,
                is_open, {fee_open} fee_open, {fee_close} fee_close,
                open_rate, {open_rate_requested} open_rate_requested, close_rate,
                {close_rate_requested} close_rate_requested, close_profit,
                stake_amount, amount, open_date, close_date, open_order_id,
                {stop_loss} stop_loss, {initial_stop_loss} initial_stop_loss,
                {max_rate} max_rate, {sell_reason} sell_reason, {strategy} strategy,
                {ticker_interval} ticker_interval
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
    fee_close = Column(Float, nullable=False, default=0.0)
    open_rate = Column(Float)
    open_rate_requested = Column(Float)
    close_rate = Column(Float)
    close_rate_requested = Column(Float)
    close_profit = Column(Float)
    stake_amount = Column(Float, nullable=False)
    amount = Column(Float)
    open_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    close_date = Column(DateTime)
    open_order_id = Column(String)
    # absolute value of the stop loss
    stop_loss = Column(Float, nullable=True, default=0.0)
    # absolute value of the initial stop loss
    initial_stop_loss = Column(Float, nullable=True, default=0.0)
    # absolute value of the highest reached price
    stoploss_order_id = Column(Integer, nullable=True, index=True)
    max_rate = Column(Float, nullable=True, default=0.0)
    sell_reason = Column(String, nullable=True)
    strategy = Column(String, nullable=True)
    ticker_interval = Column(Integer, nullable=True)

    def __repr__(self):
        open_since = arrow.get(self.open_date).humanize() if self.is_open else 'closed'

        return (f'Trade(id={self.id}, pair={self.pair}, amount={self.amount:.8f}, '
                f'open_rate={self.open_rate:.8f}, open_since={open_since})')

    def adjust_stop_loss(self, current_price: float, stoploss: float, initial: bool = False):
        """this adjusts the stop loss to it's most recently observed setting"""

        if initial and not (self.stop_loss is None or self.stop_loss == 0):
            # Don't modify if called with initial and nothing to do
            return

        new_loss = float(current_price * (1 - abs(stoploss)))

        # keeping track of the highest observed rate for this trade
        if self.max_rate is None:
            self.max_rate = current_price
        else:
            if current_price > self.max_rate:
                self.max_rate = current_price

        # no stop loss assigned yet
        if not self.stop_loss:
            logger.debug("assigning new stop loss")
            self.stop_loss = new_loss
            self.initial_stop_loss = new_loss

        # evaluate if the stop loss needs to be updated
        else:
            if new_loss > self.stop_loss:  # stop losses only walk up, never down!
                self.stop_loss = new_loss
                logger.debug("adjusted stop loss")
            else:
                logger.debug("keeping current stop loss")

        logger.debug(
            f"{self.pair} - current price {current_price:.8f}, "
            f"bought at {self.open_rate:.8f} and calculated "
            f"stop loss is at: {self.initial_stop_loss:.8f} initial "
            f"stop at {self.stop_loss:.8f}. "
            f"trailing stop loss saved us: "
            f"{float(self.stop_loss) - float(self.initial_stop_loss):.8f} "
            f"and max observed rate was {self.max_rate:.8f}")

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

        logger.info('Updating trade (id=%d) ...', self.id)

        if order_type == 'limit' and order['side'] == 'buy':
            # Update open rate and actual amount
            self.open_rate = Decimal(order['price'])
            self.amount = Decimal(order['amount'])
            logger.info('LIMIT_BUY has been fulfilled for %s.', self)
            self.open_order_id = None
        elif order_type == 'limit' and order['side'] == 'sell':
            self.close(order['price'])
        else:
            raise ValueError(f'Unknown order type: {order_type}')
        cleanup()

    def close(self, rate: float) -> None:
        """
        Sets close_rate to the given rate, calculates total profit
        and marks trade as closed
        """
        self.close_rate = Decimal(rate)
        self.close_profit = self.calc_profit_percent()
        self.close_date = datetime.utcnow()
        self.is_open = False
        self.open_order_id = None
        logger.info(
            'Marking %s as closed as the trade is fulfilled and found no open orders for it.',
            self
        )

    def calc_open_trade_price(
            self,
            fee: Optional[float] = None) -> float:
        """
        Calculate the open_rate including fee.
        :param fee: fee to use on the open rate (optional).
        If rate is not set self.fee will be used
        :return: Price in of the open trade incl. Fees
        """

        buy_trade = (Decimal(self.amount) * Decimal(self.open_rate))
        fees = buy_trade * Decimal(fee or self.fee_open)
        return float(buy_trade + fees)

    def calc_close_trade_price(
            self,
            rate: Optional[float] = None,
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

        sell_trade = (Decimal(self.amount) * Decimal(rate or self.close_rate))
        fees = sell_trade * Decimal(fee or self.fee_close)
        return float(sell_trade - fees)

    def calc_profit(
            self,
            rate: Optional[float] = None,
            fee: Optional[float] = None) -> float:
        """
        Calculate the absolute profit in stake currency between Close and Open trade
        :param fee: fee to use on the close rate (optional).
        If rate is not set self.fee will be used
        :param rate: close rate to compare with (optional).
        If rate is not set self.close_rate will be used
        :return:  profit in stake currency as float
        """
        open_trade_price = self.calc_open_trade_price()
        close_trade_price = self.calc_close_trade_price(
            rate=(rate or self.close_rate),
            fee=(fee or self.fee_close)
        )
        profit = close_trade_price - open_trade_price
        return float(f"{profit:.8f}")

    def calc_profit_percent(
            self,
            rate: Optional[float] = None,
            fee: Optional[float] = None) -> float:
        """
        Calculates the profit in percentage (including fee).
        :param rate: rate to compare with (optional).
        If rate is not set self.close_rate will be used
        :param fee: fee to use on the close rate (optional).
        :return: profit in percentage as float
        """

        open_trade_price = self.calc_open_trade_price()
        close_trade_price = self.calc_close_trade_price(
            rate=(rate or self.close_rate),
            fee=(fee or self.fee_close)
        )
        profit_percent = (close_trade_price / open_trade_price) - 1
        return float(f"{profit_percent:.8f}")
