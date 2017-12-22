import logging
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Optional, Dict

import arrow
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

_CONF = {}
_DECL_BASE = declarative_base()


def init(config: dict, engine: Optional[Engine] = None) -> None:
    """
    Initializes this module with the given config,
    registers all known command handlers
    and starts polling for message updates
    :param config: config to use
    :param engine: database engine for sqlalchemy (Optional)
    :return: None
    """
    _CONF.update(config)
    if not engine:
        if _CONF.get('dry_run', False):
            # the user wants dry run to use a DB
            if _CONF.get('dry_run_db', False):
                engine = create_engine('sqlite:///tradesv3.dry_run.sqlite')
            # Otherwise dry run will store in memory
            else:
                engine = create_engine('sqlite://',
                                       connect_args={'check_same_thread': False},
                                       poolclass=StaticPool,
                                       echo=False)
        else:
            engine = create_engine('sqlite:///tradesv3.sqlite')

    session = scoped_session(sessionmaker(bind=engine, autoflush=True, autocommit=True))
    Trade.session = session()
    Trade.query = session.query_property()
    _DECL_BASE.metadata.create_all(engine)


def cleanup() -> None:
    """
    Flushes all pending operations to disk.
    :return: None
    """
    Trade.session.flush()


class Trade(_DECL_BASE):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    exchange = Column(String, nullable=False)
    pair = Column(String, nullable=False)
    is_open = Column(Boolean, nullable=False, default=True)
    fee = Column(Float, nullable=False, default=0.0)
    open_rate = Column(Float)
    close_rate = Column(Float)
    close_profit = Column(Float)
    stake_amount = Column(Float, nullable=False)
    amount = Column(Float)
    open_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    close_date = Column(DateTime)
    open_order_id = Column(String)

    def __repr__(self):
        return 'Trade(id={}, pair={}, amount={:.8f}, open_rate={:.8f}, open_since={})'.format(
            self.id,
            self.pair,
            self.amount,
            self.open_rate,
            arrow.get(self.open_date).humanize() if self.is_open else 'closed'
        )

    def update(self, order: Dict) -> None:
        """
        Updates this entity with amount and actual open/close rates.
        :param order: order retrieved by exchange.get_order()
        :return: None
        """
        # Ignore open and cancelled orders
        if not order['closed'] or order['rate'] is None:
            return

        logger.info('Updating trade (id=%d) ...', self.id)

        getcontext().prec = 8  # Bittrex do not go above 8 decimal
        if order['type'] == 'LIMIT_BUY':
            # Update open rate and actual amount
            self.open_rate = Decimal(order['rate'])
            self.amount = Decimal(order['amount'])
            logger.info('LIMIT_BUY has been fulfilled for %s.', self)
            self.open_order_id = None
        elif order['type'] == 'LIMIT_SELL':
            self.close(order['rate'])
        else:
            raise ValueError('Unknown order type: {}'.format(order['type']))
        Trade.session.flush()

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
        Calculate the open_rate in BTC
        :param fee: fee to use on the open rate (optional).
        If rate is not set self.fee will be used
        :return: Price in BTC of the open trade
        """
        getcontext().prec = 8

        buy_trade = (Decimal(self.amount) * Decimal(self.open_rate))
        fees = buy_trade * Decimal(fee or self.fee)
        return float(buy_trade + fees)

    def calc_close_trade_price(
            self,
            rate: Optional[float] = None,
            fee: Optional[float] = None) -> float:
        """
        Calculate the close_rate in BTC
        :param fee: fee to use on the close rate (optional).
        If rate is not set self.fee will be used
        :param rate: rate to compare with (optional).
        If rate is not set self.close_rate will be used
        :return: Price in BTC of the open trade
        """
        getcontext().prec = 8

        if rate is None and not self.close_rate:
            return 0.0

        sell_trade = (Decimal(self.amount) * Decimal(rate or self.close_rate))
        fees = sell_trade * Decimal(fee or self.fee)
        return float(sell_trade - fees)

    def calc_profit(
            self,
            rate: Optional[float] = None,
            fee: Optional[float] = None) -> float:
        """
        Calculate the profit in BTC between Close and Open trade
        :param fee: fee to use on the close rate (optional).
        If rate is not set self.fee will be used
        :param rate: close rate to compare with (optional).
        If rate is not set self.close_rate will be used
        :return:  profit in BTC as float
        """
        open_trade_price = self.calc_open_trade_price()
        close_trade_price = self.calc_close_trade_price(
            rate=Decimal(rate or self.close_rate),
            fee=Decimal(fee or self.fee)
        )
        return float("{0:.8f}".format(close_trade_price - open_trade_price))

    def calc_profit_percent(
            self,
            rate: Optional[float] = None,
            fee: Optional[float] = None) -> float:
        """
        Calculates the profit in percentage (including fee).
        :param rate: rate to compare with (optional).
        If rate is not set self.close_rate will be used
        :return: profit in percentage as float
        """
        getcontext().prec = 8

        open_trade_price = self.calc_open_trade_price()
        close_trade_price = self.calc_close_trade_price(
            rate=Decimal(rate or self.close_rate),
            fee=Decimal(fee or self.fee)
        )

        return float("{0:.8f}".format((close_trade_price / open_trade_price) - 1))
