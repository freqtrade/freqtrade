import logging
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Optional, Dict

import arrow
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.orm.session import sessionmaker

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_CONF = {}
Base = declarative_base()


def init(config: dict, db_url: Optional[str] = None) -> None:
    """
    Initializes this module with the given config,
    registers all known command handlers
    and starts polling for message updates
    :param config: config to use
    :param db_url: database connector string for sqlalchemy (Optional)
    :return: None
    """
    _CONF.update(config)
    if not db_url:
        if _CONF.get('dry_run', False):
            db_url = 'sqlite:///tradesv3.dry_run.sqlite'
        else:
            db_url = 'sqlite:///tradesv3.sqlite'

    engine = create_engine(db_url, echo=False)
    session = scoped_session(sessionmaker(bind=engine, autoflush=True, autocommit=True))
    Trade.session = session()
    Trade.query = session.query_property()
    Base.metadata.create_all(engine)


def cleanup() -> None:
    """
    Flushes all pending operations to disk.
    :return: None
    """
    Trade.session.flush()


class Trade(Base):
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
        return 'Trade(id={}, pair={}, amount={}, open_rate={}, open_since={})'.format(
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
        if not order['closed']:
            return

        logger.debug('Updating trade (id=%d) ...', self.id)
        if order['type'] == 'LIMIT_BUY':
            # Update open rate and actual amount
            self.open_rate = order['rate']
            self.amount = order['amount']
        elif order['type'] == 'LIMIT_SELL':
            # Set close rate and set actual profit
            self.close_rate = order['rate']
            self.close_profit = self.calc_profit()
        else:
            raise ValueError('Unknown order type: {}'.format(order['type']))

        self.open_order_id = None

    def calc_profit(self, rate: Optional[float] = None) -> float:
        """
        Calculates the profit in percentage (including fee).
        :param rate: rate to compare with (optional).
        If rate is not set self.close_rate will be used
        :return: profit in percentage as float
        """
        getcontext().prec = 8
        return float((Decimal(rate or self.close_rate) - Decimal(self.open_rate))
                     / Decimal(self.open_rate) - Decimal(self.fee))
