from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.types import Enum

import exchange


_db_handle = None
_session = None
_conf = {}


Base = declarative_base()


def init(config: dict) -> None:
    """
    Initializes this module with the given config,
    registers all known command handlers
    and starts polling for message updates
    :param config: config to use
    :return: None
    """
    global _db_handle, _session
    _conf.update(config)
    if _conf.get('dry_run', False):
        _db_handle = 'sqlite:///tradesv2.dry_run.sqlite'
    else:
        _db_handle = 'sqlite:///tradesv2.sqlite'

    engine = create_engine(_db_handle, echo=False)
    _session = scoped_session(sessionmaker(bind=engine, autoflush=True, autocommit=True))
    Trade.session = _session
    Trade.query = _session.query_property()
    Base.metadata.create_all(engine)


def get_session():
    return _session


class Trade(Base):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    exchange = Column(Enum(exchange.Exchange), nullable=False)
    pair = Column(String, nullable=False)
    is_open = Column(Boolean, nullable=False, default=True)
    open_rate = Column(Float, nullable=False)
    close_rate = Column(Float)
    close_profit = Column(Float)
    btc_amount = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    open_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    close_date = Column(DateTime)
    open_order_id = Column(String)

    def __repr__(self):
        return 'Trade(id={}, pair={}, amount={}, open_rate={}, open_since={})'.format(
            self.id,
            self.pair,
            self.amount,
            self.open_rate,
            'closed' if not self.is_open else round((datetime.utcnow() - self.open_date).total_seconds() / 60, 2)
        )

    def exec_sell_order(self, rate: float, amount: float) -> float:
        """
        Executes a sell for the given trade and updated the entity.
        :param rate: rate to sell for
        :param amount: amount to sell
        :return: current profit as percentage
        """
        profit = 100 * ((rate - self.open_rate) / self.open_rate)

        # Execute sell and update trade record
        order_id = exchange.sell(str(self.pair), rate, amount)
        self.close_rate = rate
        self.close_profit = profit
        self.close_date = datetime.utcnow()
        self.open_order_id = order_id
        return profit

