from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.types import Enum

from freqtrade import exchange

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
            db_url = 'sqlite:///tradesv2.dry_run.sqlite'
        else:
            db_url = 'sqlite:///tradesv2.sqlite'

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
    open_rate = Column(Float, nullable=False)
    close_rate = Column(Float)
    close_profit = Column(Float)
    stake_amount = Column(Float, name='btc_amount', nullable=False)
    amount = Column(Float, nullable=False)
    open_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    close_date = Column(DateTime)
    open_order_id = Column(String)

    def __repr__(self):
        if self.is_open:
            open_since = 'closed'
        else:
            open_since = round((datetime.utcnow() - self.open_date).total_seconds() / 60, 2)
        return 'Trade(id={}, pair={}, amount={}, open_rate={}, open_since={})'.format(
            self.id,
            self.pair,
            self.amount,
            self.open_rate,
            open_since
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

        # Flush changes
        Trade.session.flush()
        return profit
