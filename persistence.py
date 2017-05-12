from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.types import Enum

from exchange import Exchange


def create_session(base, filename):
    """
    Creates sqlite database and setup tables.
    :return: sqlalchemy Session
    """
    engine = create_engine(filename, echo=False)
    base.metadata.create_all(engine)
    return scoped_session(sessionmaker(bind=engine, autoflush=True, autocommit=True))


Base = declarative_base()
Session = create_session(Base, filename='sqlite:///tradesv2.sqlite')


class Trade(Base):
    __tablename__ = 'trades'

    query = Session.query_property()

    id = Column(Integer, primary_key=True)
    exchange = Column(Enum(Exchange), nullable=False)
    pair = Column(String, nullable=False)
    is_open = Column(Boolean, nullable=False, default=True)
    open_rate = Column(Float, nullable=False)
    close_rate = Column(Float)
    close_profit = Column(Float)
    btc_amount = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    open_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    close_date = Column(DateTime)

    def __repr__(self):
        return 'Trade(id={}, pair={}, amount={}, open_rate={}, open_since={})'.format(
            self.id,
            self.pair,
            self.amount,
            self.open_rate,
            'closed' if not self.is_open else round((datetime.utcnow() - self.open_date).total_seconds() / 60, 2)
        )
