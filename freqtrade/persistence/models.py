"""
This module contains the class to persist trades into SQLite
"""
import logging

from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import NoSuchModuleError
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import StaticPool

from freqtrade.exceptions import OperationalException
from freqtrade.persistence.base import _DECL_BASE
from freqtrade.persistence.migrations import check_migrate
from freqtrade.persistence.pairlock import PairLock
from freqtrade.persistence.trade_model import Order, Trade


logger = logging.getLogger(__name__)


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

    if db_url == 'sqlite:///':
        raise OperationalException(
            f'Bad db-url {db_url}. For in-memory database, please use `sqlite://`.')
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
