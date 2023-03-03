"""
This module contains the class to persist trades into SQLite
"""
import logging
from typing import Any, Dict

from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import NoSuchModuleError
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import StaticPool

from freqtrade.exceptions import OperationalException
from freqtrade.persistence.base import ModelBase
from freqtrade.persistence.migrations import check_migrate
from freqtrade.persistence.pairlock import PairLock
from freqtrade.persistence.trade_model import Order, Trade


logger = logging.getLogger(__name__)


_SQL_DOCS_URL = 'http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls'


def init_db(db_url: str) -> None:
    """
    Initializes this module with the given config,
    registers all known command handlers
    and starts polling for message updates
    :param db_url: Database to use
    :return: None
    """
    kwargs: Dict[str, Any] = {}

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
    Trade._session = scoped_session(sessionmaker(bind=engine, autoflush=False))
    Order._session = Trade._session
    PairLock._session = Trade._session
    Trade.query = Trade._session.query_property()
    Order.query = Trade._session.query_property()
    PairLock.query = Trade._session.query_property()

    previous_tables = inspect(engine).get_table_names()
    ModelBase.metadata.create_all(engine)
    check_migrate(engine, decl_base=ModelBase, previous_tables=previous_tables)
