import logging
from typing import Any, Dict

from freqtrade.configuration.config_setup import setup_utils_configuration
from freqtrade.enums.runmode import RunMode


logger = logging.getLogger(__name__)


def start_db_convert(args: Dict[str, Any]) -> None:
    from sqlalchemy.orm import make_transient

    from freqtrade.persistence import Trade, init_db
    from freqtrade.persistence.pairlock import PairLock

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    init_db(config['db_url'], False)
    session_target = Trade._session
    init_db(config['db_url_from'], False)

    # print(f"{id(sessionA)=}, {id(sessionB)=}")
    trade_count = 0
    pairlock_count = 0
    for trade in Trade.get_trades():
        trade_count += 1
        make_transient(trade)
        for o in trade.orders:
            make_transient(o)

        session_target.add(trade)
    session_target.commit()

    for pairlock in PairLock.query:
        pairlock_count += 1
        make_transient(pairlock)
        session_target.add(pairlock)
    session_target.commit()
    logger.info(f"Migrated {trade_count} Trades, and {pairlock_count} Pairlocks.")
