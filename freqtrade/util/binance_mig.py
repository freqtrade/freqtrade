import logging

from freqtrade.constants import Config
from freqtrade.enums.tradingmode import TradingMode
from freqtrade.persistence.pairlock import PairLock
from freqtrade.persistence.trade_model import Trade


logger = logging.getLogger(__name__)


def migrate_binance_futures_names(config: Config):

    if (
        not (config['trading_mode'] == TradingMode.FUTURES
             and config['exchange']['name'] == 'binance')
    ):
        # only act on new futures
        return
    _migrate_binance_futures_db(config)


def _migrate_binance_futures_db(config: Config):
    logger.warning('Migrating binance futures pairs')
    trades = Trade.get_trades([Trade.exchange == 'binance', Trade.trading_mode == 'FUTURES']).all()
    for trade in trades:
        if ':' in trade.pair:
            # already migrated
            continue
        new_pair = f"{trade.pair}:{trade.stake_currency}"
        trade.pair = new_pair

        for order in trade.orders:
            order.ft_pair = new_pair
            # Should symbol be migrated too?
            # order.symbol = new_pair
    Trade.commit()
    pls = PairLock.query.filter(PairLock.pair.notlike('%:%'))
    for pl in pls:
        pl.pair = f"{pl.pair}:{config['stake_currency']}"
    # print(pls)
    # pls.update({'pair': concat(PairLock.pair,':USDT')})
    Trade.commit()
    logger.warning('Done migrating binance futures pairs')
