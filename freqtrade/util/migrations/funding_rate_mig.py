import logging
from typing import Optional

from freqtrade.constants import Config
from freqtrade.data.history.idatahandler import get_datahandler
from freqtrade.enums import TradingMode
from freqtrade.exchange import Exchange


logger = logging.getLogger(__name__)


def migrate_funding_fee_timeframe(config: Config, exchange: Optional[Exchange]):
    if (
        config.get('trading_mode', TradingMode.SPOT) != TradingMode.FUTURES
    ):
        # only act on futures
        return

    if not exchange:
        from freqtrade.resolvers import ExchangeResolver
        exchange = ExchangeResolver.load_exchange(config, validate=False)

    ff_timeframe = exchange.get_option('funding_fee_timeframe')

    dhc = get_datahandler(config['datadir'], config['dataformat_ohlcv'])
    dhc.fix_funding_fee_timeframe(ff_timeframe)
