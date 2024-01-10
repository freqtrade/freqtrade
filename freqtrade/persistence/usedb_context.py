
from freqtrade.persistence.pairlock_middleware import PairLocks
from freqtrade.persistence.trade_model import Trade


def disable_database_use(timeframe: str) -> None:
    """
    Disable database usage for PairLocks and Trade models.
    Used for backtesting, and some other utility commands.
    """
    PairLocks.use_db = False
    PairLocks.timeframe = timeframe
    Trade.use_db = False


def enable_database_use() -> None:
    """
    Cleanup function to restore database usage.
    """
    PairLocks.use_db = True
    PairLocks.timeframe = ''
    Trade.use_db = True


class FtNoDBContext:
    def __init__(self, timeframe: str = ''):
        self.timeframe = timeframe

    def __enter__(self):
        disable_database_use(self.timeframe)

    def __exit__(self, exc_type, exc_val, exc_tb):
        enable_database_use()
