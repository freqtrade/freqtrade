from freqtrade.constants import Config
from freqtrade.exchange import Exchange
from freqtrade.util.migrations.funding_rate_mig import migrate_funding_fee_timeframe
from freqtrade.util.migrations.migrate_wallet_history import migrate_wallet_history


def migrate_data(config: Config, exchange: Exchange | None = None) -> None:
    """
    Migrate persisted data from old formats to new formats
    """

    migrate_funding_fee_timeframe(config, exchange)


def migrate_live_content(config: Config, exchange: Exchange, starting_balance: float) -> None:
    """
    Migrate database content from old formats to new formats
    Used for dry/live mode.
    """
    migrate_wallet_history(config, exchange, starting_balance)
