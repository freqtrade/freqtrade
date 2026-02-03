from freqtrade.exchange import Exchange
from freqtrade.util.migrations.funding_rate_mig import migrate_funding_fee_timeframe


def migrate_data(config, exchange: Exchange | None = None) -> None:
    """
    Migrate persisted data from old formats to new formats
    """

    migrate_funding_fee_timeframe(config, exchange)


def migrate_live_content(config, exchange: Exchange | None = None) -> None:
    """
    Migrate database content from old formats to new formats
    Used for dry/live mode.
    """
    # Currently not used
    pass
