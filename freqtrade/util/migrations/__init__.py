from freqtrade.util.migrations.binance_mig import migrate_binance_futures_names  # noqa F401
from freqtrade.util.migrations.binance_mig import migrate_binance_futures_data


def migrate_data(config):
    migrate_binance_futures_data(config)
