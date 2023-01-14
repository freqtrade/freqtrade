

import shutil
from pathlib import Path

from freqtrade.util.binance_mig import migrate_binance_futures_data


def test_binance_mig_data_conversion(default_conf_usdt, tmpdir, testdatadir):

    # call doing nothing (spot mode)
    migrate_binance_futures_data(default_conf_usdt)
    default_conf_usdt['trading_mode'] = 'futures'
    pair_old = 'XRP_USDT'
    pair_unified = 'XRP_USDT_USDT'
    futures_src = testdatadir / 'futures'
    futures_dst = tmpdir / 'futures'
    futures_dst.mkdir()
    files = [
        '-1h-mark.json',
        '-1h-futures.json',
        '-8h-funding_rate.json',
        '-8h-mark.json',
    ]

    # Copy files to tmpdir and rename to old naming
    for file in files:
        fn_after = futures_dst / f'{pair_old}{file}'
        shutil.copy(futures_src / f'{pair_unified}{file}', fn_after)

    default_conf_usdt['datadir'] = Path(tmpdir)
    # Migrate files to unified namings
    migrate_binance_futures_data(default_conf_usdt)

    for file in files:
        fn_after = futures_dst / f'{pair_unified}{file}'
        assert fn_after.exists()
