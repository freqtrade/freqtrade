from shutil import copytree

from freqtrade.util.migrations import migrate_funding_fee_timeframe


def test_migrate_funding_rate_timeframe(default_conf_usdt, tmp_path, testdatadir):

    copytree(testdatadir / 'futures', tmp_path / 'futures')
    file_4h = tmp_path / 'futures' / 'XRP_USDT_USDT-4h-funding_rate.feather'
    file_8h = tmp_path / 'futures' / 'XRP_USDT_USDT-8h-funding_rate.feather'
    file_1h = tmp_path / 'futures' / 'XRP_USDT_USDT-1h-futures.feather'
    file_8h.rename(file_4h)
    assert file_1h.exists()
    assert file_4h.exists()
    assert not file_8h.exists()

    default_conf_usdt['datadir'] = tmp_path

    # Inactive on spot trading ...
    migrate_funding_fee_timeframe(default_conf_usdt, None)

    default_conf_usdt['trading_mode'] = 'futures'

    migrate_funding_fee_timeframe(default_conf_usdt, None)

    assert not file_4h.exists()
    assert file_8h.exists()
    # futures files is untouched.
    assert file_1h.exists()
