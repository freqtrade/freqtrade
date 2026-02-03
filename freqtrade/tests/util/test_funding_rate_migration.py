from shutil import copytree

from freqtrade.util.migrations import migrate_funding_fee_timeframe


def test_migrate_funding_rate_timeframe(default_conf_usdt, tmp_path, testdatadir):
    copytree(testdatadir / "futures", tmp_path / "futures")
    file_30m = tmp_path / "futures" / "XRP_USDT_USDT-30m-funding_rate.feather"
    file_1h_fr = tmp_path / "futures" / "XRP_USDT_USDT-1h-funding_rate.feather"
    file_1h = tmp_path / "futures" / "XRP_USDT_USDT-1h-futures.feather"
    file_1h_fr.rename(file_30m)
    assert file_1h.exists()
    assert file_30m.exists()
    assert not file_1h_fr.exists()

    default_conf_usdt["datadir"] = tmp_path

    # Inactive on spot trading ...
    migrate_funding_fee_timeframe(default_conf_usdt, None)

    default_conf_usdt["trading_mode"] = "futures"

    migrate_funding_fee_timeframe(default_conf_usdt, None)

    assert not file_30m.exists()
    assert file_1h_fr.exists()
    # futures files is untouched.
    assert file_1h.exists()
