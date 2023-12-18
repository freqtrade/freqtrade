from datetime import datetime, timezone
from shutil import copytree
from unittest.mock import PropertyMock

import pytest

from freqtrade.data.converter.trade_converter_kraken import import_kraken_trades_from_csv
from freqtrade.data.history.idatahandler import get_datahandler
from freqtrade.exceptions import OperationalException
from tests.conftest import EXMS, log_has, log_has_re, patch_exchange


def test_import_kraken_trades_from_csv(testdatadir, tmp_path, caplog, default_conf_usdt, mocker):
    with pytest.raises(OperationalException, match="This function is only for the kraken exchange"):
        import_kraken_trades_from_csv(default_conf_usdt, 'feather')

    default_conf_usdt['exchange']['name'] = 'kraken'

    patch_exchange(mocker, id='kraken')
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value={
        'BCH/EUR': {'symbol': 'BCH/EUR', 'id': 'BCHEUR', 'altname': 'BCHEUR'},
    }))
    dstfile = tmp_path / 'BCH_EUR-trades.feather'
    assert not dstfile.is_file()
    default_conf_usdt['datadir'] = tmp_path
    # There's 2 files in this tree, containing a total of 2 days.
    # tests/testdata/kraken/
    # └── trades_csv
    # ├── BCHEUR.csv       <-- 2023-01-01
    # └── incremental_q2
    #     └── BCHEUR.csv   <-- 2023-01-02

    copytree(testdatadir / 'kraken/trades_csv', tmp_path / 'trades_csv')

    import_kraken_trades_from_csv(default_conf_usdt, 'feather')
    assert log_has("Found csv files for BCHEUR.", caplog)
    assert log_has_re(r"BCH/EUR: 340 trades.* 2023-01-01.* 2023-01-02.*", caplog)

    assert dstfile.is_file()

    dh = get_datahandler(tmp_path, 'feather')
    trades = dh.trades_load('BCH_EUR')
    assert len(trades) == 340

    assert trades['date'].min().to_pydatetime() == datetime(2023, 1, 1, 0, 3, 56,
                                                            tzinfo=timezone.utc)
    assert trades['date'].max().to_pydatetime() == datetime(2023, 1, 2, 23, 17, 3,
                                                            tzinfo=timezone.utc)
    # ID is not filled
    assert len(trades.loc[trades['id'] != '']) == 0
