import pytest

from freqtrade.persistence import FtNoDBContext, PairLocks, Trade


@pytest.mark.parametrize('timeframe', ['', '5m', '1d'])
def test_FtNoDBContext(timeframe):
    PairLocks.timeframe = ''
    assert Trade.use_db is True
    assert PairLocks.use_db is True
    assert PairLocks.timeframe == ''

    with FtNoDBContext(timeframe):
        assert Trade.use_db is False
        assert PairLocks.use_db is False
        assert PairLocks.timeframe == timeframe

    with FtNoDBContext():
        assert Trade.use_db is False
        assert PairLocks.use_db is False
        assert PairLocks.timeframe == ''

    assert Trade.use_db is True
    assert PairLocks.use_db is True
