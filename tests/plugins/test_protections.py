from freqtrade.strategy.interface import SellType
from unittest.mock import MagicMock, PropertyMock
import random
import pytest
from datetime import datetime, timedelta

from freqtrade.constants import AVAILABLE_PROTECTIONS
from freqtrade.persistence import Trade
from tests.conftest import get_patched_freqtradebot, log_has_re


def generate_mock_trade(pair: str, fee: float, is_open: bool,
                        sell_reason: str = SellType.SELL_SIGNAL,
                        min_ago_open: int = None, min_ago_close: int = None,
                        ):
    open_rate = random.random()

    trade = Trade(
        pair=pair,
        stake_amount=0.01,
        fee_open=fee,
        fee_close=fee,
        open_date=datetime.utcnow() - timedelta(minutes=min_ago_open or 200),
        close_date=datetime.utcnow() - timedelta(minutes=min_ago_close or 30),
        open_rate=open_rate,
        is_open=is_open,
        amount=0.01 / open_rate,
        exchange='bittrex',
    )
    trade.recalc_open_trade_price()
    if not is_open:
        trade.close(open_rate * (1 - 0.9))
        trade.sell_reason = sell_reason
    return trade


@pytest.mark.usefixtures("init_persistence")
def test_stoploss_guard(mocker, default_conf, fee, caplog):
    default_conf['protections'] = [{
        "method": "StoplossGuard",
        "lookback_period": 60,
        "trade_limit": 2
    }]
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    message = r"Trading stopped due to .*"
    assert not freqtrade.protections.global_stop()
    assert not log_has_re(message, caplog)
    caplog.clear()

    Trade.session.add(generate_mock_trade(
        'XRP/BTC', fee.return_value, False, sell_reason=SellType.STOP_LOSS.value,
        min_ago_open=200, min_ago_close=30,
        ))

    assert not freqtrade.protections.global_stop()
    assert not log_has_re(message, caplog)
    caplog.clear()
    # This trade does not count, as it's closed too long ago
    Trade.session.add(generate_mock_trade(
        'BCH/BTC', fee.return_value, False, sell_reason=SellType.STOP_LOSS.value,
        min_ago_open=250, min_ago_close=100,
    ))

    Trade.session.add(generate_mock_trade(
        'ETH/BTC', fee.return_value, False, sell_reason=SellType.STOP_LOSS.value,
        min_ago_open=240, min_ago_close=30,
    ))
    # 3 Trades closed - but the 2nd has been closed too long ago.
    assert not freqtrade.protections.global_stop()
    assert not log_has_re(message, caplog)
    caplog.clear()

    Trade.session.add(generate_mock_trade(
        'LTC/BTC', fee.return_value, False, sell_reason=SellType.STOP_LOSS.value,
        min_ago_open=180, min_ago_close=30,
    ))

    assert freqtrade.protections.global_stop()
    assert log_has_re(message, caplog)
