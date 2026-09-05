from unittest.mock import MagicMock

import pytest

from freqtrade.enums import RPCMessageType, TradingMode
from freqtrade.exceptions import PricingError, TemporaryError
from freqtrade.persistence import Trade
from freqtrade.util import dt_now
from tests.conftest import EXMS, get_patched_freqtradebot, patch_get_signal


def _make_trade(
    trade_id: int, pair: str, liquidation_price: float, is_short: bool = False, open_rate=1.0
):
    trade = Trade(
        id=trade_id,
        pair=pair,
        stake_amount=10.0,
        amount=10.0,
        amount_requested=10.0,
        open_rate=open_rate,
        fee_open=0.0,
        fee_close=0.0,
        exchange="binance",
        is_short=is_short,
        leverage=5.0,
        trading_mode=TradingMode.FUTURES,
        open_date=dt_now(),
        is_open=True,
    )
    trade.liquidation_price = liquidation_price
    return trade


def _get_bot(mocker, conf, margin_mode: str, rate: float, trades: list[Trade], warn_ratio=0.2):
    """Helper to get a mocked bot as all tests here need a very similar same setup."""
    conf["trading_mode"] = "futures"
    conf["margin_mode"] = margin_mode
    conf["liquidation_warn_ratio"] = warn_ratio
    freqtrade = get_patched_freqtradebot(mocker, conf)
    patch_get_signal(freqtrade)
    mocker.patch(f"{EXMS}.get_rate", return_value=rate)
    mocker.patch("freqtrade.persistence.Trade.get_open_trades", return_value=trades)
    freqtrade.rpc.send_msg = MagicMock()
    return freqtrade


@pytest.mark.parametrize("is_short", [False, True])
def test_liquidation_warning_isolated(mocker, default_conf_usdt, is_short) -> None:
    # Open rate 1.0, liquidation stop 0.9 (1.1 for shorts) - at 0.919 (1.081) only 19% of that
    # distance is left. The second trade's stop is far away, so 100% is left there.
    trades = [
        _make_trade(1, "ETH/USDT:USDT", 1.1 if is_short else 0.9, is_short),
        _make_trade(2, "XRP/USDT:USDT", 1.5 if is_short else 0.5, is_short),
    ]
    rate = 1.081 if is_short else 0.919
    freqtrade = _get_bot(mocker, default_conf_usdt, "isolated", rate, trades, warn_ratio=0.2)

    freqtrade.check_liquidation_warnings()

    assert freqtrade.rpc.send_msg.call_count == 1
    msg = freqtrade.rpc.send_msg.call_args[0][0]
    assert msg["type"] == RPCMessageType.LIQUIDATION_WARNING
    assert msg["pair"] == "ETH/USDT:USDT"
    assert msg["trade_id"] == 1
    assert msg["margin_mode"] == "isolated"
    assert msg["direction"] == ("Short" if is_short else "Long")
    assert msg["liquidation_price"] == (1.1 if is_short else 0.9)
    assert msg["current_rate"] == rate
    assert msg["warn_ratio"] == 0.2
    assert pytest.approx(msg["remaining_ratio"]) == 0.19
    # Isolated warns per position
    assert msg["positions_at_risk"] == 1
    assert msg["open_positions"] == 2

    freqtrade.rpc.send_msg.reset_mock()
    # Repeated calls don't repeat the warning
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 0


def test_liquidation_warning_cross(mocker, default_conf_usdt) -> None:
    # Both within the warning ratio - cross margin aggregates into one message.
    trades = [
        _make_trade(1, "ETH/USDT:USDT", 0.9),
        _make_trade(2, "XRP/USDT:USDT", 0.91),
    ]
    freqtrade = _get_bot(mocker, default_conf_usdt, "cross", 0.919, trades, warn_ratio=0.2)

    freqtrade.check_liquidation_warnings()

    assert freqtrade.rpc.send_msg.call_count == 1
    msg = freqtrade.rpc.send_msg.call_args[0][0]
    assert msg["margin_mode"] == "cross"
    # Reports the position closest to its liquidation stop
    assert msg["pair"] == "XRP/USDT:USDT"
    assert msg["positions_at_risk"] == 2
    assert msg["open_positions"] == 2

    freqtrade.rpc.send_msg.reset_mock()
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 0


def test_liquidation_warning_escalates_and_resets(mocker, default_conf_usdt) -> None:
    # Open rate 1.0, liquidation stop 0.9 - so 0.1 of price movement is the full distance.
    trade = _make_trade(1, "ETH/USDT:USDT", 0.9)
    freqtrade = _get_bot(mocker, default_conf_usdt, "isolated", 0.919, [trade], warn_ratio=0.2)

    # 19% left - first warning
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 1

    freqtrade.rpc.send_msg.reset_mock()
    # 9% left - more than halved, warn again
    mocker.patch(f"{EXMS}.get_rate", return_value=0.909)
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 1

    # 8% left - closer, but not halved - no new message
    freqtrade.rpc.send_msg.reset_mock()
    mocker.patch(f"{EXMS}.get_rate", return_value=0.908)
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 0

    # 23% left - inside the recovery area, state is kept
    freqtrade.rpc.send_msg.reset_mock()
    mocker.patch(f"{EXMS}.get_rate", return_value=0.923)
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 0

    # 50% left - recovered beyond the recovery area, state is reset
    freqtrade.rpc.send_msg.reset_mock()
    mocker.patch(f"{EXMS}.get_rate", return_value=0.95)
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 0
    assert freqtrade._liq_warn_cache == {}

    # Approaching again warns immediately
    freqtrade.rpc.send_msg.reset_mock()
    mocker.patch(f"{EXMS}.get_rate", return_value=0.919)
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 1


def test_liquidation_warning_leverage_independent(mocker, default_conf_usdt) -> None:
    """Warns at any leverage - a plain price distance would not."""
    # 2x: liquidation at 50, 5x: at 80, 20x: at 95 - each 90% of the way to its stop.
    for liq_price, rate in [(50.0, 55.0), (80.0, 82.0), (95.0, 95.5)]:
        trade = _make_trade(1, "ETH/USDT:USDT", liq_price, open_rate=100.0)
        freqtrade = _get_bot(mocker, default_conf_usdt, "isolated", rate, [trade], warn_ratio=0.2)
        freqtrade.check_liquidation_warnings()
        assert freqtrade.rpc.send_msg.call_count == 1
        assert pytest.approx(freqtrade.rpc.send_msg.call_args[0][0]["remaining_ratio"]) == 0.1

    # And a freshly opened position never warns, no matter the leverage
    for liq_price in (50.0, 80.0, 95.0, 98.0):
        trade = _make_trade(1, "ETH/USDT:USDT", liq_price, open_rate=100.0)
        freqtrade = _get_bot(mocker, default_conf_usdt, "isolated", 100.0, [trade], warn_ratio=0.2)
        freqtrade.check_liquidation_warnings()
        assert freqtrade.rpc.send_msg.call_count == 0


@pytest.mark.parametrize(
    "warn_ratio,trading_mode,liq_price,amount,expected",
    [
        (0.2, "futures", 0.9, 10.0, 1),
        # Disabled
        (0.0, "futures", 0.9, 10.0, 0),
        # Spot has no liquidation
        (0.2, "spot", 0.9, 10.0, 0),
        # Too far away - 84% of the distance is left
        (0.2, "futures", 0.5, 10.0, 0),
        # No liquidation price known
        (0.2, "futures", None, 10.0, 0),
        # Entry order not filled yet
        (0.2, "futures", 0.9, 0.0, 0),
        # Liquidation price equals the open rate - nothing sensible to measure
        (0.2, "futures", 1.0, 10.0, 0),
    ],
)
def test_liquidation_warning_not_sent(
    mocker, default_conf_usdt, warn_ratio, trading_mode, liq_price, amount, expected
) -> None:
    trade = _make_trade(1, "ETH/USDT:USDT", liq_price)
    trade.amount = amount
    freqtrade = _get_bot(mocker, default_conf_usdt, "isolated", 0.919, [trade], warn_ratio)
    freqtrade.trading_mode = TradingMode(trading_mode)

    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == expected


@pytest.mark.parametrize("exception", [PricingError, TemporaryError])
def test_liquidation_warning_pricing_error(mocker, default_conf_usdt, exception) -> None:
    trade = _make_trade(1, "ETH/USDT:USDT", 0.9)
    freqtrade = _get_bot(mocker, default_conf_usdt, "isolated", 0.919, [trade], warn_ratio=0.2)
    mocker.patch(f"{EXMS}.get_rate", side_effect=exception())

    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 0
