from unittest.mock import MagicMock

import pytest

from freqtrade.enums import RPCMessageType, TradingMode
from freqtrade.exceptions import PricingError, TemporaryError
from freqtrade.persistence import Order, Trade
from freqtrade.util import dt_now
from tests.conftest import EXMS, get_patched_freqtradebot, patch_get_signal


def _make_trade(
    trade_id: int,
    pair: str,
    liquidation_price: float,
    is_short: bool = False,
    open_rate=1.0,
    leverage=10.0,
):
    """
    Trade with a fixed liquidation stop. At the default of 10x the margin covers a 10% move,
    so an open rate of 1.0 with a stop at 0.9 (1.1 for shorts) is roughly a 10x stop.
    """
    trade = Trade(
        id=trade_id,
        pair=pair,
        stake_amount=open_rate * 10.0 / leverage,
        amount=10.0,
        amount_requested=10.0,
        open_rate=open_rate,
        fee_open=0.0,
        fee_close=0.0,
        exchange="binance",
        is_short=is_short,
        leverage=leverage,
        trading_mode=TradingMode.FUTURES,
        open_date=dt_now(),
        is_open=True,
    )
    trade.liquidation_price = liquidation_price
    return trade


def _add_open_exit_order(trade: Trade, rate: float) -> None:
    """Attach an unfilled exit order, as placed by the liquidation exit."""
    trade.orders.append(
        Order(
            ft_order_side=trade.exit_side,
            ft_pair=trade.pair,
            ft_is_open=True,
            ft_amount=trade.amount,
            ft_price=rate,
            order_id=f"exit_{trade.id}",
            status="open",
            symbol=trade.pair,
            order_type="limit",
            side=trade.exit_side,
            price=rate,
            filled=0,
            remaining=trade.amount,
            order_date=dt_now(),
        )
    )


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
    # Open rate 1.0, liquidation stop 0.9 (1.1 for shorts) - at 0.919 (1.081) the stop is 1.9%
    # away, 19% of the 10% move the margin covers. The second trade's stop is far away.
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


def test_liquidation_warning_cross_tracks_closest(mocker, default_conf_usdt) -> None:
    # Only ETH is close to its stop - XRP's stop is far away.
    trades = [
        _make_trade(1, "ETH/USDT:USDT", 0.9),
        _make_trade(2, "XRP/USDT:USDT", 0.5),
    ]
    freqtrade = _get_bot(mocker, default_conf_usdt, "cross", 0.905, trades, warn_ratio=0.2)

    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 1
    msg = freqtrade.rpc.send_msg.call_args[0][0]
    assert msg["pair"] == "ETH/USDT:USDT"
    assert msg["positions_at_risk"] == 1
    assert msg["open_positions"] == 2

    freqtrade.rpc.send_msg.reset_mock()
    # XRP enters the warning zone (15% left) - ETH is still the closest, the account was warned.
    mocker.patch(f"{EXMS}.get_rate", side_effect=[0.905, 0.515])
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 0

    # ETH is exited - XRP is now the closest position and is warned about right away.
    mocker.patch("freqtrade.persistence.Trade.get_open_trades", return_value=trades[1:])
    mocker.patch(f"{EXMS}.get_rate", return_value=0.515)
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 1
    msg = freqtrade.rpc.send_msg.call_args[0][0]
    assert msg["pair"] == "XRP/USDT:USDT"
    assert msg["positions_at_risk"] == 1
    assert msg["open_positions"] == 1

    freqtrade.rpc.send_msg.reset_mock()
    # A new position overtakes XRP (10% left) - warned about once, then quiet.
    trades.append(_make_trade(3, "ADA/USDT:USDT", 0.5))
    mocker.patch("freqtrade.persistence.Trade.get_open_trades", return_value=trades[1:])
    for _ in range(2):
        mocker.patch(f"{EXMS}.get_rate", side_effect=[0.515, 0.51])
        freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 1
    msg = freqtrade.rpc.send_msg.call_args[0][0]
    assert msg["pair"] == "ADA/USDT:USDT"
    assert msg["positions_at_risk"] == 2

    freqtrade.rpc.send_msg.reset_mock()
    # XRP is the closest again, but was warned about already - no message
    mocker.patch(f"{EXMS}.get_rate", side_effect=[0.515, 0.52])
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 0

    freqtrade.rpc.send_msg.reset_mock()
    # The account recovers - all state is reset, ...
    mocker.patch(f"{EXMS}.get_rate", return_value=0.9)
    freqtrade.check_liquidation_warnings()
    assert freqtrade._liq_warn_cache == {}
    # ... so approaching again warns immediately, about a position warned about before.
    mocker.patch(f"{EXMS}.get_rate", side_effect=[0.515, 0.9])
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 1
    assert freqtrade.rpc.send_msg.call_args[0][0]["pair"] == "XRP/USDT:USDT"


def test_liquidation_warning_cross_many_positions(mocker, default_conf_usdt) -> None:
    """Several positions approaching liquidation together result in one message, not one each."""
    trades = [_make_trade(i, f"{i}/USDT:USDT", 0.9) for i in range(1, 6)]
    freqtrade = _get_bot(mocker, default_conf_usdt, "cross", 0.919, trades, warn_ratio=0.2)
    # Approaching together - 19%, 15% and 12% left
    for rate in (0.919, 0.915, 0.912):
        mocker.patch(f"{EXMS}.get_rate", return_value=rate)
        freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 1
    assert freqtrade.rpc.send_msg.call_args[0][0]["positions_at_risk"] == 5

    freqtrade.rpc.send_msg.reset_mock()
    # Halved (9% left) - warned again, once
    for rate in (0.909, 0.908):
        mocker.patch(f"{EXMS}.get_rate", return_value=rate)
        freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 1


@pytest.mark.parametrize("is_short", [False, True])
def test_liquidation_warning_past_stop(mocker, default_conf_usdt, is_short) -> None:
    """
    A position past its stop whose exit could not be placed (exchange error) warns once,
    not on every iteration.
    """
    trade = _make_trade(1, "ETH/USDT:USDT", 1.1 if is_short else 0.9, is_short)
    rate = 1.11 if is_short else 0.89
    freqtrade = _get_bot(mocker, default_conf_usdt, "isolated", rate, [trade], warn_ratio=0.2)

    for _ in range(3):
        freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 1
    assert freqtrade.rpc.send_msg.call_args[0][0]["remaining_ratio"] == 0.0


@pytest.mark.parametrize("is_short", [False, True])
def test_liquidation_warning_open_exit_order(mocker, default_conf_usdt, is_short) -> None:
    """
    Once the liquidation exit is placed there is nothing left to warn about - even if the
    order sits unfilled while the price keeps moving.
    """
    trade = _make_trade(1, "ETH/USDT:USDT", 1.1 if is_short else 0.9, is_short)
    rate = 1.11 if is_short else 0.89
    _add_open_exit_order(trade, rate)
    freqtrade = _get_bot(mocker, default_conf_usdt, "isolated", rate, [trade], warn_ratio=0.2)

    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 0


def test_liquidation_warning_cross_exiting_position_not_shadowing(
    mocker, default_conf_usdt
) -> None:
    """
    In cross margin only the closest position is checked. A position past its stop is always
    the closest - while its exit order rests unfilled it must not hide other positions
    approaching their own stops.
    """
    trades = [
        _make_trade(1, "ETH/USDT:USDT", 0.9),
        _make_trade(2, "XRP/USDT:USDT", 0.9),
    ]
    freqtrade = _get_bot(mocker, default_conf_usdt, "cross", 0.5, trades, warn_ratio=0.2)
    rates = {"ETH/USDT:USDT": 0.89, "XRP/USDT:USDT": 0.95}
    mocker.patch(f"{EXMS}.get_rate", side_effect=lambda pair, **kwargs: rates[pair])

    # ETH past its stop - warned about, exit order placed by exit_positions
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 1
    assert freqtrade.rpc.send_msg.call_args[0][0]["trade_id"] == 1
    _add_open_exit_order(trades[0], 0.89)
    freqtrade.rpc.send_msg.reset_mock()

    # ETH's exit still not filled - XRP enters the warning zone and must be warned about.
    for rate in (0.93, 0.915):
        rates["XRP/USDT:USDT"] = rate
        freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 1
    msg = freqtrade.rpc.send_msg.call_args[0][0]
    assert msg["trade_id"] == 2
    assert msg["positions_at_risk"] == 1
    assert pytest.approx(msg["remaining_ratio"]) == 0.15


@pytest.mark.parametrize("is_short", [False, True])
def test_liquidation_warning_stop_beyond_open_rate(mocker, default_conf_usdt, is_short) -> None:
    """
    In cross margin, losses on other positions can push the liquidation stop past the open rate.
    The distance is then measured from the stop in the direction that liquidates - against the
    current rate, as the position is in profit.
    """
    # Long: open 100, stop 105 - Short: open 100, stop 95
    trade = _make_trade(1, "ETH/USDT:USDT", 95.0 if is_short else 105.0, is_short, open_rate=100.0)
    # 5 away from the stop, 10x - about half the 10% move the margin covers is left.
    rate = 90.0 if is_short else 110.0
    freqtrade = _get_bot(mocker, default_conf_usdt, "cross", rate, [trade], warn_ratio=0.2)
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 0

    # 0.5 away from the stop - about 5% of that move is left
    mocker.patch(f"{EXMS}.get_rate", return_value=94.5 if is_short else 105.5)
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 1
    remaining = freqtrade.rpc.send_msg.call_args[0][0]["remaining_ratio"]
    assert pytest.approx(remaining, rel=1e-3) == (0.0529 if is_short else 0.0474)


@pytest.mark.parametrize("is_short", [False, True])
def test_liquidation_warning_dragged_stop(mocker, default_conf_usdt, is_short) -> None:
    """
    Cross margin: the account's collateral drains without this position moving - the stop
    creeps towards the price and the warning has to pick that up.
    """
    # Long 10x at 100 (short at 100), the stop set from a well funded account.
    trade = _make_trade(1, "ETH/USDT:USDT", 120.0 if is_short else 80.0, is_short, open_rate=100.0)
    freqtrade = _get_bot(mocker, default_conf_usdt, "cross", 100.0, [trade], warn_ratio=0.2)
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 0

    # Losses elsewhere pull the stop to within 1.5% of the price - 15% of the margin's move.
    trade.liquidation_price = 101.5 if is_short else 98.5
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 1
    assert pytest.approx(freqtrade.rpc.send_msg.call_args[0][0]["remaining_ratio"]) == 0.15

    # The position is in profit when that happens - the stop's distance from the current
    # price is what counts (1.5% here), not the distance from the open rate.
    freqtrade.rpc.send_msg.reset_mock()
    freqtrade._liq_warn_cache.clear()
    trade.liquidation_price = 50.75 if is_short else 197.0
    mocker.patch(f"{EXMS}.get_rate", return_value=50.0 if is_short else 200.0)
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 1
    assert pytest.approx(freqtrade.rpc.send_msg.call_args[0][0]["remaining_ratio"]) == 0.15


def test_liquidation_warning_escalates_and_resets(mocker, default_conf_usdt) -> None:
    # Open rate 1.0, liquidation stop 0.9 at 10x - 0.1 of price movement is the full distance.
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
    for leverage, liq_price, rate in [(2, 50.0, 55.0), (5, 80.0, 82.0), (20, 95.0, 95.5)]:
        trade = _make_trade(1, "ETH/USDT:USDT", liq_price, open_rate=100.0, leverage=leverage)
        freqtrade = _get_bot(mocker, default_conf_usdt, "isolated", rate, [trade], warn_ratio=0.2)
        freqtrade.check_liquidation_warnings()
        assert freqtrade.rpc.send_msg.call_count == 1
        assert pytest.approx(freqtrade.rpc.send_msg.call_args[0][0]["remaining_ratio"]) == 0.1

    # A freshly opened position never warns, no matter the leverage - even though the stop
    # (0.5% maintenance margin plus the 5% buffer) sits closer than 1/leverage, so what's
    # left reads below 1.0 right away. Only visible with the (maximum) ratio of 0.99.
    for leverage, liq_price, expected in [
        (2, 52.98, 0.94),
        (5, 81.48, 0.93),
        (20, 95.73, 0.85),
        (50, 98.58, 0.71),
        (100, 99.53, 0.47),
    ]:
        trade = _make_trade(1, "ETH/USDT:USDT", liq_price, open_rate=100.0, leverage=leverage)
        freqtrade = _get_bot(mocker, default_conf_usdt, "isolated", 100.0, [trade], warn_ratio=0.2)
        freqtrade.check_liquidation_warnings()
        assert freqtrade.rpc.send_msg.call_count == 0

        freqtrade = _get_bot(mocker, default_conf_usdt, "isolated", 100.0, [trade], 0.99)
        freqtrade.check_liquidation_warnings()
        assert freqtrade.rpc.send_msg.call_count == 1
        remaining = freqtrade.rpc.send_msg.call_args[0][0]["remaining_ratio"]
        assert pytest.approx(remaining, rel=1e-2) == expected


@pytest.mark.parametrize("is_short", [False, True])
def test_liquidation_warning_measured_from_open_rate_at_loss(
    mocker, default_conf_usdt, is_short
) -> None:
    """
    While at a loss, the distance is measured against the open rate - measured against the
    current rate, a 2x long would be warned about late and a 2x short early.
    """
    # 2x, open 100 - stop at 53 (147 for shorts). 10 away from the stop: 20% of the 50% move.
    trade = _make_trade(1, "ETH/USDT:USDT", 147.0 if is_short else 53.0, is_short, 100.0, 2)
    rate = 137.0 if is_short else 63.0
    freqtrade = _get_bot(mocker, default_conf_usdt, "isolated", rate, [trade], warn_ratio=0.2)
    freqtrade.check_liquidation_warnings()
    assert freqtrade.rpc.send_msg.call_count == 1
    assert pytest.approx(freqtrade.rpc.send_msg.call_args[0][0]["remaining_ratio"]) == 0.2


@pytest.mark.parametrize(
    "warn_ratio,trading_mode,liq_price,amount,expected",
    [
        (0.2, "futures", 0.9, 10.0, 1),
        # Disabled
        (0.0, "futures", 0.9, 10.0, 0),
        # Spot has no liquidation
        (0.2, "spot", 0.9, 10.0, 0),
        # Too far away
        (0.2, "futures", 0.5, 10.0, 0),
        # No liquidation price known
        (0.2, "futures", None, 10.0, 0),
        # Entry order not filled yet
        (0.2, "futures", 0.9, 0.0, 0),
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
