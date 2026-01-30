from unittest import mock

import pytest

from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT


@pytest.fixture
def paper_config(tmp_path):
    # Use a tmp dir for ledger
    # ledger_dir = tmp_path / "paper_ledger"
    return {
        "icicibreeze_paper_forward_test": True,
        "paper_slippage_bps": 10,
        "paper_fee_bps": 20,
        # Ensure we don't try to connect to real Breeze
        "key": "mock_key",
        "secret": "mock_secret",
        "icici_mode": "mock",  # To fetch mock prices for execution
    }


@pytest.fixture
def paper_exchange(paper_config):
    # Patch PaperLedger to use our tmp path
    with mock.patch("adapters.ccxt_shim.paper_ledger.PaperLedger") as MockLedger:
        # Mock instance
        mock_ledger_instance = MockLedger.return_value

        ex = BreezeCCXT(paper_config)

        # Inject our mock instance (though init likely did it)
        ex.paper_ledger = mock_ledger_instance
        yield ex


def test_paper_mode_initialization(paper_config):
    ex = BreezeCCXT(paper_config)
    assert ex.paper_mode is True
    assert ex.paper_slippage == 10
    assert ex.paper_ledger is not None


def test_paper_create_order_simulates_fill(paper_exchange):
    symbol = "RELIANCE/INR"
    side = "buy"
    amount = 10

    # Mock ticker to return 1000.0
    with mock.patch.object(paper_exchange, "fetch_ticker", return_value={"last": 1000.0}):
        # Mock MarketHours to allow order
        with mock.patch.object(paper_exchange.market_hours, "assert_can_create_order"):
            with mock.patch.object(
                paper_exchange.risk_guard, "should_block_entry", return_value=(False, "OK")
            ):
                order = paper_exchange.create_order(symbol, "limit", side, amount)

            # Checks
            assert order["status"] == "closed"
            assert order["info"]["paper"] is True
            assert order["id"].startswith("paper-")

            # Slippage: 10 bps on Buy = 1000 * (1 + 10/10000) = 1000 * 1.001 = 1001.0
            assert order["price"] == pytest.approx(1001.0)
            assert order["filled"] == 10

            # Fees: 20 bps on Notional (1001 * 10 = 10010) -> 10010 * 0.002 = 20.02
            assert order["fee"]["cost"] == pytest.approx(20.02)

            # Ledger called?
            paper_exchange.paper_ledger.record_trade.assert_called_once()
            args = paper_exchange.paper_ledger.record_trade.call_args[0][0]
            assert args["symbol"] == symbol
            assert args["base_price"] == 1000.0


def test_paper_cancel_order_returns_closed(paper_exchange):
    res = paper_exchange.cancel_order("paper-123", "RELIANCE/INR")
    assert res["status"] == "closed"
    assert res["info"]["paper_cancel"] is True


def test_real_order_blocked_in_paper_mode(paper_exchange):
    # Ensure standard mock path is NOT taken
    paper_exchange._mock_orders = {}

    # We are in paper execution mode.
    # If we call create_order, it goes to paper logic.
    # To verifying "blocking", we can check that it didn't call ANY Breeze/SDK
    # method or modify _mock_orders

    with mock.patch.object(paper_exchange, "fetch_ticker", return_value={"last": 100.0}):
        with mock.patch.object(paper_exchange.market_hours, "assert_can_create_order"):
            with mock.patch.object(
                paper_exchange.risk_guard, "should_block_entry", return_value=(False, "OK")
            ):
                paper_exchange.create_order("SBIN/INR", "limit", "buy", 1)

    assert len(paper_exchange._mock_orders) == 0


def test_ledger_persistence_integration(tmp_path):
    # Real test of PaperLedger class via integration
    from adapters.ccxt_shim.paper_ledger import PaperLedger

    ledger_dir = tmp_path / "ledger_test"
    ledger = PaperLedger(location=ledger_dir)

    trade = {
        "id": "p-1",
        "timestamp": 1672531200000,  # 2023-01-01 00:00 UTC
        "symbol": "TATA/INR",
        "side": "buy",
        "amount": 10,
        "price": 500.0,
        "cost": 5000.0,
        "fee": {"cost": 5.0},
    }

    ledger.record_trade(trade)

    assert (ledger_dir / "paper_trades.csv").exists()
    assert (ledger_dir / "paper_daily_summary.csv").exists()

    with (ledger_dir / "paper_daily_summary.csv").open("r") as f:
        content = f.read()
        # Header + 1 row
        assert "trades_count,gross_notional,total_fees" in content
        # Date should be approx 2023-01-01 (ignoring exact tz for this check)
        # 5000.0 notional
        assert "5000.00" in content
