from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from freqtrade.enums import CandleType
from freqtrade.persistence import KeyValueStore, Order, Trade, WalletHistory
from freqtrade.util import dt_now, dt_utc
from freqtrade.util.migrations.migrate_wallet_history import (
    _migrate_wallet_history,
    _prepare_balance_distribution,
    migrate_wallet_history,
)
from tests.conftest import EXMS, generate_test_data, get_patched_exchange, log_has_re


def create_closed_mock_trade(fee, pair: str, open_date: datetime, close_date: datetime):
    """Create a closed trade for wallet history testing."""
    trade = Trade(
        pair=pair,
        stake_amount=100.0,
        amount=10.0,
        amount_requested=10.0,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=10.0,
        close_rate=11.0,
        close_profit=0.1,
        close_profit_abs=9.5,
        exchange="binance",
        is_open=False,
        strategy="TestStrategy",
        timeframe=5,
        open_date=open_date,
        close_date=close_date,
        is_short=False,
    )
    order_entry = Order(
        ft_order_side="buy",
        ft_pair=pair,
        ft_is_open=False,
        ft_amount=10.0,
        ft_price=10.0,
        order_id=f"order_{pair}_entry",
        status="closed",
        symbol=pair,
        order_type="limit",
        side="buy",
        price=10.0,
        average=10.0,
        amount=10.0,
        filled=10.0,
        remaining=0.0,
        order_date=open_date,
        order_filled_date=open_date,
    )

    order_exit = Order(
        ft_order_side="sell",
        ft_pair=pair,
        ft_is_open=False,
        ft_amount=10.0,
        ft_price=11.0,
        order_id=f"order_{pair}_exit",
        status="closed",
        symbol=pair,
        order_type="limit",
        side="sell",
        price=11.0,
        average=11.0,
        amount=10.0,
        filled=10.0,
        remaining=0.0,
        order_date=close_date,
        order_filled_date=close_date,
    )

    trade.orders.append(order_entry)
    trade.orders.append(order_exit)
    return trade


@pytest.mark.usefixtures("init_persistence")
def test_migrate_wallet_history_skips_when_no_ohlcv_history(mocker, default_conf_usdt):
    """Test that migration is skipped when exchange doesn't support OHLCV history."""
    exchange = MagicMock()
    exchange.get_option.return_value = False  # ohlcv_has_history = False

    migrate_mock = mocker.patch(
        "freqtrade.util.migrations.migrate_wallet_history._migrate_wallet_history"
    )

    migrate_wallet_history(default_conf_usdt, exchange, 1000.0)

    # Should return early without setting the migration flag
    assert KeyValueStore.get_int_value("wallet_history_migration") is None
    assert not migrate_mock.called


@pytest.mark.usefixtures("init_persistence")
def test_migrate_wallet_history_skips_when_already_migrated(mocker, default_conf_usdt):
    """Test that migration is skipped if already completed."""
    exchange = MagicMock()
    exchange.get_option.return_value = True

    migrate_mock = mocker.patch(
        "freqtrade.util.migrations.migrate_wallet_history._migrate_wallet_history"
    )

    # Set migration as already completed
    KeyValueStore.store_value("wallet_history_migration", 1)

    migrate_wallet_history(default_conf_usdt, exchange, 1000.0)
    # Should not call any migration logic
    assert KeyValueStore.get_int_value("wallet_history_migration") == 1
    assert not migrate_mock.called


@pytest.mark.usefixtures("init_persistence")
def test_migrate_wallet_history_no_trades(default_conf_usdt):
    """Test migration with no trades in database."""
    exchange = MagicMock()
    exchange.get_option.return_value = True

    # Set bot_start_time
    KeyValueStore.store_value("bot_start_time", dt_now() - timedelta(days=5))

    migrate_wallet_history(default_conf_usdt, exchange, 1000.0)

    # Should complete migration (flag set) but no wallet entries
    assert KeyValueStore.get_int_value("wallet_history_migration") == 1
    assert WalletHistory.session.query(WalletHistory).count() == 0


@pytest.mark.usefixtures("init_persistence")
def test_migrate_wallet_history_with_trades(default_conf_usdt, fee, time_machine, markets):
    """Test migration with trades creates wallet history entries."""
    start_time = dt_utc(2024, 1, 10, 12, 0, 0)
    time_machine.move_to(start_time, tick=False)

    # Bot started 10 days ago
    bot_start = start_time - timedelta(days=10)
    KeyValueStore.store_value("bot_start_time", bot_start)

    # Create mock trades with dates within the range
    trade_open = start_time - timedelta(days=5)
    trade_close = start_time - timedelta(days=3)
    trade1 = create_closed_mock_trade(
        fee,
        "ETH/USDT",
        open_date=trade_open,
        close_date=trade_close,
    )
    Trade.session.add(trade1)
    Trade.commit()

    # Generate mock OHLCV data starting from bot_start
    candle_type = default_conf_usdt.get("candle_type_def", CandleType.SPOT)
    ohlcv_df = generate_test_data("1d", size=15, start=bot_start.strftime("%Y-%m-%d"))
    ohlcv_data = {("ETH/USDT", "1d", candle_type): ohlcv_df}

    exchange = MagicMock()
    exchange.get_option.return_value = True
    exchange.markets = markets
    exchange.refresh_latest_ohlcv.return_value = ohlcv_data
    exchange.get_pair_base_currency = MagicMock(side_effect=lambda pair: markets.get(pair)["base"])

    migrate_wallet_history(default_conf_usdt, exchange, 1000.0)

    # Should complete migration
    assert KeyValueStore.get_int_value("wallet_history_migration") == 1

    # Should have created wallet history entries
    wallet_entries = WalletHistory.session.query(WalletHistory).all()
    assert len(wallet_entries) > 0


@pytest.mark.usefixtures("init_persistence")
def test_migrate_wallet_history_with_multiple_pairs(default_conf_usdt, fee, time_machine, markets):
    """Test migration with multiple trading pairs."""
    start_time = dt_utc(2024, 1, 15, 12, 0, 0)
    time_machine.move_to(start_time, tick=False)

    # Bot started 15 days ago
    bot_start = start_time - timedelta(days=15)
    KeyValueStore.store_value("bot_start_time", bot_start)

    # Create mock trades for multiple pairs within the date range
    trade1 = create_closed_mock_trade(
        fee,
        "ETH/USDT",
        open_date=start_time - timedelta(days=10),
        close_date=start_time - timedelta(days=6),
    )
    trade2 = create_closed_mock_trade(
        fee,
        "BTC/USDT",
        open_date=start_time - timedelta(days=7),
        close_date=start_time - timedelta(days=5),
    )
    Trade.session.add(trade1)
    Trade.session.add(trade2)
    Trade.commit()

    # Generate mock OHLCV data for both pairs starting from bot_start
    candle_type = default_conf_usdt.get("candle_type_def", CandleType.SPOT)
    ohlcv_data = {}
    ohlcv_data[("ETH/USDT", "1d", candle_type)] = generate_test_data(
        "1d", size=20, start=bot_start.strftime("%Y-%m-%d"), base=1500
    )

    ohlcv_data[("BTC/USDT", "1d", candle_type)] = generate_test_data(
        "1d", size=20, start=bot_start.strftime("%Y-%m-%d"), base=30000
    )

    exchange = MagicMock()
    exchange.get_option.return_value = True
    exchange.markets = markets
    exchange.refresh_latest_ohlcv.return_value = ohlcv_data
    exchange.get_pair_base_currency = MagicMock(side_effect=lambda pair: markets.get(pair)["base"])

    migrate_wallet_history(default_conf_usdt, exchange, 1000.0)

    # Should complete migration
    assert KeyValueStore.get_int_value("wallet_history_migration") == 1

    # Should have wallet history entries
    wallet_entries = WalletHistory.session.query(WalletHistory).all()
    assert len(wallet_entries) > 0

    # Check that stake currency (USDT) entries exist
    usdt_entries = [e for e in wallet_entries if e.currency == "USDT"]
    assert len(usdt_entries) > 0
    assert len(wallet_entries) > len(usdt_entries)

    # Stake currency should have price = 1.0
    for entry in usdt_entries:
        assert entry.rate == 1.0

    eth_entries = [e for e in wallet_entries if e.currency == "ETH"]
    btc_entries = [e for e in wallet_entries if e.currency == "BTC"]
    assert len(eth_entries) == 4
    assert len(btc_entries) == 2
    assert all(entry.rate and entry.rate > 1400 and entry.rate < 1600 for entry in eth_entries)
    assert all(entry.rate and entry.rate > 29000 and entry.rate < 31000 for entry in btc_entries)
    assert all(entry.balance == 10 for entry in btc_entries)


@pytest.mark.usefixtures("init_persistence")
def test_migrate_wallet_history_pair_not_in_markets(
    default_conf_usdt, caplog, fee, time_machine, markets
):
    """Test migration handles pairs that are not in exchange markets."""
    start_time = dt_utc(2024, 1, 10, 12, 0, 0)
    time_machine.move_to(start_time, tick=False)

    # Bot started 10 days ago
    bot_start = start_time - timedelta(days=10)
    KeyValueStore.store_value("bot_start_time", bot_start)

    # Create a trade with a pair that won't be in markets
    trade1 = create_closed_mock_trade(
        fee,
        "UNKNOWN/USDT",
        open_date=start_time - timedelta(days=5),
        close_date=start_time - timedelta(days=3),
    )
    Trade.session.add(trade1)
    Trade.commit()

    exchange = MagicMock()
    exchange.get_option.return_value = True
    exchange.markets = markets
    exchange.refresh_latest_ohlcv.return_value = {}

    migrate_wallet_history(default_conf_usdt, exchange, 1000.0)
    assert log_has_re("No OHLCV data available for .*", caplog)


@pytest.mark.usefixtures("init_persistence")
def test_migrate_wallet_history_stores_migration_date(
    default_conf_usdt, fee, time_machine, markets
):
    """Test that migration stores the migration date."""
    start_time = dt_utc(2024, 1, 10, 12, 0, 0)
    time_machine.move_to(start_time, tick=False)

    # Bot started 10 days ago
    bot_start = start_time - timedelta(days=10)
    KeyValueStore.store_value("bot_start_time", bot_start)

    # Create a trade
    trade1 = create_closed_mock_trade(
        fee,
        "ETH/USDT",
        open_date=start_time - timedelta(days=5),
        close_date=start_time - timedelta(days=3),
    )
    Trade.session.add(trade1)
    Trade.commit()

    candle_type = default_conf_usdt.get("candle_type_def", CandleType.SPOT)
    ohlcv_data = {
        ("ETH/USDT", "1d", candle_type): generate_test_data(
            "1d", size=15, start=bot_start.strftime("%Y-%m-%d")
        )
    }

    exchange = MagicMock()
    exchange.get_option.return_value = True
    exchange.markets = markets
    exchange.refresh_latest_ohlcv.return_value = ohlcv_data

    migrate_wallet_history(default_conf_usdt, exchange, 1000.0)

    # Check migration date is stored
    migration_date = KeyValueStore.get_datetime_value("wallet_history_migration_date")
    assert migration_date is not None


@pytest.mark.usefixtures("init_persistence")
def test_internal_migrate_wallet_history_empty_trades(default_conf_usdt, time_machine):
    """Test _migrate_wallet_history returns early when no trades exist."""
    start_time = dt_utc(2024, 1, 1, 12, 0, 0)
    time_machine.move_to(start_time, tick=False)

    # Set bot_start_time
    KeyValueStore.store_value("bot_start_time", start_time - timedelta(days=5))

    exchange = MagicMock()
    exchange.get_option.return_value = True
    exchange.markets = {}
    exchange.refresh_latest_ohlcv.return_value = {}

    # Call internal function directly with no trades
    _migrate_wallet_history(default_conf_usdt, exchange, 1000.0)

    # refresh_latest_ohlcv should not be called when there are no trades
    exchange.refresh_latest_ohlcv.assert_not_called()


@pytest.mark.usefixtures("init_persistence")
def test_migrate_wallet_history_with_patched_exchange(mocker, default_conf_usdt, fee, time_machine):
    """Test migration using get_patched_exchange helper."""
    start_time = dt_utc(2024, 1, 10, 12, 0, 0)
    time_machine.move_to(start_time, tick=False)

    # Bot started 10 days ago
    bot_start = start_time - timedelta(days=10)
    KeyValueStore.store_value("bot_start_time", bot_start)

    # Create a trade
    trade1 = create_closed_mock_trade(
        fee,
        "ETH/USDT",
        open_date=start_time - timedelta(days=5),
        close_date=start_time - timedelta(days=3),
    )
    Trade.session.add(trade1)
    Trade.commit()

    # Generate mock OHLCV data starting from bot_start
    candle_type = default_conf_usdt.get("candle_type_def", CandleType.SPOT)
    ohlcv_df = generate_test_data("1d", size=15, start=bot_start.strftime("%Y-%m-%d"))
    ohlcv_data = {("ETH/USDT", "1d", candle_type): ohlcv_df}

    # Mock exchange methods
    mocker.patch.multiple(
        EXMS,
        get_option=MagicMock(return_value=True),
        refresh_latest_ohlcv=MagicMock(return_value=ohlcv_data),
    )

    exchange = get_patched_exchange(mocker, default_conf_usdt)

    migrate_wallet_history(default_conf_usdt, exchange, 1000.0)

    # Should complete migration
    assert KeyValueStore.get_int_value("wallet_history_migration") == 1


@pytest.mark.usefixtures("init_persistence")
def test_migrate_wallet_history_db_error_handling(
    mocker, default_conf_usdt, fee, time_machine, markets
):
    """Test that database errors are handled gracefully."""
    start_time = dt_utc(2024, 1, 10, 12, 0, 0)
    time_machine.move_to(start_time, tick=False)

    # Bot started 10 days ago
    bot_start = start_time - timedelta(days=10)
    KeyValueStore.store_value("bot_start_time", bot_start)

    # Create a trade
    trade1 = create_closed_mock_trade(
        fee,
        "ETH/USDT",
        open_date=start_time - timedelta(days=5),
        close_date=start_time - timedelta(days=3),
    )
    Trade.session.add(trade1)
    Trade.commit()

    candle_type = default_conf_usdt.get("candle_type_def", CandleType.SPOT)
    ohlcv_data = {
        ("ETH/USDT", "1d", candle_type): generate_test_data(
            "1d", size=15, start=bot_start.strftime("%Y-%m-%d")
        )
    }

    exchange = MagicMock()
    exchange.get_option.return_value = True
    exchange.markets = markets
    exchange.refresh_latest_ohlcv.return_value = ohlcv_data

    # Mock bulk_save_objects to raise an exception
    mocker.patch.object(
        WalletHistory.session, "bulk_save_objects", side_effect=Exception("DB Error")
    )

    # Should not raise exception, but handle error gracefully
    migrate_wallet_history(default_conf_usdt, exchange, 1000.0)

    # Migration flag should still be set even after error in _migrate
    assert KeyValueStore.get_int_value("wallet_history_migration") == 1


@pytest.mark.usefixtures("init_persistence")
def test__prepare_balance_distribution(default_conf_usdt, fee, time_machine, markets):
    """Test migration with multiple trading pairs."""
    start_time = dt_utc(2024, 1, 15, 12, 0, 0)
    time_machine.move_to(start_time, tick=False)

    # Bot started 15 days ago
    bot_start = start_time - timedelta(days=15)
    KeyValueStore.store_value("bot_start_time", bot_start)

    # Create mock trades for multiple pairs within the date range
    trade1 = create_closed_mock_trade(
        fee,
        "ETH/USDT",
        open_date=start_time - timedelta(days=10),
        close_date=start_time - timedelta(days=6),
    )
    trade2 = create_closed_mock_trade(
        fee,
        "BTC/USDT",
        open_date=start_time - timedelta(days=7),
        close_date=start_time - timedelta(days=5),
    )
    Trade.session.add(trade1)
    Trade.session.add(trade2)
    Trade.commit()

    # Generate mock OHLCV data for both pairs starting from bot_start
    candle_type = default_conf_usdt.get("candle_type_def", CandleType.SPOT)
    ohlcv_data = {}
    ohlcv_data[("ETH/USDT", "1d", candle_type)] = generate_test_data(
        "1d", size=20, start=bot_start.strftime("%Y-%m-%d"), base=1500
    )

    ohlcv_data[("BTC/USDT", "1d", candle_type)] = generate_test_data(
        "1d", size=20, start=bot_start.strftime("%Y-%m-%d"), base=30000
    )

    exchange = MagicMock()
    exchange.get_option.return_value = True
    exchange.markets = markets
    exchange.refresh_latest_ohlcv.return_value = ohlcv_data

    balance_dist, pairlist_valid = _prepare_balance_distribution(
        default_conf_usdt, exchange, 1000.0
    )
    assert not balance_dist.empty
    assert len(pairlist_valid) == 2
    assert "ETH/USDT" in pairlist_valid
    assert "BTC/USDT" in pairlist_valid

    assert len(balance_dist) == 16  # 16 days from bot_start to now
    assert balance_dist["USDT"].iloc[0] == 1000.0
    assert pd.isna(balance_dist["USDT"]).sum() == 0

    assert all(
        col in balance_dist.columns
        for col in [
            "USDT",
            "ETH/USDT",
            "ETH/USDT_collateral",
            "ETH/USDT_leverage",
            "BTC/USDT",
            "BTC/USDT_collateral",
            "BTC/USDT_leverage",
            "ETH/USDT_open",
            "BTC/USDT_open",
            "ETH/USDT_value",
            "BTC/USDT_value",
            "total_value",
        ]
    )
