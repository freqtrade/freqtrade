# pragma pylint: disable=missing-docstring, C0103
"""
Tests for the Balance model and BalanceEventType enum.
"""

from datetime import UTC, datetime, timedelta

import pytest

from freqtrade.persistence import Balance, BalanceEventType, Trade
from freqtrade.util import dt_now


@pytest.mark.usefixtures("init_persistence")
class TestBalanceEventType:
    def test_event_type_values(self):
        """Test that all expected event types exist"""
        assert BalanceEventType.BOT_START.value == "bot_start"
        assert BalanceEventType.TRADE_OPEN.value == "trade_open"
        assert BalanceEventType.TRADE_CLOSE.value == "trade_close"
        assert BalanceEventType.ORDER_FILL.value == "order_fill"
        assert BalanceEventType.ORDER_CANCEL.value == "order_cancel"
        assert BalanceEventType.POSITION_ADJUST.value == "position_adjust"
        assert BalanceEventType.SCHEDULED.value == "scheduled"
        assert BalanceEventType.MANUAL.value == "manual"

    def test_event_type_is_str(self):
        """Test that BalanceEventType inherits from str"""
        assert isinstance(BalanceEventType.BOT_START, str)
        assert BalanceEventType.BOT_START == "bot_start"


@pytest.mark.usefixtures("init_persistence")
class TestBalanceModel:
    def test_balance_creation(self):
        """Test creating a balance record"""
        balance = Balance(
            timestamp=dt_now(),
            event_type=BalanceEventType.BOT_START.value,
            total_balance=1000.0,
            free_balance=900.0,
            used_balance=100.0,
            stake_currency="USDT",
            is_dry_run=True,
        )
        Balance.session.add(balance)
        Trade.commit()

        assert balance.id is not None
        assert balance.total_balance == 1000.0
        assert balance.free_balance == 900.0
        assert balance.used_balance == 100.0
        assert balance.stake_currency == "USDT"
        assert balance.is_dry_run is True

    def test_balance_to_json(self):
        """Test converting balance to JSON"""
        now = dt_now()
        balance = Balance(
            timestamp=now,
            event_type=BalanceEventType.TRADE_CLOSE.value,
            total_balance=1500.0,
            free_balance=1400.0,
            used_balance=100.0,
            stake_currency="USDT",
            total_profit=500.0,
            open_trade_value=100.0,
            external_change=0.0,
            is_dry_run=False,
        )
        Balance.session.add(balance)
        Trade.commit()

        json_data = balance.to_json()
        assert json_data["id"] == balance.id
        assert json_data["event_type"] == "trade_close"
        assert json_data["total_balance"] == 1500.0
        assert json_data["free_balance"] == 1400.0
        assert json_data["used_balance"] == 100.0
        assert json_data["stake_currency"] == "USDT"
        assert json_data["total_profit"] == 500.0
        assert json_data["open_trade_value"] == 100.0
        assert json_data["external_change"] == 0.0
        assert json_data["is_dry_run"] is False

    def test_balance_repr(self):
        """Test balance string representation"""
        balance = Balance(
            timestamp=dt_now(),
            event_type=BalanceEventType.BOT_START.value,
            total_balance=1000.0,
            free_balance=900.0,
            stake_currency="USDT",
            is_dry_run=True,
        )
        Balance.session.add(balance)
        Trade.commit()

        repr_str = repr(balance)
        assert "Balance" in repr_str
        assert "bot_start" in repr_str
        assert "total=1000.0" in repr_str


@pytest.mark.usefixtures("init_persistence")
class TestBalanceQueries:
    def _create_test_balances(self):
        """Create test balance records"""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        balances = [
            Balance(
                timestamp=base_time,
                event_type=BalanceEventType.BOT_START.value,
                total_balance=1000.0,
                free_balance=1000.0,
                stake_currency="USDT",
                total_profit=0.0,
                is_dry_run=True,
            ),
            Balance(
                timestamp=base_time + timedelta(hours=1),
                event_type=BalanceEventType.TRADE_OPEN.value,
                total_balance=900.0,
                free_balance=800.0,
                used_balance=100.0,
                stake_currency="USDT",
                total_profit=0.0,
                ft_trade_id=1,
                is_dry_run=True,
            ),
            Balance(
                timestamp=base_time + timedelta(hours=2),
                event_type=BalanceEventType.TRADE_CLOSE.value,
                total_balance=1050.0,
                free_balance=1050.0,
                stake_currency="USDT",
                total_profit=50.0,
                ft_trade_id=1,
                is_dry_run=True,
            ),
            Balance(
                timestamp=base_time + timedelta(hours=3),
                event_type=BalanceEventType.BOT_START.value,
                total_balance=1200.0,  # External deposit of ~150
                free_balance=1200.0,
                stake_currency="USDT",
                total_profit=50.0,
                external_change=150.0,
                is_dry_run=True,
            ),
        ]
        for b in balances:
            Balance.session.add(b)
        Trade.commit()
        return balances

    def test_get_balances_all(self):
        """Test getting all balance records"""
        self._create_test_balances()
        balances = Balance.get_balances()
        assert len(balances) == 4

    def test_get_balances_by_event_type(self):
        """Test filtering by event type"""
        self._create_test_balances()
        balances = Balance.get_balances(event_type=BalanceEventType.BOT_START)
        assert len(balances) == 2
        assert all(b.event_type == "bot_start" for b in balances)

    def test_get_balances_by_event_type_string(self):
        """Test filtering by event type as string"""
        self._create_test_balances()
        balances = Balance.get_balances(event_type="trade_close")
        assert len(balances) == 1
        assert balances[0].event_type == "trade_close"

    def test_get_balances_by_trade_id(self):
        """Test filtering by trade ID"""
        self._create_test_balances()
        balances = Balance.get_balances(trade_id=1)
        assert len(balances) == 2
        assert all(b.ft_trade_id == 1 for b in balances)

    def test_get_balances_by_time_range(self):
        """Test filtering by time range"""
        self._create_test_balances()
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        balances = Balance.get_balances(
            since=base_time + timedelta(minutes=30),
            until=base_time + timedelta(hours=2, minutes=30),
        )
        assert len(balances) == 2
        assert all(b.event_type in ["trade_open", "trade_close"] for b in balances)

    def test_get_balances_with_limit(self):
        """Test limiting results"""
        self._create_test_balances()
        balances = Balance.get_balances(limit=2)
        assert len(balances) == 2

    def test_get_latest(self):
        """Test getting the most recent balance"""
        self._create_test_balances()
        latest = Balance.get_latest()
        assert latest is not None
        assert latest.total_balance == 1200.0

    def test_get_latest_by_currency(self):
        """Test getting latest balance by currency"""
        self._create_test_balances()
        latest = Balance.get_latest(stake_currency="USDT")
        assert latest is not None
        assert latest.stake_currency == "USDT"

        # Non-existent currency
        latest_btc = Balance.get_latest(stake_currency="BTC")
        assert latest_btc is None

    def test_get_latest_before(self):
        """Test getting balance before a timestamp"""
        self._create_test_balances()
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        before = Balance.get_latest_before(base_time + timedelta(hours=2, minutes=30))
        assert before is not None
        assert before.event_type == "trade_close"

    def test_get_latest_before_none(self):
        """Test getting balance before first record returns None"""
        self._create_test_balances()
        base_time = datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC)
        before = Balance.get_latest_before(base_time)
        assert before is None


@pytest.mark.usefixtures("init_persistence")
class TestExternalChangeDetection:
    def test_calculate_external_change_no_previous(self):
        """Test external change with no previous record"""
        balance = Balance(
            timestamp=dt_now(),
            event_type=BalanceEventType.BOT_START.value,
            total_balance=1000.0,
            free_balance=1000.0,
            stake_currency="USDT",
            total_profit=0.0,
            is_dry_run=True,
        )
        Balance.session.add(balance)
        Trade.commit()

        external_change = Balance.calculate_external_change(balance)
        assert external_change is None

    def test_calculate_external_change_normal_trade(self):
        """Test external change with normal trade profit"""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Initial balance
        b1 = Balance(
            timestamp=base_time,
            event_type=BalanceEventType.BOT_START.value,
            total_balance=1000.0,
            free_balance=1000.0,
            stake_currency="USDT",
            total_profit=0.0,
            is_dry_run=True,
        )
        Balance.session.add(b1)
        Trade.commit()

        # Balance after profit (no external change)
        b2 = Balance(
            timestamp=base_time + timedelta(hours=1),
            event_type=BalanceEventType.TRADE_CLOSE.value,
            total_balance=1050.0,  # 50 profit
            free_balance=1050.0,
            stake_currency="USDT",
            total_profit=50.0,  # 50 profit
            is_dry_run=True,
        )

        external_change = Balance.calculate_external_change(b2)
        assert external_change == 0.0  # Expected: 1000 + 50 = 1050, actual = 1050

    def test_calculate_external_change_deposit(self):
        """Test detecting a deposit"""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Initial balance
        b1 = Balance(
            timestamp=base_time,
            event_type=BalanceEventType.BOT_START.value,
            total_balance=1000.0,
            free_balance=1000.0,
            stake_currency="USDT",
            total_profit=0.0,
            is_dry_run=True,
        )
        Balance.session.add(b1)
        Trade.commit()

        # Balance after deposit
        b2 = Balance(
            timestamp=base_time + timedelta(hours=1),
            event_type=BalanceEventType.BOT_START.value,
            total_balance=1500.0,  # 500 deposited
            free_balance=1500.0,
            stake_currency="USDT",
            total_profit=0.0,  # No profit change
            is_dry_run=True,
        )

        external_change = Balance.calculate_external_change(b2)
        assert external_change == 500.0  # Expected: 1000, actual: 1500

    def test_calculate_external_change_withdrawal(self):
        """Test detecting a withdrawal"""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Initial balance
        b1 = Balance(
            timestamp=base_time,
            event_type=BalanceEventType.BOT_START.value,
            total_balance=1000.0,
            free_balance=1000.0,
            stake_currency="USDT",
            total_profit=0.0,
            is_dry_run=True,
        )
        Balance.session.add(b1)
        Trade.commit()

        # Balance after withdrawal
        b2 = Balance(
            timestamp=base_time + timedelta(hours=1),
            event_type=BalanceEventType.BOT_START.value,
            total_balance=700.0,  # 300 withdrawn
            free_balance=700.0,
            stake_currency="USDT",
            total_profit=0.0,  # No profit change
            is_dry_run=True,
        )

        external_change = Balance.calculate_external_change(b2)
        assert external_change == -300.0  # Expected: 1000, actual: 700

    def test_detect_deposits_withdrawals(self):
        """Test detecting deposits and withdrawals in batch"""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        balances = [
            Balance(
                timestamp=base_time,
                event_type=BalanceEventType.BOT_START.value,
                total_balance=1000.0,
                free_balance=1000.0,
                stake_currency="USDT",
                total_profit=0.0,
                external_change=None,  # First record
                is_dry_run=True,
            ),
            Balance(
                timestamp=base_time + timedelta(hours=1),
                event_type=BalanceEventType.TRADE_CLOSE.value,
                total_balance=1050.0,
                free_balance=1050.0,
                stake_currency="USDT",
                total_profit=50.0,
                external_change=0.0,  # No external change
                is_dry_run=True,
            ),
            Balance(
                timestamp=base_time + timedelta(hours=2),
                event_type=BalanceEventType.BOT_START.value,
                total_balance=1550.0,
                free_balance=1550.0,
                stake_currency="USDT",
                total_profit=50.0,
                external_change=500.0,  # Deposit
                is_dry_run=True,
            ),
            Balance(
                timestamp=base_time + timedelta(hours=3),
                event_type=BalanceEventType.MANUAL.value,
                total_balance=1350.0,
                free_balance=1350.0,
                stake_currency="USDT",
                total_profit=50.0,
                external_change=-200.0,  # Withdrawal
                is_dry_run=True,
            ),
        ]
        for b in balances:
            Balance.session.add(b)
        Trade.commit()

        changes = Balance.detect_deposits_withdrawals()
        assert len(changes) == 2

        # Check deposit
        deposit = next((c for c in changes if c["type"] == "deposit"), None)
        assert deposit is not None
        assert deposit["amount"] == 500.0

        # Check withdrawal
        withdrawal = next((c for c in changes if c["type"] == "withdrawal"), None)
        assert withdrawal is not None
        assert withdrawal["amount"] == -200.0

    def test_detect_deposits_withdrawals_with_threshold(self):
        """Test filtering by threshold"""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        balances = [
            Balance(
                timestamp=base_time,
                event_type=BalanceEventType.BOT_START.value,
                total_balance=1000.0,
                free_balance=1000.0,
                stake_currency="USDT",
                external_change=50.0,  # Small deposit
                is_dry_run=True,
            ),
            Balance(
                timestamp=base_time + timedelta(hours=1),
                event_type=BalanceEventType.BOT_START.value,
                total_balance=2000.0,
                free_balance=2000.0,
                stake_currency="USDT",
                external_change=500.0,  # Large deposit
                is_dry_run=True,
            ),
        ]
        for b in balances:
            Balance.session.add(b)
        Trade.commit()

        # With threshold of 100, should only see the large deposit
        changes = Balance.detect_deposits_withdrawals(threshold=100.0)
        assert len(changes) == 1
        assert changes[0]["amount"] == 500.0
