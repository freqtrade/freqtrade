"""
Balance tracking model for wallet balance snapshots at significant events.
"""

import logging
from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar

from sqlalchemy import Float, ForeignKey, Integer, String, desc, select
from sqlalchemy.orm import Mapped, mapped_column

from freqtrade.persistence.base import ModelBase, SessionType


logger = logging.getLogger(__name__)


class BalanceEventType(str, Enum):
    """Types of events that trigger balance snapshots."""

    BOT_START = "bot_start"
    TRADE_OPEN = "trade_open"
    TRADE_CLOSE = "trade_close"
    ORDER_FILL = "order_fill"
    ORDER_CANCEL = "order_cancel"
    POSITION_ADJUST = "position_adjust"  # DCA
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class Balance(ModelBase):
    """
    Balance database model.
    Tracks wallet balance snapshots at significant events to enable:
    - Detection of deposits/withdrawals
    - Accurate long-term PnL tracking
    - Historical balance reconstruction
    """

    __tablename__ = "balances"
    __allow_unmapped__ = True
    session: ClassVar[SessionType]

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Balance data (stake currency)
    total_balance: Mapped[float] = mapped_column(Float(), nullable=False)
    free_balance: Mapped[float] = mapped_column(Float(), nullable=False)
    used_balance: Mapped[float | None] = mapped_column(Float(), nullable=True)
    stake_currency: Mapped[str] = mapped_column(String(25), nullable=False)

    # Trade/Order association (optional)
    ft_trade_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("trades.id", ondelete="CASCADE"), nullable=True, index=True
    )
    ft_order_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("orders.id", ondelete="CASCADE"), nullable=True
    )

    # Profit metrics at this point
    total_profit: Mapped[float | None] = mapped_column(Float(), nullable=True)
    open_trade_value: Mapped[float | None] = mapped_column(Float(), nullable=True)

    # Deposit/withdrawal detection
    external_change: Mapped[float | None] = mapped_column(Float(), nullable=True)

    is_dry_run: Mapped[bool] = mapped_column(nullable=False, default=False)
    context: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    def __repr__(self) -> str:
        return (
            f"Balance(id={self.id}, timestamp={self.timestamp}, "
            f"event_type={self.event_type}, total={self.total_balance}, "
            f"external_change={self.external_change})"
        )

    def to_json(self) -> dict[str, Any]:
        """Convert balance record to JSON-serializable dict."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "event_type": self.event_type,
            "total_balance": self.total_balance,
            "free_balance": self.free_balance,
            "used_balance": self.used_balance,
            "stake_currency": self.stake_currency,
            "ft_trade_id": self.ft_trade_id,
            "ft_order_id": self.ft_order_id,
            "total_profit": self.total_profit,
            "open_trade_value": self.open_trade_value,
            "external_change": self.external_change,
            "is_dry_run": self.is_dry_run,
            "context": self.context,
        }

    @staticmethod
    def get_balances(
        *,
        event_type: BalanceEventType | str | None = None,
        trade_id: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> Sequence["Balance"]:
        """
        Query balance records with optional filters.

        :param event_type: Filter by event type
        :param trade_id: Filter by trade ID
        :param since: Filter records after this timestamp
        :param until: Filter records before this timestamp
        :param limit: Maximum number of records to return
        :return: List of Balance records
        """
        filters = []
        if event_type:
            event_str = event_type.value if isinstance(event_type, BalanceEventType) else event_type
            filters.append(Balance.event_type == event_str)
        if trade_id is not None:
            filters.append(Balance.ft_trade_id == trade_id)
        if since:
            filters.append(Balance.timestamp >= since)
        if until:
            filters.append(Balance.timestamp <= until)

        query = select(Balance).filter(*filters).order_by(desc(Balance.timestamp))

        if limit:
            query = query.limit(limit)

        return Balance.session.scalars(query).all()

    @staticmethod
    def get_latest(stake_currency: str | None = None) -> "Balance | None":
        """
        Get the most recent balance record.

        :param stake_currency: Optionally filter by stake currency
        :return: Latest Balance record or None
        """
        filters = []
        if stake_currency:
            filters.append(Balance.stake_currency == stake_currency)

        query = select(Balance).filter(*filters).order_by(desc(Balance.timestamp)).limit(1)
        return Balance.session.scalars(query).first()

    @staticmethod
    def get_latest_before(
        timestamp: datetime, stake_currency: str | None = None
    ) -> "Balance | None":
        """
        Get the most recent balance record before the given timestamp.

        :param timestamp: Get balance before this time
        :param stake_currency: Optionally filter by stake currency
        :return: Balance record or None
        """
        filters = [Balance.timestamp < timestamp]
        if stake_currency:
            filters.append(Balance.stake_currency == stake_currency)

        query = select(Balance).filter(*filters).order_by(desc(Balance.timestamp)).limit(1)
        return Balance.session.scalars(query).first()

    @staticmethod
    def calculate_external_change(current: "Balance") -> float | None:
        """
        Calculate external change (deposit/withdrawal) between previous and current balance.

        External change = actual balance - expected balance
        Expected = previous_total + profit_change

        :param current: Current balance record
        :return: External change amount or None if no previous record exists
        """
        previous = Balance.get_latest_before(current.timestamp, current.stake_currency)
        if not previous:
            return None

        # Expected = previous_total + profit_change
        profit_change = (current.total_profit or 0) - (previous.total_profit or 0)
        expected_balance = previous.total_balance + profit_change

        # External change = actual - expected
        external_change = current.total_balance - expected_balance

        # Ignore floating point noise (< 0.01% of balance)
        if current.total_balance > 0 and abs(external_change) > current.total_balance * 0.0001:
            return external_change
        return 0.0

    @staticmethod
    def detect_deposits_withdrawals(
        since: datetime | None = None,
        until: datetime | None = None,
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Detect deposits and withdrawals by finding balance records with significant
        external changes.

        :param since: Start of time range
        :param until: End of time range
        :param threshold: Minimum absolute change to report (default 0.0)
        :return: List of records with external changes
        """
        balances = Balance.get_balances(since=since, until=until)

        changes = []
        for balance in balances:
            if balance.external_change is not None and abs(balance.external_change) > threshold:
                change_type = "deposit" if balance.external_change > 0 else "withdrawal"
                changes.append(
                    {
                        "timestamp": balance.timestamp.isoformat() if balance.timestamp else None,
                        "type": change_type,
                        "amount": balance.external_change,
                        "event_type": balance.event_type,
                        "total_balance": balance.total_balance,
                        "balance_id": balance.id,
                    }
                )

        return changes
