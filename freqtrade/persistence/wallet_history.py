from datetime import datetime
from typing import ClassVar

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from freqtrade.persistence.base import ModelBase, SessionType


class WalletHistory(ModelBase):
    """
    Daily wallet state tracking with minimal fields
    """

    __tablename__ = "wallet_history"
    session: ClassVar[SessionType]

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    currency: Mapped[str] = mapped_column(String(25), nullable=False)
    # Rate: price of 1 unit of `currency` quoted in `quote_currency`.
    # e.g., USDT/ETH -> USDT per ETH
    rate: Mapped[float] = mapped_column(Float, nullable=True)
    # Quote currency for rate/total fields (e.g., 'USDT')
    quote_currency: Mapped[str] = mapped_column(String(25), nullable=False)

    # Balance in `currency` units
    balance: Mapped[float] = mapped_column(Float, nullable=False)

    # Canonical total wallet equity/value denominated in `quote_currency` (if available)
    # For futures positions, collateral + PnL is used to compute this value.
    total_quote: Mapped[float] = mapped_column(Float, nullable=True)
    # Total position value in `quote_currency` - including leverage
    total_position_value: Mapped[float] = mapped_column(Float, nullable=True)
    collateral: Mapped[float] = mapped_column(Float, nullable=True)
    leverage: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    bot_managed: Mapped[bool] = mapped_column(nullable=False, default=True)

    __table_args__ = (
        # Ensure one record per currency per day
        UniqueConstraint("timestamp", "currency", name="unique_wallet_daily"),
    )

    def __repr__(self) -> str:
        return (
            f"WalletHistory(timestamp={self.timestamp}, currency={self.currency}, "
            f"rate={self.rate}, total_quote={self.total_quote}, "
            f"balance={self.balance}, leverage={self.leverage})"
        )
