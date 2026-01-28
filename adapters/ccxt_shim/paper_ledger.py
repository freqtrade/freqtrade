import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PaperLedger:
    def __init__(self, data_dir: Path | None = None):
        if data_dir:
            self.base_dir = data_dir
        else:
            self.base_dir = Path("user_data") / "generated" / "paper_ledger"

        self.trades_file = self.base_dir / "paper_trades.csv"
        self.daily_file = self.base_dir / "paper_daily_summary.csv"

        self._ensure_dir()
        self._ensure_headers()

    def _ensure_dir(self):
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create paper ledger directory {self.base_dir}: {e}")

    def _ensure_headers(self):
        if not self.trades_file.exists():
            with self.trades_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "utc_ts",
                        "local_ts_ist",
                        "order_id",
                        "symbol",
                        "side",
                        "amount",
                        "avg_price",
                        "base_price",
                        "slippage_bps",
                        "fee",
                        "notional",
                    ]
                )

        if not self.daily_file.exists():
            with self.daily_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["date_ist", "trades_count", "gross_notional", "total_fees"])

    def record_trade(self, trade: Dict[str, Any]):
        """
        Append a trade to the ledger and update daily summary.
        Expected trade dict keys matching the CSV header where applicable.
        """
        try:
            ts = trade.get("timestamp", int(time.time() * 1000))
            utc_dt = datetime.utcfromtimestamp(ts / 1000)
            # Simple IST approximation for display (UTC+5:30)
            # In a real app we might use pytz, but let's keep it simple and dependency-free if possible
            # or rely on system provided timezone if needed.
            # Only for logging purposes.
            # Let's strictly use UTC for calculation logic, but user requested local_ts_ist.
            # 5.5 hours = 19800 seconds
            ist_ts = ts / 1000 + 19800
            ist_dt = datetime.utcfromtimestamp(ist_ts)

            row = [
                ts,
                ist_dt.strftime("%Y-%m-%d %H:%M:%S"),
                trade.get("id"),
                trade.get("symbol"),
                trade.get("side"),
                trade.get("amount"),
                trade.get("price"),  # avg_price
                trade.get("base_price", ""),
                trade.get("slippage_bps", ""),
                trade.get("fee", {}).get("cost", 0) if isinstance(trade.get("fee"), dict) else 0,
                trade.get("cost", 0),
            ]

            with self.trades_file.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            self._update_daily_summary(ist_dt.strftime("%Y-%m-%d"), trade)

        except Exception as e:
            logger.error(f"Failed to record paper trade: {e}")

    def _update_daily_summary(self, date_ist: str, trade: Dict[str, Any]):
        # Read all daily summaries
        summary = {}
        if self.daily_file.exists():
            try:
                with self.daily_file.open("r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        summary[row["date_ist"]] = row
            except Exception:
                pass  # corrupted or empty

        # Update specific date
        if date_ist not in summary:
            summary[date_ist] = {
                "date_ist": date_ist,
                "trades_count": "0",
                "gross_notional": "0.0",
                "total_fees": "0.0",
            }

        current = summary[date_ist]

        # Add new values
        try:
            cnt = int(current["trades_count"]) + 1
            notional = float(current["gross_notional"]) + float(trade.get("cost", 0))
            fees = float(current["total_fees"]) + (
                float(trade.get("fee", {}).get("cost", 0))
                if isinstance(trade.get("fee"), dict)
                else 0
            )

            summary[date_ist] = {
                "date_ist": date_ist,
                "trades_count": str(cnt),
                "gross_notional": f"{notional:.2f}",
                "total_fees": f"{fees:.2f}",
            }

            # Write back sorted
            sorted_dates = sorted(summary.keys())
            with self.daily_file.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["date_ist", "trades_count", "gross_notional", "total_fees"]
                )
                writer.writeheader()
                for d in sorted_dates:
                    writer.writerow(summary[d])

        except Exception as e:
            logger.error(f"Failed to update daily summary: {e}")
