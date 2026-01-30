import csv
import logging
import os
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PaperLedger:
    def __init__(self, location: Path | None = None):
        """
        :param location: Path to sqlite file (e.g. paper.sqlite) OR directory for CSVs.
        """
        self.mode = "csv"

        if location and str(location).endswith(".sqlite"):
            self.mode = "sqlite"
            self.db_path = location
            self._ensure_db_dir()
            self._init_db()
        else:
            if location:
                self.base_dir = location
            else:
                self.base_dir = Path("user_data") / "generated" / "paper_ledger"

            self.trades_file = self.base_dir / "paper_trades.csv"
            self.daily_file = self.base_dir / "paper_daily_summary.csv"
            self._ensure_dir()
            self._ensure_headers()

    def _ensure_db_dir(self):
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create db directory: {e}")

    def _init_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    timestamp INTEGER,
                    datetime TEXT,
                    symbol TEXT,
                    side TEXT,
                    amount REAL,
                    price REAL,
                    cost REAL,
                    fee_cost REAL,
                    details JSON
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to init sqlite db: {e}")

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
        Record trade to backend (CSV or SQLite).
        """
        if self.mode == "sqlite":
            self._record_sqlite(trade)
        else:
            self._record_csv(trade)

    def _record_sqlite(self, trade: Dict[str, Any]):
        try:
            import json

            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            ts = trade.get("timestamp", int(time.time() * 1000))
            dt = datetime.utcfromtimestamp(ts / 1000).isoformat()

            details = {
                "base_price": trade.get("base_price"),
                "slippage_bps": trade.get("slippage_bps"),
                "fee": trade.get("fee"),
            }

            c.execute(
                """
                INSERT OR REPLACE INTO trades 
                (id, timestamp, datetime, symbol, side, amount, price, cost, fee_cost, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trade.get("id"),
                    ts,
                    dt,
                    trade.get("symbol"),
                    trade.get("side"),
                    trade.get("amount"),
                    trade.get("price"),
                    trade.get("cost"),
                    trade.get("fee", {}).get("cost", 0)
                    if isinstance(trade.get("fee"), dict)
                    else 0,
                    json.dumps(details),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to record sqlite trade: {e}")

    def _record_csv(self, trade: Dict[str, Any]):
        try:
            ts = trade.get("timestamp", int(time.time() * 1000))
            utc_dt = datetime.utcfromtimestamp(ts / 1000)
            # IST approximation
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
