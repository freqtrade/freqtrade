#!/usr/bin/env python3
"""
Metrics Exporter for ICICI Breeze Adapter.
Collects metrics from:
- Health Snapshot (health.json)
- Runtime Alerts (alerts.jsonl)
- Paper Ledger (sqlite/csv)

Exports to:
- metrics.json
- metrics.prom (Prometheus TextFile format)
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any

from adapters.ccxt_shim import health_snapshot

logger = logging.getLogger(__name__)

RUNTIME_DIR = Path("user_data/generated/runtime")
ALERTS_FILE = RUNTIME_DIR / "alerts.jsonl"
PAPER_LEDGER_SQLITE = Path("user_data/generated/paper_ledger/paper.sqlite")
PAPER_LEDGER_CSV = Path("user_data/generated/paper_ledger/paper_trades.csv")

METRICS_JSON = RUNTIME_DIR / "metrics.json"
METRICS_PROM = RUNTIME_DIR / "metrics.prom"


def get_paper_order_count() -> int:
    """Count total paper orders from SQLite or CSV."""
    if PAPER_LEDGER_SQLITE.exists():
        try:
            conn = sqlite3.connect(PAPER_LEDGER_SQLITE)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM trades")
            count = c.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.error(f"Failed to read paper sqlite: {e}")
            return 0

    if PAPER_LEDGER_CSV.exists():
        try:
            with open(PAPER_LEDGER_CSV, "r") as f:
                # Subtract header
                return max(0, sum(1 for _ in f) - 1)
        except Exception as e:
            logger.error(f"Failed to read paper csv: {e}")
            return 0

    return 0


def get_alert_count() -> int:
    """Count total alerts from alerts.jsonl."""
    if not ALERTS_FILE.exists():
        return 0
    try:
        with open(ALERTS_FILE, "r") as f:
            return sum(1 for _ in f)
    except Exception as e:
        logger.error(f"Failed to read alerts.jsonl: {e}")
        return 0


def collect_metrics() -> Dict[str, Any]:
    """Gather all metrics."""
    # 1. Health Snapshot data
    health_data = health_snapshot.load()
    counters = health_data.get("counters", {})
    durations = health_data.get("durations", {})
    circuit_breaker = health_data.get("circuit_breaker", {})

    # Calculate p50 latencies
    latencies = {
        "fetch_ticker": health_snapshot.get_p50_latency("fetch_ticker"),
        "fetch_ohlcv": health_snapshot.get_p50_latency("fetch_ohlcv"),
        "create_order": health_snapshot.get_p50_latency("create_order"),
    }

    # 2. External counts
    paper_orders = get_paper_order_count()
    alerts_total = get_alert_count()  # Potentially mapping to degraded_failures? Not necessarily.

    metrics = {
        "policy_blocks_total": counters.get("policy_blocks", 0),
        "degraded_failures_total": counters.get("degraded_failures", 0),
        "circuit_open_total": 1
        if circuit_breaker.get("state") == "open"
        else 0,  # Or a counter if we add one, requirement says total.
        "orders_paper_total": paper_orders,
        "orders_live_blocked_total": counters.get(
            "policy_blocks", 0
        ),  # Reuse policy blocks for now as proxy
        "latency_ms_p50": latencies,
        "alerts_total": alerts_total,
    }

    return metrics


def write_json(metrics: Dict[str, Any]):
    """Write metrics to atomic JSON file."""
    try:
        tmp_path = METRICS_JSON.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(metrics, f, indent=2)
        tmp_path.rename(METRICS_JSON)
    except Exception as e:
        logger.error(f"Failed to write metrics.json: {e}")


def write_prom(metrics: Dict[str, Any]):
    """Write metrics to atomic Prometheus TextFile."""
    lines = []

    # Counters
    lines.append(f"policy_blocks_total {metrics['policy_blocks_total']}")
    lines.append(f"degraded_failures_total {metrics['degraded_failures_total']}")
    lines.append(f"circuit_open_total {metrics['circuit_open_total']}")
    lines.append(f"orders_paper_total {metrics['orders_paper_total']}")
    lines.append(f"orders_live_blocked_total {metrics['orders_live_blocked_total']}")
    lines.append(f"alerts_total {metrics['alerts_total']}")

    # Latencies
    for method, value in metrics["latency_ms_p50"].items():
        lines.append(f'latency_ms_p50{{method="{method}"}} {value}')

    # Sort lines for determinism
    lines.sort()

    content = "\n".join(lines) + "\n"

    try:
        tmp_path = METRICS_PROM.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            f.write(content)
        tmp_path.rename(METRICS_PROM)
    except Exception as e:
        logger.error(f"Failed to write metrics.prom: {e}")


def export_metrics():
    """Main export function."""
    metrics = collect_metrics()
    write_json(metrics)
    write_prom(metrics)
    logger.info("Metrics exported successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    export_metrics()
