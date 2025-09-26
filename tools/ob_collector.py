#!/usr/bin/env python3

"""
Orderbook collector: WebSocket -> 1s last aggregation -> 1-minute parquet batches
- ZSTD compression, partitioned by year/month/day
- Reconnect with heartbeat timeout and exponential backoff
- Fixed schema with schema_version
"""

import asyncio
import contextlib
import os
import secrets
from collections import deque
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


try:
    import ccxtpro
except Exception:  # pragma: no cover - handled gracefully for environments without ccxtpro
    ccxtpro = None

EXCHANGE = os.getenv("OB_EXCHANGE", "bybit")
PAIR = os.getenv("OB_PAIR", "BTC/USDT:USDT")
DEPTH = int(os.getenv("OB_DEPTH", "200"))
ROOT_DIR = os.getenv("OB_ROOT", "user_data/featurestore/bybit/BTCUSDT/1s")
SCHEMA_VER = int(os.getenv("OB_SCHEMA_VER", "1"))
HEARTBEAT_TOUT_SEC = int(os.getenv("OB_HEARTBEAT", "15"))
BACKOFF_BASE = float(os.getenv("OB_BACKOFF_BASE", "3.0"))
BATCH_SECONDS = int(os.getenv("OB_BATCH_SECONDS", "60"))


def utc_now_floor_s() -> datetime:
    return pd.Timestamp.utcnow().floor("S").to_pydatetime().replace(tzinfo=UTC)


def _minute_key(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M")


async def _write_minute_batch(df_minute: pd.DataFrame) -> None:
    if df_minute.empty:
        return
    out = df_minute.copy()
    out["year"] = out.index.year
    out["month"] = out.index.month
    out["day"] = out.index.day
    table = pa.Table.from_pandas(out, preserve_index=True)
    ds.write_dataset(
        data=table,
        base_dir=ROOT_DIR,
        format="parquet",
        partitioning=["year", "month", "day"],
        file_options=pq.ParquetFileWriteOptions(compression="zstd"),
        min_rows_per_group=60,
        max_rows_per_group=600,
        existing_data_behavior="create",
    )


async def run() -> None:
    if ccxtpro is None:
        raise RuntimeError("ccxtpro is not installed. Please install extra requirements.")

    ex = getattr(ccxtpro, EXCHANGE)({"enableRateLimit": True})
    buf: deque[dict[str, Any]] = deque()
    last_sec: datetime | None = None
    current_minute: str | None = None

    minute_df: pd.DataFrame | None = None

    while True:
        try:
            ob = await asyncio.wait_for(
                ex.watch_order_book(PAIR, limit=DEPTH), timeout=HEARTBEAT_TOUT_SEC
            )
            ts = utc_now_floor_s()
            bids, asks = ob.get("bids", [])[:DEPTH], ob.get("asks", [])[:DEPTH]
            if not bids or not asks:
                continue

            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            mid = (best_ask + best_bid) / 2.0
            bid_vol = float(sum(q for _, q in bids))
            ask_vol = float(sum(q for _, q in asks))
            imb = bid_vol / max(bid_vol + ask_vol, 1e-9)
            depth_delta = bid_vol - ask_vol

            buf.append(
                {
                    "ts": ts,
                    "schema_version": SCHEMA_VER,
                    "exchange": EXCHANGE,
                    "pair": PAIR,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": spread,
                    "mid": mid,
                    f"imb_{DEPTH}": imb,
                    f"depth_delta_{DEPTH}": depth_delta,
                    "top_bid_qty": float(bids[0][1]),
                    "top_ask_qty": float(asks[0][1]),
                }
            )

            if last_sec is None:
                last_sec = ts
                current_minute = _minute_key(ts)

            if ts > last_sec:
                sec_df = pd.DataFrame([x for x in buf if x["ts"] == last_sec])
                if not sec_df.empty:
                    out = sec_df.groupby("ts").last().sort_index()

                    minute_key = _minute_key(last_sec)
                    if minute_key != current_minute and minute_df is not None:
                        await _write_minute_batch(minute_df)
                        minute_df = None
                        current_minute = minute_key

                    if minute_df is None:
                        minute_df = out
                    else:
                        minute_df = pd.concat([minute_df, out], axis=0)

                # cleanup buffer
                buf = deque([x for x in buf if x["ts"] > last_sec])
                last_sec = ts

        except TimeoutError:
            # Heartbeat timeout - reconnect with exponential backoff
            with contextlib.suppress(Exception):
                await ex.close()
            backoff = BACKOFF_BASE + (secrets.randbelow(1000) / 1000.0) * 2.0
            await asyncio.sleep(backoff)
            ex = getattr(ccxtpro, EXCHANGE)({"enableRateLimit": True})
        except Exception as e:
            print(f"[collector] error: {e}")
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    asyncio.run(run())
