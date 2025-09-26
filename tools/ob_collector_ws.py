#!/usr/bin/env python3

"""
Bybit v5 WebSocket orderbook collector (free stack)
- Connects to Bybit public WS (linear/spot) and subscribes to orderbook.<DEPTH>.<SYMBOL>
- Initializes with REST snapshot, then applies WS deltas using pu/u continuity check
- Aggregates to per-second "last" records and writes 1-minute parquet batches
- ZSTD compression, partitioned by year/month/day, with schema_version column
"""

import asyncio
import os
import secrets
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import aiohttp
import httpx
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


EXCHANGE = os.getenv("EXCHANGE", "bybit")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
DEPTH = int(os.getenv("DEPTH", "200"))
CATEGORY = os.getenv("CATEGORY", "linear")  # "linear" or "spot"
ROOT_DIR = os.getenv("ROOT_DIR", f"user_data/featurestore/{EXCHANGE}/{SYMBOL}/1s")
# Auto-pick WS url if not provided
_default_ws = {
    "linear": "wss://stream.bybit.com/v5/public/linear",
    "spot": "wss://stream.bybit.com/v5/public/spot",
}
WS_URL = os.getenv("WS_URL", _default_ws.get(CATEGORY, _default_ws["linear"]))
REST_URL = os.getenv("REST_URL", "https://api.bybit.com")
HEARTBEAT_TIMEOUT = int(os.getenv("HEARTBEAT_TIMEOUT", "15"))
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "3.0"))
SCHEMA_VER = 1
PING_INTERVAL = int(os.getenv("PING_INTERVAL", "10"))
RUN_SECONDS = int(os.getenv("RUN_SECONDS", "0"))  # 0=run forever


@dataclass
class OBState:
    bids: dict[float, float]
    asks: dict[float, float]
    update_id: int | None


class OrderBookBuilder:
    def __init__(self, depth: int = 200):
        self.depth = depth
        self.state = OBState(bids={}, asks={}, update_id=None)
        self.needs_resync = True

    def apply_snapshot(self, bids: list[list[Any]], asks: list[list[Any]], update_id: int) -> None:
        def _val(x, idx):
            # handle both list and dict entries
            if isinstance(x, dict):
                return x.get("price") if idx == 0 else x.get("size")
            return x[idx]

        self.state.bids = {
            float(_val(e, 0)): float(_val(e, 1)) for e in bids if float(_val(e, 1) or 0) > 0
        }
        self.state.asks = {
            float(_val(e, 0)): float(_val(e, 1)) for e in asks if float(_val(e, 1) or 0) > 0
        }
        self.state.update_id = int(update_id)
        self.needs_resync = False

    def apply_delta(
        self,
        bids: list[list[Any]] | None,
        asks: list[list[Any]] | None,
        update_id: int,
        prev_update_id: int | None = None,
    ) -> None:
        # Continuity: Prefer pu-based continuity when provided
        if self.state.update_id is not None:
            if prev_update_id is not None:
                if int(prev_update_id) != int(self.state.update_id):
                    self.needs_resync = True
                    return
            else:
                # Fallback to +1 continuity if pu not provided
                if int(update_id) != int(self.state.update_id) + 1:
                    self.needs_resync = True
                    return

        def _apply(side: str, entries: list[list[Any]]):
            book = self.state.bids if side == "bids" else self.state.asks
            for e in entries:
                if isinstance(e, dict):
                    price = float(e.get("price"))
                    size = float(e.get("size"))
                    action = e.get("action")
                else:
                    price = float(e[0])
                    size = float(e[1])
                    action = e[2] if len(e) > 2 else None
                if action == "d" or size == 0.0:
                    book.pop(price, None)
                else:
                    book[price] = size

        if bids:
            _apply("bids", bids)
        if asks:
            _apply("asks", asks)

        self.state.update_id = int(update_id)
        self.needs_resync = False

    def top_stats(self) -> tuple[float, float, float, float, float, float, float, float]:
        bids_sorted = sorted(self.state.bids.items(), key=lambda x: x[0], reverse=True)[
            : self.depth
        ]
        asks_sorted = sorted(self.state.asks.items(), key=lambda x: x[0])[: self.depth]
        if not bids_sorted or not asks_sorted:
            raise ValueError("Empty orderbook levels")
        best_bid, top_bid_qty = bids_sorted[0][0], bids_sorted[0][1]
        best_ask, top_ask_qty = asks_sorted[0][0], asks_sorted[0][1]
        spread = best_ask - best_bid
        mid = (best_ask + best_bid) / 2.0
        bid_vol = float(sum(q for _, q in bids_sorted))
        ask_vol = float(sum(q for _, q in asks_sorted))
        return best_bid, best_ask, spread, mid, bid_vol, ask_vol, top_bid_qty, top_ask_qty


async def fetch_snapshot(symbol: str, depth: int) -> tuple[list[list[Any]], list[list[Any]], int]:
    url = f"{REST_URL}/v5/market/orderbook"
    params = {"category": CATEGORY, "symbol": symbol, "limit": str(depth)}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    res = data.get("result", {})
    asks = res.get("a", [])
    bids = res.get("b", [])
    update_id = int(res.get("u") or res.get("updateId") or 0)
    if update_id == 0:
        raise RuntimeError("Snapshot missing update id")
    return bids, asks, update_id


def utc_now_floor_s() -> datetime:
    return pd.Timestamp.utcnow().floor("s").to_pydatetime().replace(tzinfo=UTC)


def _minute_key(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M")


async def _write_minute_batch(df_minute: pd.DataFrame) -> None:
    if df_minute is None or df_minute.empty:
        return
    out = df_minute.copy()
    out["year"] = out.index.year
    out["month"] = out.index.month
    out["day"] = out.index.day
    table = pa.Table.from_pandas(out, preserve_index=True)

    # Build write options compatible across pyarrow versions
    fmt = ds.ParquetFileFormat()
    try:
        write_opts = fmt.make_write_options(compression="zstd")
    except Exception:
        try:
            write_opts = fmt.make_write_options(compression="snappy")
        except Exception:
            write_opts = fmt.make_write_options()

    # Unique-ish basename per minute to avoid collisions across runs
    try:
        minute_str = pd.to_datetime(out.index.min()).strftime("%Y%m%d%H%M")
    except Exception:
        minute_str = "unknown"
    basename = f"ob_{minute_str}_{secrets.token_hex(2)}_{{i}}.parquet"

    ds.write_dataset(
        data=table,
        base_dir=ROOT_DIR,
        format=fmt,
        partitioning=["year", "month", "day"],
        file_options=write_opts,
        min_rows_per_group=60,
        max_rows_per_group=600,
        existing_data_behavior="overwrite_or_ignore",
        basename_template=basename,
    )


async def run() -> None:  # noqa: C901
    # TODO: Refactor per docs/TODO_PRECOMMIT_AND_WS_OB_AUTOMATION.md
    builder = OrderBookBuilder(depth=DEPTH)

    async def resync() -> None:
        bids, asks, u = await fetch_snapshot(SYMBOL, DEPTH)
        builder.apply_snapshot(bids=bids, asks=asks, update_id=u)

    await resync()

    buf: deque[dict[str, Any]] = deque()
    last_sec: datetime | None = None
    current_minute: str | None = None
    minute_df: pd.DataFrame | None = None

    subscribe_msg = {"op": "subscribe", "args": [f"orderbook.{DEPTH}.{SYMBOL}"]}

    loop = asyncio.get_running_loop()
    deadline = loop.time() + RUN_SECONDS if RUN_SECONDS > 0 else None

    async with aiohttp.ClientSession() as session:
        while True:
            ping_task: asyncio.Task | None = None
            try:
                if deadline is not None and loop.time() >= deadline:
                    # Flush pending minute
                    if minute_df is not None and not minute_df.empty:
                        await _write_minute_batch(minute_df)
                    return

                async with session.ws_connect(
                    WS_URL, heartbeat=None, timeout=HEARTBEAT_TIMEOUT
                ) as ws:
                    await ws.send_bytes(orjson.dumps(subscribe_msg))

                    async def _pinger():
                        while True:
                            try:
                                await asyncio.sleep(PING_INTERVAL)
                                await ws.send_bytes(orjson.dumps({"op": "ping"}))
                            except Exception:
                                break

                    ping_task = asyncio.create_task(_pinger())

                    while True:
                        if deadline is not None and loop.time() >= deadline:
                            if ping_task is not None and not ping_task.done():
                                ping_task.cancel()
                            if minute_df is not None and not minute_df.empty:
                                await _write_minute_batch(minute_df)
                            return

                        msg = await asyncio.wait_for(ws.receive(), timeout=HEARTBEAT_TIMEOUT)
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            payload = orjson.loads(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            payload = orjson.loads(msg.data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            raise TimeoutError("WS closed/error")
                        else:
                            continue

                        # Bybit ping/pong
                        if payload.get("op") == "pong":
                            continue

                        topic = payload.get("topic", "")
                        if not topic.startswith("orderbook"):
                            continue

                        dtype = payload.get("type")
                        data = payload.get("data", {})
                        bids = data.get("b") or data.get("bid") or []
                        asks = data.get("a") or data.get("ask") or []
                        u = int(data.get("u") or 0)
                        pu = data.get("pu")
                        pu = int(pu) if pu is not None else None

                        if dtype == "snapshot":
                            builder.apply_snapshot(bids=bids, asks=asks, update_id=u)
                        elif dtype in ("delta", "update"):
                            builder.apply_delta(
                                bids=bids, asks=asks, update_id=u, prev_update_id=pu
                            )
                            if builder.needs_resync:
                                await resync()
                                continue
                        else:
                            continue

                        # Build per-second last record
                        ts = utc_now_floor_s()
                        try:
                            (
                                best_bid,
                                best_ask,
                                spread,
                                mid,
                                bid_vol,
                                ask_vol,
                                top_bid_qty,
                                top_ask_qty,
                            ) = builder.top_stats()
                        except Exception as e:
                            print(f"[collector_ws] top_stats error: {e}")
                            continue

                        imb = bid_vol / max(bid_vol + ask_vol, 1e-9)
                        depth_delta = bid_vol - ask_vol

                        buf.append(
                            {
                                "ts": ts,
                                "schema_version": SCHEMA_VER,
                                "exchange": EXCHANGE,
                                "pair": SYMBOL,
                                "best_bid": best_bid,
                                "best_ask": best_ask,
                                "spread": spread,
                                "mid": mid,
                                f"imb_{DEPTH}": imb,
                                f"depth_delta_{DEPTH}": depth_delta,
                                "top_bid_qty": float(top_bid_qty),
                                "top_ask_qty": float(top_ask_qty),
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

                                minute_df = (
                                    out
                                    if minute_df is None
                                    else pd.concat([minute_df, out], axis=0)
                                )

                            buf = deque([x for x in buf if x["ts"] > last_sec])
                            last_sec = ts

            except TimeoutError:
                jitter = secrets.randbelow(1000) / 1000.0
                await asyncio.sleep(BACKOFF_BASE + jitter * 2.0)
                continue
            except Exception as e:
                print(f"[collector_ws] error: {e}")
                await asyncio.sleep(1.0)
                continue
            finally:
                if ping_task is not None and not ping_task.done():
                    ping_task.cancel()


if __name__ == "__main__":
    asyncio.run(run())
