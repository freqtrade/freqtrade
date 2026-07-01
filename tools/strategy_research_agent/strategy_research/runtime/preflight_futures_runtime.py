#!/usr/bin/env python3
"""Preflight checks for Binance USDT-M futures dry-run/live runtime.

This script intentionally uses public endpoints only. It verifies that the
same ccxt async network stack used by Freqtrade can reach Binance futures
through the current process environment, including VPN/proxy variables.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import ccxt.async_support as ccxt


async def _check_binance_futures(pair: str, timeframe: str, timeout_ms: int) -> int:
    exchange = ccxt.binanceusdm(
        {
            "enableRateLimit": True,
            "timeout": timeout_ms,
            "aiohttp_trust_env": True,
            "options": {"defaultType": "future"},
        }
    )
    try:
        started = time.time()
        await exchange.fetch_time()
        candles = await exchange.fetch_ohlcv(pair, timeframe, limit=2)
        elapsed = time.time() - started
        if len(candles) < 2:
            print(
                f"preflight failed: Binance futures returned {len(candles)} candles "
                f"for {pair} {timeframe}",
                file=sys.stderr,
            )
            return 1
        print(
            f"preflight ok: Binance futures ccxt fetch succeeded for "
            f"{pair} {timeframe} in {elapsed:.2f}s"
        )
        return 0
    except Exception as exc:
        print(f"preflight failed: {type(exc).__name__}: {str(exc)[:240]}", file=sys.stderr)
        return 1
    finally:
        await exchange.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="BTC/USDT:USDT")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(_check_binance_futures(args.pair, args.timeframe, args.timeout_ms))


if __name__ == "__main__":
    raise SystemExit(main())
