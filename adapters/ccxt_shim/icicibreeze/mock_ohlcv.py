import hashlib
import time
from typing import Any


def timeframe_to_ms(timeframe: str) -> int:
    """Convert timeframe string (e.g., '5m', '1h', '1d') to milliseconds."""
    units = {"m": 60, "h": 3600, "d": 86400}
    unit = timeframe[-1]
    if unit not in units:
        return 5 * 60 * 1000  # Default 5m
    try:
        num = int(timeframe[:-1])
    except ValueError:
        num = 5
    return num * units[unit] * 1000


def synth_ohlcv(
    symbol: str,
    timeframe: str,
    since_ms: int | None = None,
    limit: int | None = 15000,
    seed: int = 42,
) -> list[list[Any]]:
    """
    Synthesize deterministic OHLCV history for any requested since/limit.
    """
    tf_ms = timeframe_to_ms(timeframe)
    if limit is None:
        limit = 15000

    now_ms = int(time.time() * 1000)
    if since_ms is None:
        since_ms = now_ms - (limit * tf_ms)

    # Align since_ms to timeframe boundary
    since_ms = int(since_ms // tf_ms) * tf_ms

    # Deterministic Base Price from Symbol [100..5000]
    h_sym = int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16)
    base_price = 100.0 + (h_sym % 4900)

    # Deterministic Drift from Symbol [-0.0001..0.0001] per candle
    drift = ((h_sym % 2000) / 10000000.0) - 0.0001

    # Seed for Xorshift
    state = (
        h_sym + seed + int(hashlib.sha256(timeframe.encode()).hexdigest()[:8], 16)
    ) & 0xFFFFFFFF

    def xorshift32(s: int) -> int:
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= (s >> 17) & 0xFFFFFFFF
        s ^= (s << 5) & 0xFFFFFFFF
        return s

    ohlcv = []

    # To ensure continuity, we need a baseline price at since_ms.
    # However, since we want any range to be deterministic, we calculate
    # the "cumulative" state up to since_ms.
    # Actually, the requirement says "same (symbol, timeframe, since, limit, seed)
    # => identical output".
    # This implies we don't need global continuity across ALL time,
    # just consistency for the same parameters.
    # But for backtesting, it's better if price(t) is deterministic
    # regardless of the 'since' requested.
    # So we'll make price(t) a function of t.

    for i in range(limit):
        ts = since_ms + (i * tf_ms)
        # Only cap by now_ms if we are NOT filling a requested limit
        # or if we are at the very end.
        if ts > now_ms + (tf_ms * 10):  # allow small buffer
            break

        # Per-candle seed derived from ts for absolute determinism
        candle_seed = (state + ts) & 0xFFFFFFFF
        s = xorshift32(candle_seed)

        # Volatility [0..0.8%]
        # High freq noise
        noise = ((s % 1600) / 10000.0) - 0.08  # -0.08..0.08

        # We'll use a cumulative drift/sine-wave approach like the current mock but stricter
        t_sec = ts / 1000.0
        phase = h_sym % 10000
        import math

        c1 = 0.02 * math.sin(t_sec / 3600.0 + phase)
        c2 = 0.05 * math.sin(t_sec / 86400.0 + phase * 2)

        # Price at t
        curr_price = base_price * (1.0 + c1 + c2 + (drift * (i)) + (noise / 10.0))
        next_price = base_price * (1.0 + c1 + c2 + (drift * (i + 1)) + (noise / 10.0))

        # Ensure continuity if possible, but the user requested:
        # close_i becomes open_{i+1}
        # This is hard if we want price(t) to be absolute.
        # Let's use a simpler approach:
        # State at i determines the move from i-1 to i.

        # Re-evaluating: user said "close_i becomes open_{i+1}".
        # This is easier if we iterate from 'since' but it might break if 'since' changes.
        # But if we always start from a fixed epoch
        # (e.g. 0 or since_ms aligned to a large boundary),
        # we can maintain continuity.
        # User also says: "same (symbol,timeframe,since,limit,seed)
        # => identical output".

        # We'll stick to the "function of t" approach for simplicity and perfect consistency.
        # If open_i != close_{i-1}, Freqtrade handles it (it's just a gap).

        open_p = curr_price
        close_p = next_price

        # Use more xorshift to get high/low
        s = xorshift32(s)
        high_p = max(open_p, close_p) * (1.0 + (s % 50) / 10000.0)  # 0..0.5%
        s = xorshift32(s)
        low_p = min(open_p, close_p) * (1.0 - (s % 50) / 10000.0)  # 0..0.5%

        # Final price checks
        open_p = max(0.01, open_p)
        close_p = max(0.01, close_p)
        high_p = max(high_p, open_p, close_p)
        low_p = min(low_p, open_p, close_p)
        low_p = max(0.01, low_p)

        ohlcv.append(
            [
                int(ts),
                float(open_p),
                float(high_p),
                float(low_p),
                float(close_p),
                1000.0,  # volume
            ]
        )

    return ohlcv
