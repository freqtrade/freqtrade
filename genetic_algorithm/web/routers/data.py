"""
Data API — serves OHLCV candle data and available trading pairs.

Reads from freqtrade's downloaded data directory (user_data/data/{exchange}/).
Supports feather and JSON data formats.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/data", tags=["data"])

# Default data directory — freqtrade convention
_DATA_DIR = Path("user_data/data")

# Max candles per request to prevent huge responses
_MAX_CANDLES = 10_000


def _find_exchange_dirs() -> List[Path]:
    """Return all exchange subdirectories that contain data files."""
    if not _DATA_DIR.exists():
        return []
    return [
        d for d in _DATA_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and not d.name.endswith("_backup")
    ]


def _parse_pair_timeframe(filename: str) -> tuple[str, str] | None:
    """
    Extract (pair, timeframe) from a filename like 'BTC_USDT-5m.feather'.

    Returns None if the filename doesn't match the expected pattern.
    """
    m = re.match(r"^(.+)-(\d+\w+)\.(feather|json|json\.gz)$", filename)
    if not m:
        return None
    pair_raw = m.group(1)      # BTC_USDT
    timeframe = m.group(2)     # 5m
    # Convert underscore back to slash: BTC_USDT → BTC/USDT
    pair = pair_raw.replace("_", "/")
    return pair, timeframe


@router.get("/pairs")
async def list_pairs(exchange: Optional[str] = None):
    """
    List available trading pairs with their timeframes and exchanges.

    Returns:
        List of dicts with exchange, pair, timeframe, file_format.
    """
    results = []
    dirs = _find_exchange_dirs()

    for edir in dirs:
        if exchange and edir.name != exchange:
            continue
        for f in sorted(edir.iterdir()):
            if not f.is_file():
                continue
            parsed = _parse_pair_timeframe(f.name)
            if parsed:
                pair, tf = parsed
                fmt = "feather" if f.suffix == ".feather" else "json"
                results.append({
                    "exchange": edir.name,
                    "pair": pair,
                    "timeframe": tf,
                    "format": fmt,
                })

    # Also scan futures subdirectory
    for edir in dirs:
        if exchange and edir.name != exchange:
            continue
        futures_dir = edir / "futures"
        if futures_dir.exists():
            for f in sorted(futures_dir.iterdir()):
                if not f.is_file():
                    continue
                parsed = _parse_pair_timeframe(f.name)
                if parsed:
                    pair, tf = parsed
                    fmt = "feather" if f.suffix == ".feather" else "json"
                    results.append({
                        "exchange": edir.name,
                        "pair": pair,
                        "timeframe": tf,
                        "format": fmt,
                        "trading_mode": "futures",
                    })

    if not results:
        return {"pairs": [], "message": "No data files found. Download data with freqtrade first."}

    return {"pairs": results}


@router.get("/ohlcv")
async def get_ohlcv(
    pair: str = Query(..., description="Trading pair (e.g., BTC/USDT)"),
    timeframe: str = Query(..., description="Timeframe (e.g., 5m, 1h, 4h)"),
    exchange: str = Query("binance", description="Exchange name"),
    start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD or YYYYMMDD)"),
    end: Optional[str] = Query(None, description="End date (YYYY-MM-DD or YYYYMMDD)"),
    limit: int = Query(5000, description="Max candles to return", le=_MAX_CANDLES),
):
    """
    Return OHLCV candles for a trading pair.

    Returns array of candles: [timestamp_ms, open, high, low, close, volume].
    """
    # Build file path: BTC/USDT → BTC_USDT
    pair_file = pair.replace("/", "_")
    exchange_dir = _DATA_DIR / exchange

    if not exchange_dir.exists():
        raise HTTPException(404, f"Exchange directory not found: {exchange}")

    # Try feather first, then json
    feather_path = exchange_dir / f"{pair_file}-{timeframe}.feather"
    json_path = exchange_dir / f"{pair_file}-{timeframe}.json"
    json_gz_path = exchange_dir / f"{pair_file}-{timeframe}.json.gz"

    candles = None

    if feather_path.exists():
        candles = _load_feather(feather_path, start, end, limit)
    elif json_path.exists():
        candles = _load_json(json_path, start, end, limit)
    elif json_gz_path.exists():
        candles = _load_json(json_gz_path, start, end, limit)
    else:
        raise HTTPException(
            404,
            f"No data file found for {pair} {timeframe} on {exchange}. "
            f"Download data with: freqtrade download-data --pairs {pair} --timeframes {timeframe}",
        )

    return {
        "pair": pair,
        "timeframe": timeframe,
        "exchange": exchange,
        "count": len(candles),
        "candles": candles,
    }


def _load_feather(
    path: Path,
    start: str | None,
    end: str | None,
    limit: int,
) -> list[list]:
    """Load OHLCV data from a feather file."""
    try:
        import pandas as pd

        df = pd.read_feather(path)
        # Standard columns: date, open, high, low, close, volume
        df.columns = ["date", "open", "high", "low", "close", "volume"][:len(df.columns)]

        # Convert date to timestamp ms
        if pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = df["date"].astype("int64") // 10**6
        else:
            df["date"] = df["date"].astype("int64")
            # If dates are in seconds (< 1e12), convert to ms
            if len(df) > 0 and df["date"].iloc[0] < 1e12:
                df["date"] = df["date"] * 1000

        # Apply date filters
        if start:
            start_ts = _parse_date_to_ms(start)
            if start_ts:
                df = df[df["date"] >= start_ts]
        if end:
            end_ts = _parse_date_to_ms(end)
            if end_ts:
                df = df[df["date"] <= end_ts]

        # Limit
        df = df.tail(limit)

        return df[["date", "open", "high", "low", "close", "volume"]].values.tolist()

    except Exception as e:
        logger.exception("Failed to load feather %s", path)
        raise HTTPException(500, f"Failed to load data: {e}")


def _load_json(
    path: Path,
    start: str | None,
    end: str | None,
    limit: int,
) -> list[list]:
    """Load OHLCV data from a JSON file (possibly gzipped)."""
    try:
        import pandas as pd

        compression = "gzip" if path.suffix == ".gz" else None
        df = pd.read_json(path, orient="values", compression=compression)
        df.columns = ["date", "open", "high", "low", "close", "volume"][:len(df.columns)]

        # Dates in JSON are typically ms timestamps already
        if len(df) > 0 and df["date"].iloc[0] < 1e12:
            df["date"] = df["date"] * 1000

        if start:
            start_ts = _parse_date_to_ms(start)
            if start_ts:
                df = df[df["date"] >= start_ts]
        if end:
            end_ts = _parse_date_to_ms(end)
            if end_ts:
                df = df[df["date"] <= end_ts]

        df = df.tail(limit)
        return df[["date", "open", "high", "low", "close", "volume"]].values.tolist()

    except Exception as e:
        logger.exception("Failed to load json %s", path)
        raise HTTPException(500, f"Failed to load data: {e}")


def _parse_date_to_ms(date_str: str) -> int | None:
    """Parse a date string to millisecond timestamp."""
    import datetime

    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            dt = datetime.datetime.strptime(date_str, fmt).replace(
                tzinfo=datetime.timezone.utc
            )
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    return None


# ── Indicator Computation ──────────────────────────────────────


# Supported indicator types and their computation functions
_INDICATOR_REGISTRY: Dict[str, Any] = {}


def _ensure_indicator_registry():
    """Lazily build the indicator registry so we don't import pandas at module load."""
    if _INDICATOR_REGISTRY:
        return

    import pandas as pd
    import numpy as np

    def _ema(df: pd.DataFrame, period: int = 20, **_kw) -> dict:
        col = df["close"].ewm(span=period, adjust=False).mean()
        return {f"EMA_{period}": col}

    def _sma(df: pd.DataFrame, period: int = 20, **_kw) -> dict:
        col = df["close"].rolling(window=period).mean()
        return {f"SMA_{period}": col}

    def _rsi(df: pd.DataFrame, period: int = 14, **_kw) -> dict:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss
        rs = rs.replace([np.inf, -np.inf], np.nan)
        rsi = 100 - (100 / (1 + rs))
        # When loss=0 (pure uptrend), RSI=100; when gain=0 (pure downtrend), RSI=0
        rsi = rsi.where(loss > 0, 100.0)
        rsi = rsi.where(gain > 0, other=rsi.where(loss > 0, 100.0))
        return {f"RSI_{period}": rsi}

    def _macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, **_kw) -> dict:
        fast = df["close"].ewm(span=fast_period, adjust=False).mean()
        slow = df["close"].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast - slow
        signal = macd_line.ewm(span=signal_period, adjust=False).mean()
        hist = macd_line - signal
        return {f"MACD_{fast_period}_{slow_period}": macd_line, f"MACDSignal_{signal_period}": signal, f"MACDHist": hist}

    def _bbands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, **_kw) -> dict:
        sma = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return {f"BB_upper_{period}": upper, f"BB_middle_{period}": sma, f"BB_lower_{period}": lower}

    def _adx(df: pd.DataFrame, period: int = 14, **_kw) -> dict:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
        adx_val = dx.rolling(window=period).mean()
        return {f"ADX_{period}": adx_val}

    def _atr(df: pd.DataFrame, period: int = 14, **_kw) -> dict:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return {f"ATR_{period}": atr}

    def _stoch(df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3, **_kw) -> dict:
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        k = 100 * (df["close"] - low_min) / (high_max - low_min)
        k_smooth = k.rolling(window=smooth_k).mean()
        d = k_smooth.rolling(window=smooth_d).mean()
        return {f"STOCH_K_{period}": k_smooth, f"STOCH_D_{period}": d}

    def _cci(df: pd.DataFrame, period: int = 20, **_kw) -> dict:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: (x - x.mean()).abs().mean(), raw=True)
        cci = (tp - sma) / (0.015 * mad)
        return {f"CCI_{period}": cci}

    _INDICATOR_REGISTRY.update({
        "EMA": _ema,
        "SMA": _sma,
        "RSI": _rsi,
        "MACD": _macd,
        "BBANDS": _bbands,
        "ADX": _adx,
        "ATR": _atr,
        "STOCH": _stoch,
        "CCI": _cci,
    })


from typing import Dict


@router.get("/indicators")
async def get_indicators(
    pair: str = Query(..., description="Trading pair (e.g., BTC/USDT)"),
    timeframe: str = Query(..., description="Timeframe (e.g., 5m, 1h)"),
    exchange: str = Query("binance", description="Exchange name"),
    indicators: str = Query(..., description="Comma-separated indicator specs (e.g., EMA_20,RSI_14,BBANDS_20_2.0)"),
    start: Optional[str] = Query(None, description="Start date"),
    end: Optional[str] = Query(None, description="End date"),
    limit: int = Query(5000, description="Max data points", le=_MAX_CANDLES),
):
    """
    Compute technical indicator values on OHLCV data.

    Indicator format: TYPE_PARAM1_PARAM2 (e.g., EMA_20, RSI_14, BBANDS_20_2.0, MACD_12_26_9).
    
    Returns dict of indicator name → array of [timestamp_ms, value] pairs.
    """
    import pandas as pd
    import numpy as np

    _ensure_indicator_registry()

    # Load OHLCV data first
    pair_file = pair.replace("/", "_")
    exchange_dir = _DATA_DIR / exchange
    if not exchange_dir.exists():
        raise HTTPException(404, f"Exchange directory not found: {exchange}")

    feather_path = exchange_dir / f"{pair_file}-{timeframe}.feather"
    json_path = exchange_dir / f"{pair_file}-{timeframe}.json"
    json_gz_path = exchange_dir / f"{pair_file}-{timeframe}.json.gz"

    df = None
    for path in [feather_path, json_path, json_gz_path]:
        if path.exists():
            if path.suffix == ".feather":
                df = pd.read_feather(path)
            else:
                compression = "gzip" if path.suffix == ".gz" else None
                df = pd.read_json(path, orient="values", compression=compression)
            break

    if df is None:
        raise HTTPException(404, f"No data file found for {pair} {timeframe} on {exchange}")

    df.columns = ["date", "open", "high", "low", "close", "volume"][:len(df.columns)]

    # Convert date to timestamp ms
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = df["date"].astype("int64") // 10**6
    else:
        df["date"] = df["date"].astype("int64")
        if len(df) > 0 and df["date"].iloc[0] < 1e12:
            df["date"] = df["date"] * 1000

    # Apply date filters
    if start:
        start_ts = _parse_date_to_ms(start)
        if start_ts:
            df = df[df["date"] >= start_ts]
    if end:
        end_ts = _parse_date_to_ms(end)
        if end_ts:
            df = df[df["date"] <= end_ts]

    df = df.tail(limit).reset_index(drop=True)

    # Parse and compute each indicator
    result_dict: dict = {}
    for spec in indicators.split(","):
        spec = spec.strip()
        if not spec:
            continue
        parts = spec.split("_")
        ind_type = parts[0].upper()

        # Special case for multi-word types like STOCH
        if ind_type not in _INDICATOR_REGISTRY:
            # Try combining first two parts (e.g., for future types)
            if len(parts) >= 2 and f"{parts[0]}_{parts[1]}".upper() in _INDICATOR_REGISTRY:
                ind_type = f"{parts[0]}_{parts[1]}".upper()
                parts = [ind_type] + parts[2:]
            else:
                continue  # Unknown indicator, skip

        compute_fn = _INDICATOR_REGISTRY[ind_type]
        # Parse numeric params from the spec
        params = {}
        param_values = parts[1:]
        
        # Map positional params to named params based on indicator type
        param_map = {
            "EMA": ["period"],
            "SMA": ["period"],
            "RSI": ["period"],
            "MACD": ["fast_period", "slow_period", "signal_period"],
            "BBANDS": ["period", "std_dev"],
            "ADX": ["period"],
            "ATR": ["period"],
            "STOCH": ["period", "smooth_k", "smooth_d"],
            "CCI": ["period"],
        }
        names = param_map.get(ind_type, [])
        for i, val in enumerate(param_values):
            if i < len(names):
                try:
                    params[names[i]] = int(val) if "." not in val else float(val)
                except ValueError:
                    pass

        try:
            columns = compute_fn(df, **params)
            for col_name, series in columns.items():
                data_points = []
                for idx in range(len(df)):
                    v = series.iloc[idx]
                    if pd.notna(v) and np.isfinite(v):
                        data_points.append([int(df["date"].iloc[idx]), round(float(v), 6)])
                
                # Determine pane: oscillators go to separate pane
                pane = "price"
                if ind_type in ("RSI", "MACD", "STOCH", "CCI", "ADX"):
                    pane = "separate"
                # MACDHist and MACDSignal also go to separate
                if "MACD" in col_name:
                    pane = "separate"

                # Use dict format matching frontend expectations
                # Frontend accesses: resp.indicators[name].values / .pane
                result_dict[col_name] = {
                    "values": data_points,
                    "pane": pane,
                }
        except Exception as e:
            logger.warning("Failed to compute %s: %s", spec, e)

    return {
        "pair": pair,
        "timeframe": timeframe,
        "exchange": exchange,
        "indicators": result_dict,
    }
