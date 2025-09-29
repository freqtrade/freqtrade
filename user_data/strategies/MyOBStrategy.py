from __future__ import annotations

from typing import Any

import pandas as pd

from freqtrade.strategy.interface import IStrategy


try:
    from freqtrade_ext.feature_store import load_orderbook_features
except Exception:  # pragma: no cover - allow strategy import without extras during tests

    def load_orderbook_features(*args, **kwargs):  # type: ignore
        return pd.DataFrame()


class MyOBStrategy(IStrategy):
    timeframe = "1m"
    can_short = True
    use_exit_signal = True

    FEAT_PREFIX = "feat__"
    FEAT_COLS = [
        "spread_bps",
        "microprice",
        "ob_imbalance",
        "ob_depth_delta",
        "ofi_top",
        "book_slope",
    ]

    def _timerange(self, df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (
            pd.to_datetime(df.index.min()).tz_convert("UTC"),
            pd.to_datetime(df.index.max()).tz_convert("UTC"),
        )

    def populate_indicators(self, df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
        pair = metadata.get("pair", "BTC/USDT:USDT")
        exid = getattr(getattr(self, "exchange", None), "id", "bybit")

        # Join external features
        try:
            feats = load_orderbook_features(
                exid, pair, self.timeframe, timerange=self._timerange(df), embargo_secs=1, depth=200
            )
        except Exception:
            feats = pd.DataFrame()

        if not feats.empty:
            df = df.join(feats, how="left")

        # Simple NA handling for runtime stability
        df = df.replace([float("inf"), float("-inf")], pd.NA).fillna(method="ffill").fillna(0)

        # Prefix feature columns for FreqAI
        for c in self.FEAT_COLS:
            if c in df.columns:
                df[f"{self.FEAT_PREFIX}{c}"] = df[c]

        # Regime filter example
        if "spread_bps" in df.columns:
            q_spread = df["spread_bps"].rolling(1440, min_periods=60).quantile(0.95)
            df["no_trade"] = (df["spread_bps"] > q_spread).astype(int).fillna(0)
        else:
            df["no_trade"] = 0

        return df

    def populate_buy_trend(self, df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
        df.loc[:, "buy"] = 0
        cond = (df["close"] > df["close"].rolling(20).mean()) & (df["no_trade"] == 0)
        df.loc[cond, "buy"] = 1
        return df

    def populate_sell_trend(self, df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
        df.loc[:, "sell"] = 0
        cond = (df["close"] < df["close"].rolling(20).mean()) & (df["no_trade"] == 0)
        df.loc[cond, "sell"] = 1
        return df

    # Futures/derivatives mode requires new-style entry/exit methods
    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
        out = self.populate_buy_trend(df.copy(), metadata)
        if "buy" not in out.columns:
            out["buy"] = 0
        # Map legacy buy -> enter_long. No short logic defined here.
        out["enter_long"] = out["buy"].astype(int)
        out["enter_short"] = 0
        return out

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
        out = self.populate_sell_trend(df.copy(), metadata)
        if "sell" not in out.columns:
            out["sell"] = 0
        # Map legacy sell -> exit_long. No short logic defined here.
        out["exit_long"] = out["sell"].astype(int)
        out["exit_short"] = 0
        return out

    # freqai_info is provided via config file (self.config["freqai"]).
