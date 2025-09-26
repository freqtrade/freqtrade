from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Lightweight ATR calculation from OHLC without external deps.
    Returns the latest ATR value, or np.nan if unavailable.
    """
    for c in ("high", "low", "close"):
        if c not in df.columns or len(df[c]) == 0:
            return float("nan")
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = np.maximum.reduce(
        [
            (high - low).abs().values,
            (high - prev_close).abs().values,
            (low - prev_close).abs().values,
        ]
    )
    tr_s = pd.Series(tr, index=df.index)
    atr = tr_s.rolling(period, min_periods=max(2, period // 2)).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else float("nan")


@dataclass
class VolTargetConfig:
    enabled: bool = False
    mode: str = "unit_atr"  # "unit_atr" | "ewma_vol" (reserved)
    risk_pct_per_trade: float = 0.002  # 0.2% of available balance by default
    atr_period: int = 14
    atr_k: float = 1.5
    min_stake: float | None = None
    max_stake: float | None = None
    max_leverage: float | None = None


class VolatilityTargetSizer:
    """
    Derives stake (margin) and leverage to target a constant risk using ATR.
    - Works in spot (leverage=1) and futures (leverage>1) modes.
    - Returns safe fallbacks when inputs are incomplete.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.cfg = VolTargetConfig(
            enabled=bool(cfg.get("enabled", False)),
            mode=str(cfg.get("mode", "unit_atr")),
            risk_pct_per_trade=float(cfg.get("risk_pct_per_trade", 0.002)),
            atr_period=int(cfg.get("atr_period", 14)),
            atr_k=float(cfg.get("atr_k", 1.5)),
            min_stake=cfg.get("min_stake"),
            max_stake=cfg.get("max_stake"),
            max_leverage=cfg.get("max_leverage"),
        )

    # ---- public API ----------------------------------------------------
    def suggest_stake(
        self,
        *,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        side: str,
        ohlcv: pd.DataFrame | None,
        edge_score: float | None = None,
    ) -> float:
        if not self.cfg.enabled:
            return proposed_stake

        lev = max(1.0, float(leverage or 1.0))
        # Use ATR as stop distance proxy.
        atr = _atr(ohlcv, self.cfg.atr_period) if ohlcv is not None else float("nan")
        if not atr or not np.isfinite(atr) or atr <= 0 or not current_rate or current_rate <= 0:
            return self._clamp_stake(proposed_stake, min_stake, max_stake)

        stop_distance = self.cfg.atr_k * atr
        # Risk budget in stake currency (quote): use available balance as equity proxy (max_stake)
        risk_budget = max_stake * float(self.cfg.risk_pct_per_trade)
        if risk_budget <= 0:
            return self._clamp_stake(proposed_stake, min_stake, max_stake)

        # Position sizing with fixed-R using ATR stop.
        # units = risk_budget / stop_distance
        # notional = units * price = risk_budget * price / stop_distance
        notional = (risk_budget * float(current_rate)) / float(stop_distance)
        # Convert notional to margin stake by dividing leverage in futures.
        stake_margin = notional / lev
        stake_margin = self._apply_edge_factor(stake_margin, edge_score)
        return self._clamp_stake(stake_margin, min_stake, max_stake)

    def suggest_leverage(
        self,
        *,
        proposed_leverage: float,
        max_leverage: float,
        current_rate: float,
        max_stake: float,
        min_stake: float | None,
        ohlcv: pd.DataFrame | None,
    ) -> float:
        if not self.cfg.enabled:
            return proposed_leverage

        max_lev = float(self.cfg.max_leverage or max_leverage or 1.0)
        max_lev = max(1.0, min(max_lev, max_leverage))

        # Optional leverage auto-scaling: increase leverage when risk-based notional would exceed
        # available margin. Keep simple and conservative.
        atr = _atr(ohlcv, self.cfg.atr_period) if ohlcv is not None else float("nan")
        if not atr or not np.isfinite(atr) or atr <= 0 or not current_rate or current_rate <= 0:
            return max(1.0, min(max_lev, float(proposed_leverage or 1.0)))

        stop_distance = self.cfg.atr_k * atr
        risk_budget = max_stake * float(self.cfg.risk_pct_per_trade)
        if risk_budget <= 0:
            return max(1.0, min(max_lev, float(proposed_leverage or 1.0)))

        notional = (risk_budget * float(current_rate)) / float(stop_distance)
        # If the margin requirement (notional/leverage) would be larger than available funds,
        # increase leverage (within constraints) to fit the margin to max_stake.
        # leverage_needed = notional / max_stake
        lev_needed = (notional / max(max_stake, 1e-9)) if max_stake else 1.0
        lev = max(1.0, min(max_lev, lev_needed))
        # Fall back to proposed if it's higher than needed (more conservative on margin use)
        if proposed_leverage:
            lev = (
                max(1.0, min(max_lev, float(proposed_leverage))) if proposed_leverage > lev else lev
            )
        return lev

    # ---- helpers -------------------------------------------------------
    def _clamp_stake(self, stake: float, min_stake: float | None, max_stake: float) -> float:
        lo = float(self.cfg.min_stake) if self.cfg.min_stake is not None else (min_stake or 0.0)
        hi = float(self.cfg.max_stake) if self.cfg.max_stake is not None else float(max_stake)
        return float(max(lo, min(hi, float(stake))))

    def _apply_edge_factor(self, stake: float, edge_score: float | None) -> float:
        """Optionally scale stake by signal strength.
        Configuration (optional, under vol_target):
          edge_enabled: bool
          edge_scale: float  (score at which factor hits max)
          edge_min: float   (floor multiplier)
          edge_max: float   (cap multiplier)
        edge_score is assumed >= 0 (excess over threshold)."""
        try:
            # self.cfg is available as dataclass; keep 'raw' for optional extras
            # Read raw dict too for extra keys
            raw = self.__dict__.get("cfg_raw") if hasattr(self, "cfg_raw") else None
        except Exception:
            raw = None

        # Recover edge settings from original init dict if present
        edge = getattr(self, "_edge_cfg", None)
        if edge is None:
            edge = {
                "enabled": bool((getattr(self, "cfg_raw", {}) or {}).get("edge_enabled", False))
                if raw is not None
                else False,
                "scale": float((getattr(self, "cfg_raw", {}) or {}).get("edge_scale", 0.002))
                if raw is not None
                else 0.002,
                "min": float((getattr(self, "cfg_raw", {}) or {}).get("edge_min", 0.75))
                if raw is not None
                else 0.75,
                "max": float((getattr(self, "cfg_raw", {}) or {}).get("edge_max", 1.5))
                if raw is not None
                else 1.5,
            }
            self._edge_cfg = edge

        if not edge.get("enabled", False) or edge_score is None or not np.isfinite(edge_score):
            return stake
        scale = max(1e-9, float(edge.get("scale", 0.002)))
        fmin = float(edge.get("min", 0.75))
        fmax = float(edge.get("max", 1.5))
        factor = fmin + float(edge_score) / scale
        factor = float(max(fmin, min(fmax, factor)))
        return float(stake) * factor
