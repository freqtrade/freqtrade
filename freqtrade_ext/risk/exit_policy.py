from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PartialTpStep:
    profit_pct: float
    reduce_pct: float
    min_hold_candles: int = 0


@dataclass
class ExitPolicyConfig:
    enabled: bool = False
    time_stop_candles: int | None = None
    partial_tps: list[PartialTpStep] = field(default_factory=list)
    trail_from_profit_pct: float | None = None
    trail_step_pct: float = 0.003
    hard_stop_dd_pct: float | None = None


@dataclass
class _TradeState:
    steps_hit: set[int] = field(default_factory=set)
    max_profit_seen: float = 0.0


class ExitPolicy:
    """
    Time-stop and partial take-profit policy for any strategy.
    Keeps minimal in-memory state per trade.id for hit steps and max-profit trailing.
    """

    def __init__(self, cfg: dict[str, Any] | None = None, timeframe_minutes: int = 1) -> None:
        cfg = cfg or {}
        steps_raw = cfg.get("partial_tps", []) or []
        steps: list[PartialTpStep] = []
        for s in steps_raw:
            profit = s.get("profit_pct")
            reduce = s.get("reduce_pct")
            if profit is None or reduce is None:
                continue
            try:
                steps.append(
                    PartialTpStep(
                        profit_pct=float(profit),
                        reduce_pct=float(reduce),
                        min_hold_candles=int(s.get("min_hold_candles", 0)),
                    )
                )
            except Exception as e:
                print(f"[exit_policy] skip invalid partial_tp step: {e}")
                continue
        tsp = cfg.get("time_stop_candles")
        tfp = cfg.get("trail_from_profit_pct")
        hdd = cfg.get("hard_stop_dd_pct")
        self.cfg = ExitPolicyConfig(
            enabled=bool(cfg.get("enabled", False)),
            time_stop_candles=int(tsp) if tsp is not None else None,
            partial_tps=sorted(steps, key=lambda x: x.profit_pct),
            trail_from_profit_pct=float(tfp) if tfp is not None else None,
            trail_step_pct=float(cfg.get("trail_step_pct", 0.003)),
            hard_stop_dd_pct=float(hdd) if hdd is not None else None,
        )
        self.timeframe_minutes = int(timeframe_minutes or 1)
        self._state: dict[int, _TradeState] = {}

    # ---- public API ----------------------------------------------------
    def evaluate_custom_exit(
        self,
        *,
        trade_id: int,
        trade_open_time: datetime,
        current_time: datetime,
        current_profit: float,
    ) -> str | None:
        if not self.cfg.enabled:
            return None

        st = self._state.setdefault(trade_id, _TradeState())
        # Update trailing max profit
        if current_profit is not None:
            st.max_profit_seen = max(float(st.max_profit_seen), float(current_profit))

        # Time stop
        if self.cfg.time_stop_candles is not None and self.cfg.time_stop_candles >= 1:
            elapsed = max(
                0,
                int(
                    (current_time - trade_open_time).total_seconds()
                    // (self.timeframe_minutes * 60)
                ),
            )
            if elapsed >= int(self.cfg.time_stop_candles):
                return "time_stop"

        # Hard drawdown from peak (realized on current_profit metric)
        if self.cfg.hard_stop_dd_pct is not None and self.cfg.hard_stop_dd_pct > 0:
            dd = float(st.max_profit_seen) - float(current_profit or 0.0)
            if dd >= float(self.cfg.hard_stop_dd_pct):
                return "hard_dd_stop"

        # Trailing from profit: if we had profit above threshold, and now retraces by trail_step
        if (
            self.cfg.trail_from_profit_pct is not None
            and st.max_profit_seen >= self.cfg.trail_from_profit_pct
        ):
            if (st.max_profit_seen - float(current_profit or 0.0)) >= float(
                self.cfg.trail_step_pct
            ):
                return "trail_take"

        return None

    def evaluate_adjustment(
        self,
        *,
        trade_id: int,
        trade_open_time: datetime,
        current_time: datetime,
        current_profit: float,
        current_stake_amount: float,
    ) -> float | None:
        if not self.cfg.enabled:
            return None

        st = self._state.setdefault(trade_id, _TradeState())

        # Evaluate partial take profit steps
        for i, step in enumerate(self.cfg.partial_tps):
            if i in st.steps_hit:
                continue

            # Optional minimal holding period before this step can trigger
            if step.min_hold_candles and step.min_hold_candles > 0:
                elapsed = max(
                    0,
                    int(
                        (current_time - trade_open_time).total_seconds()
                        // (self.timeframe_minutes * 60)
                    ),
                )
                if elapsed < int(step.min_hold_candles):
                    continue

            if current_profit is not None and float(current_profit) >= float(step.profit_pct):
                # Mark step hit and request a reduction of stake
                st.steps_hit.add(i)
                reduce_amt = -abs(float(current_stake_amount) * float(step.reduce_pct))
                return reduce_amt if reduce_amt != 0.0 else None

        return None
