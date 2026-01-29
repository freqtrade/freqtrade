"""
Smart Money / Option Chain Analysis (FR-203)
Pure logic module for evaluating option chain snapshots.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StrikeRow:
    strike: float
    right: str  # "CE" or "PE"
    oi_change_pct: float
    volume: int
    ltp_change_pct: float
    iv_change_pct: float = 0.0


@dataclass
class OptionChainSnapshot:
    underlying: str
    ts_utc: str
    strikes: List[StrikeRow]


@dataclass
class SmartMoneyDecision:
    allow_trade: bool
    bias_strength: int = 0
    reasons: List[str] = field(default_factory=list)


class SmartMoneyEngine:
    # Configuration (Hardcoded or injected)
    MIN_OI_CHANGE_PCT = 10.0
    MIN_VOLUME = 100000

    @staticmethod
    def evaluate(snapshot: Optional[OptionChainSnapshot]) -> SmartMoneyDecision:
        """
        Evaluates the snapshot and returns a trading decision.
        If snapshot is None, returns safe default (Allow, but weak bias).
        """
        if snapshot is None:
            return SmartMoneyDecision(
                allow_trade=True, bias_strength=0, reasons=["NO_SNAPSHOT_BYPASS"]
            )

        reasons = []
        bias_score = 0
        allow = True

        # We aggregate metrics across relevant strikes (e.g., ATM/OTM).
        # For simplicity in this pure logic phase, we iterate ALL provided strikes
        # and enforce rules on the "best" or "average" or strictly "any" meeting criteria.
        # Requirement: "require_min_oi_change_pct: 10.0"
        # Let's assume we check if *significant* activity exists.

        # Implementation: Check if ANY strike meets the "Smart Money" criteria
        # If no strike meets criteria, block trade? Or just low score?
        # "reject_if_ltp_change_pct <= 0.0" -> implies looking for directional move validation.

        strong_strikes_count = 0

        for strike in snapshot.strikes:
            # 1. Volume Check
            if strike.volume < SmartMoneyEngine.MIN_VOLUME:
                continue

            # 2. LTP Decay Check (Must be positive for directional buying of options)
            # Assuming we are buying options here.
            if strike.ltp_change_pct <= 0:
                continue

            # 3. OI Change Check (Smart Money presence)
            if strike.oi_change_pct >= SmartMoneyEngine.MIN_OI_CHANGE_PCT:
                strong_strikes_count += 1
                bias_score += 10  # Arbitrary scoring per strong strike

        # Clamp Score
        bias_score = min(100, bias_score)

        # Decision Logic
        if strong_strikes_count == 0:
            allow = False
            reasons.append("NO_STRONG_STRIKES")
        else:
            reasons.append(f"FOUND_{strong_strikes_count}_STRONG_STRIKES")

        # Final Bias Gate
        # "final_allow_trade: allow_trade = (bias_strength >= 60) AND thresholds_met"
        # Since we just summed 10 points per strike, we need 6 strikes?
        # Or maybe simplified logic for this first pass.
        # Let's stick to the prompt requirement broadly but ensure testability.
        # Let's say we need at least 1 strong strike for now, and map bias to that.
        # Prompt says: "bias_strength >= 60".
        # If we have < 6 strong strikes, we block? That implies a very wide chain.
        # Let's adjust scoring: 20 points per strong strike -> need 3.

        adjusted_score = min(100, strong_strikes_count * 20)

        if adjusted_score < 60:
            allow = False
            reasons.append("LOW_BIAS_STRENGTH")

        return SmartMoneyDecision(allow_trade=allow, bias_strength=adjusted_score, reasons=reasons)
