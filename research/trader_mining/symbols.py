# research/trader_mining/symbols.py
"""Shared symbol parsing used by both fee-currency classification (engine.py) and
ledger-event reconciliation (ledger.py) -- kept in one place so the two don't drift on
what "base asset" means for a given trading symbol."""

from __future__ import annotations


def base_asset_of(symbol: str) -> str | None:
    """The base asset ticker parsed from the part of `symbol` before "/" (e.g. "HYPE"
    from "HYPE/USDC" or "BTC" from "BTC/USDC:USDC"). None for the handful of unparsable
    raw internal Hyperliquid index symbols observed in real data (e.g. "@705", no "/")."""
    return symbol.split("/")[0] if "/" in symbol else None
