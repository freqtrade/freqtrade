from __future__ import annotations


def fill_price(
    mid: float,
    spread: float,
    side: str,
    qty: float,
    top_depth: float,
    rel_fee_bp: float,
    funding_bp: float = 0.0,
) -> tuple[float, float]:
    """
        Simple yet practical execution approximation:
            base  = mid ± 0.5*spread
            impact= (qty / top_depth)^gamma * (spread/mid) * k * mid
      fee   = px * rel_fee_bp * 1e-4
      funding = px * funding_bp * 1e-4
    Returns: (exec_price, total_cost)
    """
    rel_spread = spread / max(mid, 1e-9)
    k, gamma = 1.0, 0.7
    impact = (qty / max(top_depth, 1e-9)) ** gamma * rel_spread * k * mid
    base = mid + (0.5 if side.lower() == "buy" else -0.5) * spread
    px = base + (impact if side.lower() == "buy" else -impact)
    fee = px * rel_fee_bp * 1e-4
    funding = px * funding_bp * 1e-4
    return px, fee + funding


def limit_fill(best_bid: float, best_ask: float, price: float, side: str) -> bool:
    """
    Basic limit order fill condition.
    buy: price >= best_ask, sell: price <= best_bid
    """
    if side.lower() == "buy":
        return price >= best_ask
    return price <= best_bid
