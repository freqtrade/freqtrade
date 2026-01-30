"""
Policy Codes for BreezeCCXT Shim.
Defines standardized blocking codes that should NOT trigger Degraded Mode.
"""


class PolicyCode:
    LIVE_BLOCKED = "Live Trading Guard: Blocked"
    MARKET_CLOSED = "market_closed"
    RISK_BLOCK = "risk_block"
    DEGRADED_BLOCK = "degraded_block"

    # Legacy/Fallback
    LIVE_BLOCKED_UC = "LIVE_BLOCKED"


# Set of codes that represent intentional safety blocks, not system failures.
SAFETY_BLOCKS = {
    PolicyCode.LIVE_BLOCKED,
    PolicyCode.MARKET_CLOSED,
    PolicyCode.RISK_BLOCK,
    PolicyCode.DEGRADED_BLOCK,
    PolicyCode.LIVE_BLOCKED_UC,
}


def is_safety_block(message: str) -> bool:
    """
    Check if an exception message corresponds to a safety block.
    """
    for code in SAFETY_BLOCKS:
        if code in message:
            return True
    return False
