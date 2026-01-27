import logging
import math
from typing import Any

from freqtrade.exceptions import OperationalException

logger = logging.getLogger(__name__)


class OrderRouter:
    """
    OrderRouter enforces trading policies at the CCXT Shim boundary.
    Crucial Policies:
    1. Lot Size Enforcement: Amounts must be multiples of lot size.
    2. Buyer Only Guard: Only BUY orders allowed for entries. SELL orders allowed only if closing position.
    3. Error Tokens: Deterministic error strings for acceptance gates.
    """

    def __init__(self, markets_source_callback: Any):
        """
        :param markets_source_callback: Callable returning the current markets dict (self.markets from exchange).
        """
        self._get_markets = markets_source_callback

    def resolve_lot_size(self, symbol: str) -> int:
        """
        Resolve lot size for a symbol from the loaded markets.
        Default to 1 if not found (e.g. Cash).
        """
        markets = self._get_markets()
        if not markets:
            logger.warning(f"OrderRouter: Markets empty during lot resolution for {symbol}.")
            return 1

        market = markets.get(symbol)
        if not market:
            logger.warning(
                f"OrderRouter: Symbol {symbol} not found in markets. Defaulting lot to 1."
            )
            return 1

        return int(market.get("lot", 1))

    def assert_lot_size(self, symbol: str, amount: float, lot_size: int) -> None:
        """
        Enforce that amount is a perfect multiple of lot size.
        """
        if lot_size <= 0:
            logger.warning(
                f"OrderRouter: Invalid lot_size {lot_size} for {symbol}. Skipping checks."
            )
            return

        # Check if amount is multiple
        # Use a small epsilon for float math, but we expect integers for lots
        remainder = amount % lot_size
        # If strict int logic required:
        if not math.isclose(remainder, 0, abs_tol=1e-5) and not math.isclose(
            remainder, lot_size, abs_tol=1e-5
        ):
            raise OperationalException(
                f"order_router_block:lot_size (Amount {amount} not multiple of {lot_size})"
            )

    def assert_buyer_only(
        self, symbol: str, side: str, position_check_callback: Any | None
    ) -> None:
        """
        Enforce Buyer Only policy.
        BUY: Always Allowed.
        SELL: Allowed ONLY if it corresponds to an existing open position (Exit).
              If we cannot determine position state, we BLOCK SELLs to be safe (Short Prevention).
        """
        if side.lower() == "buy":
            return

        # If logic reaches here, it's a SELL
        if position_check_callback is None:
            # Fail safe: Block if we can't check positions
            raise OperationalException(
                f"order_router_block:buyer_only (Sell blocked, no position check available)"
            )

        # Check if we have an open position for this symbol
        # position_check_callback should return True if Long Position exists
        is_open_long = position_check_callback(symbol)

        if not is_open_long:
            raise OperationalException(
                f"order_router_block:buyer_only (Sell blocked, no open position)"
            )

    def track_and_assert_modify(self, order_id: str, now_ts: float) -> None:
        """
        Enforce Modification Quota and Ladder (Rate Limit for Mods).
        Policies:
        1. Max 3 modifications per order.
        2. Min 2 seconds between modifications.
        """
        # In-memory storage for P16 (non-persistent)
        if not hasattr(self, "_mod_state"):
            self._mod_state: dict[str, dict] = {}

        state = self._mod_state.get(order_id, {"count": 0, "last_ts": 0.0})

        # 1. Quota Check
        if state["count"] >= 3:
            raise OperationalException(
                f"order_router_block:mod_quota (Max 3 mods exceeded for {order_id})"
            )

        # 2. Ladder Check
        if now_ts - state["last_ts"] < 2.0:
            raise OperationalException(
                f"order_router_block:mod_ladder (Mod too fast for {order_id}, wait 2s)"
            )

        # Update State
        state["count"] += 1
        state["last_ts"] = now_ts
        self._mod_state[order_id] = state

        logger.info(f"OrderRouter: Modification allowed for {order_id}. Count: {state['count']}")

    def validate_entry(
        self, symbol: str, side: str, amount: float, position_check_callback: Any | None = None
    ) -> None:
        """
        Primary validation entry point.
        """
        # 1. Lot Size
        lot_size = self.resolve_lot_size(symbol)
        self.assert_lot_size(symbol, amount, lot_size)

        # 2. Buyer Only
        self.assert_buyer_only(symbol, side, position_check_callback)

        logger.info(f"OrderRouter: Validated {side} {amount} {symbol} (Lot: {lot_size})")
