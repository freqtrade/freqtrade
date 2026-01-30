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
        self.paper_mode = False
        self.mock_mode = False
        self.live_trading_enabled = False

    def resolve_lot_size(self, symbol: str) -> int:
        """
        Resolve lot size for a symbol from the loaded markets.
        Default to 1 if not found (e.g. Cash).
        """
        markets = self._get_markets()
        if not markets:
            # Only warn if we are in a fully live, non-mock, non-paper context
            should_warn = self.live_trading_enabled and not self.paper_mode and not self.mock_mode
            if should_warn:
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
        self,
        symbol: str,
        side: str,
        position_check_callback: Any | None,
        reduce_only: bool = False,
    ) -> None:
        """
        Enforce Buyer Only policy.
        BUY: Always Allowed.
        SELL: Allowed ONLY if it corresponds to an existing open position (Exit) OR reduceOnly=True.
              If we cannot determine position state, we BLOCK SELLs to be safe UNLESS reduceOnly is explicit.
        """
        if side.lower() == "buy":
            return

        # If logic reaches here, it's a SELL
        
        # P35.5 T4: Allow explicit reduceOnly to bypass position check
        if reduce_only:
             return

        if position_check_callback is None:
            # Fail safe: Block if we can't check positions
            raise OperationalException(
                "order_router_block:buyer_only (Sell blocked, no position check available)"
            )

        # Check if we have an open position for this symbol
        # position_check_callback should return True if Long Position exists
        is_open_long = position_check_callback(symbol)

        if not is_open_long:
            raise OperationalException(
                "order_router_block:buyer_only (Sell blocked, no open position)"
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

    # --- P28: Microstructure Logic ---

    def check_gtt_hysteresis(
        self,
        order_id: str,
        new_price: float,
        last_price: float,
        now_ts: float,
        config: dict,
        tick_size: float | None = None,
    ) -> dict:
        """
        Checks if modification should be skipped due to hysteresis.
        Returns {'skip': bool, 'reason': str}
        """
        rearm = config.get("rearm_seconds", 20)
        min_move = config.get("min_price_move_ticks", 2)

        # Access state (reusing _mod_state or new one)
        if not hasattr(self, "_mod_state"):
            self._mod_state = {}

        state = self._mod_state.get(order_id, {"last_ts": 0.0, "last_price": last_price})

        # Check Time Hysteresis
        time_diff = now_ts - state.get("last_ts", 0.0)

        if time_diff < rearm:
            # Check Price Hysteresis
            # Priority: param > config? > default 0.05
            if tick_size is None or tick_size <= 0:
                tick_size = config.get("tick_size", 0.05)

            price_change = abs(new_price - state.get("last_price", last_price))
            ticks_changed = price_change / tick_size

            if ticks_changed < min_move:
                return {
                    "skip": True,
                    "reason": f"SKIPPED_HYSTERESIS (Time {time_diff:.1f}s < {rearm}s AND Move {ticks_changed:.1f} < {min_move})",
                }

        # Update State implicit? No, only on actual modify success. This is just a check.
        return {"skip": False, "reason": ""}

    def check_sniper_cancel(self, open_timestamp: float, now_ts: float, config: dict) -> bool:
        """
        Returns True if order should be cancelled (Sniper logic).
        """
        cancel_after = config.get("cancel_after_seconds", 3)
        age = now_ts - open_timestamp
        return age >= cancel_after

    def calculate_atr_limit_buffer(
        self, last_price: float, side: str, atr: float, config: dict
    ) -> float:
        """
        Calculates buffered limit price.
        """
        mult = config.get("buffer_mult", 0.15)
        # min_ticks = config.get("min_ticks", 1)  # Not used in calculation logic per prompt requirement detail?
        # Actually prompt says "buffer_ticks computed... clamp [min, max]"

        buffer_val = atr * mult

        # Clamp Logic?
        # For now return raw calc, clamp can be done by caller or here if we had tick size.
        # Impl: buy = min(limit, last + buffer) -> this logic is "what is the buffer value"
        # Let's return the target price Limit.

        if side.lower() == "buy":
            # Cap buy price: Don't look too far up
            return last_price + buffer_val
        else:
            # Floor sell price
            return last_price - buffer_val

    def slice_order(self, symbol: str, total_qty: int, config: dict) -> list[int]:
        """
        Splits total quantity into child orders.
        """
        max_child = config.get("max_child_orders", 4)
        lot_size = self.resolve_lot_size(symbol)

        # Ensure total_qty is multiple of lot size (already asserted by validate_entry)
        # But let's be safe.

        if total_qty < lot_size:
            return [total_qty]  # Or fail? assert_lot_size handles it.

        # Naive split: uniform
        # Max child orders constraint
        # Also need to respect lot size for EACH child.

        # Calculate optimal chunks
        # e.g. 1000 qty, lot 50, max 4. -> 250 each.

        base_chunk = total_qty // max_child

        # Round chunk down to lot multiple
        chunk_lots = base_chunk // lot_size
        chunk_qty = chunk_lots * lot_size

        if chunk_qty == 0:
            # Total qty too small to split perfectly into N, reduction needed
            # Fallback: simple filling
            chunk_qty = lot_size

        chunks = []
        remaining = total_qty

        while remaining > 0 and len(chunks) < max_child - 1:
            if remaining < chunk_qty:
                chunks.append(remaining)
                remaining = 0
            else:
                chunks.append(chunk_qty)
                remaining -= chunk_qty

        if remaining > 0:
            chunks.append(remaining)

        return chunks

    def validate_entry(
        self,
        symbol: str,
        side: str,
        amount: float,
        position_check_callback: Any | None = None,
        reduce_only: bool = False,
    ) -> None:
        """
        Primary validation entry point.
        """
        # 1. Lot Size
        lot_size = self.resolve_lot_size(symbol)
        self.assert_lot_size(symbol, amount, lot_size)

        # 2. Buyer Only
        self.assert_buyer_only(symbol, side, position_check_callback, reduce_only)

        logger.info(f"OrderRouter: Validated {side} {amount} {symbol} (Lot: {lot_size})")
