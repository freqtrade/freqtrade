# Phase 16: Order Router & Guardrails

## Overview

The Order Router (`adapters/ccxt_shim/order_router.py`) acts as a critical choke point for all order operations in the `BreezeCCXT` shim. It enforces strict trading policies to prevent invalid or risky orders from reaching the exchange or the mock adapter.

## Policies Enforced

### 1. Lot Size

- **Source**: `SecurityMaster` (via `BreezeCCXT.markets`)
- **Rule**: Order amount must be a perfect integer multiple of the contract's lot size.
- **Behavior**: Rejects non-compliant orders with `order_router_block:lot_size`.

### 2. Buyer Only Guard

- **Rule**:
  - **BUY**: Always allowed.
  - **SELL**: Allowed **ONLY** if an open position (Long) exists for the symbol.
- **Purpose**: Strictly prevents short selling or opening short positions.
- **Behavior**: Rejects invalid sells with `order_router_block:buyer_only`.

### 3. Modification Rules

- **Method**: `edit_order` (Simulated via Cancel/Replace)
- **Quota**: Max **3** modifications per order ID.
- **Ladder**: Min **2 seconds** between modifications for the same order.
- **Behavior**: Rejects excessive or rapid mods with `order_router_block:mod_quota` or `order_router_block:mod_ladder`.

## Acceptance Gates

### P16 (Positive)

- **Command**: `bash scripts/accept_all.sh p16_order_router`
- **Behavior**: Places a valid BUY order and cancels it. Expects Success.

### P16 (Negative)

- **Command**: `bash scripts/accept_all.sh --neg p16_order_router`
- **Behavior**: Attempts a SELL order without an open position.
- **Expectation**: Gate PASSES if the Order Router **BLOCKS** the sell with `order_router_block:buyer_only`.

## Implementation Details

- **Class**: `OrderRouter`
- **Integration**: `BreezeCCXT` initializes `OrderRouter` and calls `validate_entry` in `create_order` and `track_and_assert_modify` in `edit_order`.
