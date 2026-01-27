# P16 Audit Report: Context & Gap Analysis

## 1. Choke Points (Order Execution Surface)

The following methods in `adapters/ccxt_shim/breeze_ccxt.py` form the critical path for all order operations. The Order Router must intercept these.

| Method | Status | Path:Line | Notes |
| :--- | :--- | :--- | :--- |
| `create_order` | **EXISTING** | `adapters/ccxt_shim/breeze_ccxt.py:554` | Primary entry point (Sync). Async wraps this. |
| `cancel_order` | **EXISTING** | `adapters/ccxt_shim/breeze_ccxt.py:610` | Cancellation entry point. |
| `edit_order` | **MISSING** | N/A | **GAP**: Must be implemented to support Cancel/Replace or Native Modify. |
| `fetch_order` | **EXISTING** | `adapters/ccxt_shim/breeze_ccxt.py:620` | Read path. |
| `fetch_open_orders` | **EXISTING** | `adapters/ccxt_shim/breeze_ccxt.py:627` | Read path. |

**Observation**: `edit_order` is absent and requires implementation for P16-2 modification quotas.

## 2. Lot Size Source of Truth

Lot sizes are authoritative in `SecurityMaster` and propagated to the CCXT market structure.

- **Source**: `adapters/ccxt_shim/security_master.py`
  - Field: `LotSize` (from CSV) -> `lot_size` (dict key)
  - Default: `1` (for Cash/Equity if missing)
- **Usage**:
  - `BreezeCCXT._fetch_future_market`: Uses `info["lot_size"]`.
  - `BreezeCCXT._fetch_cash_market`: Uses `info.get("lot_size", 1)`.
  - `BreezeCCXT.fetch_markets`: Exposes `lot` in the market dictionary.

**Decision**: The `OrderRouter` should resolve lot size by looking up the symbol in `self.markets` (which is populated via `fetch_markets` -> `SecurityMaster`).

## 3. CCXT Constructor Status

Verification confirmed that the upstream `ccxt` module has no knowledge of `icicibreeze`.

- `ccxt.icicibreeze`: **MISSING** (AttributeError)
- `ccxt.async_support.icicibreeze`: **MISSING** (Assumed)

**Decision**: P16 tests and gates **MUST NOT** rely on `ccxt.icicibreeze`. They should explicitly load the exchange class from `adapters.ccxt_shim.breeze_ccxt` or use Freqtrade's exchange resolver mechanism.

## 4. Implementation Strategy

- **OrderRouter**: Will be a new class instantiated in `BreezeCCXT.__init__`.
- **Validation**: `create_order` will call `router.validate_entry(symbol, amount, side)`.
- **Modifications**: `edit_order` will be implemented in `BreezeCCXT` and delegate policy checks to `router`.
