# Phase P27: Smart Money & Integrity (FR-203/202)

## Objective

Implement "Smart Money" decision logic (FR-203) using Option Chain Snapshots and (optional) basic OHLCV integrity checks (FR-202). Pure logic implementation requiring no external I/O.

## Components

### 1. Smart Money Module (`user_data/strategies/smart_money_fr203.py`)

- **Input**: OptionChainSnapshot (Underlying, Strikes, OI/Volume Change).
- **Logic**:
  - `require_min_oi_change_pct`: 10.0
  - `require_min_volume`: 100,000
  - `reject_if_ltp_change_pct <= 0` (Decay guard)
- **Output**: `SmartMoneyDecision(allow_trade: bool, bias_strength: int, reasons: list)`
- **Determinism**: Bias scoring must be deterministic integer math.

### 2. Data Integrity Module (`user_data/strategies/data_integrity_fr202.py`)

- **Logic**:
  - Monotonic timestamps.
  - OHLC sanity (High >= Low/Open/Close, etc.)
  - Non-negative volume.
- **Output**: List of violation strings.

### 3. Strategy Integration

- Strategies will:
  - Check integrity of DF.
  - evaluate `FR203` using a fixture (or empty/bypass if missing).
  - Block entries if `allow_trade` is False or integrity fails.

## Acceptance Criteria (Gate `p27_smart_money`)

- **Positive**:
  - Unit tests verify "Good" snapshot allows trade.
  - Marker: `P27_POS_PASS`
- **Negative**:
  - Unit tests verify "Bad" snapshot (Low OI) blocks trade.
  - Marker: `P27_NEG_EXPECTED_FAIL`
