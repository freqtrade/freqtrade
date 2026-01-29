# Phase P26: Indicator Governance & Guards

## Objective

Establish a single source of truth for strategy indicators, enforcing warmup requirements, stale-data prevention, and lookahead guards (index vs. stocks specific).

## Components

### 1. Indicator Registry (`user_data/strategies/indicator_registry.py`)

- **Responsibility**: Define canonical indicators, defaults, and startup candle counts.
- **Scope**: STOCK_OPT, INDEX_OPT, AUTO_OPT.
- **Contract**:
  - `get_required_indicators(strategy_id)`
  - `get_defaults(strategy_id)`
  - `get_startup_candle_count(strategy_id)`
- **Defaults**:
  - `startup_candle_count`: 50
  - `stale_tolerance_seconds`: 600 (10 mins)

### 2. Guards (`user_data/strategies/guards.py`)

- **Responsibility**: Enforce data integrity and safety rules.
- **Functions**:
  - `enforce_warmup(df, count)`: Blocks signals during warmup.
  - `check_stale_informative(df, tolerance)`: Blocks signals if data is old.
  - `no_lookahead_sanity(df)`: Ensures no forward-filling of future data.

### 3. Strategy Integration

- Strategies must fetch configuration from the registry.
- `populate_indicators` must apply guards to the dataframe.
- `populate_entry_trend` must respect guard outputs (e.g., `data_stale` flag).

## Acceptance Criteria (Gate `p26_indicator_governance`)

- **Positive**:
  - Unit tests for registry and guards pass.
  - Mock backtest loads strategies without error.
  - Marker: `P26_POS_PASS`
- **Negative**:
  - Lookahead violation triggers explicit failure.
  - Marker: `P26_NEG_EXPECTED_FAIL`
