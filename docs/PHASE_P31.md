# Phase 31: Runtime Health Snapshot

## Objective

Provide a persisted, atomic snapshot of the runtime health and state of the ICICI Breeze integration. This file (`user_data/generated/runtime/health.json`) serves as a source of truth for external monitoring and for the bot itself to recover context (if needed).

## Schema

File Path: `user_data/generated/runtime/health.json`

```json
{
  "meta": {
    "generated_at_utc": "ISO8601",
    "commit": "git_hash_or_unknown"
  },
  "runtime": {
    "mode": {
      "breeze_mock": bool,
      "paper_trading": bool,
      "live_trading_enabled": bool
    }
  },
  "last_calls": {
    "fetch_ticker_utc": "ISO8601|null",
    "fetch_ohlcv_utc": "ISO8601|null",
    "create_order_utc": "ISO8601|null"
  },
  "counters": {
    "policy_blocks": int,
    "degraded_failures": int
  },
  "last_error": {
    "code": "str|null",
    "message": "str|null"
  }
}
```

## Implementation Details

- **Atomic Write**: Write to `.tmp` then rename to ensure readers never see partial files.
- **No Secrets**: Strictly forbidden to write API keys or tokens.
- **Updates**: Triggered on key events (API calls, errors, policy blocks).

## Integration Points

- `BreezeCCXT`: Calls `health_snapshot.update()` on `fetch_ticker`, `fetch_ohlcv`, `create_order`.
- `DegradedModeGuard`: Updates failure counters? (Actually P31 says BreezeCCXT does it).
- `Policy Blocks`: Increments `policy_blocks` counter (live_guard, market_hours, risk_guard).

## Verification

- **P31 Gate (`p31_health_snapshot`)**:
  - **Pos**: Run shim operations -> Assert `health.json` exists and contains expected data.
  - **Neg**: Corrupt the file -> Assert `load()` recovers gracefully (empty dict or default).
