# Risk Guard Configuration Schema (P15)

The `risk_guard` section in `config.json` provides strict, deterministic guardrails at the broker shim level. These rules are enforced *before* any order reaches the exchange API.

## Schema

```json
{
  "risk_guard": {
    "enabled": true,
    "max_trades_per_day": 10,
    "max_open_positions": 1,
    "green_day_profit_lock_pct": 1.0,
    "intraday_entry_cutoff_ist": "15:05",
    "intraday_force_exit_ist": "15:15",
    "spread_guard": {
      "enabled": true,
      "max_spread_pct": 0.40
    },
    "allow_exits_when_blocked": true
  }
}
```

## Field Definitions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Master switch for the Risk Guard module. |
| `max_trades_per_day` | int | `10` | Hard cap on total orders (entries) per trading session. |
| `max_open_positions` | int | `1` | Maximum concurrent open positions allowed. |
| `green_day_profit_lock_pct` | float | `1.0` | Daily profit percentage at which to lock further entries (1.0 = 1%). |
| `intraday_entry_cutoff_ist` | string | `"15:05"` | Time (IST) after which new entries are blocked. |
| `intraday_force_exit_ist` | string | `"15:15"` | Time (IST) to trigger force exit logic (reserved for P16). |
| `spread_guard.enabled` | bool | `true` | Enable Bid/Ask spread checks. |
| `spread_guard.max_spread_pct` | float | `0.40` | Block entry if `(Ask - Bid) / Mid * 100` exceeds this value. |
| `allow_exits_when_blocked` | bool | `true` | If true, REDUCE/EXIT orders are allowed even if the guard is blocking entries. |

## Behavior

- **Blocking**: Enforced via `BreezeCCXT.create_order`. Raises `OperationalException` prefixed with `risk_block:`.
- **Timezone**: All time-based rules operate in **IST (Asia/Kolkata)**.
- **Overrides**: Can be overridden by env vars for testing (e.g., `FT_IST_NOW`).
