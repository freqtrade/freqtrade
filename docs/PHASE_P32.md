# Phase 32: Alerting Transitions

## Objective

Implement intelligent alerting for critical state transitions within the ICICI Breeze shim. The goal is to notify operators of significant events (circuit breaker trips, risk blocks) without creating alert fatigue.

## Logic

- **State Based**: Alert only when state *changes* (e.g., Green -> Red).
- **Suppression**: If detailed error logs are spamming, alerts should be throttled (e.g., max 1 per minute per category).
- **Routing**: Currently routes to `logger.error` / `logger.warning` with a specific prefix `[ALERT]` for log monitoring filters.

## Categories

1. **Degraded Mode**:
   - `DEGRADED_ENTER`: Circuit breaker tripped.
   - `DEGRADED_EXIT`: System recovered (optional, maybe too noisy).
2. **Risk Block**:
   - `RISK_BLOCK`: Order rejected by RiskGuard.
3. **Policy Block**:
   - `POLICY_BLOCK`: Live Guard or Market Hours blocking orders (often expected, maybe Lower Priority).

## Components

- `adapters/ccxt_shim/alerts.py`: `AlertManager` class.
- Integration into `DegradedModeGuard` and `RiskGuard`.

## Verification

- **Gate**: `p32_alerting_transitions.sh`
- **Tests**: `tests/test_p32_alerting_transitions.py`
