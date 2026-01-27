# Phase 15: Risk Guardrails

## Overview

We have implemented a deterministic Risk Guard at the broker shim level (`BreezeCCXT`). This ensures that all orders pass strict risk checks before reaching the exchange API.

## Implementation

- **Guard**: `adapters/ccxt_shim/risk_guard.py`
- **Hook**: `BreezeCCXT.create_order` calls `should_block_entry`.
- **Config**: `risk_guard` section in `config.json`.

## Acceptance Gates

### P15 (Positive)

- **Command**: `bash scripts/accept_all.sh p15`
- **Config**: `user_data/examples/config_p15_pos_overlay.json`
- **Behavior**: Trading ALLOWED. No `risk_block:` errors.

### P15 (Negative)

- **Command**: `bash scripts/gates/p15_risk_guardrails.sh --mode=neg`
- **Config**: `user_data/examples/config_p15_neg_overlay.json`
- **Behavior**: Trading BLOCKED. `risk_block:` errors expected.

## Verification

Run the following commands to verify:

```bash
# Verify Positive Case
bash scripts/accept_all.sh p15

# Verify Negative Case
bash scripts/gates/p15_risk_guardrails.sh --mode=neg
```
