# Phase 22: Real-Mode Market Data Validation

## 1. Objective

Validate the Freqtrade + Breeze adapter integration in **Real Mode** (`BREEZE_MOCK=0`) for read-only market data operations.
Ensure that:

- Authentication works with real API keys and session tokens.
- `list-markets` returns valid instrument data.
- `download-data` fetches OHLCV candles successfully for a sample pair (RELIANCE/INR).

## 2. Invariants & Scope

- **NO Live Orders**: This phase strictly forbids placing real orders. All logic must be read-only.
- **Secrets Hygiene**: Credentials must come from environment variables only (never logged or printed).
- **Minimal Config**: Use a generated minimal config to strictly limit scope to `RELIANCE/INR`.

## 3. Pre-requisites

- `BREEZE_API_KEY`, `BREEZE_API_SECRET`, and `BREEZE_SESSION_TOKEN` must be present in environment for positive case.
- If pre-requisites are missing, the gate MUST skip (soft fail), not crash or hang.

## 4. Verification

Run the acceptance gate:

```bash
bash scripts/accept_all.sh p22_real_mode_market_data
```
