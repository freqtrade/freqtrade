# Third Party Review: Outside Freqtrade Core

This document outlines the scope for third-party auditing of components that exist **outside** the standard Freqtrade codebase but are critical for the ICICI Breeze integration.

## Scope

### 1. Adapter Layer (`adapters/ccxt_shim/`)

The `ccxt_shim` is a custom adapter implementing the CCXT interface for ICICI Breeze. It is NOT part of standard Freqtrade.

- [ ] **Auth Hygiene**: Ensure Session Tokens and API Keys are never logged (verified in P21).
- [ ] **Rate Limiting**: Verify leaky bucket implementation respects 1 req/sec strict limit.
- [ ] **Error Handling**: Verify translation of Breeze exceptions to `OperationalException` (fail-safe).
- [ ] **Market Hours**: Verify IST time conversion logic (critical for India markets).

### 2. Guardrails (`adapters/ccxt_shim/*_guard.py`)

Custom safety logic injected into the adapter.

- [ ] **Risk Guard**: Daily trade limit, Max Drawdown logic. Is it stateful? Is persistence robust?
- [ ] **Deadman Switch**: Is the file check atomic? Race conditions?
- [ ] **Live Readiness**: Disk space check thresholds.

### 3. Operational Scripts (`scripts/`)

- [ ] **Gate Scripts**: Verify that gate scripts (`p*_*.sh`) accurately simulate production constraints.
- [ ] **Common Utils**: Review `common.sh` for environment variable leakage.

### 4. Infrastructure & Environment

- [ ] **Secrets Management**: check `user_data/secrets` permissions (0600).
- [ ] **Network Egress**: Verify no unauthorized calls (only to `api.icicidirect.com`).

## Release Certification

Before "Live Beta", a code review by an independent developer (not the author) is required for:

- `breeze_ccxt.py` (Core Logic)
- `order_router.py` (Routing Logic)
- `live_readiness.py` (Safety Logic)
