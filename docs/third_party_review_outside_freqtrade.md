# Third Party Review: High Frequency Trading System (ICICI Breeze Adapter)

**Revision:** 1.0.0
**Date:** 2026-01-31
**Scope:** `adapters/ccxt_shim`, `freqtrade` integration, Security & Risk Modules.

## 1. System Overview

This system integrates Freqtrade with ICICI Breeze API via a custom CCXT-compatible Shim ("BreezeCCXT").
It implements Strict Risk Gates (P15), Clean Architecture (P35), and Fail-Closed Security (P40).

### Architecture (Ports & Adapters)

```mermaid
graph TD
    User[Trader / Ops] -->|SSH / API| FT[Freqtrade Core]
    FT -->|CCXT Interface| Shim[BreezeCCXT Shim]
    
    subgraph Adapters [adapters/ccxt_shim]
        Shim --> Router[OrderRouter (P16)]
        Shim --> Risk[RiskGuard (P15/P40)]
        Shim --> Ready[LiveReadiness (P40)]
        Shim --> Idem[Idempotency (P40)]
        Shim --> Circuit[CircuitBreaker]
    end
    
    subgraph Infrastructure
        Risk -->|Persist| Disk[user_data/generated]
        Idem -->|Persist| Disk
        Ready -->|Check| Secrets[Deadman Switch]
    end
    
    Shim -->|HTTP| Breeze[ICICI Breeze API]
```

## 2. Security Posture

### 2.1 Secrets Hygiene (Gate P21)

- **Policy:** Zero secrets in code/logs.
- **Implementation:** `p21_secrets_hygiene.sh` scans all artifacts post-run.
- **Evidence:** Logs are sanitized. API keys injected via Env/File only.

### 2.2 Network Security (Gate P20)

- **Policy:** Zero open ports on public interfaces.
- **Implementation:** `p20_no_open_ports_pos.sh` calls `p20_scan_port_exposure.py`.
- **Status:** PASS (Localhost binding forced).

## 3. Operational Hardening

### 3.1 Live Readiness (Gate P40)

- **Mechanism:** `LiveReadiness.check()` enforces Fail-Closed.
- **Checks:**
  1. Deadman Switch (`user_data/secrets/deadman_live.ok`) presence & freshness (<10m).
  2. Disk Space (>2GB).
  3. Session Token validity.
- **Failure Mode:** `OperationalException` (Trade Blocked).

### 3.2 Order Idempotency (Gate P40)

- **Mechanism:** `OrderIdempotency` class with persistence.
- **Logic:** Request -> Hash(Fields) -> Cache Check -> Block if Exists.
- **Persistence:** `runtime/order_id_cache.json` (survives restarts).

### 3.3 Capital Risk Guard (Gate P15)

- **Mechanism:** `RiskGuard` enforces:
  - Max Loss Per Day.
  - Max Consecutive Losses.
  - Max Open Positions.
- **Persistence:** `runtime/live_halt.json`. A halt triggers a persistent block until manual reset or next day.

## 4. Operational Invariants

| Invariant | Implementation | Failure Handling |
|-----------|----------------|------------------|
| **No Duplicate Orders** | `OrderIdempotency.is_duplicate` | Raise `DUPLICATE_SUPPRESSED` |
| **No Unattended Runs** | `check_deadman` | Raise `DEADMAN_FAIL` |
| **No Infinite Loss** | `RiskGuard.record_loss` | Persistent Halt (Entry Block) |
| **No Broken Code** | `P39 Hygiene Gate` | CI Failure on FIXME/TODO |

## 5. Verification Evidence

- **Soak Testing:** P38 gate runs 90s soak, verifying `health.json` metrics.
- **Audit Logs:** All runs archive artifacts to `generated/accept_runs/$RUN_ID`.
- **Gate Suite:** `scripts/accept_all.sh` runs 40+ checks daily.

## 6. Runbook Reference

See `docs/ops/RUNBOOK.md` for:

- resetting risk halts
- renewing deadman switch
- rotating API keys
