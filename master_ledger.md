# Master Ledger

This document serves as a registry for major project phases, decision records, and acceptance gates.

## Phase Registry

| Phase ID | Description | Scope Document | Status |
|----------|-------------|----------------|--------|
| P20      | UI Readiness & Safe Exposure | [PHASE_P20.md](docs/PHASE_P20.md) | COMPLETED |
| P21      | Live Session & Secrets Hardening | [PHASE_P21.md](docs/PHASE_P21.md) | IN_PROGRESS |

## Acceptance Gates Registry

| Gate ID | Description | Script | Mode Support |
|---------|-------------|--------|--------------|
| P00-P19 | Previous Phases | `scripts/accept_all.sh` | Pos/Neg |
| P20     | No Open Ports | `scripts/gates/p20_no_open_ports_pos.sh` | Pos/Neg |
| P21     | Secrets Hygiene | `scripts/gates/p21_secrets_hygiene.sh` | Pos/Neg |

## Decision Records

- **P20 Decision**: Defer Custom UI. Prioritize Safe API Enablement. Default bind strictly to `127.0.0.1`.
- **P21 Decision**: Strict Secrets Hygiene. No secrets in logs/artifacts. Env vars only for credentials.
