# Phase 20: UI Readiness and Safe Exposure (Scope Contract)

**Phase ID**: p20_ui_readiness_and_safe_exposure
**Status**: IN_PROGRESS

## Goal

Keeping the custom UI implementation deferred, this phase focuses on making the Freqtrade API enablement safe, explicit, and auditable. We provide a controlled monitoring surface (REST/WebSocket API) ONLY when explicitly enabled by the user.

## Non-Goals (Out of Scope)

- **NO** Custom Streamlit or Gradio dashboards.
- **NO** New trading features or algorithm changes.
- **NO** Opening ports to the internet (0.0.0.0) by default.

## Constraints

- **Minimal Invasion**: Prefer configuration, scripts, and documentation over modifying Freqtrade core code.
- **Security-First**: Default everything to `127.0.0.1`. Strong authentication (username/password) is mandatory if API is enabled.
- **Auditable**: All exposed ports must be tracked.

## Deliverables

1. **Safety Contract**: `docs/P20_API_SAFETY.md` defining safe usage patterns.
2. **Smoke Test**: `scripts/p20_api_smoke.sh` verifying API works on localhost without exposing ports.
3. **Guardrails**: `scripts/ops/p20_scan_port_exposure.py` and `scripts/gates/p20_no_open_ports_pos.sh` to prevent accidental exposure.
4. **Documentation**: Updated `docs/OPS_RUNBOOK.md` noting existing host ports (e.g., 6080) out of repo scope.

## Decision Freeze

- **API Server**: We use the standard Freqtrade API server (FastAPI/Uvicorn).
- **Binding**: Strictly `127.0.0.1` unless explicitly overridden by user (which triggers warnings).
- **Auth**: Basic Auth / JWT required.

---
*This document serves as the decision record for Phase 20.*
