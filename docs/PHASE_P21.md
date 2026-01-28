# Phase 21: Live Session and Secrets Hardening (Scope Contract)

**Phase ID**: p21_live_session_and_secrets_hardening
**Status**: IN_PROGRESS

## Goal

Make the live session bootstrap safe, repeatable, and auditably secure. Harden secrets hygiene to ensure no accidental exposure in logs, artifacts, or console output.

## Secrets Policy (Violations will fail gates)

1. **Golden Rule**: Secrets MUST NEVER be committed to the repository.
2. **Env Only**: `BREEZE_API_KEY`, `BREEZE_API_SECRET`, and `BREEZE_SESSION_TOKEN` must be sourced from environment variables.
3. **Logs**: Secrets MUST NEVER be printed to `stdout` or written to log files.
    - *Exception*: Placeholder strings like `your_key_here` in `.example` files are permitted.
4. **Artifacts**: Acceptance test artifacts (tarballs) must be free of live credentials.

## Operational Contracts

### Real Mode

- Requires: `BREEZE_API_KEY`, `BREEZE_API_SECRET`, `BREEZE_SESSION_TOKEN` in env.
- Readiness: Use `scripts/p21_session_check.py` to verify credentials *before* starting the bot.

### Mock Mode

- Command: `export BREEZE_MOCK=1`
- Behavior: Bypasses credential checks. Safe for CI and local testing without keys.

### Session Rotation

- The `BREEZE_SESSION_TOKEN` expires daily (approx 24h).
- Ops Procedure:
    1. Generate new token via ICICI Breeze login.
    2. Export new `BREEZE_SESSION_TOKEN`.
    3. Run `scripts/p21_session_check.py` to verify.
    4. Restart Freqtrade.

## Deliverables

- `scripts/p21_session_check.py`: Verification script suitable for CI and pre-start hooks.
- `scripts/gates/p21_secrets_hygiene.sh`: Acceptance gate enforcing strict no-leak policy.
