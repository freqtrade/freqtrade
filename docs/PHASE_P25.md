# Phase 25: Daily Security Master Refresh

## 1. Objective

Automate the daily retrieval and normalization of ICICIBreeze Security Master files (NSE Cash and NFO).
Produce a unified, fast-loading JSON artifact (`latest.json`) to replace runtime CSV parsing.

## 2. Components

### A. Fetch Script (`scripts/p25_fetch_security_master.py`)

- **Real Mode**: Downloads `NSEScripMaster.txt` and `FONSEScripMaster.txt` from Breeze API (via `https://directlink.icicidirect.com/NewSecurityMaster/SecurityMaster.asp`).
- **Mock Mode**: Copies sample fixtures from `user_data/data/icicibreeze`.
- **Output**: `user_data/cache/security_master/NSEScripMaster.txt` (and FONSEScripMaster.txt).

### B. Build Script (`scripts/p25_build_security_master_json.py`)

- **Input**: The raw TXT files fetched above.
- **Logic**: Reuses `adapters.ccxt_shim.security_master` parsing logic to ensure consistency.
- **Output**: `user_data/cache/security_master/latest.json`.
- **Schema**:

  ```json
  {
    "meta": {"generated_at": "...", "counts": {...}},
    "cash": [{"symbol": "RELIANCE", "token": "...", "lot_size": 1}],
    "options": [{"underlying": "NIFTY", "expiry": "20260226", "strike": 22000, "right": "CE", "token": "..."}],
    "futures": [{"underlying": "NIFTY", "expiry": "20260226", "token": "..."}]
  }
  ```

## 3. Invariants

- **Atomic Write**: `latest.json` is written to a temp file and renamed.
- **Determinism**: Lists are sorted by stable keys (Symbol/Underlying + Expiry + Strike).
- **Validation**: Building fails if files are missing or empty.

## 4. Verification

Run the acceptance gate:

```bash
bash scripts/accept_all.sh p25_security_master_refresh
```
