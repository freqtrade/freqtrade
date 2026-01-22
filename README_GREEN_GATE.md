# Green Gate Verification

This document describes the automated verification gate `scripts/green_gate.sh`.

## PASS Definition

The gate checks the health of the Icicibreeze integration. It is considered **PASS** (`GREEN_GATE=PASS`) if and only if:

1. **Compilation**: `python -m compileall` passes without error.
2. **Configuration**: `freqtrade show-config` runs successfully.
3. **Market Listing**: `freqtrade list-markets` runs successfully.
4. **Ticker Shim**: `scripts/smoke_icicibreeze_ticker.py` executes without error (verifying `fetch_ticker`).
5. **Data Download**: `freqtrade download-data` runs specifically for `BTC/USDT` (verifying `fetchOHLCV` and `fetch_markets` filtering).
6. **Dry Run**: `freqtrade trade --dry-run` starts up, correctly resolves the `Icicibreeze` exchange class, and reaches the "Wallets synced" state.

## FAILURE Patterns

If the gate fails, examine the output and the generated temporary files (`/tmp/*.txt`).

### Common Failures

* **"INR not available as stake"**:
  * **Cause**: `load_markets` in `IcicibreezeAsyncShim` (used by Freqtrade) does not match `IcicibreezeShim` (used by scripts), or `active` flag is missing from INR pairs.
  * **Fix**: Ensure `IcicibreezeAsyncShim.load_markets` explicitly sets `active=True` and populates `self.currencies["INR"]`.

* **"Exchange not resolved"**:
  * **Cause**: `ccxt` registry issue or import error.
  * **Fix**: Check `setattr(ccxt, ...)` lines in `icicibreeze.py`.

* **"Wallets not synced" (Timeout)**:
  * **Cause**: Startup crashed before reaching wallet sync or hung.
  * **Fix**: Check `/tmp/trade.txt` for tracebacks.
