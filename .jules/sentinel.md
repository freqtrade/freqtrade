## 2026-02-08 - RateLimiter Persistence and Secrets Redaction
**Vulnerability:** The `RateLimiter` class in `freqtrade/rpc/api_server/deps.py` was instantiated per-request by FastAPI's dependency injection system without sharing state, rendering rate limiting ineffective for concurrent requests or when using `Depends(RateLimiter(...))`. Additionally, sensitive keys `jwt_secret_key` and `ws_token` were not redacted in configuration outputs.
**Learning:** FastAPI's `Depends` creates a new instance of the dependency class for each request unless it's explicitly managed as a singleton or cached. Passing an instance to `Depends` works, but modifying the class to use a shared cache keyed by parameters ensures robustness against instantiation patterns.
**Prevention:** When implementing stateful dependencies in FastAPI (like rate limiters), always ensure the state is stored in a global or singleton structure, or verify that the dependency provider is reused correctly. For secrets, always audit configuration output logic when adding new sensitive configuration keys.

## 2026-02-09 - Pandas 3.0+ Timezone-Aware Datetime Conversion
**Vulnerability:** Compatibility regression in timestamp conversion logic.
**Learning:** Pandas 3.0+ enforces stricter type conversion. Using `.astype(int64)` on timezone-aware datetimes can fail or produce different precisions (us vs ns) depending on the underlying storage (Feather vs CSV vs Parquet).
**Prevention:** Explicitly check for `pd.DatetimeTZDtype`. Use `.astype("datetime64[ms, UTC]")` for aware data and `.astype("datetime64[ms]")` for naive data before casting to int64 to ensure consistent millisecond timestamps.
