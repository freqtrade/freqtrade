# P20: Safe API Enablement Contract

**Phase ID**: p20_ui_readiness_and_safe_exposure

## Safety Rules

1. **Bind Address**: The API server MUST bind to `127.0.0.1` (localhost) only.
    - **NEVER** bind to `0.0.0.0` (all interfaces) unless you have a specific, secured reverse proxy in front and understand the risks.
2. **Authentication**: Authentication is **mandatory**.
    - `jwt_secret_key`: Must be generated and secure.
    - `CORS`: Restrict origins if accessed from a browser.
3. **Exposure**:
    - Do **NOT** expose port `8080` (or your chosen API port) to the public internet via Docker mapping or firewall rules.
    - Docker ports should be mapped as `127.0.0.1:8080:8080`.

## Configuration Example

To enable the API safely, add the following block to your `config.json`. Note that `api_server` is disabled by default.

```json
    "api_server": {
        "enabled": true,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "username": "freqtrader",
        "password": "SuperSecurePassword123!",
        "jwt_secret_key": "replaceme_with_super_secure_random_string",
        "CORS_origins": [],
        "verbosity": "info"
    }
```

## Monitoring

Use `scripts/ops/p20_scan_port_exposure.py` (once implemented) to verify that no dangerous binds exist in your configuration or codebase.
