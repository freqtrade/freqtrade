# Operational Runbook: Freqtrade ICICI Breeze Adapter

This runbook describes how to operate the Freqtrade bot with the ICICI Breeze adapter in production.

## 1. Prerequisites

- **Python 3.10+**
- **Systemd** (for Linux service management)
- **ICICI Securities Account** (API Key, Secret, Session Token)

## 2. Configuration

- Ensure `user_data/config_icicibreeze.json` matches your prod environment.
- **NEVER** commit secrets to git. Use environment variables or a separate secrets file.

## 3. Running in Production

```bash
# Activate environment
source .venv/bin/activate

# Start Freqtrade
freqtrade trade -c user_data/config_icicibreeze.json --strategy IndiaOptionsAutoStrategy
```

## 4. Mock Mode

To run in mock mode (safe for testing):

```bash
export BREEZE_MOCK=1
# Dry-run is recommended with mock mode
freqtrade trade --dry-run -c user_data/config_icicibreeze.json ...
```

## 5. Verification

Run the acceptance suite before any major deployment:

```bash
bash scripts/accept_all.sh
```

## Host Ports (Safety Note)

During P20 inventory, the following ports were observed listening on the host:

- `0.0.0.0:6080` (Websockify / NoVNC)
- `0.0.0.0:22` (SSH)

**NOTE**: These services are part of the host infrastructure and are **OUTSIDE** the scope of this repository. This repository only manages the Freqtrade application and its direct dependencies. The presence of these ports is acknowledged but not managed by Freqtrade or its acceptance gates (except to ensure we don't accidentally conflict with or expose them further).

## 6. Troubleshooting

- **Logs**: Check `user_data/logs/freqtrade.log`.
- **Connectivity**: Verify internet access and DNS resolution for `api.icicidirect.com`.
- **Session**: If session expires, regenerate session token via `scripts/get_session_token.py` (if implemented) or manually.
