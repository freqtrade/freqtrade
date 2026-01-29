# Phase 23: Secure Session Token via Telegram

## 1. Objective

Allow automated ingestion of the daily Breeze Session Token via a secure Telegram bot channel, avoiding manual copy-paste onto the server console.

## 2. Invariants

- **Zero-Knowledge Logs**: The token value MUST NEVER be printed to logs, console, or artifacts.
- **Strict Storage**: Token is stored in `user_data/secrets/breeze_session_token` with permissions `0400` (or `0600` for dev).
- **One-Way Flow**: The bot must reply "Session updated" without echoing the token back.

## 3. Components

1. `scripts/p23_session_store.py`: A CLI utility that reads a token from STDIN (preferred) or argument (discouraged), validates it, normalizes it, and atomically writes it to the secrets file.
2. `scripts/telegram/p23_token_bot.py`: A pure-python script (using `python-telegram-bot`) that runs as a daemon/service, listens for `/session <token>`, and pipes it to the storage utility.
3. `scripts/gates/p23_session_token_telegram.sh`: Verification gate ensuring the storage utility works as expected and logs are clean.

## 4. Verification

Run the acceptance gate:

```bash
bash scripts/accept_all.sh p23_session_token_telegram
```
