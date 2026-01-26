# Security Hygiene

## 1. Secrets Management

- **Do not commit secrets** (API keys, passwords, private keys) to the repository.
- Use `.env` files (added to `.gitignore`) or system environment variables.
- The `scripts/security/secret_scan_strict.sh` script runs in CI to catch accidental leaks.

## 2. File Permissions

- Ensure `user_data/secrets` and `.env` files are **NOT** world-readable (`chmod 600`).
- The `scripts/security/file_perms_audit.sh` script verifies this.

## 3. Network

- Bind the API server to `127.0.0.1` unless behind a secure proxy/VPN.
- Use HTTPS for all external communication (handled by `breeze_connect` via SSL).

## 4. Dependencies

- Regularly update dependencies to patch vulnerabilities.
- Review `requirements.txt` changes carefully.
