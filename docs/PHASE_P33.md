# Phase 33: Backup & Restore

## Objective

Provide operational tools to backup and restore the bot's state. This is critical for disaster recovery and migrations.

## Scope

- `user_data/` is the primary state directory.
- We must backup:
  - `data/` (market data, although redownloadable, valuable for backtest continuity)
  - `generated/` (runtime health, gate artifacts)
  - `strategies/` (if any user strategies exist)
  - `config.json` (if present in user_data)
- We likely exclude:
  - `logs/` (too large)
  - `backups/` (recursive)
  - `.tmp`/`.temp` (transient)

## Scripts

1. **`scripts/backup_state.sh`**:
   - Creates `user_data/backups/backup_YYYYMMDD_HHMMSS.tar.gz`.
   - Returns absolute path of backup.

2. **`scripts/restore_state.sh <backup_file>`**:
   - Validates tarfile.
   - Extracts to `user_data/.restore_tmp`.
   - Renames `user_data` -> `user_data.old_TIMESTAMP`.
   - Renames `.restore_tmp` -> `user_data`.
   - (Verification step?)

## Gate (`p33_backup_restore`)

- **Pos**:
    1. Write `user_data/test_marker`
    2. Run Backup.
    3. Delete `user_data/test_marker`.
    4. Run Restore.
    5. Assert `user_data/test_marker` exists.
- **Neg**:
    1. Pass invalid/corrupt archive to Restore.
    2. Assert restore fails and `user_data` is untouched.
