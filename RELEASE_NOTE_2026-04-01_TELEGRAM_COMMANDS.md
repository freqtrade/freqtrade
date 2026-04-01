# Release Note 2026-04-01

## Scope

Telegram command experience update for `/data/ft_pm_userdata/freqtrade`.

## Changes

- Added `/closeall` as a dedicated one-command shortcut to force-exit all open trades.
- Kept existing `/forceexit all` behavior intact for backward compatibility.
- Added `/logs` to the default Telegram keyboard so log access is visible without custom keyboard setup.
- Updated Telegram help text and usage documentation to include the new shortcut.
- Added regression tests covering `/closeall` and the updated default keyboard layout.

## User Impact

- Operators can now close all open positions with `/closeall`.
- Operators can access recent logs faster from the default Telegram keyboard.

## Compatibility

- Backward compatible with existing `/forceexit`, `/fx`, and `/logs` usage.
- No config migration required.
