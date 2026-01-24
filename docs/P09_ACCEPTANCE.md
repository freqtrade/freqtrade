# P09x Acceptance Criteria

This document defines the acceptance criteria for the P09x (Universe Scan) component in mock mode (`BREEZE_MOCK=1`).

## Scanner Robustness

- **Priority**: Stocks must be processed before indices in the whitelist generation.
- **Fail-Safe**: If an underlying fails to scan (e.g., missing data in mock master), the scanner must log the error to `skipped_underlyings` and continue with the rest of the universe.
- **Empty Whitelist**: If no pairs are selected, the output file must contain an empty list `[]` and the report must indicate `status: empty`.

## Deterministic Behavior

- **Sorted Output**: The generated `pairs.json` must always be sorted alphabetically.
- **Consistent SHASUM**: Running the scanner multiple times on the same date with the same config and master file must yield the same SHA256 sum for the output files.

## Report Content

- **Selected Components**: The report must explicitly list `selected_indices` and `selected_stocks`.
- **Reasoning**: Any skipped underlyings must have a clear `reason` (e.g., `no options`, `no CE+PE available`, or `error: ...`).
- **Policy Metadata**: The report should reflect the `chosen_expiry` and `chosen_atm_strike` for each selected underlying.

## Verification Command

```bash
unset PYTHONPATH && export BREEZE_MOCK=1 && .venv/bin/python scripts/universe_scan_and_generate_pairs.py \
  --mode mock \
  --strategy-config user_data/india_strategy.yaml \
  --out user_data/generated/p09x_pairs.json
```
