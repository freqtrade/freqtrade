# ICICI Breeze SecurityMaster Mapping

This document describes how SecurityMaster files are used to map canonical pairs
to ICICI Breeze instruments.

## Files Used

The adapter reads the latest available files from the default search paths:

- `NSEScripMaster.txt` (cash equities)
- `FONSEScripMaster.txt` (futures & options)

## Required Fields (Token Mapping)

The following columns are required to build deterministic token mappings:

### Cash (NSE)

- `Token`
- `ShortName`
- `Series`
- `Underlyer`
- `LotSize`
- `TickSize` (optional but used when present)

### Futures & Options (NFO)

- `Token`
- `ShortName`
- `ExpiryDate`
- `StrikePrice`
- `OptionType`
- `Underlyer`
- `LotSize`
- `TickSize` (optional but used when present)

## Normalization Rules

### Expiry Normalization

SecurityMaster `ExpiryDate` values are normalized into:

- `expiry_yyyymmdd` → `YYYYMMDD` (canonical)
- `expiry_iso` → `YYYY-MM-DD` (kept for API parameters where needed)

### Strike Normalization

`StrikePrice` is converted to a floating point number. Canonical formatting
uses a compact string representation (e.g., `22500` instead of `22500.0`).

### CE/PE Normalization

`OptionType` values are normalized as:

- `CE` or `Call` → `CE`
- `PE` or `Put` → `PE`

## Canonical Contract Keys

The canonical indexes used for mapping are:

- Options: `(UNDERLYING, EXPIRY_YYYYMMDD, STRIKE, RIGHT)`
- Futures: `(UNDERLYING, EXPIRY_YYYYMMDD)`

These keys are the single source of truth for token lookups and must align
with the canonical pair schema in `docs/PAIR_NAMING.md`.
