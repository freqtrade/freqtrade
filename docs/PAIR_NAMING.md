# Canonical Pair Naming (Single Source of Truth)

This document defines the *only* accepted canonical symbol formats used across
the ICICI Breeze adapter (markets, ticker, OHLCV, and config validation).

## Cash (Spot)

```
{UNDERLYING}/INR
```

Example:

```
RELIANCE/INR
```

## Futures

```
{UNDERLYING}-{EXPIRY_YYYYMMDD}-FUT/INR
```

Example:

```
NIFTY-20260226-FUT/INR
```

## Options

```
{UNDERLYING}-{EXPIRY_YYYYMMDD}-{STRIKE}-{RIGHT}/INR
```

Where:
- `{RIGHT}` ∈ `{CE, PE}`

Example:

```
NIFTY-20260226-22500-CE/INR
```
