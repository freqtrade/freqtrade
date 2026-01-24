import logging
import re
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)

_LEGACY_EXPIRY_WARNING_EMITTED = False


def _strip_dataframe_strings(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    object_columns = df.select_dtypes(include=["object"]).columns
    for column in object_columns:
        df[column] = df[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return df


def _normalize_expiry_date(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y%m%d"), parsed.dt.strftime("%Y-%m-%d")


def _normalize_option_type(value: Any) -> str | None:
    mapping = {"CE": "CE", "CALL": "CE", "PE": "PE", "PUT": "PE"}
    if isinstance(value, str):
        return mapping.get(value.strip().upper())
    return None


def _warn_legacy_expiry_once() -> None:
    global _LEGACY_EXPIRY_WARNING_EMITTED
    if not _LEGACY_EXPIRY_WARNING_EMITTED:
        logger.warning(
            "Legacy expiry format YYYY-MM-DD detected in whitelist. "
            "Use canonical YYYYMMDD going forward."
        )
        _LEGACY_EXPIRY_WARNING_EMITTED = True


def find_latest_master_file(
    filename: str = "FONSEScripMaster.txt", search_paths: list[str] | None = None
) -> str | None:
    if search_paths is None:
        search_paths = [
            "user_data/data/icicibreeze/",
            "src/data/icicibreeze/",
            "user_data/data/",
            "src/data/",
        ]

    from pathlib import Path

    found_files = []
    for path in search_paths:
        full_path = Path.cwd() / path / filename
        if full_path.exists():
            found_files.append((str(full_path), full_path.stat().st_mtime))

    if not found_files:
        return None

    found_files.sort(key=lambda x: x[1], reverse=True)
    return found_files[0][0]


def load_nfo_options_master(file_path: str) -> dict[str, Any]:
    logger.info(f"Loading SecurityMaster from {file_path}")
    try:
        df = pd.read_csv(file_path)
        df = _strip_dataframe_strings(df)
    except Exception as e:
        logger.error(f"Failed to read SecurityMaster {file_path}: {e}")
        return {
            "by_contract": {},
            "by_future": {},
            "by_underlying": {},
            "company_search": {},
        }

    required = [
        "Token",
        "ShortName",
        "ExpiryDate",
        "StrikePrice",
        "OptionType",
        "Underlyer",
        "LotSize",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(
            f"SecurityMaster missing required columns: {missing}. Found: {list(df.columns)}"
        )
        return {
            "by_contract": {},
            "by_future": {},
            "by_underlying": {},
            "company_search": {},
        }

    if "DeleteFlag" in df.columns:
        df = df[df["DeleteFlag"].astype(str) == "0"]

    inst_col = "InstrumentName" if "InstrumentName" in df.columns else "Series"
    if inst_col in df.columns:
        df = df[df[inst_col].isin(["OPTIDX", "OPTSTK", "FUTIDX", "FUTSTK"])]

    df["expiry_yyyymmdd"], df["expiry_iso"] = _normalize_expiry_date(df["ExpiryDate"])
    df = df.dropna(subset=["expiry_yyyymmdd"])

    df["OptionType"] = df["OptionType"].map(_normalize_option_type)
    df["StrikePrice"] = pd.to_numeric(df["StrikePrice"], errors="coerce")
    df["LotSize"] = pd.to_numeric(df["LotSize"], errors="coerce")
    if "TickSize" not in df.columns:
        df["TickSize"] = 0.05
    df["TickSize"] = pd.to_numeric(df["TickSize"], errors="coerce").fillna(0.05)

    options_df = df[df["OptionType"].isin(["CE", "PE"]) & df["StrikePrice"].notna()].copy()
    futures_df = df[df["OptionType"].isna()].copy()

    by_contract = {}
    by_future = {}
    by_underlying = {}
    company_search = {}

    for _, row in options_df.iterrows():
        underlying = str(row["Underlyer"]).upper()
        expiry_yyyymmdd = row["expiry_yyyymmdd"]
        expiry_iso = row["expiry_iso"]
        strike = float(row["StrikePrice"])
        right = row["OptionType"]
        token = str(row["Token"])
        contract_key = (underlying, expiry_yyyymmdd, strike, right)
        info = {
            "token": token,
            "underlying": underlying,
            "expiry_yyyymmdd": expiry_yyyymmdd,
            "expiry_iso": expiry_iso,
            "strike": strike,
            "right": right,
            "lot_size": int(row["LotSize"]) if pd.notna(row["LotSize"]) else 1,
            "tick_size": float(row["TickSize"]),
            "short_name": str(row["ShortName"]),
            "company_name": str(row.get("CompanyName", row["ShortName"])).lower(),
        }
        by_contract[contract_key] = info
        if underlying not in by_underlying:
            by_underlying[underlying] = {"expiries": set(), "strikes": set()}
        by_underlying[underlying]["expiries"].add(expiry_yyyymmdd)
        by_underlying[underlying]["strikes"].add(strike)
        company_search[info["company_name"]] = underlying

    for _, row in futures_df.iterrows():
        underlying = str(row["Underlyer"]).upper()
        expiry_yyyymmdd = row["expiry_yyyymmdd"]
        expiry_iso = row["expiry_iso"]
        token = str(row["Token"])
        future_key = (underlying, expiry_yyyymmdd)
        info = {
            "token": token,
            "underlying": underlying,
            "expiry_yyyymmdd": expiry_yyyymmdd,
            "expiry_iso": expiry_iso,
            "lot_size": int(row["LotSize"]) if pd.notna(row["LotSize"]) else 1,
            "tick_size": float(row["TickSize"]),
            "short_name": str(row["ShortName"]),
            "company_name": str(row.get("CompanyName", row["ShortName"])).lower(),
        }
        by_future[future_key] = info
        company_search[info["company_name"]] = underlying

    logger.info(
        "Parsed %s active option contracts and %s futures",
        len(by_contract),
        len(by_future),
    )
    return {
        "by_contract": by_contract,
        "by_future": by_future,
        "by_underlying": by_underlying,
        "company_search": company_search,
    }


def load_nse_cash_master(file_path: str) -> dict[str, Any]:
    logger.info(f"Loading NSE Cash SecurityMaster from {file_path}")
    try:
        df = pd.read_csv(file_path)
        df = _strip_dataframe_strings(df)
    except Exception as e:
        logger.error(f"Failed to read NSE Cash SecurityMaster {file_path}: {e}")
        return {"by_symbol": {}, "company_search": {}}

    required = ["Token", "ShortName", "Series", "Underlyer", "LotSize"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(
            "NSE Cash SecurityMaster missing required columns: %s. Found: %s",
            missing,
            list(df.columns),
        )
        return {"by_symbol": {}, "company_search": {}}

    df = df[df["Series"].isin(["EQ", "BE", "SM", "ST"])]
    if "DeleteFlag" in df.columns:
        df = df[df["DeleteFlag"].astype(str) == "0"]

    by_symbol = {}
    company_search = {}

    for _, row in df.iterrows():
        symbol = str(row["ShortName"]).upper()
        token = str(row["Token"])
        info = {
            "token": token,
            "symbol": symbol,
            "lot_size": int(row["LotSize"]) if pd.notna(row["LotSize"]) else 1,
            "tick_size": float(row.get("TickSize", 0.05)),
            "company_name": str(row.get("CompanyName", row["ShortName"])).lower(),
        }
        by_symbol[symbol] = info
        company_search[info["company_name"]] = symbol

    logger.info(f"Parsed {len(by_symbol)} active NSE Cash scrips")
    return {"by_symbol": by_symbol, "company_search": company_search}


def parse_pair_whitelist_for_options(pairs: list[str]) -> list[dict[str, Any]]:
    specs = []
    for pair in pairs:
        canonical_opt = re.match(
            r"^(?P<is_cn>CN:)?(?P<prefix>.+?)-(?P<expiry>\d{8})-(?P<strike>\d+(?:\.\d+)?)-(?P<right>CE|PE)/INR$",
            pair,
            re.IGNORECASE,
        )
        if canonical_opt:
            specs.append(
                {
                    "original": pair,
                    "type": "option",
                    "is_company": bool(canonical_opt.group("is_cn")),
                    "prefix": canonical_opt.group("prefix").strip(),
                    "expiry_yyyymmdd": canonical_opt.group("expiry"),
                    "strike": float(canonical_opt.group("strike")),
                    "right": canonical_opt.group("right").upper(),
                }
            )
            continue
        canonical_fut = re.match(
            r"^(?P<is_cn>CN:)?(?P<prefix>.+?)-(?P<expiry>\d{8})-FUT/INR$",
            pair,
            re.IGNORECASE,
        )
        if canonical_fut:
            specs.append(
                {
                    "original": pair,
                    "type": "future",
                    "is_company": bool(canonical_fut.group("is_cn")),
                    "prefix": canonical_fut.group("prefix").strip(),
                    "expiry_yyyymmdd": canonical_fut.group("expiry"),
                }
            )
            continue

        legacy_opt = re.match(
            r"^(?P<is_cn>CN:)?(?P<prefix>.+?)-(?P<expiry>\d{4}-\d{2}-\d{2})-(?P<strike>\d+(?:\.\d+)?)-(?P<right>CE|PE)$",
            pair,
            re.IGNORECASE,
        )
        if legacy_opt:
            _warn_legacy_expiry_once()
            expiry_yyyymmdd, _ = _normalize_expiry_date(pd.Series([legacy_opt.group("expiry")]))
            expiry_value = expiry_yyyymmdd.iloc[0]
            if isinstance(expiry_value, str):
                specs.append(
                    {
                        "original": pair,
                        "type": "option",
                        "is_company": bool(legacy_opt.group("is_cn")),
                        "prefix": legacy_opt.group("prefix").strip(),
                        "expiry_yyyymmdd": expiry_value,
                        "strike": float(legacy_opt.group("strike")),
                        "right": legacy_opt.group("right").upper(),
                    }
                )
                continue

        legacy_fut = re.match(
            r"^(?P<is_cn>CN:)?(?P<prefix>.+?)-(?P<expiry>\d{4}-\d{2}-\d{2})-FUT$",
            pair,
            re.IGNORECASE,
        )
        if legacy_fut:
            _warn_legacy_expiry_once()
            expiry_yyyymmdd, _ = _normalize_expiry_date(pd.Series([legacy_fut.group("expiry")]))
            expiry_value = expiry_yyyymmdd.iloc[0]
            if isinstance(expiry_value, str):
                specs.append(
                    {
                        "original": pair,
                        "type": "future",
                        "is_company": bool(legacy_fut.group("is_cn")),
                        "prefix": legacy_fut.group("prefix").strip(),
                        "expiry_yyyymmdd": expiry_value,
                    }
                )
                continue

        if "/" in pair:
            base, quote = pair.split("/", maxsplit=1)
            specs.append(
                {"original": pair, "type": "cash", "prefix": base.strip(), "quote": quote.strip()}
            )
            continue
        logger.warning("Invalid pair format in whitelist: %s", pair)
    return specs


def resolve_underlying(
    specs: list[dict[str, Any]], nfo_master: dict[str, Any], nse_master: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    resolved = []
    unresolved = []
    nfo_company_search = nfo_master.get("company_search", {})
    for spec in specs:
        if spec["type"] in {"option", "future"}:
            if spec["is_company"]:
                company_name = spec["prefix"].lower()
                if company_name in nfo_company_search:
                    spec["underlying"] = nfo_company_search[company_name]
                    resolved.append(spec)
                else:
                    unresolved.append(spec["original"])
            else:
                spec["underlying"] = spec["prefix"].upper()
                resolved.append(spec)
        elif spec["type"] == "cash":
            spec["underlying"] = spec["prefix"].upper()
            resolved.append(spec)
    return resolved, unresolved
