import logging
import os
import re
from typing import Any, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


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

    found_files = []
    for path in search_paths:
        full_path = os.path.join(os.getcwd(), path, filename)
        if os.path.exists(full_path):
            found_files.append((full_path, os.path.getmtime(full_path)))

    if not found_files:
        return None

    found_files.sort(key=lambda x: x[1], reverse=True)
    return found_files[0][0]


def load_nfo_options_master(file_path: str) -> dict[str, Any]:
    logger.info(f"Loading SecurityMaster from {file_path}")
    try:
        df = pd.read_csv(file_path)
        # Strip both columns and string data
        df.columns = [c.strip() for c in df.columns]
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    except Exception as e:
        logger.error(f"Failed to read SecurityMaster {file_path}: {e}")
        return {"by_contract": {}, "by_underlying": {}, "company_search": {}}

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
        return {"by_contract": {}, "by_underlying": {}, "company_search": {}}

    if "DeleteFlag" in df.columns:
        df = df[df["DeleteFlag"].astype(str) == "0"]

    inst_col = "InstrumentName" if "InstrumentName" in df.columns else "Series"
    if inst_col in df.columns:
        df = df[df[inst_col].isin(["OPTIDX", "OPTSTK"])]

    df = df[df["OptionType"].isin(["CE", "PE", "Call", "Put"])]
    df["OptionType"] = df["OptionType"].map({"CE": "CE", "Call": "CE", "PE": "PE", "Put": "PE"})

    def parse_expiry(date_str):
        try:
            return pd.to_datetime(date_str).strftime("%Y-%m-%d")
        except:
            return None

    df["expiry_norm"] = df["ExpiryDate"].apply(parse_expiry)
    df = df.dropna(subset=["expiry_norm"])

    by_contract = {}
    by_underlying = {}
    company_search = {}

    for _, row in df.iterrows():
        underlying = str(row["Underlyer"]).upper()
        expiry = row["expiry_norm"]
        strike = float(row["StrikePrice"])
        right = row["OptionType"]
        token = str(row["Token"])
        contract_key = (underlying, expiry, strike, right)
        info = {
            "token": token,
            "underlying": underlying,
            "expiry": expiry,
            "strike": strike,
            "right": right,
            "lot_size": int(row["LotSize"]),
            "tick_size": float(row.get("TickSize", 0.05)),
            "short_name": str(row["ShortName"]),
            "company_name": str(row.get("CompanyName", row["ShortName"])).lower(),
        }
        by_contract[contract_key] = info
        if underlying not in by_underlying:
            by_underlying[underlying] = {"expiries": set(), "strikes": set()}
        by_underlying[underlying]["expiries"].add(expiry)
        by_underlying[underlying]["strikes"].add(strike)
        company_search[info["company_name"]] = underlying

    logger.info(f"Parsed {len(by_contract)} active option contracts")
    return {
        "by_contract": by_contract,
        "by_underlying": by_underlying,
        "company_search": company_search,
    }


def load_nse_cash_master(file_path: str) -> dict[str, Any]:
    logger.info(f"Loading NSE Cash SecurityMaster from {file_path}")
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.strip() for c in df.columns]
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    except Exception as e:
        logger.error(f"Failed to read NSE Cash SecurityMaster {file_path}: {e}")
        return {"by_symbol": {}, "company_search": {}}

    required = ["Token", "ShortName", "Series", "Underlyer", "LotSize"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(
            f"NSE Cash SecurityMaster missing required columns: {missing}. Found: {list(df.columns)}"
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
            "lot_size": int(row["LotSize"]),
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
        pattern = r"^(?P<is_cn>CN:)?(?P<prefix>.*?)-(?P<expiry>\d{4}-\d{2}-\d{2})-(?P<strike>\d+(?:\.\d+)?)-(?P<right>CE|PE)$"
        match = re.match(pattern, pair, re.IGNORECASE)
        if match:
            specs.append(
                {
                    "original": pair,
                    "type": "option",
                    "is_company": bool(match.group("is_cn")),
                    "prefix": match.group("prefix").strip(),
                    "expiry": match.group("expiry"),
                    "strike": float(match.group("strike")),
                    "right": match.group("right").upper(),
                }
            )
            continue
        if "/" in pair:
            base, quote = pair.split("/")
            specs.append(
                {"original": pair, "type": "cash", "prefix": base.strip(), "quote": quote.strip()}
            )
            continue
        logger.warning(f"Invalid pair format in whitelist: {pair}")
    return specs


def resolve_underlying(
    specs: list[dict[str, Any]], nfo_master: dict[str, Any], nse_master: dict[str, Any]
) -> Tuple[list[dict[str, Any]], list[str]]:
    resolved = []
    unresolved = []
    nfo_company_search = nfo_master.get("company_search", {})
    nse_company_search = nse_master.get("company_search", {})
    for spec in specs:
        if spec["type"] == "option":
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
