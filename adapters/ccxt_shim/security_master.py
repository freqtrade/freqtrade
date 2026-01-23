import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def find_latest_master_file(
    filename: str = "FONSEScripMaster.txt", search_paths: Optional[List[str]] = None
) -> Optional[str]:
    """
    Search for the latest security master file in prioritized paths.
    """
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

    # Sort by mtime descending
    found_files.sort(key=lambda x: x[1], reverse=True)
    return found_files[0][0]


def load_nfo_options_master(file_path: str) -> Dict[str, Any]:
    """
    Load and parse NFO options from SecurityMaster file.
    Only keeps active OPTIDX/OPTSTK contracts.
    """
    logger.info(f"Loading SecurityMaster from {file_path}")

    try:
        # Breeze scrip masters are usually comma-separated
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to read SecurityMaster {file_path}: {e}")
        return {"by_contract": {}, "by_underlying": {}, "company_search": {}}

    # Normalize column names (sometimes they have spaces or different case)
    df.columns = [c.strip() for c in df.columns]

    # Required columns check
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
        logger.error(f"SecurityMaster missing required columns: {missing}")
        return {"by_contract": {}, "by_underlying": {}, "company_search": {}}

    # Filter:
    # 1. DeleteFlag == 0 (if column exists)
    if "DeleteFlag" in df.columns:
        df = df[df["DeleteFlag"].astype(str) == "0"]

    # 2. Only Options (OPTIDX, OPTSTK)
    # Some older files use InstrumentName, others Series
    inst_col = "InstrumentName" if "InstrumentName" in df.columns else "Series"
    if inst_col in df.columns:
        df = df[df[inst_col].isin(["OPTIDX", "OPTSTK"])]

    # 3. Only CE/PE
    df = df[df["OptionType"].isin(["CE", "PE", "Call", "Put"])]

    # Normalize OptionType
    df["OptionType"] = df["OptionType"].map({"CE": "CE", "Call": "CE", "PE": "PE", "Put": "PE"})

    # Normalize ExpiryDate (expects DD-MMM-YYYY or similar, want YYYY-MM-DD)
    # ICICI format often: 26-Feb-2026
    def parse_expiry(date_str):
        try:
            return pd.to_datetime(date_str).strftime("%Y-%m-%d")
        except:
            return None

    df["expiry_norm"] = df["ExpiryDate"].apply(parse_expiry)
    df = df.dropna(subset=["expiry_norm"])

    # Build indexes
    by_contract = {}
    by_underlying = {}
    company_search = {}

    for _, row in df.iterrows():
        underlying = str(row["Underlyer"]).strip().upper()
        expiry = row["expiry_norm"]
        strike = float(row["StrikePrice"])
        right = row["OptionType"]
        token = str(row["Token"])

        # Key: (underlying, expiry, strike, right)
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

        # Index by underlying for discovery
        if underlying not in by_underlying:
            by_underlying[underlying] = {"expiries": set(), "strikes": set()}
        by_underlying[underlying]["expiries"].add(expiry)
        by_underlying[underlying]["strikes"].add(strike)

        # Index for company search
        company_search[info["company_name"]] = underlying

    logger.info(f"Parsed {len(by_contract)} active option contracts from SecurityMaster")

    return {
        "by_contract": by_contract,
        "by_underlying": by_underlying,
        "company_search": company_search,
    }


def parse_pair_whitelist_for_options(pairs: List[str]) -> List[Dict[str, Any]]:
    """
    Parse whitelist pairs into spec dictionaries.
    Forms:
    - RELIANCE-2026-02-26-2800-CE
    - CN:RELIANCE INDUSTRIES LTD-2026-02-26-2800-CE
    """
    specs = []
    for pair in pairs:
        # Regex to match: [CN:]prefix-expiry-strike-right
        # Expiry: YYYY-MM-DD
        # Strike: Number
        # Right: CE|PE
        pattern = r"^(?P<is_cn>CN:)?(?P<prefix>.*?)-(?P<expiry>\d{4}-\d{2}-\d{2})-(?P<strike>\d+(?:\.\d+)?)-(?P<right>CE|PE)$"
        match = re.match(pattern, pair, re.IGNORECASE)
        if match:
            specs.append(
                {
                    "original": pair,
                    "is_company": bool(match.group("is_cn")),
                    "prefix": match.group("prefix").strip(),
                    "expiry": match.group("expiry"),
                    "strike": float(match.group("strike")),
                    "right": match.group("right").upper(),
                }
            )
        else:
            logger.warning(f"Invalid option pair format in whitelist: {pair}")

    return specs


def resolve_underlying(
    specs: List[Dict[str, Any]], master: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Resolve company names to symbols.
    """
    resolved = []
    unresolved = []

    company_search = master.get("company_search", {})

    for spec in specs:
        if spec["is_company"]:
            company_name = spec["prefix"].lower()
            if company_name in company_search:
                spec["underlying"] = company_search[company_name]
                resolved.append(spec)
            else:
                # Try partial match if exact fails? Requirement says resolution, doesn't specify partial.
                # Let's stick to exact for now as per "authoritative".
                unresolved.append(spec["original"])
        else:
            spec["underlying"] = spec["prefix"].upper()
            resolved.append(spec)

    return resolved, unresolved
