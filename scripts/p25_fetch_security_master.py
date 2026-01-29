#!/usr/bin/env python3
"""
P25 Fetch Security Master
Fetches NSEScripMaster.txt and FONSEScripMaster.txt.
Supports Mock Mode (copy fixtures) vs Real Mode (HTTP Download).
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import requests

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("P25Fetch")

DEFAULT_CACHE_DIR = Path("user_data/cache/security_master")
FIXTURE_DIR = Path("user_data/data/icicibreeze")

# URLs
URL_NSE = "https://scriptmaster.icicidirect.com/Content/File/txt/NSEScripMaster.txt"
URL_FNO = "https://scriptmaster.icicidirect.com/Content/File/txt/FONSEScripMaster.txt"


def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def fetch_mock(output_dir: Path):
    logger.info(f"Fetching in MOCK mode (copying fixtures) to {output_dir}...")
    ensure_dir(output_dir)

    files = ["NSEScripMaster.txt", "FONSEScripMaster.txt"]
    for fname in files:
        src = FIXTURE_DIR / fname
        dst = output_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"Copied {src} -> {dst}")
        else:
            logger.warning(f"Fixture {src} not found!")


def fetch_real(output_dir: Path):
    logger.info(f"Fetching in REAL mode (downloading) to {output_dir}...")
    ensure_dir(output_dir)

    targets = [(URL_NSE, "NSEScripMaster.txt"), (URL_FNO, "FONSEScripMaster.txt")]

    for url, fname in targets:
        dst = output_dir / fname
        logger.info(f"Downloading {url} -> {dst}...")
        try:
            resp = requests.get(url, timeout=30, stream=True)
            resp.raise_for_status()
            with dst.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded {fname}")
        except Exception:
            logger.error(f"Failed to download {url}", exc_info=True)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use mock fixtures instead of download")
    parser.add_argument("--output", help="Output directory", default=str(DEFAULT_CACHE_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Check env var too
    is_mock = args.mock or os.environ.get("BREEZE_MOCK", "0") == "1"

    if is_mock:
        fetch_mock(output_dir)
    else:
        fetch_real(output_dir)


if __name__ == "__main__":
    main()
