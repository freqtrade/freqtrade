import logging
from pathlib import Path
from typing import Optional

import requests


logger = logging.getLogger(__name__)

# Timeout for requests
req_timeout = 30


def clean_ui_subdir(directory: Path):
    if directory.is_dir():
        logger.info("Removing UI directory content.")

        for p in reversed(list(directory.glob("**/*"))):  # iterate contents from leaves to root
            if p.name in (".gitkeep", "fallback_file.html"):
                continue
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()


def read_ui_version(dest_folder: Path) -> Optional[str]:
    file = dest_folder / ".uiversion"
    if not file.is_file():
        return None

    with file.open("r") as f:
        return f.read()


def download_and_install_ui(dest_folder: Path, dl_url: str, version: str):
    from io import BytesIO
    from zipfile import ZipFile

    logger.info(f"Downloading {dl_url}")
    resp = requests.get(dl_url, timeout=req_timeout).content
    dest_folder.mkdir(parents=True, exist_ok=True)
    with ZipFile(BytesIO(resp)) as zf:
        for fn in zf.filelist:
            with zf.open(fn) as x:
                destfile = dest_folder / fn.filename
                if fn.is_dir():
                    destfile.mkdir(exist_ok=True)
                else:
                    destfile.write_bytes(x.read())
    with (dest_folder / ".uiversion").open("w") as f:
        f.write(version)


def get_ui_download_url(version: Optional[str] = None) -> tuple[str, str]:
    base_url = "https://api.github.com/repos/freqtrade/frequi/"
    # Get base UI Repo path

    resp = requests.get(f"{base_url}releases", timeout=req_timeout)
    resp.raise_for_status()
    r = resp.json()

    if version:
        tmp = [x for x in r if x["name"] == version]
        if tmp:
            latest_version = tmp[0]["name"]
            assets = tmp[0].get("assets", [])
        else:
            raise ValueError("UI-Version not found.")
    else:
        latest_version = r[0]["name"]
        assets = r[0].get("assets", [])
    dl_url = ""
    if assets and len(assets) > 0:
        dl_url = assets[0]["browser_download_url"]

    # URL not found - try assets url
    if not dl_url:
        assets = r[0]["assets_url"]
        resp = requests.get(assets, timeout=req_timeout)
        r = resp.json()
        dl_url = r[0]["browser_download_url"]

    return dl_url, latest_version
