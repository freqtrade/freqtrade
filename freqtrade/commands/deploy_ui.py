import logging
from pathlib import Path

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


def read_ui_version(dest_folder: Path) -> str | None:
    file = dest_folder / ".uiversion"
    if not file.is_file():
        return None

    with file.open("r") as f:
        return f.read()


def download_and_install_ui(dest_folder: Path, dl_url: str, version: str):
    from io import BytesIO
    from zipfile import ZipFile

    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    logger.info(f"Downloading {dl_url}")

    content = BytesIO()
    with requests.get(dl_url, stream=True, timeout=req_timeout) as resp:
        resp.raise_for_status()
        total_length = int(resp.headers.get("content-length", 0))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Downloading FreqUI...", total=total_length)

            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    content.write(chunk)
                    progress.update(task, advance=len(chunk))

    dest_folder.mkdir(parents=True, exist_ok=True)
    content.seek(0)
    with ZipFile(content) as zf:
        for fn in zf.filelist:
            with zf.open(fn) as x:
                destfile = dest_folder / fn.filename
                if fn.is_dir():
                    destfile.mkdir(exist_ok=True)
                else:
                    destfile.write_bytes(x.read())
    with (dest_folder / ".uiversion").open("w") as f:
        f.write(version)


def get_ui_download_url(version: str | None, prerelease: bool) -> tuple[str, str]:
    base_url = "https://api.github.com/repos/freqtrade/frequi/"
    # Get base UI Repo path

    resp = requests.get(f"{base_url}releases", timeout=req_timeout)
    resp.raise_for_status()
    r = resp.json()

    if version:
        tmp = [x for x in r if x["name"] == version]
    else:
        tmp = [x for x in r if prerelease or not x.get("prerelease")]

    if tmp:
        # Ensure we have the latest version
        if version is None:
            tmp.sort(key=lambda x: x["created_at"], reverse=True)
        latest_version = tmp[0]["name"]
        assets = tmp[0].get("assets", [])
    else:
        raise ValueError("UI-Version not found.")

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
