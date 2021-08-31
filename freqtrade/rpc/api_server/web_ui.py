from pathlib import Path

from fastapi import APIRouter
from fastapi.exceptions import HTTPException
from starlette.responses import FileResponse


router_ui = APIRouter()


@router_ui.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(str(Path(__file__).parent / 'ui/favicon.ico'))


@router_ui.get('/fallback_file.html', include_in_schema=False)
async def fallback():
    return FileResponse(str(Path(__file__).parent / 'ui/fallback_file.html'))


@router_ui.get('/ui_version', include_in_schema=False)
async def ui_version():
    from freqtrade.commands.deploy_commands import read_ui_version
    uibase = Path(__file__).parent / 'ui/installed/'
    version = read_ui_version(uibase)

    return {
        "version": version if version else "not_installed",
    }


def is_relative_to(path, base) -> bool:
    # Helper function simulating behaviour of is_relative_to, which was only added in python 3.9
    try:
        path.relative_to(base)
        return True
    except ValueError:
        pass
    return False


@router_ui.get('/{rest_of_path:path}', include_in_schema=False)
async def index_html(rest_of_path: str):
    """
    Emulate path fallback to index.html.
    """
    if rest_of_path.startswith('api') or rest_of_path.startswith('.'):
        raise HTTPException(status_code=404, detail="Not Found")
    uibase = Path(__file__).parent / 'ui/installed/'
    filename = uibase / rest_of_path
    # It's security relevant to check "relative_to".
    # Without this, Directory-traversal is possible.
    if filename.is_file() and is_relative_to(filename, uibase):
        return FileResponse(str(filename))

    index_file = uibase / 'index.html'
    if not index_file.is_file():
        return FileResponse(str(uibase.parent / 'fallback_file.html'))
    # Fall back to index.html, as indicated by vue router docs
    return FileResponse(str(index_file))
