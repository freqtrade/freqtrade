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


@router_ui.get('/{rest_of_path:path}', include_in_schema=False)
async def index_html(rest_of_path: str):
    """
    Emulate path fallback to index.html.
    """
    if rest_of_path.startswith('api') or rest_of_path.startswith('.'):
        raise HTTPException(status_code=404, detail="Not Found")
    uibase = Path(__file__).parent / 'ui/installed/'
    if (uibase / rest_of_path).is_file():
        return FileResponse(str(uibase / rest_of_path))

    index_file = uibase / 'index.html'
    if not index_file.is_file():
        return FileResponse(str(uibase.parent / 'fallback_file.html'))
    # Fall back to index.html, as indicated by vue router docs
    return FileResponse(str(index_file))
