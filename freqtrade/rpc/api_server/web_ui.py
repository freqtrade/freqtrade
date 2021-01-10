from pathlib import Path

from fastapi import APIRouter
from fastapi.exceptions import HTTPException
from starlette.responses import FileResponse


router_ui = APIRouter()


@router_ui.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(Path(__file__).parent / 'ui/favicon.ico')


@router_ui.get('/{rest_of_path:path}', include_in_schema=False)
async def index_html(rest_of_path: str):
    """
    Emulate path fallback to index.html.
    """
    if rest_of_path.startswith('api') or rest_of_path.startswith('.'):
        raise HTTPException(status_code=404, detail="Not Found")
    uibase = Path(__file__).parent / 'ui'
    if (uibase / rest_of_path).is_file():
        return FileResponse(uibase / rest_of_path)

    # Fall back to index.html, as indicated by vue router docs
    return FileResponse(uibase / 'index.html')
