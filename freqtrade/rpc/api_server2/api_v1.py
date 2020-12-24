from typing import Dict

from fastapi import APIRouter


router = APIRouter()


@router.get('/ping')
def _ping() -> Dict[str, str]:
    """simple ping version"""
    return {"status": "pong"}


