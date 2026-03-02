"""
Generations API — drill down into a specific generation of a run.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from genetic_algorithm.web.models.generation import GenerationDetail

router = APIRouter(prefix="/api/runs/{run_id}/generations", tags=["generations"])


def _data(request: Request):
    return request.app.state.data_service


@router.get("/{gen}", response_model=GenerationDetail)
async def get_generation(run_id: str, gen: int, request: Request):
    """Get all individuals in a specific generation."""
    detail = _data(request).get_generation(run_id, gen)
    if not detail:
        raise HTTPException(404, f"Generation {gen} not found for run {run_id}")
    return detail
