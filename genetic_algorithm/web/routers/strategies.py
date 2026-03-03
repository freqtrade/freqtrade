"""
Strategies API — inspect individual strategies, view code, compare.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

from genetic_algorithm.web.models.strategy import StrategyDetail, LineageResponse

router = APIRouter(prefix="/api", tags=["strategies"])


def _data(request: Request):
    return request.app.state.data_service


@router.get("/runs/{run_id}/strategies/{strategy_id}", response_model=StrategyDetail)
async def get_strategy(run_id: str, strategy_id: str, request: Request):
    """Get detailed info about a specific strategy."""
    detail = _data(request).get_strategy(run_id, strategy_id)
    if not detail:
        raise HTTPException(404, f"Strategy {strategy_id} not found in run {run_id}")
    return detail


@router.get("/runs/{run_id}/strategies/{strategy_id}/code")
async def get_strategy_code(run_id: str, strategy_id: str, request: Request):
    """Get generated Python code for a strategy."""
    code = _data(request).get_strategy_code(run_id, strategy_id)
    if code is None:
        raise HTTPException(404, f"Cannot generate code for {strategy_id}")
    return {"strategy_id": strategy_id, "code": code}


@router.get("/hall-of-fame")
async def get_hall_of_fame(request: Request):
    """Get all Hall of Fame entries."""
    return _data(request).get_hall_of_fame()


@router.get("/runs/{run_id}/lineage/{strategy_id}", response_model=LineageResponse)
async def get_lineage(run_id: str, strategy_id: str, request: Request):
    """Trace a strategy's ancestral lineage through parent chain."""
    chain = _data(request).get_lineage(run_id, strategy_id)
    return LineageResponse(strategy_id=strategy_id, run_id=run_id, chain=chain)
