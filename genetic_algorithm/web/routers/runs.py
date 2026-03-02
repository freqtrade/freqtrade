"""
Runs API — list, start, stop, pause, resume, checkpoint, inject.

All endpoints delegate to RunManager and DataService.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

from genetic_algorithm.web.models.run import (
    InjectStrategyRequest,
    RunDetail,
    RunSummary,
    StartRunRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/runs", tags=["runs"])


def _data(request: Request):
    return request.app.state.data_service


def _mgr(request: Request):
    return request.app.state.run_manager


# ── List / Get ─────────────────────────────────────────────────

@router.get("", response_model=List[RunSummary])
async def list_runs(request: Request):
    """List all runs (active + past)."""
    return _data(request).list_runs()


@router.get("/{run_id}", response_model=RunDetail)
async def get_run(run_id: str, request: Request):
    """Get detailed information about a run."""
    detail = _data(request).get_run_detail(run_id)
    if not detail:
        raise HTTPException(404, f"Run {run_id} not found")
    return detail


@router.get("/{run_id}/config")
async def get_run_config(run_id: str, request: Request):
    """Get the config used for a run."""
    detail = _data(request).get_run_detail(run_id)
    if not detail:
        raise HTTPException(404, f"Run {run_id} not found")
    return detail.config


# ── Start ──────────────────────────────────────────────────────

@router.post("", response_model=RunSummary)
async def start_run(body: StartRunRequest, request: Request):
    """Start a new evolution run."""
    try:
        handle = _mgr(request).start_run(
            config=body.config,
            run_id=body.run_id,
            resume_from=body.resume_from,
        )
        return handle.to_summary()
    except Exception as e:
        logger.exception("Failed to start run")
        raise HTTPException(500, f"Failed to start run: {e}")


# ── Control ────────────────────────────────────────────────────

@router.post("/{run_id}/stop")
async def stop_run(run_id: str, request: Request):
    """Stop a running evolution (graceful — saves checkpoint)."""
    ok = _mgr(request).stop_run(run_id)
    if not ok:
        raise HTTPException(400, f"Cannot stop run {run_id} (not running)")
    return {"status": "stopping", "run_id": run_id}


@router.post("/{run_id}/pause")
async def pause_run(run_id: str, request: Request):
    """Pause a running evolution."""
    ok = _mgr(request).pause_run(run_id)
    if not ok:
        raise HTTPException(400, f"Cannot pause run {run_id}")
    return {"status": "paused", "run_id": run_id}


@router.post("/{run_id}/resume")
async def resume_run(run_id: str, request: Request):
    """Resume a paused evolution."""
    ok = _mgr(request).resume_run(run_id)
    if not ok:
        raise HTTPException(400, f"Cannot resume run {run_id}")
    return {"status": "running", "run_id": run_id}


@router.post("/{run_id}/checkpoint")
async def save_checkpoint(run_id: str, request: Request):
    """Request an immediate checkpoint save."""
    ok = _mgr(request).save_checkpoint(run_id)
    if not ok:
        raise HTTPException(400, f"Cannot checkpoint run {run_id}")
    return {"status": "checkpoint_requested", "run_id": run_id}


@router.post("/{run_id}/inject")
async def inject_strategy(run_id: str, body: InjectStrategyRequest, request: Request):
    """Inject a strategy into a running evolution."""
    ok = _mgr(request).inject_strategy(run_id, body.strategy_gene)
    if not ok:
        raise HTTPException(400, f"Cannot inject into run {run_id}")
    return {"status": "injected", "run_id": run_id}
