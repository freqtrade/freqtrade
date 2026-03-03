"""
Config API — list templates, load, validate, and save configs.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/config", tags=["config"])


def _data(request: Request):
    return request.app.state.data_service


@router.get("/templates")
async def list_templates(request: Request):
    """List available config templates."""
    return _data(request).get_config_templates()


@router.get("/templates/{name}")
async def get_template(name: str, request: Request):
    """Load a specific config template."""
    config = _data(request).load_config_template(name)
    if config is None:
        raise HTTPException(404, f"Template {name} not found")
    return config


@router.post("/validate")
async def validate_config(body: Dict[str, Any], request: Request):
    """
    Validate a GA config dict.

    Returns {"valid": true/false, "errors": [...], "warnings": [...]}.
    """
    errors = []
    warnings = []

    ga = body.get("genetic_algorithm")
    if not ga:
        errors.append("Missing 'genetic_algorithm' section")
    else:
        if not isinstance(ga.get("population_size"), int) or ga["population_size"] < 2:
            errors.append("population_size must be an integer >= 2")
        if not isinstance(ga.get("generations"), int) or ga["generations"] < 1:
            errors.append("generations must be an integer >= 1")
        mr = ga.get("mutation_rate")
        if mr is not None and not (0 <= mr <= 1):
            errors.append("mutation_rate must be between 0 and 1")
        cr = ga.get("crossover_rate")
        if cr is not None and not (0 <= cr <= 1):
            errors.append("crossover_rate must be between 0 and 1")
        es = ga.get("elite_size", 0)
        ps = ga.get("population_size", 0)
        if es >= ps:
            errors.append("elite_size must be < population_size")

    bt = body.get("backtesting")
    if not bt:
        errors.append("Missing 'backtesting' section")
    else:
        if not bt.get("pairs"):
            errors.append("backtesting.pairs must be a non-empty list")
        if not bt.get("timerange"):
            warnings.append("backtesting.timerange is empty — will use all available data")

    fw = body.get("fitness_weights")
    if fw:
        total = sum(v for v in fw.values() if isinstance(v, (int, float)))
        if abs(total - 1.0) > 0.05:
            warnings.append(f"fitness_weights sum to {total:.2f} (expected ~1.0)")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
