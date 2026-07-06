import asyncio
import logging
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.exceptions import HTTPException

from freqtrade.constants import Config
from freqtrade.enums import RunMode
from freqtrade.exceptions import ConfigurationError, DependencyException, OperationalException
from freqtrade.misc import deep_merge_dicts
from freqtrade.rpc.api_server.api_schemas import (
    BgJobStarted,
    LookaheadAnalysisRequest,
    LookaheadAnalysisResponse,
    RecursiveAnalysisRequest,
    RecursiveAnalysisResponse,
)
from freqtrade.rpc.api_server.deps import get_config, verify_strategy
from freqtrade.rpc.api_server.webserver_bgwork import ApiBG
from freqtrade.util import get_progress_tracker


logger = logging.getLogger(__name__)

router_recursive = APIRouter()
router_lookahead = APIRouter()


def __resolve_strategy_object(config_loc: Config) -> dict:
    """Resolve the strategy object the way analysis functions expect it"""
    from freqtrade.resolvers.strategy_resolver import StrategyResolver

    strategy_obj = next(
        (
            s
            for s in StrategyResolver.search_all_objects(
                config_loc,
                enum_failed=False,
                recursive=config_loc.get("recursive_strategy_search", False),
            )
            if s["name"] == config_loc["strategy"]
        ),
        None,
    )
    if not strategy_obj:
        raise ConfigurationError(f"Strategy {config_loc['strategy']} not found.")
    return strategy_obj


def __run_analysis_bg(
    config_loc: Config,
    job_id: str,
    analysis_name: Literal["Recursive", "Lookahead"],
    run_analysis: Callable[[Config, Any], dict],
):
    """
    :param config_loc: The configuration to use for the analysis
    :param job_id: The job ID for the analysis
    :param analysis_name: The name of the analysis |
    :param run_analysis: The function to run the analysis.
    """
    job = ApiBG.jobs[job_id]
    job["is_running"] = True
    asyncio.set_event_loop(asyncio.new_event_loop())
    try:

        def ft_callback(task) -> None:
            job["progress_tasks"][str(task.id)] = {
                "progress": task.completed,
                "total": task.total,
                "description": task.description,
            }

        pt = get_progress_tracker(ft_callback=ft_callback)
        job["result"] = run_analysis(config_loc, pt)
        job["status"] = "success"
    except ConfigurationError as e:
        logger.error(f"{analysis_name} analysis encountered a configuration Error: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
    except (Exception, OperationalException, DependencyException) as e:
        logger.exception(f"{analysis_name} analysis caused an error: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
    finally:
        job["is_running"] = False
        ApiBG.analysis_running = False


def __run_recursive_analysis_bg(config_loc: Config, job_id: str):
    from freqtrade.optimize.analysis.recursive_helpers import RecursiveAnalysisSubFunctions

    def run_analysis(config_loc: Config, pt: Any) -> dict:
        config_loc = RecursiveAnalysisSubFunctions.calculate_config_overrides(config_loc)
        strategy_obj = __resolve_strategy_object(config_loc)
        instance = RecursiveAnalysisSubFunctions.initialize_single_recursive_analysis(
            config_loc, strategy_obj, pt
        )
        return {
            "strategy": instance.strategy_obj["name"],
            "startup_candles": instance._startup_candle,
            "strategy_scc": instance._strat_scc,
            "results": {
                indicator: {str(candle): diff for candle, diff in values.items()}
                for indicator, values in instance.dict_recursive.items()
            },
        }

    __run_analysis_bg(config_loc, job_id, "Recursive", run_analysis)


@router_recursive.post("/recursive_analysis", response_model=BgJobStarted)
def api_start_recursive_analysis(
    payload: RecursiveAnalysisRequest,
    background_tasks: BackgroundTasks,
    config=Depends(get_config),
):
    if ApiBG.analysis_running:
        raise HTTPException(status_code=400, detail="Analysis is already running.")

    verify_strategy(payload.strategy)

    config_loc = deepcopy(config)
    config_loc["runmode"] = RunMode.UTIL_NO_EXCHANGE
    settings = dict(payload)
    config_loc = deep_merge_dicts(settings, config_loc, allow_null_overrides=False)

    job_id = ApiBG.get_job_id()
    ApiBG.jobs[job_id] = {
        "category": "recursive_analysis",
        "status": "pending",
        "progress": None,
        "progress_tasks": {},
        "is_running": False,
        "result": {},
        "error": None,
    }
    background_tasks.add_task(__run_recursive_analysis_bg, config_loc, job_id)
    ApiBG.analysis_running = True

    return {
        "status": "Recursive analysis started in background.",
        "job_id": job_id,
    }


def __api_get_status(jobid: str, jobcategory=str):
    if not (job := ApiBG.jobs.get(jobid)) or job["category"] != jobcategory:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["is_running"] or job["status"] == "pending":
        return {"status": "running", "running": True, "status_msg": "Analysis running"}
    if job["status"] == "failed":
        return {
            "status": "error",
            "running": False,
            "status_msg": f"Analysis failed with {job['error']}",
        }
    return {
        "status": "ended",
        "running": False,
        "status_msg": "Analysis ended",
        "result": job["result"],
    }


@router_recursive.get("/recursive_analysis/{jobid}", response_model=RecursiveAnalysisResponse)
def api_get_recursive_analysis(jobid: str):
    return __api_get_status(jobid, "recursive_analysis")


@router_lookahead.get("/lookahead_analysis/{jobid}", response_model=LookaheadAnalysisResponse)
def api_get_lookahead_analysis(jobid: str):
    return __api_get_status(jobid, "lookahead_analysis")


def __run_lookahead_analysis_bg(config_loc: Config, job_id: str):
    from freqtrade.optimize.analysis.lookahead_helpers import LookaheadAnalysisSubFunctions

    def run_analysis(config_loc: Config, pt: Any) -> dict:
        config_loc = LookaheadAnalysisSubFunctions.calculate_config_overrides(config_loc)
        strategy_obj = __resolve_strategy_object(config_loc)
        instance = LookaheadAnalysisSubFunctions.initialize_single_lookahead_analysis(
            config_loc, strategy_obj, pt
        )
        analysis = instance.current_analysis
        return {
            "strategy": instance.strategy_obj["name"],
            "has_bias": analysis.has_bias,
            "total_signals": int(analysis.total_signals),
            "biased_entry_signals": int(analysis.false_entry_signals),
            "biased_exit_signals": int(analysis.false_exit_signals),
            "biased_indicators": analysis.false_indicators,
        }

    __run_analysis_bg(config_loc, job_id, "Lookahead", run_analysis)


@router_lookahead.post("/lookahead_analysis", response_model=BgJobStarted)
def api_start_lookahead_analysis(
    payload: LookaheadAnalysisRequest,
    background_tasks: BackgroundTasks,
    config=Depends(get_config),
):
    if ApiBG.analysis_running:
        raise HTTPException(status_code=400, detail="Analysis is already running.")

    verify_strategy(payload.strategy)

    config_loc = deepcopy(config)
    config_loc["runmode"] = RunMode.UTIL_NO_EXCHANGE
    settings = dict(payload)
    config_loc = deep_merge_dicts(settings, config_loc, allow_null_overrides=False)

    job_id = ApiBG.get_job_id()
    ApiBG.jobs[job_id] = {
        "category": "lookahead_analysis",
        "status": "pending",
        "progress": None,
        "progress_tasks": {},
        "is_running": False,
        "result": {},
        "error": None,
    }
    background_tasks.add_task(__run_lookahead_analysis_bg, config_loc, job_id)
    ApiBG.analysis_running = True

    return {
        "status": "Lookahead analysis started in background.",
        "job_id": job_id,
    }
