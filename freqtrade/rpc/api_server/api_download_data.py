import logging
import time
from copy import deepcopy

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.exceptions import HTTPException

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import FtNoDBContext
from freqtrade.rpc.api_server.api_pairlists import handleExchangePayload
from freqtrade.rpc.api_server.api_schemas import BgJobStarted, DownloadDataPayload
from freqtrade.rpc.api_server.deps import RateLimiter, get_config, get_exchange
from freqtrade.rpc.api_server.webserver_bgwork import ApiBG
from freqtrade.util.progress_tracker import get_progress_tracker


logger = logging.getLogger(__name__)

# Private API, protected by authentication and webserver_mode dependency
router = APIRouter()


def __run_download(job_id: str, config_loc: Config):
    try:
        ApiBG.jobs[job_id]["is_running"] = True
        from freqtrade.data.history.history_utils import download_data

        with FtNoDBContext():
            exchange = get_exchange(config_loc)
            last_refresh = [0.0]

            def ft_callback(task) -> None:
                ApiBG.jobs[job_id]["progress_tasks"][str(task.id)] = {
                    "progress": task.completed,
                    "total": task.total,
                    "description": task.description,
                }
                if time.time() - last_refresh[0] > 60:
                    if job := ApiBG.jobs.get(job_id):
                        ApiBG.jobs[job_id] = job
                        last_refresh[0] = time.time()

            pt = get_progress_tracker(ft_callback=ft_callback)

            download_data(config_loc, exchange, progress_tracker=pt)
            ApiBG.jobs[job_id]["status"] = "success"
    except (OperationalException, Exception) as e:
        logger.exception(e)
        ApiBG.jobs[job_id]["error"] = str(e)
        ApiBG.jobs[job_id]["status"] = "failed"
    finally:
        if job := ApiBG.jobs.get(job_id):
            job["is_running"] = False
            ApiBG.jobs[job_id] = job
        ApiBG.download_data_running = False


@router.post(
    "/download_data",
    response_model=BgJobStarted,
    dependencies=[Depends(RateLimiter(max_calls=2, time_seconds=600))],
)
def pairlists_evaluate(
    payload: DownloadDataPayload, background_tasks: BackgroundTasks, config=Depends(get_config)
):
    if ApiBG.download_data_running:
        raise HTTPException(status_code=400, detail="Data Download is already running.")
    config_loc = deepcopy(config)
    config_loc["stake_currency"] = ""
    config_loc["pairs"] = payload.pairs
    if payload.timerange:
        config_loc["timerange"] = payload.timerange
    config_loc["days"] = payload.days
    config_loc["timeframes"] = payload.timeframes
    config_loc["erase"] = payload.erase
    config_loc["download_trades"] = payload.download_trades
    config_loc["prepend_data"] = payload.prepend_data
    if payload.candle_types is not None:
        config_loc["candle_types"] = payload.candle_types

    handleExchangePayload(payload, config_loc)

    job_id = ApiBG.get_job_id()

    ApiBG.jobs[job_id] = {
        "category": "download_data",
        "status": "pending",
        "progress": None,
        "progress_tasks": {},
        "is_running": False,
        "result": {},
        "error": None,
    }
    background_tasks.add_task(__run_download, job_id, config_loc)
    ApiBG.download_data_running = True

    return {
        "status": "Data Download started in background.",
        "job_id": job_id,
    }
