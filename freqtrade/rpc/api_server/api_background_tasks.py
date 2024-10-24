import logging

from fastapi import APIRouter
from fastapi.exceptions import HTTPException

from freqtrade.rpc.api_server.api_schemas import BackgroundTaskStatus
from freqtrade.rpc.api_server.webserver_bgwork import ApiBG


logger = logging.getLogger(__name__)

# Private API, protected by authentication and webserver_mode dependency
router = APIRouter()


@router.get("/background", response_model=list[BackgroundTaskStatus], tags=["webserver"])
def background_job_list():
    return [
        {
            "job_id": jobid,
            "job_category": job["category"],
            "status": job["status"],
            "running": job["is_running"],
            "progress": job.get("progress"),
            "error": job.get("error", None),
        }
        for jobid, job in ApiBG.jobs.items()
    ]


@router.get("/background/{jobid}", response_model=BackgroundTaskStatus, tags=["webserver"])
def background_job(jobid: str):
    if not (job := ApiBG.jobs.get(jobid)):
        raise HTTPException(status_code=404, detail="Job not found.")

    return {
        "job_id": jobid,
        "job_category": job["category"],
        "status": job["status"],
        "running": job["is_running"],
        "progress": job.get("progress"),
        "error": job.get("error", None),
    }
