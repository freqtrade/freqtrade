import logging

from fastapi import APIRouter
from fastapi.exceptions import HTTPException

from freqtrade.rpc.api_server.api_schemas import BackgroundTaskStatus
from freqtrade.rpc.api_server.webserver_bgwork import ApiBG, JobsContainer


logger = logging.getLogger(__name__)

# Private API, protected by authentication and webserver_mode dependency
router = APIRouter()


def _create_background_task_response(jobid: str, job: JobsContainer) -> BackgroundTaskStatus:
    return BackgroundTaskStatus(
        job_id=jobid,
        job_category=job["category"],
        status=job["status"],
        running=job["is_running"],
        progress=job.get("progress"),
        progress_tasks=job.get("progress_tasks"),
        error=job.get("error", None),
    )


@router.get("/background", response_model=list[BackgroundTaskStatus])
def background_job_list():
    return [_create_background_task_response(jobid, job) for jobid, job in ApiBG.jobs.items()]


@router.get("/background/{jobid}", response_model=BackgroundTaskStatus)
def background_job(jobid: str):
    if not (job := ApiBG.jobs.get(jobid)):
        raise HTTPException(status_code=404, detail="Job not found.")

    return _create_background_task_response(jobid, job)


@router.delete(
    "/background/clear",
    response_model=list[BackgroundTaskStatus],
    description="Delete all background jobs that are not running. Returns not deleted jobs.",
)
def background_job_delete_all():
    for jobid, job in list(ApiBG.jobs.items()):
        if job["is_running"]:
            continue
        del ApiBG.jobs[jobid]
    return [_create_background_task_response(jobid, job) for jobid, job in ApiBG.jobs.items()]


@router.delete("/background/{jobid}", response_model=BackgroundTaskStatus)
def background_job_delete(jobid: str):
    if not (job := ApiBG.jobs.get(jobid)):
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["is_running"]:
        raise HTTPException(status_code=400, detail="Job is still running.")

    del ApiBG.jobs[jobid]

    return _create_background_task_response(jobid, job)
