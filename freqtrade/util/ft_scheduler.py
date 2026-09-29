import logging
from datetime import datetime

from schedule import Job, Scheduler

from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


class FtScheduler(Scheduler):
    """
    Scheduler which logs exceptions raised by a job and reschedules it.
    """

    def _run_job(self, job: Job) -> None:
        try:
            super()._run_job(job)
        except OperationalException:
            raise
        except Exception:
            logger.exception(f"Error in scheduled job {job}.")
            # Job.run() only reschedules after the job returned.
            # Reschedule manually, otherwise the job stays due and re-runs on every iteration.
            job.last_run = datetime.now()  # noqa: DTZ005 - schedule uses naive local time
            job._schedule_next_run()
