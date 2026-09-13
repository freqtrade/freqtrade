from datetime import datetime, timedelta

import pytest

from freqtrade.exceptions import OperationalException, TemporaryError
from freqtrade.util import FtScheduler
from tests.conftest import log_has_re


@pytest.mark.parametrize("exception", [AttributeError, TemporaryError])
def test_ft_scheduler_error(caplog, exception):
    scheduler = FtScheduler()
    calls = []

    def failing_job():
        raise exception("'str' object has no attribute 'keys'")

    job1 = scheduler.every().day.do(failing_job)
    job2 = scheduler.every().day.do(lambda: calls.append(1))
    # Make both jobs due - the failing job first (jobs run sorted by next_run).
    job1.next_run = datetime.now() - timedelta(seconds=2)
    job2.next_run = datetime.now() - timedelta(seconds=1)

    scheduler.run_pending()

    assert log_has_re(r"Error in scheduled job Job\(.*do=failing_job.*", caplog)
    # The job scheduled after the failing one still ran
    assert calls == [1]
    # Both jobs were rescheduled - and are not due anymore
    assert job1.next_run > datetime.now()
    assert job2.next_run > datetime.now()
    assert job1.last_run is not None
    caplog.clear()

    scheduler.run_pending()
    assert calls == [1]
    assert not log_has_re(r"Error in scheduled job.*", caplog)


def test_ft_scheduler_propagates_operational_exception():
    scheduler = FtScheduler()

    def failing_job():
        raise OperationalException("stop")

    job = scheduler.every().day.do(failing_job)
    job.next_run = datetime.now() - timedelta(seconds=1)

    with pytest.raises(OperationalException, match="stop"):
        scheduler.run_pending()
    # Job was not rescheduled - and will run again when the bot resumes
    assert job.next_run < datetime.now()
