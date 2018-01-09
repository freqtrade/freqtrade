from freqtrade.watchdog import Watchdog


def test_watchdog_timeout(caplog):
    watchdog = Watchdog(1)
    assert(watchdog.run(0) is False)
    log = ["Watchdog started", "Kill process due to timeout"]
    for line in log:
        assert line in caplog.text


def test_watchdog_kill(caplog):
    watchdog = Watchdog(1)
    watchdog.exit_gracefully(1, 0)
    assert(watchdog.run(0) is False)
    log = ["Watchdog started", "Watchdog stopped"]
    for line in log:
        assert line in caplog.text


def test_try_kill_failed(mocker):
    mocker.patch("os.kill")
    mocker.patch("os.waitpid", return_value=(0, 0))
    watchdog = Watchdog(1, 1)
    assert watchdog.try_kill(0) is False


def test_try_kill_success(mocker):
    mocker.patch("os.kill")
    mocker.patch("os.waitpid", return_value=(0, 1))
    watchdog = Watchdog(1, 1)
    assert watchdog.try_kill(0) is True


def test_try_kill_error(mocker):
    mocker.patch("os.kill")
    mocker.patch("os.waitpid", side_effect=OSError)
    watchdog = Watchdog(1, 1)
    assert watchdog.try_kill(0) is True
