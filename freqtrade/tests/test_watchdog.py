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
