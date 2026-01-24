import scripts.list_icici_contracts as lic


def _setup_master(monkeypatch, options, futures) -> None:
    monkeypatch.setattr(lic, "find_latest_master_file", lambda _: "dummy")
    monkeypatch.setattr(
        lic,
        "load_nfo_options_master",
        lambda _: {"by_contract": options, "by_future": futures},
    )


def test_list_contracts_filters_option_output(monkeypatch, capsys) -> None:
    options = {
        ("RELIANCE", "20240101", 2500.0, "CE"): {},
        ("RELIANCE", "20240101", 2500.0, "PE"): {},
    }
    futures = {("RELIANCE", "20240101"): {}}
    _setup_master(monkeypatch, options, futures)

    exit_code = lic.list_contracts(
        expiries=1,
        strikes=1,
        underlyings=5,
        underlying_filter=None,
        kind="opt",
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "OPT" in output
    assert "FUT" not in output


def test_list_contracts_filters_future_output(monkeypatch, capsys) -> None:
    options = {("RELIANCE", "20240101", 2500.0, "CE"): {}}
    futures = {("RELIANCE", "20240101"): {}}
    _setup_master(monkeypatch, options, futures)

    exit_code = lic.list_contracts(
        expiries=1,
        strikes=1,
        underlyings=5,
        underlying_filter=None,
        kind="fut",
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "FUT" in output
    assert "OPT" not in output


def test_list_contracts_respects_underlying_filter_order(monkeypatch, capsys) -> None:
    options = {("RELIANCE", "20240101", 2500.0, "CE"): {}}
    futures = {("RELIANCE", "20240101"): {}, ("TCS", "20240105"): {}}
    _setup_master(monkeypatch, options, futures)

    exit_code = lic.list_contracts(
        expiries=1,
        strikes=1,
        underlyings=1,
        underlying_filter=["TCS", "RELIANCE"],
        kind="both",
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    first_index = output.find("Underlying: TCS")
    second_index = output.find("Underlying: RELIANCE")
    assert first_index != -1
    assert second_index != -1
    assert first_index < second_index
