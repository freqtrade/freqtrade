from __future__ import annotations

from datetime import date

import pytest

from adapters.ccxt_shim.instrument import InstrumentSpec, InstrumentType, format_pair
from adapters.ccxt_shim.security_master import SecurityMaster
from scripts.gen_option_whitelist import OptionSelection
from scripts.universe_scan_and_generate_pairs import (
    OptionPolicy,
    UniverseConfig,
    _build_pairs_report,
)


def _make_option_pair(underlying: str, expiry: str, strike: float, right: str) -> str:
    return format_pair(
        InstrumentSpec(
            type=InstrumentType.OPT,
            underlying=underlying,
            quote="INR",
            expiry_yyyymmdd=expiry,
            strike=strike,
            right=right,
        )
    )


def _make_selection(
    option_pairs: list[str],
    option_count: int,
    ce_pe_pairs_count: int,
    expiry: str = "20250101",
    strike: float = 100.0,
) -> OptionSelection:
    return OptionSelection(
        option_pairs=option_pairs,
        selected_expiries=[expiry],
        atm_strike_by_expiry={expiry: strike},
        option_count=option_count,
        ce_pe_pairs_count=ce_pe_pairs_count,
    )


def _run_report_with_selection(
    monkeypatch: pytest.MonkeyPatch,
    selections: dict[str, OptionSelection],
    universe: UniverseConfig,
    option_policy: OptionPolicy,
) -> tuple[list[str], dict[str, object]]:
    def _mock_scan(_security_master, underlying: str, _today: date, _policy: OptionPolicy) -> OptionSelection:
        return selections[underlying]

    monkeypatch.setattr(
        "scripts.universe_scan_and_generate_pairs._scan_underlying",
        _mock_scan,
    )
    security_master = SecurityMaster({})
    return _build_pairs_report(universe, option_policy, security_master, date(2025, 1, 1))


def test_index_skips_no_options_records_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    universe = UniverseConfig(indices=["BANKNIFTY"], stocks=[], top_n_stocks=None, total_pairs_cap=None)
    option_policy = OptionPolicy(
        expiry_policy="nearest",
        atm_breadth=2,
        total_pairs_cap=None,
        require_two_sided=True,
        include_cash_pair=None,
    )
    selection = _make_selection(option_pairs=[], option_count=0, ce_pe_pairs_count=0)

    pairs, report = _run_report_with_selection(
        monkeypatch,
        {"BANKNIFTY": selection},
        universe,
        option_policy,
    )

    assert pairs == []
    assert report["selected_indices"] == []
    assert report["skipped_underlyings"] == [
        {"underlying": "BANKNIFTY", "reason": "no options"}
    ]
    assert report["chosen_expiry"]["BANKNIFTY"] == "20250101"
    assert report["chosen_atm_strike"]["BANKNIFTY"] == 100.0


def test_index_skips_when_missing_two_sided(monkeypatch: pytest.MonkeyPatch) -> None:
    universe = UniverseConfig(indices=["NIFTY"], stocks=[], top_n_stocks=None, total_pairs_cap=None)
    option_policy = OptionPolicy(
        expiry_policy="nearest",
        atm_breadth=2,
        total_pairs_cap=None,
        require_two_sided=True,
        include_cash_pair=None,
    )
    option_pairs = [_make_option_pair("NIFTY", "20250101", 100.0, "CE")]
    selection = _make_selection(option_pairs=option_pairs, option_count=1, ce_pe_pairs_count=0)

    pairs, report = _run_report_with_selection(
        monkeypatch,
        {"NIFTY": selection},
        universe,
        option_policy,
    )

    assert pairs == []
    assert report["selected_indices"] == []
    assert report["skipped_underlyings"] == [
        {"underlying": "NIFTY", "reason": "no CE+PE available"}
    ]


def test_index_includes_cash_pair_only_when_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    universe = UniverseConfig(indices=["NIFTY"], stocks=[], top_n_stocks=None, total_pairs_cap=None)
    option_policy = OptionPolicy(
        expiry_policy="nearest",
        atm_breadth=2,
        total_pairs_cap=None,
        require_two_sided=True,
        include_cash_pair=None,
    )
    option_pairs = [
        _make_option_pair("NIFTY", "20250101", 100.0, "CE"),
        _make_option_pair("NIFTY", "20250101", 100.0, "PE"),
    ]
    selection = _make_selection(option_pairs=option_pairs, option_count=2, ce_pe_pairs_count=1)

    pairs, report = _run_report_with_selection(
        monkeypatch,
        {"NIFTY": selection},
        universe,
        option_policy,
    )

    cash_pair = format_pair(
        InstrumentSpec(type=InstrumentType.CASH, underlying="NIFTY", quote="INR")
    )
    assert pairs == [cash_pair, *option_pairs]
    assert report["selected_indices"] == ["NIFTY"]


def test_stock_default_excludes_cash_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    universe = UniverseConfig(indices=[], stocks=["RELIANCE"], top_n_stocks=None, total_pairs_cap=None)
    option_policy = OptionPolicy(
        expiry_policy="nearest",
        atm_breadth=2,
        total_pairs_cap=None,
        require_two_sided=True,
        include_cash_pair=None,
    )
    option_pairs = [
        _make_option_pair("RELIANCE", "20250101", 100.0, "CE"),
        _make_option_pair("RELIANCE", "20250101", 100.0, "PE"),
    ]
    selection = _make_selection(option_pairs=option_pairs, option_count=2, ce_pe_pairs_count=1)

    pairs, report = _run_report_with_selection(
        monkeypatch,
        {"RELIANCE": selection},
        universe,
        option_policy,
    )

    assert pairs == option_pairs
    assert report["selected_stocks"] == ["RELIANCE"]
