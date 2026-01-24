from datetime import date

from pathlib import Path

from adapters.ccxt_shim.security_master import SecurityMaster
from scripts.gen_option_whitelist import WhitelistInputs, _compute_step, generate_option_whitelist


def test_compute_step_prefers_most_common_diff() -> None:
    strikes = [100.0, 110.0, 120.0, 140.0]
    assert _compute_step(strikes) == 10.0


def test_generate_option_whitelist_uses_median_spot_when_ticker_missing() -> None:
    contracts = {
        ("RELIANCE", "20250130", 2400.0, "CE"): {},
        ("RELIANCE", "20250130", 2400.0, "PE"): {},
        ("RELIANCE", "20250130", 2500.0, "CE"): {},
        ("RELIANCE", "20250130", 2500.0, "PE"): {},
        ("RELIANCE", "20250130", 2600.0, "CE"): {},
        ("RELIANCE", "20250130", 2600.0, "PE"): {},
    }
    security_master = SecurityMaster(contracts)
    inputs = WhitelistInputs(
        underlying="RELIANCE",
        expiry_policy="nearest",
        atm_breadth=2,
        n_expiries=1,
        out_path=Path("unused.json"),
        mode="mock",
    )

    pairs = generate_option_whitelist(
        security_master=security_master,
        inputs=inputs,
        today=date(2025, 1, 1),
        spot_fetcher=None,
    )

    assert "RELIANCE/INR" in pairs
    assert "RELIANCE-20250130-2400-CE/INR" in pairs
    assert "RELIANCE-20250130-2400-PE/INR" in pairs
    assert len(pairs) == 7


def test_generate_option_whitelist_skips_missing_rights() -> None:
    contracts = {
        ("RELIANCE", "20250130", 100.0, "CE"): {},
        ("RELIANCE", "20250130", 150.0, "CE"): {},
        ("RELIANCE", "20250130", 150.0, "PE"): {},
    }
    security_master = SecurityMaster(contracts)
    inputs = WhitelistInputs(
        underlying="RELIANCE",
        expiry_policy="nearest",
        atm_breadth=2,
        n_expiries=1,
        out_path=Path("unused.json"),
        mode="mock",
    )

    pairs = generate_option_whitelist(
        security_master=security_master,
        inputs=inputs,
        today=date(2025, 1, 1),
        spot_fetcher=None,
    )

    assert "RELIANCE-20250130-100-CE/INR" in pairs
    assert "RELIANCE-20250130-100-PE/INR" not in pairs
    assert "RELIANCE-20250130-150-CE/INR" in pairs
    assert "RELIANCE-20250130-150-PE/INR" in pairs
    assert len(pairs) == 1 + 3


def test_generate_option_whitelist_requires_ce_pe_pair() -> None:
    contracts = {
        ("RELIANCE", "20250130", 100.0, "CE"): {},
        ("RELIANCE", "20250130", 150.0, "CE"): {},
    }
    security_master = SecurityMaster(contracts)
    inputs = WhitelistInputs(
        underlying="RELIANCE",
        expiry_policy="nearest",
        atm_breadth=2,
        n_expiries=1,
        out_path=Path("unused.json"),
        mode="mock",
    )

    pairs = generate_option_whitelist(
        security_master=security_master,
        inputs=inputs,
        today=date(2025, 1, 1),
        spot_fetcher=None,
    )

    assert pairs == []
