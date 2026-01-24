from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from adapters.ccxt_shim.instrument import InstrumentType, parse_pair
from adapters.ccxt_shim.security_master import SecurityMaster, load_nfo_options_master
from scripts.gen_option_whitelist import WhitelistInputs, generate_option_whitelist


def _load_security_master() -> SecurityMaster:
    master_path = Path("user_data/data/icicibreeze/FONSEScripMaster.txt")
    if not master_path.exists():
        pytest.skip("FONSEScripMaster.txt not available in this environment.")
    master = load_nfo_options_master(str(master_path))
    return SecurityMaster(master.get("by_contract", {}))


def _pick_underlying_with_ce_pe(security_master: SecurityMaster) -> str:
    rights_by_contract: dict[tuple[str, str, float], set[str]] = {}
    for underlying, expiry, strike, right in security_master.by_contract:
        rights_by_contract.setdefault((underlying, expiry, float(strike)), set()).add(right)
    candidates = [
        underlying
        for (underlying, _expiry, _strike), rights in rights_by_contract.items()
        if rights >= {"CE", "PE"}
    ]
    if not candidates:
        pytest.skip("No underlying with CE+PE pair found in SecurityMaster.")
    return sorted(set(candidates))[0]


def test_generated_pairs_exist_in_master() -> None:
    security_master = _load_security_master()
    underlying = _pick_underlying_with_ce_pe(security_master)
    inputs = WhitelistInputs(
        underlying=underlying,
        expiry_policy="nearest",
        atm_breadth=2,
        n_expiries=1,
        out_path=Path("unused.json"),
        mode="mock",
    )

    pairs = generate_option_whitelist(
        security_master=security_master,
        inputs=inputs,
        today=date.today(),
        spot_fetcher=None,
    )

    for pair in pairs:
        spec = parse_pair(pair)
        if spec.type != InstrumentType.OPT:
            continue
        key = (
            spec.underlying,
            spec.expiry_yyyymmdd,
            float(spec.strike),
            spec.right,
        )
        assert key in security_master.by_contract
