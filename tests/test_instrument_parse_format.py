import pytest

from adapters.ccxt_shim.instrument import InstrumentSpec, InstrumentType, format_pair, parse_pair


@pytest.mark.parametrize(
    ("pair", "expected_type"),
    [
        ("RELIANCE/INR", InstrumentType.CASH),
        ("NIFTY-20260226-FUT/INR", InstrumentType.FUT),
        ("NIFTY-20260226-22500-CE/INR", InstrumentType.OPT),
    ],
)
def test_parse_pair_canonical(pair: str, expected_type: InstrumentType) -> None:
    spec = parse_pair(pair)
    assert spec.type == expected_type
    assert format_pair(spec) == pair


def test_parse_pair_invalid() -> None:
    with pytest.raises(ValueError):
        parse_pair("NIFTY-20260226-22500/INR")


def test_format_pair_validation() -> None:
    spec = InstrumentSpec(type=InstrumentType.OPT, underlying="NIFTY", expiry_yyyymmdd="20260226")
    with pytest.raises(ValueError):
        format_pair(spec)
