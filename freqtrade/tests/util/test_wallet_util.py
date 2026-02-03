import pytest

from freqtrade.util import get_dry_run_wallet


@pytest.mark.parametrize(
    "wallet,stake_currency,expected",
    [
        (1000, "USDT", 1000),
        ({"USDT": 1000, "USDC": 500}, "USDT", 1000),
        ({"USDT": 1000, "USDC": 500}, "USDC", 500),
        ({"USDT": 1000, "USDC": 500}, "NOCURR", 0.0),
    ],
)
def test_get_dry_run_wallet(default_conf_usdt, wallet, stake_currency, expected):
    # As int
    default_conf_usdt["dry_run_wallet"] = wallet
    default_conf_usdt["stake_currency"] = stake_currency
    assert get_dry_run_wallet(default_conf_usdt) == expected
