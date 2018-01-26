# pragma pylint: disable=missing-docstring,W0212

from user_data.hyperopt_conf import hyperopt_optimize_conf


def test_hyperopt_optimize_conf():
    hyperopt_conf = hyperopt_optimize_conf()

    assert "max_open_trades" in hyperopt_conf
    assert "stake_currency" in hyperopt_conf
    assert "stake_amount" in hyperopt_conf
    assert "minimal_roi" in hyperopt_conf
    assert "stoploss" in hyperopt_conf
    assert "bid_strategy" in hyperopt_conf
    assert "exchange" in hyperopt_conf
    assert "pair_whitelist" in hyperopt_conf['exchange']
