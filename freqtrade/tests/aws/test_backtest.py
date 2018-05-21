import pytest

from freqtrade.aws.backtesting_lambda import backtest


def test_backtest():
    backtest({}, {})
