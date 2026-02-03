# pragma pylint: disable=missing-docstring, C0103

import pytest

from freqtrade.enums import HyperoptState
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt_tools import HyperoptStateContainer
from freqtrade.strategy.parameters import (
    BaseParameter,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
)


def test_hyperopt_int_parameter():
    from optuna.distributions import IntDistribution

    HyperoptStateContainer.set_state(HyperoptState.INDICATORS)

    with pytest.raises(OperationalException, match=r"Name is determined.*"):
        IntParameter(low=0, high=5, default=1, name="hello")

    with pytest.raises(OperationalException, match=r"IntParameter space must be.*"):
        IntParameter(low=0, default=5, space="buy")

    with pytest.raises(OperationalException, match=r"IntParameter space invalid\."):
        IntParameter([0, 10], high=7, default=5, space="buy")

    intpar = IntParameter(low=0, high=5, default=1, space="buy")
    assert intpar.value == 1
    assert isinstance(intpar.get_space(""), IntDistribution)
    assert isinstance(intpar.range, range)
    assert len(list(intpar.range)) == 1
    # Range contains ONLY the default / value.
    assert list(intpar.range) == [intpar.value]
    intpar.in_space = True

    assert len(list(intpar.range)) == 6
    assert list(intpar.range) == [0, 1, 2, 3, 4, 5]

    HyperoptStateContainer.set_state(HyperoptState.OPTIMIZE)
    assert len(list(intpar.range)) == 1


def test_hyperopt_real_parameter():
    HyperoptStateContainer.set_state(HyperoptState.INDICATORS)
    from optuna.distributions import FloatDistribution

    with pytest.raises(OperationalException, match=r"RealParameter space must be.*"):
        RealParameter(low=0, default=5, space="buy")

    with pytest.raises(OperationalException, match=r"RealParameter space invalid\."):
        RealParameter([0, 10], high=7, default=5, space="buy")

    fltpar = RealParameter(low=0.0, high=5.5, default=1.0, space="buy")
    assert fltpar.value == 1.0
    assert isinstance(fltpar.get_space(""), FloatDistribution)

    assert not hasattr(fltpar, "range")


def test_hyperopt_decimal_parameter():
    HyperoptStateContainer.set_state(HyperoptState.INDICATORS)
    from freqtrade.optimize.space import SKDecimal

    with pytest.raises(OperationalException, match=r"DecimalParameter space must be.*"):
        DecimalParameter(low=0, default=5, space="buy")

    with pytest.raises(OperationalException, match=r"DecimalParameter space invalid\."):
        DecimalParameter([0, 10], high=7, default=5, space="buy")

    decimalpar = DecimalParameter(low=0.0, high=0.5, default=0.14, decimals=1, space="buy")
    assert decimalpar.value == 0.1
    assert isinstance(decimalpar.get_space(""), SKDecimal)
    assert isinstance(decimalpar.range, list)
    assert len(list(decimalpar.range)) == 1
    # Range contains ONLY the default / value.
    assert list(decimalpar.range) == [decimalpar.value]
    decimalpar.in_space = True
    assert len(list(decimalpar.range)) == 6
    assert list(decimalpar.range) == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    decimalpar2 = DecimalParameter(low=0.01, high=0.03, decimals=3, default=0.02, space="buy")
    decimalpar2.in_space = True
    assert len(list(decimalpar2.range)) == 21
    expected_range = [round(0.01 + i * 0.001, 3) for i in range(21)]
    assert list(decimalpar2.range) == expected_range
    assert decimalpar2.value == 0.02
    decimalpar2.value = 0.022222
    assert decimalpar2.value == 0.022

    HyperoptStateContainer.set_state(HyperoptState.OPTIMIZE)
    assert len(list(decimalpar.range)) == 1


def test_hyperopt_categorical_parameter():
    HyperoptStateContainer.set_state(HyperoptState.INDICATORS)
    from optuna.distributions import CategoricalDistribution

    with pytest.raises(OperationalException, match=r"CategoricalParameter space must.*"):
        CategoricalParameter(["aa"], default="aa", space="buy")

    with pytest.raises(TypeError):
        BaseParameter(opt_range=[0, 1], default=1, space="buy")

    catpar = CategoricalParameter(
        ["buy_rsi", "buy_macd", "buy_none"], default="buy_macd", space="buy"
    )
    assert catpar.value == "buy_macd"
    assert isinstance(catpar.get_space(""), CategoricalDistribution)
    assert isinstance(catpar.range, list)
    assert len(list(catpar.range)) == 1
    # Range contains ONLY the default / value.
    assert list(catpar.range) == [catpar.value]
    catpar.in_space = True
    assert len(list(catpar.range)) == 3
    assert list(catpar.range) == ["buy_rsi", "buy_macd", "buy_none"]

    boolpar = BooleanParameter(default=True, space="buy")
    assert boolpar.value is True
    assert isinstance(boolpar.get_space(""), CategoricalDistribution)
    assert isinstance(boolpar.range, list)
    assert len(list(boolpar.range)) == 1

    boolpar.in_space = True
    assert len(list(boolpar.range)) == 2

    assert list(boolpar.range) == [True, False]

    HyperoptStateContainer.set_state(HyperoptState.OPTIMIZE)
    assert len(list(catpar.range)) == 1
    assert len(list(boolpar.range)) == 1
