from math import cos, exp, pi, sqrt

import numpy as np
import talib as ta
from pandas import Series


def went_up(series: Series) -> bool:
    return series > series.shift(1)


def went_down(series: Series) -> bool:
    return series < series.shift(1)


def ehlers_super_smoother(series: Series, smoothing: float = 6) -> Series:
    magic = pi * sqrt(2) / smoothing
    a1 = exp(-magic)
    coeff2 = 2 * a1 * cos(magic)
    coeff3 = -a1 * a1
    coeff1 = (1 - coeff2 - coeff3) / 2

    filtered = series.copy()

    for i in range(2, len(series)):
        filtered.iloc[i] = coeff1 * (series.iloc[i] + series.iloc[i-1]) + \
            coeff2 * filtered.iloc[i-1] + coeff3 * filtered.iloc[i-2]

    return filtered


def fishers_inverse(series: Series, smoothing: float = 0) -> np.ndarray:
    """ Does a smoothed fishers inverse transformation.
        Can be used with any oscillator that goes from 0 to 100 like RSI or MFI """
    v1 = 0.1 * (series - 50)
    if smoothing > 0:
        v2 = ta.WMA(v1.values, timeperiod=smoothing)
    else:
        v2 = v1
    return (np.exp(2 * v2)-1) / (np.exp(2 * v2) + 1)
