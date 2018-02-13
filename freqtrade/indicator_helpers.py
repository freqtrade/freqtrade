from math import exp, pi, sqrt, cos

from pandas import Series


def went_up(series: Series) -> Series:
    return series > series.shift(1)


def went_down(series: Series) -> Series:
    return series < series.shift(1)


def ehlers_super_smoother(series: Series, smoothing: float = 6):
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
