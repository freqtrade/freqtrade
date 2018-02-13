from pandas import Series


def went_up(series: Series) -> Series:
    return series > series.shift(1)


def went_down(series: Series) -> Series:
    return series < series.shift(1)
