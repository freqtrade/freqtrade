# QTPyLib: Quantitative Trading Python Library
# https://github.com/ranaroussi/qtpylib
#
# Copyright 2016-2018 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas.core.base import PandasObject


# =============================================
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# =============================================


def numpy_rolling_window(data, window):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = (*data.strides, data.strides[-1])
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def numpy_rolling_series(func):
    def func_wrapper(data, window, as_source=False):
        series = data.values if isinstance(data, pd.Series) else data

        new_series = np.empty(len(series)) * np.nan
        calculated = func(series, window)
        new_series[-len(calculated) :] = calculated

        if as_source and isinstance(data, pd.Series):
            return pd.Series(index=data.index, data=new_series)

        return new_series

    return func_wrapper


@numpy_rolling_series
def numpy_rolling_mean(data, window, as_source=False):
    return np.mean(numpy_rolling_window(data, window), axis=-1)


@numpy_rolling_series
def numpy_rolling_std(data, window, as_source=False):
    return np.std(numpy_rolling_window(data, window), axis=-1, ddof=1)


# ---------------------------------------------


def session(df, start="17:00", end="16:00"):
    """remove previous globex day from df"""
    if df.empty:
        return df

    # get start/end/now as decimals
    int_start = list(map(int, start.split(":")))
    int_start = (int_start[0] + int_start[1] - 1 / 100) - 0.0001
    int_end = list(map(int, end.split(":")))
    int_end = int_end[0] + int_end[1] / 100
    int_now = df[-1:].index.hour[0] + (df[:1].index.minute[0]) / 100

    # same-dat session?
    is_same_day = int_end > int_start

    # set pointers
    curr = prev = df[-1:].index[0].strftime("%Y-%m-%d")

    # globex/forex session
    if not is_same_day:
        prev = (datetime.strptime(curr, "%Y-%m-%d") - timedelta(1)).strftime("%Y-%m-%d")

    # slice
    if int_now >= int_start:
        df = df[df.index >= curr + " " + start]
    else:
        df = df[df.index >= prev + " " + start]

    return df.copy()


# ---------------------------------------------


def heikinashi(bars):
    bars = bars.copy()
    bars["ha_close"] = (bars["open"] + bars["high"] + bars["low"] + bars["close"]) / 4

    # ha open
    bars.at[0, "ha_open"] = (bars.at[0, "open"] + bars.at[0, "close"]) / 2
    for i in range(1, len(bars)):
        bars.at[i, "ha_open"] = (bars.at[i - 1, "ha_open"] + bars.at[i - 1, "ha_close"]) / 2

    bars["ha_high"] = bars.loc[:, ["high", "ha_open", "ha_close"]].max(axis=1)
    bars["ha_low"] = bars.loc[:, ["low", "ha_open", "ha_close"]].min(axis=1)

    return pd.DataFrame(
        index=bars.index,
        data={
            "open": bars["ha_open"],
            "high": bars["ha_high"],
            "low": bars["ha_low"],
            "close": bars["ha_close"],
        },
    )


# ---------------------------------------------


def tdi(series, rsi_lookback=13, rsi_smooth_len=2, rsi_signal_len=7, bb_lookback=34, bb_std=1.6185):
    rsi_data = rsi(series, rsi_lookback)
    rsi_smooth = sma(rsi_data, rsi_smooth_len)
    rsi_signal = sma(rsi_data, rsi_signal_len)

    bb_series = bollinger_bands(rsi_data, bb_lookback, bb_std)

    return pd.DataFrame(
        index=series.index,
        data={
            "rsi": rsi_data,
            "rsi_signal": rsi_signal,
            "rsi_smooth": rsi_smooth,
            "rsi_bb_upper": bb_series["upper"],
            "rsi_bb_lower": bb_series["lower"],
            "rsi_bb_mid": bb_series["mid"],
        },
    )


# ---------------------------------------------


def awesome_oscillator(df, weighted=False, fast=5, slow=34):
    midprice = (df["high"] + df["low"]) / 2

    if weighted:
        ao = (midprice.ewm(fast).mean() - midprice.ewm(slow).mean()).values
    else:
        ao = numpy_rolling_mean(midprice, fast) - numpy_rolling_mean(midprice, slow)

    return pd.Series(index=df.index, data=ao)


# ---------------------------------------------


def nans(length=1):
    mtx = np.empty(length)
    mtx[:] = np.nan
    return mtx


# ---------------------------------------------


def typical_price(bars):
    res = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    return pd.Series(index=bars.index, data=res)


# ---------------------------------------------


def mid_price(bars):
    res = (bars["high"] + bars["low"]) / 2.0
    return pd.Series(index=bars.index, data=res)


# ---------------------------------------------


def ibs(bars):
    """Internal bar strength"""
    res = np.round((bars["close"] - bars["low"]) / (bars["high"] - bars["low"]), 2)
    return pd.Series(index=bars.index, data=res)


# ---------------------------------------------


def true_range(bars):
    return pd.DataFrame(
        {
            "hl": bars["high"] - bars["low"],
            "hc": abs(bars["high"] - bars["close"].shift(1)),
            "lc": abs(bars["low"] - bars["close"].shift(1)),
        }
    ).max(axis=1)


# ---------------------------------------------


def atr(bars, window=14, exp=False):
    tr = true_range(bars)

    if exp:
        res = rolling_weighted_mean(tr, window)
    else:
        res = rolling_mean(tr, window)

    return pd.Series(res)


# ---------------------------------------------


def crossed(series1, series2, direction=None):
    if isinstance(series1, np.ndarray):
        series1 = pd.Series(series1)

    if isinstance(series2, float | int | np.ndarray | np.integer | np.floating):
        series2 = pd.Series(index=series1.index, data=series2)

    if direction is None or direction == "above":
        above = pd.Series((series1 > series2) & (series1.shift(1) <= series2.shift(1)))

    if direction is None or direction == "below":
        below = pd.Series((series1 < series2) & (series1.shift(1) >= series2.shift(1)))

    if direction is None:
        return above | below

    return above if direction == "above" else below


def crossed_above(series1, series2):
    return crossed(series1, series2, "above")


def crossed_below(series1, series2):
    return crossed(series1, series2, "below")


# ---------------------------------------------


def rolling_std(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    if min_periods == window and len(series) > window:
        return numpy_rolling_std(series, window, True)
    else:
        try:
            return series.rolling(window=window, min_periods=min_periods).std()
        except Exception as e:  # noqa: F841
            return pd.Series(series).rolling(window=window, min_periods=min_periods).std()


# ---------------------------------------------


def rolling_mean(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    if min_periods == window and len(series) > window:
        return numpy_rolling_mean(series, window, True)
    else:
        try:
            return series.rolling(window=window, min_periods=min_periods).mean()
        except Exception as e:  # noqa: F841
            return pd.Series(series).rolling(window=window, min_periods=min_periods).mean()


# ---------------------------------------------


def rolling_min(series, window=14, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        return series.rolling(window=window, min_periods=min_periods).min()
    except Exception as e:  # noqa: F841
        return pd.Series(series).rolling(window=window, min_periods=min_periods).min()


# ---------------------------------------------


def rolling_max(series, window=14, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        return series.rolling(window=window, min_periods=min_periods).max()
    except Exception as e:  # noqa: F841
        return pd.Series(series).rolling(window=window, min_periods=min_periods).max()


# ---------------------------------------------


def rolling_weighted_mean(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        return series.ewm(span=window, min_periods=min_periods).mean()
    except Exception as e:  # noqa: F841
        return pd.ewma(series, span=window, min_periods=min_periods)


# ---------------------------------------------


def hull_moving_average(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    ma = (2 * rolling_weighted_mean(series, window / 2, min_periods)) - rolling_weighted_mean(
        series, window, min_periods
    )
    return rolling_weighted_mean(ma, np.sqrt(window), min_periods)


# ---------------------------------------------


def sma(series, window=200, min_periods=None):
    return rolling_mean(series, window=window, min_periods=min_periods)


# ---------------------------------------------


def wma(series, window=200, min_periods=None):
    return rolling_weighted_mean(series, window=window, min_periods=min_periods)


# ---------------------------------------------


def hma(series, window=200, min_periods=None):
    return hull_moving_average(series, window=window, min_periods=min_periods)


# ---------------------------------------------


def vwap(bars):
    """
    calculate vwap of entire time series
    (input can be pandas series or numpy array)
    bars are usually mid [ (h+l)/2 ] or typical [ (h+l+c)/3 ]
    """
    raise ValueError(
        "using `qtpylib.vwap` facilitates lookahead bias. Please use "
        "`qtpylib.rolling_vwap` instead, which calculates vwap in a rolling manner."
    )
    # typical = ((bars['high'] + bars['low'] + bars['close']) / 3).values
    # volume = bars['volume'].values

    # return pd.Series(index=bars.index,
    #                  data=np.cumsum(volume * typical) / np.cumsum(volume))


# ---------------------------------------------


def rolling_vwap(bars, window=200, min_periods=None):
    """
    calculate vwap using moving window
    (input can be pandas series or numpy array)
    bars are usually mid [ (h+l)/2 ] or typical [ (h+l+c)/3 ]
    """
    min_periods = window if min_periods is None else min_periods

    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    volume = bars["volume"]

    left = (volume * typical).rolling(window=window, min_periods=min_periods).sum()
    right = volume.rolling(window=window, min_periods=min_periods).sum()

    return (
        pd.Series(index=bars.index, data=(left / right))
        .replace([np.inf, -np.inf], float("NaN"))
        .ffill()
    )


# ---------------------------------------------


def rsi(series, window=14):
    """
    compute the n period relative strength indicator
    """

    # 100-(100/relative_strength)
    deltas = np.diff(series)
    seed = deltas[: window + 1]

    # default values
    ups = seed[seed > 0].sum() / window
    downs = -seed[seed < 0].sum() / window
    rsival = np.zeros_like(series)
    rsival[:window] = 100.0 - 100.0 / (1.0 + ups / downs)

    # period values
    for i in range(window, len(series)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta

        ups = (ups * (window - 1) + upval) / window
        downs = (downs * (window - 1.0) + downval) / window
        rsival[i] = 100.0 - 100.0 / (1.0 + ups / downs)

    # return rsival
    return pd.Series(index=series.index, data=rsival)


# ---------------------------------------------


def macd(series, fast=3, slow=10, smooth=16):
    """
    compute the MACD (Moving Average Convergence/Divergence)
    using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    macd_line = rolling_weighted_mean(series, window=fast) - rolling_weighted_mean(
        series, window=slow
    )
    signal = rolling_weighted_mean(macd_line, window=smooth)
    histogram = macd_line - signal
    # return macd_line, signal, histogram
    return pd.DataFrame(
        index=series.index,
        data={"macd": macd_line.values, "signal": signal.values, "histogram": histogram.values},
    )


# ---------------------------------------------


def bollinger_bands(series, window=20, stds=2):
    ma = rolling_mean(series, window=window, min_periods=1)
    std = rolling_std(series, window=window, min_periods=1)
    upper = ma + std * stds
    lower = ma - std * stds

    return pd.DataFrame(index=series.index, data={"upper": upper, "mid": ma, "lower": lower})


# ---------------------------------------------


def weighted_bollinger_bands(series, window=20, stds=2):
    ema = rolling_weighted_mean(series, window=window)
    std = rolling_std(series, window=window)
    upper = ema + std * stds
    lower = ema - std * stds

    return pd.DataFrame(
        index=series.index, data={"upper": upper.values, "mid": ema.values, "lower": lower.values}
    )


# ---------------------------------------------


def returns(series):
    try:
        res = (series / series.shift(1) - 1).replace([np.inf, -np.inf], float("NaN"))
    except Exception as e:  # noqa: F841
        res = nans(len(series))

    return pd.Series(index=series.index, data=res)


# ---------------------------------------------


def log_returns(series):
    try:
        res = np.log(series / series.shift(1)).replace([np.inf, -np.inf], float("NaN"))
    except Exception as e:  # noqa: F841
        res = nans(len(series))

    return pd.Series(index=series.index, data=res)


# ---------------------------------------------


def implied_volatility(series, window=252):
    try:
        logret = np.log(series / series.shift(1)).replace([np.inf, -np.inf], float("NaN"))
        res = numpy_rolling_std(logret, window) * np.sqrt(window)
    except Exception as e:  # noqa: F841
        res = nans(len(series))

    return pd.Series(index=series.index, data=res)


# ---------------------------------------------


def keltner_channel(bars, window=14, atrs=2):
    typical_mean = rolling_mean(typical_price(bars), window)
    atrval = atr(bars, window) * atrs

    upper = typical_mean + atrval
    lower = typical_mean - atrval

    return pd.DataFrame(
        index=bars.index,
        data={"upper": upper.values, "mid": typical_mean.values, "lower": lower.values},
    )


# ---------------------------------------------


def roc(series, window=14):
    """
    compute rate of change
    """
    res = (series - series.shift(window)) / series.shift(window)
    return pd.Series(index=series.index, data=res)


# ---------------------------------------------


def cci(series, window=14):
    """
    compute commodity channel index
    """
    price = typical_price(series)
    typical_mean = rolling_mean(price, window)
    res = (price - typical_mean) / (0.015 * np.std(typical_mean))
    return pd.Series(index=series.index, data=res)


# ---------------------------------------------


def stoch(df, window=14, d=3, k=3, fast=False):
    """
    compute the n period relative strength indicator
    http://excelta.blogspot.co.il/2013/09/stochastic-oscillator-technical.html
    """

    my_df = pd.DataFrame(index=df.index)

    my_df["rolling_max"] = df["high"].rolling(window).max()
    my_df["rolling_min"] = df["low"].rolling(window).min()

    my_df["fast_k"] = (
        100 * (df["close"] - my_df["rolling_min"]) / (my_df["rolling_max"] - my_df["rolling_min"])
    )
    my_df["fast_d"] = my_df["fast_k"].rolling(d).mean()

    if fast:
        return my_df.loc[:, ["fast_k", "fast_d"]]

    my_df["slow_k"] = my_df["fast_k"].rolling(k).mean()
    my_df["slow_d"] = my_df["slow_k"].rolling(d).mean()

    return my_df.loc[:, ["slow_k", "slow_d"]]


# ---------------------------------------------


def zlma(series, window=20, min_periods=None, kind="ema"):
    """
    John Ehlers' Zero lag (exponential) moving average
    https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average
    """
    min_periods = window if min_periods is None else min_periods

    lag = (window - 1) // 2
    series = 2 * series - series.shift(lag)
    if kind in ["ewm", "ema"]:
        return wma(series, lag, min_periods)
    elif kind == "hma":
        return hma(series, lag, min_periods)
    return sma(series, lag, min_periods)


def zlema(series, window, min_periods=None):
    return zlma(series, window, min_periods, kind="ema")


def zlsma(series, window, min_periods=None):
    return zlma(series, window, min_periods, kind="sma")


def zlhma(series, window, min_periods=None):
    return zlma(series, window, min_periods, kind="hma")


# ---------------------------------------------


def zscore(bars, window=20, stds=1, col="close"):
    """get zscore of price"""
    std = numpy_rolling_std(bars[col], window)
    mean = numpy_rolling_mean(bars[col], window)
    return (bars[col] - mean) / (std * stds)


# ---------------------------------------------


def pvt(bars):
    """Price Volume Trend"""
    trend = ((bars["close"] - bars["close"].shift(1)) / bars["close"].shift(1)) * bars["volume"]
    return trend.cumsum()


def chopiness(bars, window=14):
    atrsum = true_range(bars).rolling(window).sum()
    highs = bars["high"].rolling(window).max()
    lows = bars["low"].rolling(window).min()
    return 100 * np.log10(atrsum / (highs - lows)) / np.log10(window)


# =============================================


PandasObject.session = session
PandasObject.atr = atr
PandasObject.bollinger_bands = bollinger_bands
PandasObject.cci = cci
PandasObject.crossed = crossed
PandasObject.crossed_above = crossed_above
PandasObject.crossed_below = crossed_below
PandasObject.heikinashi = heikinashi
PandasObject.hull_moving_average = hull_moving_average
PandasObject.ibs = ibs
PandasObject.implied_volatility = implied_volatility
PandasObject.keltner_channel = keltner_channel
PandasObject.log_returns = log_returns
PandasObject.macd = macd
PandasObject.returns = returns
PandasObject.roc = roc
PandasObject.rolling_max = rolling_max
PandasObject.rolling_min = rolling_min
PandasObject.rolling_mean = rolling_mean
PandasObject.rolling_std = rolling_std
PandasObject.rsi = rsi
PandasObject.stoch = stoch
PandasObject.zscore = zscore
PandasObject.pvt = pvt
PandasObject.chopiness = chopiness
PandasObject.tdi = tdi
PandasObject.true_range = true_range
PandasObject.mid_price = mid_price
PandasObject.typical_price = typical_price
PandasObject.vwap = vwap
PandasObject.rolling_vwap = rolling_vwap
PandasObject.weighted_bollinger_bands = weighted_bollinger_bands
PandasObject.rolling_weighted_mean = rolling_weighted_mean

PandasObject.sma = sma
PandasObject.wma = wma
PandasObject.ema = wma
PandasObject.hma = hma

PandasObject.zlsma = zlsma
PandasObject.zlwma = zlema
PandasObject.zlema = zlema
PandasObject.zlhma = zlhma
PandasObject.zlma = zlma
