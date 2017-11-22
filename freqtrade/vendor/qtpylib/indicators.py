#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# QTPyLib: Quantitative Trading Python Library
# https://github.com/ranaroussi/qtpylib
#
# Copyright 2016 Ran Aroussi
#
# Licensed under the GNU Lesser General Public License, v3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/lgpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas.core.base import PandasObject

# =============================================
# check min, python version
if sys.version_info < (3, 4):
    raise SystemError("QTPyLib requires Python version >= 3.4")

# =============================================
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# =============================================


def numpy_rolling_window(data, window):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def numpy_rolling_series(func):
    def func_wrapper(data, window, as_source=False):
        series = data.values if isinstance(data, pd.Series) else data

        new_series = np.empty(len(series)) * np.nan
        calculated = func(series, window)
        new_series[-len(calculated):] = calculated

        if as_source and isinstance(data, pd.Series):
            return pd.Series(index=data.index, data=new_series)

        return new_series

    return func_wrapper


@numpy_rolling_series
def numpy_rolling_mean(data, window, as_source=False):
    return np.mean(numpy_rolling_window(data, window), -1)


@numpy_rolling_series
def numpy_rolling_std(data, window, as_source=False):
    return np.std(numpy_rolling_window(data, window), -1)

# ---------------------------------------------


def session(df, start='17:00', end='16:00'):
    """ remove previous globex day from df """
    if len(df) == 0:
        return df

    # get start/end/now as decimals
    int_start = list(map(int, start.split(':')))
    int_start = (int_start[0] + int_start[1] - 1 / 100) - 0.0001
    int_end = list(map(int, end.split(':')))
    int_end = int_end[0] + int_end[1] / 100
    int_now = (df[-1:].index.hour[0] + (df[:1].index.minute[0]) / 100)

    # same-dat session?
    is_same_day = int_end > int_start

    # set pointers
    curr = prev = df[-1:].index[0].strftime('%Y-%m-%d')

    # globex/forex session
    if not is_same_day:
        prev = (datetime.strptime(curr, '%Y-%m-%d') -
                timedelta(1)).strftime('%Y-%m-%d')

    # slice
    if int_now >= int_start:
        df = df[df.index >= curr + ' ' + start]
    else:
        df = df[df.index >= prev + ' ' + start]

    return df.copy()


# ---------------------------------------------

def heikinashi(bars):
    bars = bars.copy()
    bars['ha_close'] = (bars['open'] + bars['high'] +
                        bars['low'] + bars['close']) / 4
    bars['ha_open'] = (bars['open'].shift(1) + bars['close'].shift(1)) / 2
    bars.loc[:1, 'ha_open'] = bars['open'].values[0]
    bars.loc[1:, 'ha_open'] = (
        (bars['ha_open'].shift(1) + bars['ha_close'].shift(1)) / 2)[1:]
    bars['ha_high'] = bars.loc[:, ['high', 'ha_open', 'ha_close']].max(axis=1)
    bars['ha_low'] = bars.loc[:, ['low', 'ha_open', 'ha_close']].min(axis=1)

    return pd.DataFrame(
        index=bars.index,
        data={
            'open': bars['ha_open'],
            'high': bars['ha_high'],
            'low': bars['ha_low'],
            'close': bars['ha_close']})


# ---------------------------------------------

def tdi(series, rsi_len=13, bollinger_len=34, rsi_smoothing=2,
        rsi_signal_len=7, bollinger_std=1.6185):
    rsi_series = rsi(series, rsi_len)
    bb_series = bollinger_bands(rsi_series, bollinger_len, bollinger_std)
    signal = sma(rsi_series, rsi_signal_len)
    rsi_series = sma(rsi_series, rsi_smoothing)

    return pd.DataFrame(index=series.index, data={
        "rsi": rsi_series,
        "signal": signal,
        "bbupper": bb_series['upper'],
        "bblower": bb_series['lower'],
        "bbmid": bb_series['mid']
    })

# ---------------------------------------------


def awesome_oscillator(df, weighted=False, fast=5, slow=34):
    midprice = (df['high'] + df['low']) / 2

    if weighted:
        ao = (midprice.ewm(fast).mean() - midprice.ewm(slow).mean()).values
    else:
        ao = numpy_rolling_mean(midprice, fast) - \
            numpy_rolling_mean(midprice, slow)

    return pd.Series(index=df.index, data=ao)


# ---------------------------------------------

def nans(len=1):
    mtx = np.empty(len)
    mtx[:] = np.nan
    return mtx


# ---------------------------------------------

def typical_price(bars):
    res = (bars['high'] + bars['low'] + bars['close']) / 3.
    return pd.Series(index=bars.index, data=res)


# ---------------------------------------------

def mid_price(bars):
    res = (bars['high'] + bars['low']) / 2.
    return pd.Series(index=bars.index, data=res)


# ---------------------------------------------

def ibs(bars):
    """ Internal bar strength """
    res = np.round((bars['close'] - bars['low']) /
                   (bars['high'] - bars['low']), 2)
    return pd.Series(index=bars.index, data=res)


# ---------------------------------------------

def true_range(bars):
    return pd.DataFrame({
        "hl": bars['high'] - bars['low'],
        "hc": abs(bars['high'] - bars['close'].shift(1)),
        "lc": abs(bars['low'] - bars['close'].shift(1))
    }).max(axis=1)


# ---------------------------------------------

def atr(bars, window=14, exp=False):
    tr = true_range(bars)

    if exp:
        res = rolling_weighted_mean(tr, window)
    else:
        res = rolling_mean(tr, window)

    res = pd.Series(res)
    return (res.shift(1) * (window - 1) + res) / window


# ---------------------------------------------

def crossed(series1, series2, direction=None):
    if isinstance(series1, np.ndarray):
        series1 = pd.Series(series1)

    if isinstance(series2, int) or isinstance(series2, float) or isinstance(series2, np.ndarray):
        series2 = pd.Series(index=series1.index, data=series2)

    if direction is None or direction == "above":
        above = pd.Series((series1 > series2) & (
            series1.shift(1) <= series2.shift(1)))

    if direction is None or direction == "below":
        below = pd.Series((series1 < series2) & (
            series1.shift(1) >= series2.shift(1)))

    if direction is None:
        return above or below

    return above if direction is "above" else below


def crossed_above(series1, series2):
    return crossed(series1, series2, "above")


def crossed_below(series1, series2):
    return crossed(series1, series2, "below")

# ---------------------------------------------


def rolling_std(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        if min_periods == window:
            return numpy_rolling_std(series, window, True)
        else:
            try:
                return series.rolling(window=window, min_periods=min_periods).std()
            except BaseException:
                return pd.Series(series).rolling(window=window, min_periods=min_periods).std()
    except BaseException:
        return pd.rolling_std(series, window=window, min_periods=min_periods)


# ---------------------------------------------

def rolling_mean(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        if min_periods == window:
            return numpy_rolling_mean(series, window, True)
        else:
            try:
                return series.rolling(window=window, min_periods=min_periods).mean()
            except BaseException:
                return pd.Series(series).rolling(window=window, min_periods=min_periods).mean()
    except BaseException:
        return pd.rolling_mean(series, window=window, min_periods=min_periods)


# ---------------------------------------------

def rolling_min(series, window=14, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        try:
            return series.rolling(window=window, min_periods=min_periods).min()
        except BaseException:
            return pd.Series(series).rolling(window=window, min_periods=min_periods).min()
    except BaseException:
        return pd.rolling_min(series, window=window, min_periods=min_periods)


# ---------------------------------------------

def rolling_max(series, window=14, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        try:
            return series.rolling(window=window, min_periods=min_periods).min()
        except BaseException:
            return pd.Series(series).rolling(window=window, min_periods=min_periods).min()
    except BaseException:
        return pd.rolling_min(series, window=window, min_periods=min_periods)


# ---------------------------------------------

def rolling_weighted_mean(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        return series.ewm(span=window, min_periods=min_periods).mean()
    except BaseException:
        return pd.ewma(series, span=window, min_periods=min_periods)


# ---------------------------------------------

def hull_moving_average(series, window=200):
    wma = (2 * rolling_weighted_mean(series, window=window / 2)) - \
        rolling_weighted_mean(series, window=window)
    return rolling_weighted_mean(wma, window=np.sqrt(window))


# ---------------------------------------------

def sma(series, window=200, min_periods=None):
    return rolling_mean(series, window=window, min_periods=min_periods)


# ---------------------------------------------

def wma(series, window=200, min_periods=None):
    return rolling_weighted_mean(series, window=window, min_periods=min_periods)


# ---------------------------------------------

def hma(series, window=200):
    return hull_moving_average(series, window=window)


# ---------------------------------------------

def vwap(bars):
    """
    calculate vwap of entire time series
    (input can be pandas series or numpy array)
    bars are usually mid [ (h+l)/2 ] or typical [ (h+l+c)/3 ]
    """
    typical = ((bars['high'] + bars['low'] + bars['close']) / 3).values
    volume = bars['volume'].values

    return pd.Series(index=bars.index,
                     data=np.cumsum(volume * typical) / np.cumsum(volume))


# ---------------------------------------------

def rolling_vwap(bars, window=200, min_periods=None):
    """
    calculate vwap using moving window
    (input can be pandas series or numpy array)
    bars are usually mid [ (h+l)/2 ] or typical [ (h+l+c)/3 ]
    """
    min_periods = window if min_periods is None else min_periods

    typical = ((bars['high'] + bars['low'] + bars['close']) / 3)
    volume = bars['volume']

    left = (volume * typical).rolling(window=window,
                                      min_periods=min_periods).sum()
    right = volume.rolling(window=window, min_periods=min_periods).sum()

    return pd.Series(index=bars.index, data=(left / right))


# ---------------------------------------------

def rsi(series, window=14):
    """
    compute the n period relative strength indicator
    """
    # 100-(100/relative_strength)
    deltas = np.diff(series)
    seed = deltas[:window + 1]

    # default values
    ups = seed[seed > 0].sum() / window
    downs = -seed[seed < 0].sum() / window
    rsival = np.zeros_like(series)
    rsival[:window] = 100. - 100. / (1. + ups / downs)

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
        downs = (downs * (window - 1.) + downval) / window
        rsival[i] = 100. - 100. / (1. + ups / downs)

    # return rsival
    return pd.Series(index=series.index, data=rsival)


# ---------------------------------------------

def macd(series, fast=3, slow=10, smooth=16):
    """
    compute the MACD (Moving Average Convergence/Divergence)
    using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    macd = rolling_weighted_mean(series, window=fast) - \
        rolling_weighted_mean(series, window=slow)
    signal = rolling_weighted_mean(macd, window=smooth)
    histogram = macd - signal
    # return macd, signal, histogram
    return pd.DataFrame(index=series.index, data={
        'macd': macd.values,
        'signal': signal.values,
        'histogram': histogram.values
    })


# ---------------------------------------------

def bollinger_bands(series, window=20, stds=2):
    sma = rolling_mean(series, window=window)
    std = rolling_std(series, window=window)
    upper = sma + std * stds
    lower = sma - std * stds

    return pd.DataFrame(index=series.index, data={
        'upper': upper,
        'mid': sma,
        'lower': lower
    })


# ---------------------------------------------

def weighted_bollinger_bands(series, window=20, stds=2):
    ema = rolling_weighted_mean(series, window=window)
    std = rolling_std(series, window=window)
    upper = ema + std * stds
    lower = ema - std * stds

    return pd.DataFrame(index=series.index, data={
        'upper': upper.values,
        'mid': ema.values,
        'lower': lower.values
    })


# ---------------------------------------------

def returns(series):
    try:
        res = (series / series.shift(1) -
               1).replace([np.inf, -np.inf], float('NaN'))
    except BaseException:
        res = nans(len(series))

    return pd.Series(index=series.index, data=res)


# ---------------------------------------------

def log_returns(series):
    try:
        res = np.log(series / series.shift(1)
                     ).replace([np.inf, -np.inf], float('NaN'))
    except BaseException:
        res = nans(len(series))

    return pd.Series(index=series.index, data=res)


# ---------------------------------------------

def implied_volatility(series, window=252):
    try:
        logret = np.log(series / series.shift(1)
                        ).replace([np.inf, -np.inf], float('NaN'))
        res = numpy_rolling_std(logret, window) * np.sqrt(window)
    except BaseException:
        res = nans(len(series))

    return pd.Series(index=series.index, data=res)


# ---------------------------------------------

def keltner_channel(bars, window=14, atrs=2):
    typical_mean = rolling_mean(typical_price(bars), window)
    atrval = atr(bars, window) * atrs

    upper = typical_mean + atrval
    lower = typical_mean - atrval

    return pd.DataFrame(index=bars.index, data={
        'upper': upper.values,
        'mid': typical_mean.values,
        'lower': lower.values
    })


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
    res = (price - typical_mean) / (.015 * np.std(typical_mean))
    return pd.Series(index=series.index, data=res)


# ---------------------------------------------

def stoch(df, window=14, d=3, k=3, fast=False):
    """
    compute the n period relative strength indicator
    http://excelta.blogspot.co.il/2013/09/stochastic-oscillator-technical.html
    """
    highs_ma = pd.concat([df['high'].shift(i)
                          for i in np.arange(window)], 1).apply(list, 1)
    highs_ma = highs_ma.T.max().T

    lows_ma = pd.concat([df['low'].shift(i)
                         for i in np.arange(window)], 1).apply(list, 1)
    lows_ma = lows_ma.T.min().T

    fast_k = ((df['close'] - lows_ma) / (highs_ma - lows_ma)) * 100
    fast_d = numpy_rolling_mean(fast_k, d)

    if fast:
        data = {
            'k': fast_k,
            'd': fast_d
        }

    else:
        slow_k = numpy_rolling_mean(fast_k, k)
        slow_d = numpy_rolling_mean(slow_k, d)
        data = {
            'k': slow_k,
            'd': slow_d
        }

    return pd.DataFrame(index=df.index, data=data)


# ---------------------------------------------

def zscore(bars, window=20, stds=1, col='close'):
    """ get zscore of price """
    std = numpy_rolling_std(bars[col], window)
    mean = numpy_rolling_mean(bars[col], window)
    return (bars[col] - mean) / (std * stds)

# ---------------------------------------------


def pvt(bars):
    """ Price Volume Trend """
    pvt = ((bars['close'] - bars['close'].shift(1)) /
           bars['close'].shift(1)) * bars['volume']
    return pvt.cumsum()


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
PandasObject.hma = hma
