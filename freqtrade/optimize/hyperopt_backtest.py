import logging

import arrow
import gc
from typing import Dict, List, Tuple, Union
from enum import IntEnum

from numba import njit
from numpy import (
    repeat,
    ones,
    nan,
    concatenate,
    ndarray,
    array,
    where,
    transpose,
    maximum,
    full,
    unique,
    insert,
    isfinite,
    isnan,
)
from pandas import (
    Timedelta,
    Series,
    DataFrame,
    Categorical,
    Index,
    MultiIndex,
    # SparseArray,
    set_option,
    to_timedelta,
    to_datetime,
)

from freqtrade.optimize.backtesting import Backtesting, BacktestResult
from freqtrade.exceptions import OperationalException
from freqtrade.strategy.interface import SellType


logger = logging.getLogger(__name__)

# import os
# import psutil
# process = psutil.Process(os.getpid())
set_option("display.max_rows", 1000)


class Candle(IntEnum):
    BOUGHT = 2
    SOLD = 5
    NOOP = 0
    END = 11  # references the last candle of a pair
    # STOPLOSS = 17


@njit  # fastmath=True ? there is no math involved here though..
def for_trail_idx(index, bos, rate, stop_idx):
    last = -2
    col = [0] * len(index)
    for i in range(len(index)):
        if bos[i] == Candle.BOUGHT:
            if index[i] > last and last != -1:
                if rate[i] > 0:
                    last = stop_idx[i]
                else:
                    last = -1
            col[i] = last
        else:
            last = -2
            col[i] = -1
    return col


def union_eq(arr: ndarray, vals: List) -> ndarray:
    """ union of equalities from a starting value and a list of values to compare """
    res = arr == vals[0]
    for v in vals[1:]:
        res = res | (arr == v)
    return res


class HyperoptBacktesting(Backtesting):

    empty_results = DataFrame.from_records([], columns=BacktestResult._fields)
    debug = False

    td_zero = Timedelta(0)
    td_half_timeframe: Timedelta
    pairs_offset: List[int]
    position_stacking: bool
    stoploss_enabled: bool
    sold_repeats: List[int]

    def __init__(self, config):
        if config.get("backtesting_engine") == "vectorized":
            self.backtest_stock = self.backtest
            self.backtest = (
                self._wrap_backtest if self.debug else self.vectorized_backtest
            )
            self.beacktesting_engine = "vectorized"
            self.td_half_timeframe = (
                Timedelta(config.get("timeframe", config["timeframe"])) / 2
            )
        super().__init__(config)

        backtesting_amounts = self.config.get("backtesting_amounts", {})
        self.stoploss_enabled = backtesting_amounts.get("stoploss", False)
        self.trailing_enabled = backtesting_amounts.get("trailing", False)
        self.roi_enabled = backtesting_amounts.get("roi", False)

        self.position_stacking = self.config.get("position_stacking", False)
        if self.config.get("max_open_trades", 0) > 0:
            logger.warn("Ignoring max open trades...")

    def get_results(self, events_buy: DataFrame, events_sell: DataFrame) -> DataFrame:
        # choose sell rate depending on sell reason and set sell_reason
        events_sell = events_sell.reindex(
            [*events_sell.columns, "close_rate", "sell_reason"], axis=1, copy=False
        )
        events_sold = events_sell.loc[
            events_sell["bought_or_sold"].values == Candle.SOLD
        ]
        # add new columns to allow multi col assignments of new columns
        result_cols = ["close_rate", "sell_reason", "ohlc"]
        # can't pass the index here because indexes are duplicated with position_stacking,
        # would have to reindex beforehand
        events_sell.loc[
            events_sold.index
            if not self.position_stacking
            else events_sell.index.isin(events_sold.index.drop_duplicates()),
            result_cols,
        ] = [
            events_sold["open"].values,
            SellType.SELL_SIGNAL,
            events_sold["ohlc"].values,
        ]
        if self.stoploss_enabled:
            events_stoploss = events_sell.loc[isfinite(events_sell["stoploss_ofs"])]
            events_sell.loc[events_stoploss.index, result_cols] = [
                events_stoploss["stoploss_rate"].values,
                SellType.STOP_LOSS,
                events_stoploss["stoploss_ofs"].values,
            ]

        open_rate = events_buy["open"].values
        close_rate = events_sell["close_rate"].values
        profits = (close_rate - close_rate * self.fee) / (
            open_rate + open_rate * self.fee
        ) - 1
        trade_duration = to_timedelta(
            Series(events_sell["date"].values - events_buy["date"].values)
        )
        # replace trade duration of same candle trades with half the timeframe reduce to minutes
        trade_duration.loc[trade_duration == self.td_zero] = self.td_half_timeframe

        return DataFrame(
            {
                "pair": events_buy["pair"].values,
                "profit_percent": profits,
                "profit_abs": self.config["stake_amount"] * profits,
                "open_time": to_datetime(events_buy["date"].values),
                "close_time": to_datetime(events_sell["date"].values),
                "open_index": events_buy["ohlc"].values,
                "close_index": events_sell["ohlc"].values,
                "trade_duration": trade_duration.dt.seconds / 60,
                "open_at_end": False,
                "open_rate": open_rate,
                "close_rate": close_rate,
                "sell_reason": events_sell["sell_reason"].values,
            }
        )

    def _shift_paw(
        self,
        data: Union[DataFrame, Series],
        period=1,
        fill_v=nan,
        null_v=nan,
        ofs=None,
    ) -> Union[DataFrame, Series]:
        """ pair aware shifting nulls rows that cross over the next pair data in concat data """
        shifted = data.shift(period, fill_value=fill_v)
        shifted.iloc[
            ofs if ofs is not None else self.pairs_ofs_end + 1 + period
        ] = null_v
        return shifted

    @staticmethod
    def _diff_indexes(arr: ndarray, with_start=False) -> ndarray:
        """ returns the indexes where consecutive values are not equal,
        used for finding pairs ends """
        return where(arr != insert(arr[:-1], 0, nan if with_start else arr[0]))[0]

    def advise_pair_df(self, df: DataFrame, pair: str) -> DataFrame:
        """ Execute strategy signals and return df for given pair """
        meta = {"pair": pair}
        df = self.strategy.advise_buy(df, meta)
        df = self.strategy.advise_sell(df, meta)
        df.fillna({"buy": 0, "sell": 0}, inplace=True)
        # cast date as intent to prevent TZ conversion when accessing values
        df["date"] = df["date"].astype(int)
        return df

    @staticmethod
    def _get_multi_index(pairs: list, idx: ndarray) -> MultiIndex:
        # if a list of [idx, pairs] is passed to from_product , the df would infer
        # the counter as the columns, when we want it as the rows, so we have to pass
        # a swapped mi to the df, there surely is a better way for this...
        return MultiIndex.from_product([pairs, idx], names=["pair", "ohlc"]).swaplevel(
            0, 1
        )

    def merge_pairs_df(self, processed: Dict[str, DataFrame]) -> DataFrame:
        """ join all the pairs data into one concatenate df adding needed columns """
        advised = {}
        data = []
        max_len = 0
        pairs_end = []
        nan_data_pairs = []

        # get the df with the longest ohlc data since all the pairs will be padded to it
        max_df = max(processed.values(), key=len)
        max_len = len(max_df)
        for pair, df in processed.items():
            # make sure to copy the df to not clobber the source data since it is accessed globally
            advised[pair] = self.advise_pair_df(df.copy(), pair)
            apv = advised[pair].values
            lapv = len(apv)
            pairs_end.append(lapv)
            if lapv < max_len:
                # pad shorter data, with an empty array of same shape (columns)
                data.extend(
                    concatenate([apv, full((max_len - lapv, apv.shape[1]), nan)])
                )
                nan_data_pairs.append(pair)
            else:
                data.extend(apv)
        self.pairs = {p: n for n, p in enumerate(advised.keys())}
        # the index shouldn't change after the advise call, so we can take the pre-advised index
        # to create the multiindex where each pair is indexed with max len
        self.n_rows = len(max_df.index.values)
        self.mi = self._get_multi_index(list(advised.keys()), max_df.index.values)
        # take a post advised df for the right columns count as the advise call
        # adds new columns
        df = DataFrame(data, index=self.mi, columns=advised[pair].columns)
        # set startup offset from the first index (should be equal for all pairs)
        self.startup_offset = df.index.get_level_values(0)[0]
        # add a column for pairs offsets to make the index unique
        offsets_arr, self.pairs_offset = self._calc_pairs_offsets(df, return_ofs=True)
        self.pairs_ofs_end = self.pairs_offset + array(pairs_end, dtype=int) - 1
        # loop over the missing data pairs and calculate the point where data ends
        # plus the absolute offset
        self.nan_data_ends = [
            self.pairs_ofs_end[self.pairs[p]] + 1 for p in nan_data_pairs
        ]
        df["ofs"] = Categorical(offsets_arr, self.pairs_offset)
        # could as easily be arange(len(df)) ...
        df["ohlc_ofs"] = (
            df.index.get_level_values(0).values + offsets_arr - self.startup_offset
        )
        return df

    def bought_or_sold(self, df: DataFrame) -> Tuple[DataFrame, bool]:
        """ Set bought_or_sold columns according to buy and sell signals """
        # set bought candles
        # skip if no valid bought candles are found
        # df["bought_or_sold"] = (df["buy"] - df["sell"]).groupby(level=1).shift().values
        df["bought_or_sold"] = self._shift_paw(
            df["buy"] - df["sell"], fill_v=Candle.NOOP
        ).values

        df.loc[df["bought_or_sold"].values == 1, "bought_or_sold"] = Candle.BOUGHT
        # set sold candles
        df.loc[df["bought_or_sold"].values == -1, "bought_or_sold"] = Candle.SOLD
        df["bought_or_sold"] = Categorical(
            df["bought_or_sold"].values, categories=list(map(int, Candle))
        )
        # set END candles as the last non nan candle of each pair data
        bos_loc = df.columns.get_loc("bought_or_sold")
        df.iloc[self.pairs_ofs_end, bos_loc] = Candle.END
        # Since bought_or_sold is shifted, null the row after the last non-nan one
        # as it doesn't have data, exclude pairs which data matches the max_len since
        # they have no nans
        df.iloc[self.nan_data_ends, bos_loc] = Candle.NOOP
        return df, len(df.loc[df["bought_or_sold"].values == Candle.BOUGHT]) < 1

    def boughts_to_sold(self, df: DataFrame) -> DataFrame:
        """
        reduce df such that there are many bought interleaved by one sold candle
        NOTE: does not modify input df
        """
        bos_df = df.loc[
            union_eq(
                df["bought_or_sold"].values, [Candle.BOUGHT, Candle.SOLD, Candle.END]
            )
        ]
        bos_df = bos_df.loc[
            # exclude duplicate sold
            ~(
                (bos_df["bought_or_sold"].values == Candle.SOLD)
                & (
                    # bos_df["bought_or_sold"]
                    # .groupby(level=1)
                    # .shift(fill_value=Candle.SOLD)
                    # .values
                    self._shift_paw(
                        bos_df["bought_or_sold"],
                        fill_v=Candle.SOLD,
                        null_v=Candle.NOOP,
                        ofs=self._diff_indexes(bos_df.index.get_level_values(1)),
                    ).values
                    == Candle.SOLD
                )
            )
        ]
        return bos_df

    def _pd_calc_sold_repeats(self, bts_df: DataFrame, sold: DataFrame) -> list:
        """ deprecated; pandas version of the next_sold_ofs calculation """
        first_bought = bts_df.groupby(level=1).first()

        def repeats(x, rep):
            vals = x.index.get_level_values(0).values
            # prepend the first range subtracting the index of the first bought
            rep.append(vals[0] - first_bought.at[x.name, "bts_index"] + 1)
            rep.extend(vals[1:] - vals[:-1])

        sold_repeats: List = []
        sold.groupby(level=1).apply(repeats, rep=sold_repeats)
        return sold_repeats

    def _np_calc_sold_repeats(self, bts_df: DataFrame, sold: DataFrame) -> list:
        """ numpy version of the next_sold_ofs calculation """
        first_bought_idx = bts_df.iloc[
            self._diff_indexes(bts_df["pair"].values, with_start=True),
            # index calling is not needed because bts_df has the full index,
            # but keep it for clarity
        ].index.values
        sold_idx = sold.index.values
        first_sold_loc = self._diff_indexes(sold["pair"].values, with_start=True)
        first_sold_idx = sold_idx[first_sold_loc]
        # the bulk of the repetitions, append an empty value
        sold_repeats = concatenate([[0], sold_idx[1:] - sold_idx[:-1]])
        # override the first repeats of each pair (will always override the value at idx 0)
        sold_repeats[first_sold_loc] = first_sold_idx - first_bought_idx + 1
        return sold_repeats

    def set_sold(self, df: DataFrame) -> DataFrame:
        # recompose the multi index swapping the ohlc count with a contiguous range
        bts_df = self.boughts_to_sold(df)
        bts_df.reset_index(inplace=True)
        # align sold to bought
        sold = bts_df.loc[
            union_eq(bts_df["bought_or_sold"].values, [Candle.SOLD, Candle.END])
        ]
        # if no sell sig is provided a limit on the trade duration could be applied..
        # if len(sold) < 1:
        # bts_df, sold = self.fill_stub_sold(df, bts_df)
        # calc the repetitions of each sell signal for each bought signal
        self.sold_repeats = self._np_calc_sold_repeats(bts_df, sold)
        # NOTE: use the "ohlc_ofs" col with offsetted original indexes
        # for stoploss calculation, consider the last candle of each pair as a sell,
        # even thought the bought will be valid only if an amount condition is triggered
        bts_df["next_sold_ofs"] = repeat(sold["ohlc_ofs"].values, self.sold_repeats)
        return bts_df, sold

    def set_stoploss(self, df: DataFrame) -> DataFrame:
        """
        returns the df of valid boughts where stoploss triggered, with matching stoploss
        index of each bought
        """
        bts_df, sold = self.set_sold(df)
        bought = bts_df.loc[bts_df["bought_or_sold"].values == Candle.BOUGHT]
        # get the index ranges of each bought->sold spans
        bought_ranges = bought["next_sold_ofs"].values - bought["ohlc_ofs"].values
        # could also just use the sum...
        if bought_ranges.mean() < 100:
            # intervals are short compute everything in one round
            bts_df = self._pd_select_triggered_stoploss(
                df, bought, bought_ranges, bts_df
            )
        else:
            # intervals are too long, jump over candles
            args = [df, bought, bought_ranges, sold, bts_df]
            bts_df = (
                self._pd_2_select_triggered_stoploss(*args)
                if not self.position_stacking
                else self._pd_2_select_triggered_stoploss_stack(*args)
            )
        return bts_df

    def _pd_2_select_triggered_stoploss_stack(
        self,
        df: DataFrame,
        bought: DataFrame,
        bought_ranges: ndarray,
        sold: DataFrame,
        bts_df: DataFrame,
    ):
        """ loop version of stoploss selection for position stacking, simply loops
        over all the bought candles of the bts dataframe """
        stoploss_index = []
        stoploss_rate = []
        bought_stoploss_ofs = []
        # copy cols for faster index accessing
        bofs = bought["ohlc_ofs"].values
        bopen = bought["open"].values
        b = 0
        stoploss_bought_ofs = bofs[b]

        ohlc_low = df["low"].values
        ohlc_ofs = df["ohlc_ofs"].values
        ohlc_ofs_start = 0
        ohlc_idx = df.index.get_level_values(0)
        end_ofs = ohlc_ofs[-1]

        while stoploss_bought_ofs < end_ofs:
            # calculate the rate from the bought candle
            stoploss_triggered_rate = self._calc_stoploss_rate_value(bopen[b])
            # check trigger for the range of the current bought
            ohlc_ofs_start += ohlc_ofs[ohlc_ofs_start:].searchsorted(
                stoploss_bought_ofs, "left"
            )
            stoploss_triggered = (
                ohlc_low[ohlc_ofs_start : ohlc_ofs_start + bought_ranges[b]]
                <= stoploss_triggered_rate
            )
            # get the position where stoploss triggered relative to the current bought slice
            stop_max_idx = stoploss_triggered.argmax()
            # check that the index returned by argmax is True
            if stoploss_triggered[stop_max_idx]:
                # set the offset of the triggered stoploss index
                stoploss_index.append(ohlc_idx[stoploss_bought_ofs + stop_max_idx])
                stoploss_rate.append(stoploss_triggered_rate)
                bought_stoploss_ofs.append(stoploss_bought_ofs)
            try:
                b += 1
                stoploss_bought_ofs = bofs[b]
            except IndexError:
                break
        # set the index to the offset and add the columns to set the stoploss
        # data points on the relevant boughts
        bts_df.set_index("ohlc_ofs", inplace=True)
        stoploss_cols = ["stoploss_ofs", "stoploss_rate"]
        bts_df.assign(**{c: nan for c in stoploss_cols})
        bts_df = bts_df.reindex(columns=[*bts_df.columns, *stoploss_cols], copy=False)
        bts_df.loc[bought_stoploss_ofs, stoploss_cols,] = [
            [stoploss_index],
            [stoploss_rate],
        ]
        return bts_df

    def _pd_2_select_triggered_stoploss(
        self,
        df: DataFrame,
        bought: DataFrame,
        bought_ranges: ndarray,
        sold: DataFrame,
        bts_df: DataFrame,
    ):
        stoploss_index = []
        stoploss_rate = []
        bought_stoploss_ofs = []
        last_stoploss_ofs: List = []
        # copy cols for faster index accessing
        bofs = bought["ohlc_ofs"].values
        bsold = bought["next_sold_ofs"].values
        bopen = bought["open"].values
        b = 0
        stoploss_bought_ofs = bofs[b]

        ohlc_low = df["low"].values
        ohlc_ofs = df["ohlc_ofs"].values
        ohlc_ofs_start = 0
        ohlc_idx = df.index.get_level_values(0)
        current_ofs = stoploss_bought_ofs
        end_ofs = ohlc_ofs[-1]

        while stoploss_bought_ofs < end_ofs:
            # calculate the rate from the bought candle
            stoploss_triggered_rate = self._calc_stoploss_rate_value(bopen[b])
            # check trigger for the range of the current bought
            ohlc_ofs_start += ohlc_ofs[ohlc_ofs_start:].searchsorted(
                stoploss_bought_ofs, "left"
            )
            stoploss_triggered = (
                ohlc_low[ohlc_ofs_start : ohlc_ofs_start + bought_ranges[b]]
                <= stoploss_triggered_rate
            )
            # get the position where stoploss triggered relative to the current bought slice
            stop_max_idx = stoploss_triggered.argmax()
            # check that the index returned by argmax is True
            if stoploss_triggered[stop_max_idx]:
                # set the offset of the triggered stoploss index
                current_ofs = stoploss_bought_ofs + stop_max_idx
                stop_ohlc_idx = ohlc_idx[current_ofs]
                stoploss_index.append(stop_ohlc_idx)
                stoploss_rate.append(stoploss_triggered_rate)
                bought_stoploss_ofs.append(stoploss_bought_ofs)
                try:
                    # get the first row where the bought index is
                    # higher than the current stoploss index
                    b += bofs[b:].searchsorted(current_ofs, "right")
                    # repeat the stoploss index for the boughts in between the stoploss
                    # and the bought with higher idx
                    last_stoploss_ofs.extend(
                        [stop_ohlc_idx] * (b - len(last_stoploss_ofs))
                    )
                    stoploss_bought_ofs = bofs[b]
                except IndexError:
                    break
            else:  # if stoploss did not trigger, jump to the first bought after next sold idx
                try:
                    b += bofs[b:].searchsorted(bsold[b], "right")
                    last_stoploss_ofs.extend([-1] * (b - len(last_stoploss_ofs)))
                    stoploss_bought_ofs = bofs[b]
                except IndexError:
                    break
        # pad the last stoploss array with the remaining boughts
        last_stoploss_ofs.extend([-1] * (len(bought) - len(last_stoploss_ofs)))
        # set the index to the offset and add the columns to set the stoploss
        # data points on the relevant boughts
        bts_df.set_index("ohlc_ofs", inplace=True)
        stoploss_cols = ["stoploss_ofs", "stoploss_rate", "last_stoploss"]
        bts_df = bts_df.reindex(columns=[*bts_df.columns, *stoploss_cols], copy=False)
        bts_df.loc[bought["ohlc_ofs"], "last_stoploss"] = last_stoploss_ofs
        bts_df.loc[bought_stoploss_ofs, stoploss_cols,] = [
            [stoploss_index],
            [stoploss_rate],
            [stoploss_index],
        ]
        bts_df["last_stoploss"].fillna(-1, inplace=True)
        return bts_df

    def _remove_pairs_offsets(self, df: DataFrame, cols: List):
        ofs_vals = df["ofs"].values.tolist()
        for c in cols:
            # use to list in case of category
            df[c] = df[c].values - ofs_vals + self.startup_offset

    def _calc_pairs_offsets(
        self, df: DataFrame, group=None, return_ofs=False
    ) -> ndarray:
        # all the pairs with df candles
        gb = df.groupby(group) if group else df.groupby(level=1)
        df_pairs = [self.pairs[p] for p in gb.indices.keys()]
        # since pairs are concatenated, their candles start at their ordered position
        pairs_offset = [self.n_rows * n for n in df_pairs]
        pairs_offset_arr = repeat(pairs_offset, gb.size().values)
        if return_ofs:
            return pairs_offset_arr, pairs_offset
        else:
            return pairs_offset_arr - self.startup_offset

    def _columns_indexes(self, df: DataFrame) -> Dict[str, int]:
        cols_idx = {}
        for col in ("open", "low", "ohlc_ofs"):
            cols_idx[col] = df.columns.get_loc(col)
        return cols_idx

    def _np_calc_triggered_stoploss(
        self, df: DataFrame, bought: DataFrame, bought_ranges: ndarray,
    ) -> ndarray:
        """ numpy equivalent of _pd_calc_triggered_stoploss that is more memory efficient """
        # clear up memory
        gc.collect()
        # expand bought ranges into ohlc processed
        ohlc_cols = list(self._columns_indexes(df).values())
        # prefetch the columns of interest to avoid querying
        # the index over the loop (avoid nd indexes)
        ohlc_vals = df.iloc[:, ohlc_cols].values
        stoploss_rate = self._calc_stoploss_rate(bought)

        # 0: open, 1: low, 2: stoploss_ofs, 3: stoploss_bought_ofs, 4: stoploss_rate
        stoploss = concatenate(
            [
                concatenate(
                    [
                        ohlc_vals[i : i + bought_ranges[n]]
                        # the array position of each bought row comes from the offset
                        # of each pair from the beginning (adjusted to the startup candles count)
                        # plus the ohlc (actual order of the initial df of concatenated pairs)
                        for n, i in enumerate(bought["ohlc_ofs"].values)
                    ]
                ),
                # stoploss_bought_ofs and stoploss_rate to the expanded columns
                transpose(
                    repeat(
                        [bought["ohlc_ofs"].values, stoploss_rate],
                        bought_ranges,
                        axis=1,
                    )
                ),
            ],
            axis=1,
        )

        # low (1) <= stoploss_rate (4)
        stoploss = stoploss[stoploss[:, 1] <= stoploss[:, 4], :]
        if len(stoploss) < 1:
            # keep shape since return value is accessed without reference
            return full((0, stoploss.shape[1]), nan)
        # only where the stoploss_bought_ofs (3) is not the same as the previous
        stoploss_bought_ofs_triggered_s1 = insert(stoploss[:-1, 3], 0, nan)
        stoploss = stoploss[where((stoploss[:, 3] != stoploss_bought_ofs_triggered_s1))]
        # exclude stoplosses that where bought past the max index of the triggers
        if not self.position_stacking:
            stoploss = stoploss[
                where(stoploss[:, 3] >= maximum.accumulate(stoploss[:, 3]))[0]
            ]
        # mark objects for gc
        del (
            stoploss_bought_ofs_triggered_s1,
            df,
            ohlc_vals,
        )
        gc.collect()
        return stoploss

    def _pd_calc_triggered_stoploss(
        self, df: DataFrame, bought: DataFrame, bought_ranges: ndarray,
    ):
        """ Expand the ohlc dataframe for each bought candle to check if stoploss was triggered """
        gc.collect()

        ohlc_vals = df["ohlc_ofs"].values

        # create a df with just the indexes to expand
        stoploss_ofs_expd = DataFrame(
            (
                concatenate(
                    [
                        ohlc_vals[i : i + bought_ranges[n]]
                        # loop over the pair/offsetted indexes that will be used as merge key
                        for n, i in enumerate(bought["ohlc_ofs"].values)
                    ]
                )
            ),
            columns=["stoploss_ofs"],
        )
        # add the row data to the expanded indexes
        stoploss = stoploss_ofs_expd.merge(
            # reset level 1 to preserve pair column
            df.reset_index(level=1),
            how="left",
            left_on="stoploss_ofs",
            right_on="ohlc_ofs",
        )
        # set bought idx for each bought timerange, so that we know to which bought candle
        # the row belongs to, and stoploss rates relative to each bought
        stoploss["stoploss_bought_ofs"], stoploss["stoploss_rate"] = repeat(
            [bought["ohlc_ofs"].values, self._calc_stoploss_rate(bought),],
            bought_ranges,
            axis=1,
        )

        stoploss = stoploss.loc[
            stoploss["low"].values <= stoploss["stoploss_rate"].values
        ]
        # filter out duplicate subsequent triggers
        # of the same bought candle as only the first ones matters
        stoploss = stoploss.loc[
            (
                stoploss["stoploss_bought_ofs"].values
                != stoploss["stoploss_bought_ofs"].shift().values
            )
        ]
        if not self.position_stacking:
            # filter out "late" stoplosses that wouldn't be applied because a previous stoploss
            # would still be active at that time
            # since stoplosses are sorted by trigger date,
            # any stoploss having a bought index older than
            # the ohlc index are invalid
            stoploss = stoploss.loc[
                stoploss["stoploss_bought_ofs"]
                >= stoploss["stoploss_bought_ofs"].cummax().values
            ]
        # select columns
        stoploss = stoploss[["stoploss_ofs", "stoploss_bought_ofs", "stoploss_rate"]]

        # mark objects for gc
        del (
            df,
            stoploss_ofs_expd,
            ohlc_vals,
        )
        gc.collect()
        return stoploss

    @staticmethod
    def _last_stoploss_apply(df: DataFrame):
        """ Loop over each row of the dataframe and only select stoplosses for boughts that
        happened after the last set stoploss """
        last = [0]

        def trail_idx(x, last):
            if x.bought_or_sold == Candle.BOUGHT:
                # if a bought candle happens after the last active stoploss index
                if x.ohlc > last[0]:
                    # if stoploss is triggered
                    if x.stoploss_rate > 0:
                        # set the new active stoploss to the current stoploss index
                        last[0] = x.stoploss_ofs
                    else:
                        last[0] = nan
                return last[0]
            else:
                # if the candle is sold, reset the last active stoploss
                last[0] = 0
                return nan

        return df.apply(trail_idx, axis=1, raw=True, args=[last]).values

    @staticmethod
    def _last_stoploss_numba(bts_df: DataFrame):
        """ numba version of _last_stoploss_apply """

        return for_trail_idx(
            bts_df["ohlc"].astype(int).values,
            bts_df["bought_or_sold"].astype(int).values,
            bts_df["stoploss_rate"].fillna(0).astype(float).values,
            # when calling this function, stoploss_ofs should have the offset removed
            bts_df["stoploss_ofs"].fillna(-1).astype(int).values,
        )

    @staticmethod
    def start_pyinst():
        from pyinstrument import Profiler

        global profiler
        profiler = Profiler()
        profiler.start()

    @staticmethod
    def stop_pyinst():
        global profiler
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        exit()

    def _pd_select_triggered_stoploss(
        self,
        df: DataFrame,
        bought: DataFrame,
        bought_ranges: ndarray,
        bts_df: DataFrame,
    ) -> DataFrame:

        # compute all the stoplosses for the buy signals and filter out clear invalids
        stoploss = DataFrame(
            self._np_calc_triggered_stoploss(df, bought, bought_ranges)[:, 2:],
            columns=["stoploss_ofs", "stoploss_bought_ofs", "stoploss_rate"],
            copy=False,
        )
        # stoploss = self._pd_calc_triggered_stoploss(df, bought, bought_ranges)

        # add stoploss data to the bought/sold dataframe
        bts_df = bts_df.merge(
            stoploss, left_on="ohlc_ofs", right_on="stoploss_bought_ofs", how="left",
        ).set_index("ohlc_ofs")
        # don't apply stoploss to sold candles
        bts_df.loc[bts_df["bought_or_sold"].values == Candle.SOLD, "stoploss_ofs"] = nan
        # align original index
        self._remove_pairs_offsets(bts_df, ["stoploss_ofs", "stoploss_bought_ofs"])
        if not self.position_stacking:
            # exclude nested boughts
            # --> | BUY1 | BUY2..STOP2 | STOP1 | -->
            # -->      V    X      X       V     -->
            # bts_df["last_stoploss"] = concatenate(
            #     bts_df.groupby("pair").apply(self._last_stoploss_numba).values
            # )
            bts_df["last_stoploss"] = self._last_stoploss_numba(bts_df)
            bts_df.loc[
                ~(  # last active stoploss matches the current stoploss, otherwise it's stale
                    (bts_df["stoploss_ofs"].values == bts_df["last_stoploss"].values)
                    # it must be the first bought matching that stoploss index,
                    # in case of subsequent boughts that triggers on the same index
                    # which wouldn't happen without position stacking
                    & (
                        bts_df["last_stoploss"].values
                        != bts_df["last_stoploss"].shift().values
                    )
                ),
                ["stoploss_ofs", "stoploss_rate"],
            ] = [nan, nan]
        gc.collect()
        return bts_df

    def _set_stoploss_rate(self, df: DataFrame):
        """ Adds a column for the stoploss rate """
        df["stoploss_rate"] = self._calc_stoploss_rate(df)

    def _calc_stoploss_rate(self, df: DataFrame) -> ndarray:
        return df["open"].values * (1 + self.config["stoploss"])

    def _calc_stoploss_rate_value(self, open_price: float) -> float:
        return open_price * (1 + self.config["stoploss"])

    def vectorized_backtest_buy_sell(
        self,
        processed: Dict[str, DataFrame],
        start_date: arrow.Arrow,
        end_date: arrow.Arrow,
        **kwargs,
    ) -> DataFrame:
        return None

    def split_events(self, bts_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        if self.stoploss_enabled:
            bts_ls_s1 = self._shift_paw(
                bts_df["last_stoploss"], ofs=self._diff_indexes(bts_df["pair"].values)
            )
            events_buy = bts_df.loc[
                (bts_df["bought_or_sold"].values == Candle.BOUGHT)
                & (
                    (
                        bts_df["bought_or_sold"].shift(fill_value=Candle.SOLD).values
                        == Candle.SOLD
                    )
                    # last_stoploss is only valid if == shift(1)
                    # if the previous candle is SOLD it is covered by the previous case
                    # this also covers the case the previous candle == Candle.END
                    | ((bts_df["last_stoploss"].values != bts_ls_s1))
                )
                # exclude the last boughts that are not stoploss and which next sold is
                # END sold candle
                & ~(
                    (isnan(bts_df["stoploss_ofs"].values))
                    & union_eq(bts_df["next_sold_ofs"].values, self.pairs_ofs_end)
                )
            ]
            events_sell = bts_df.loc[
                (
                    (bts_df["bought_or_sold"].values == Candle.SOLD)
                    # select only sold candles that are not preceded by a stoploss
                    & (bts_ls_s1 == -1)
                )
                # and stoplosses (all candles with notna stoploss_ofs should be valid)
                | (isfinite(bts_df["stoploss_ofs"].values))
            ]
        else:
            events_buy = bts_df.loc[
                (bts_df["bought_or_sold"].values == Candle.BOUGHT)
                & (
                    union_eq(
                        bts_df["bought_or_sold"].shift(fill_value=Candle.SOLD)
                        # check for END too otherwise the first bought of mid-pairs
                        # wouldn't be included
                        .values,
                        [Candle.SOLD, Candle.END],
                    )
                )
                # exclude the last boughts that are not stoploss and which next sold is
                # END sold candle
                & ~(union_eq(bts_df["next_sold_ofs"].values, self.pairs_ofs_end))
            ]
            events_sell = bts_df.loc[(bts_df["bought_or_sold"].values == Candle.SOLD)]

        return (events_buy, events_sell)

    def split_events_stack(self, bts_df: DataFrame):
        """"""
        if self.stoploss_enabled:
            events_buy = bts_df.loc[
                (bts_df["bought_or_sold"].values == Candle.BOUGHT)
                # exclude the last boughts that are not stoploss and which next sold is
                # END sold candle
                & ~(
                    (isnan(bts_df["stoploss_ofs"].values))
                    & union_eq(bts_df["next_sold_ofs"].values, self.pairs_ofs_end)
                )
            ]
            # compute the number of sell repetitions for non stoplossed boughts
            nso, sell_repeats = unique(
                events_buy.loc[isnan(events_buy["stoploss_ofs"].values)][
                    "next_sold_ofs"
                ],
                return_counts=True,
            )
            # need to check for membership against the bought candles next_sold_ofs here because
            # some sold candles can be void if all the preceding bought candles
            # (after the previous sold) are triggered by a stoploss
            # (otherwise would just be an eq check == Candle.SOLD)
            events_sell = bts_df.loc[
                bts_df.index.isin(nso) | isfinite(bts_df["stoploss_ofs"].values)
            ]
            events_sell_repeats = ones(len(events_sell))
            events_sell_repeats[events_sell.index.isin(nso)] = sell_repeats
            events_sell = events_sell.reindex(
                events_sell.index.repeat(events_sell_repeats)
            )
        else:
            events_buy = bts_df.loc[
                (bts_df["bought_or_sold"].values == Candle.BOUGHT)
                # exclude the last boughts that are not stoploss and which next sold is
                # END sold candle
                & ~(union_eq(bts_df["next_sold_ofs"].values, self.pairs_ofs_end))
            ]
            events_sell = bts_df.loc[bts_df["bought_or_sold"].values == Candle.SOLD]
            _, sold_repeats = unique(
                events_buy["next_sold_ofs"].values, return_counts=True
            )
            events_sell = events_sell.reindex(events_sell.index.repeat(sold_repeats))
        return (events_buy, events_sell)

    def vectorized_backtest(
        self, processed: Dict[str, DataFrame], **kwargs,
    ) -> DataFrame:
        """ NOTE: can't have default values as arguments since it is an overridden function
        TODO: benchmark if rewriting without use of df masks for
        readability gives a worthwhile speedup
        """
        df = self.merge_pairs_df(processed)

        df, empty = self.bought_or_sold(df)

        if empty:  # if no bought signals
            return self.empty_results

        if self.stoploss_enabled:
            bts_df = self.set_stoploss(df)
        else:
            bts_df, _ = self.set_sold(df)

        if len(bts_df) < 1:
            return self.empty_results

        events_buy, events_sell = (
            self.split_events(bts_df)
            if not self.position_stacking
            else self.split_events_stack(bts_df)
        )

        self._validate_results(events_buy, events_sell)
        return self.get_results(events_buy, events_sell)

    def _validate_results(self, events_buy: DataFrame, events_sell: DataFrame):
        try:
            assert len(events_buy) == len(events_sell)
        except AssertionError:
            print("Buy and sell events not matching")
            print(len(events_buy), len(events_sell))
            print(events_buy.iloc[-10:], events_sell.iloc[-10:])
            raise OperationalException

    def _wrap_backtest(self, processed: Dict[str, DataFrame], **kwargs,) -> DataFrame:
        """ debugging """
        import pickle

        # results = self.backtest_stock(
        #     processed,
        #     **kwargs,
        # )
        results = self.vectorized_backtest(processed)
        with open("/tmp/backtest.pkl", "rb+") as fp:
            # pickle.dump(results, fp)
            saved_results: DataFrame = pickle.load(fp)
        to_print = []
        # for i in results["open_index"].values:
        #     if i not in saved_results["open_index"].values:
        #         to_print.append(i)
        for i in saved_results["open_index"].values:
            if i not in results["open_index"].values:
                to_print.append(i)
        # print(saved_results.sort_values(["pair", "open_time"]).iloc[:10])
        # print(
        #     "to_print count: ",
        #     len(to_print),
        #     "computed res: ",
        #     len(results),
        #     "saved res: ",
        #     len(saved_results),
        # )
        # print(to_print[:10])
        if to_print:
            print(saved_results.loc[saved_results["open_index"].isin(to_print)])
        return results

    # @staticmethod
    # def fill_stub_sold(df: DataFrame, bts_df: DataFrame) -> DataFrame:
    #     """ Helper function to limit trades duration """
    #     sold = (
    #         df.loc[~df.index.isin(bts_df.set_index("index").index)]
    #         .iloc[::1000]
    #         .reset_index()
    #     )

    #     sold["bought_or_sold"] = Candle.SOLD
    #     bts_df = bts_df.merge(sold, how="outer", on=sold.columns.tolist()).sort_values(
    #         by="index"
    #     )
    #     bts_df.drop(
    #         bts_df.loc[
    #             (bts_df["bought_or_sold"].values == Candle.SOLD)
    #             & (bts_df["bought_or_sold"].shift().values == Candle.SOLD)
    #         ].index,
    #     )
    #     # ensure the latest candle is always sold
    #     if bts_df.iloc[-1]["bought_or_sold"] == Candle.BOUGHT:
    #         sold.iloc[len(sold)] = df.iloc[-1]
    #         sold.iloc[-1]["bought_or_sold"] = Candle.SOLD
    #     return (bts_df, sold)
