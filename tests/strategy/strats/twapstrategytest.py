from datetime import datetime, timedelta

from pandas import DataFrame
from technical import qtpylib

from freqtrade.persistence import Trade

from .strategy_test_v3 import StrategyTestV3


class TWAPStrategyTest(StrategyTestV3):
    position_adjustment_enable = True
    max_entry_position_adjustment = 10
    twap_num_slices = 10
    twap_interval_minutes = 1

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return proposed_stake / self.twap_num_slices

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] < self.buy_rsi.value)
                & (dataframe["fastd"] < 35)
                & (dataframe["adx"] > 30)
                & (dataframe["plus_di"] > self.buy_plusdi.value)
            )
            | ((dataframe["adx"] > 65) & (dataframe["plus_di"] > self.buy_plusdi.value)),
            "enter_long",
        ] = 1
        dataframe.loc[
            (qtpylib.crossed_below(dataframe["rsi"], self.sell_rsi.value)),
            ("enter_short", "enter_tag"),
        ] = (1, "short_Tag")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def ome_populate_exit_trend(self, trade: Trade, current_time: datetime) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return False

        df = dataframe.copy()
        if "rsi" not in df or "fastd" not in df:
            return False

        df.loc[
            (
                (qtpylib.crossed_above(df["rsi"], self.sell_rsi.value))
                | (qtpylib.crossed_above(df["fastd"], 70))
            )
            & (df["adx"] > 10),
            "exit_long",
        ] = 1

        df.loc[
            qtpylib.crossed_above(df["rsi"], self.buy_rsi.value),
            "exit_short",
        ] = 1

        last_candle = df.iloc[-1]
        signal_col = "exit_short" if trade.is_short else "exit_long"
        return bool(last_candle.get(signal_col, 0) == 1)

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str | None]:

        if trade.has_open_orders:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        entry_slices_done = len(filled_entries)
        filled_exits = trade.select_filled_orders(trade.exit_side)
        exit_slices_done = len(filled_exits)

        already_exiting = exit_slices_done > 0

        if already_exiting or self.ome_populate_exit_trend(trade, current_time):
            return self._next_exit_slice(trade, current_time, filled_exits, exit_slices_done)

        if entry_slices_done < self.twap_num_slices:
            return self._next_entry_slice(trade, current_time, filled_entries, entry_slices_done)

        return None

    def _next_entry_slice(
        self, trade: Trade, current_time: datetime, filled_entries: list, slices_done: int
    ) -> float | None | tuple[float | None, str | None]:

        last_fill_time = (
            filled_entries[-1].order_filled_utc if filled_entries else trade.open_date_utc
        )
        next_slice_due_at = last_fill_time + timedelta(minutes=self.twap_interval_minutes)
        if current_time < next_slice_due_at:
            return None

        stake_already_filled = sum(o.stake_amount_filled for o in filled_entries)
        twap_total_stake = (
            filled_entries[0].stake_amount_filled * self.twap_num_slices
            if filled_entries
            else trade.stake_amount * self.twap_num_slices
        )
        remaining_stake = twap_total_stake - stake_already_filled
        remaining_slices = self.twap_num_slices - slices_done

        next_slice_stake = (
            remaining_stake if remaining_slices <= 1 else remaining_stake / remaining_slices
        )
        if next_slice_stake < 0:
            return None

        return next_slice_stake, f"twap_entry_{slices_done + 1}_of_{self.twap_num_slices}"

    def _next_exit_slice(
        self, trade: Trade, current_time: datetime, filled_exits: list, slices_done: int
    ) -> float | None | tuple[float | None, str | None]:

        if slices_done >= self.twap_num_slices:
            return None

        last_fill_time = filled_exits[-1].order_filled_utc if filled_exits else current_time
        next_slice_due_at = last_fill_time + timedelta(minutes=self.twap_interval_minutes)
        if slices_done > 0 and current_time < next_slice_due_at:
            return None

        remaining_slices = self.twap_num_slices - slices_done

        if remaining_slices <= 1:
            return -trade.stake_amount

        slice_stake = trade.stake_amount / remaining_slices
        return -slice_stake
