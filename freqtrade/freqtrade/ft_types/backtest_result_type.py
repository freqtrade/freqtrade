from copy import deepcopy
from typing import Any, cast

from pandas import DataFrame
from typing_extensions import TypedDict

from freqtrade.constants import Config


class BacktestMetadataType(TypedDict):
    run_id: str
    backtest_start_time: int


class BacktestResultType(TypedDict):
    metadata: dict[str, Any]  # BacktestMetadataType
    strategy: dict[str, Any]
    strategy_comparison: list[Any]


def get_BacktestResultType_default() -> BacktestResultType:
    return cast(
        BacktestResultType,
        deepcopy(
            {
                "metadata": {},
                "strategy": {},
                "strategy_comparison": [],
            }
        ),
    )


class BacktestHistoryEntryType(BacktestMetadataType):
    filename: str
    strategy: str
    notes: str
    backtest_start_ts: int | None
    backtest_end_ts: int | None
    timeframe: str | None
    timeframe_detail: str | None


class BacktestContentTypeIcomplete(TypedDict, total=False):
    results: DataFrame
    config: Config
    locks: Any
    rejected_signals: int
    timedout_entry_orders: int
    timedout_exit_orders: int
    canceled_trade_entries: int
    canceled_entry_orders: int
    replaced_entry_orders: int
    final_balance: float
    backtest_start_time: int
    backtest_end_time: int
    run_id: str


class BacktestContentType(BacktestContentTypeIcomplete, total=True):
    pass
