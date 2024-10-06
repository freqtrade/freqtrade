from typing import Any, Optional

from typing_extensions import TypedDict


class BacktestMetadataType(TypedDict):
    run_id: str
    backtest_start_time: int


class BacktestResultType(TypedDict):
    metadata: dict[str, Any]  # BacktestMetadataType
    strategy: dict[str, Any]
    strategy_comparison: list[Any]


def get_BacktestResultType_default() -> BacktestResultType:
    return {
        "metadata": {},
        "strategy": {},
        "strategy_comparison": [],
    }


class BacktestHistoryEntryType(BacktestMetadataType):
    filename: str
    strategy: str
    notes: str
    backtest_start_ts: Optional[int]
    backtest_end_ts: Optional[int]
    timeframe: Optional[str]
    timeframe_detail: Optional[str]
