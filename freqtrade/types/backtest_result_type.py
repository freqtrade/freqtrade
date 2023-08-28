from typing import Any, Dict, List

from typing_extensions import TypedDict


class BacktestMetadataType(TypedDict):
    run_id: str
    backtest_start_time: int


class BacktestResultType(TypedDict):
    metadata: Dict[str, Any]  # BacktestMetadataType
    strategy: Dict[str, Any]
    strategy_comparison: List[Any]


def get_BacktestResultType_default() -> BacktestResultType:
    return {
        'metadata': {},
        'strategy': {},
        'strategy_comparison': [],
    }


class BacktestHistoryEntryType(BacktestMetadataType):
    filename: str
    strategy: str
    notes: str
