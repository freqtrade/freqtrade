from typing import Any, Dict, List

from typing_extensions import TypedDict


class BacktestResultType(TypedDict):
    metadata: Dict[str, Any]
    strategy: Dict[str, Any]
    strategy_comparison: List[Any]


def get_BacktestResultType_default() -> BacktestResultType:
    return {
        'metadata': {},
        'strategy': {},
        'strategy_comparison': [],
    }
