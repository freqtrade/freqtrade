from pathlib import Path
from unittest.mock import MagicMock

from freqtrade.resolvers import StrategyResolver


def test_india_options_auto_strategy_signals(default_conf, dataframe_1m):
    default_conf.update(
        {
            "strategy": "IndiaOptionsAutoStrategy",
            "strategy_path": str(Path(__file__).parents[2] / "user_data" / "strategies"),
        }
    )
    strategy = StrategyResolver.load_strategy(default_conf)
    strategy.dp = MagicMock()
    strategy.dp.get_pair_dataframe.return_value = dataframe_1m.copy()

    metadata = {"pair": "RELIANCE-20250130-2500-CE/INR"}
    indicators = strategy.advise_indicators(dataframe_1m.copy(), metadata=metadata)
    assert {"ema_5_underlying", "ema_20_underlying", "rsi_14_underlying"}.issubset(
        indicators.columns
    )

    entries = strategy.advise_entry(indicators.copy(), metadata=metadata)
    assert "enter_long" in entries.columns

    exits = strategy.advise_exit(indicators.copy(), metadata=metadata)
    assert "exit_long" in exits.columns
