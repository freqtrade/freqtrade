from pathlib import Path

from freqtrade.resolvers import StrategyResolver


def test_india_equity_smoke_strategy_signals(default_conf, dataframe_1m):
    default_conf.update(
        {
            "strategy": "IndiaEquitySmokeStrategy",
            "strategy_path": str(Path(__file__).parents[2] / "user_data" / "strategies"),
        }
    )
    strategy = StrategyResolver.load_strategy(default_conf)

    metadata = {"pair": "ETH/BTC"}
    indicators = strategy.advise_indicators(dataframe_1m.copy(), metadata=metadata)
    assert {"ema_9", "ema_21", "rsi_14"}.issubset(indicators.columns)

    entries = strategy.advise_entry(indicators.copy(), metadata=metadata)
    assert "enter_long" in entries.columns

    exits = strategy.advise_exit(indicators.copy(), metadata=metadata)
    assert "exit_long" in exits.columns
