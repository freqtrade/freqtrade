from __future__ import annotations

import pandas as pd

from user_data.strategies.MyOBStrategy import MyOBStrategy


def _ohlcv_df():
    idx = pd.date_range("2025-01-01", periods=180, freq="T", tz="UTC")
    df = pd.DataFrame(
        {
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100,
            "volume": 1.0,
        },
        index=idx,
    )
    return df


def test_strategy_adds_prefixed_columns(monkeypatch):
    strat = MyOBStrategy(config={})

    # Monkeypatch loader to return simple features aligned to df index
    def fake_loader(
        exchange, pair, timeframe, timerange=None, embargo_secs=1, depth=200, root_dir=None
    ):
        start, end = timerange if timerange else (df.index.min(), df.index.max())
        idx = pd.date_range(start=start, end=end, freq="T", tz="UTC")
        feats = pd.DataFrame(
            {
                "spread_bps": 1.0,
                "microprice": 100.0,
                "ob_imbalance": 0.5,
                "ob_depth_delta": 0.0,
                "ofi_top": 0.0,
                "book_slope": 0.0,
            },
            index=idx,
        )
        feats.index.name = "date"
        return feats

    df = _ohlcv_df()
    monkeypatch.setattr(
        "user_data.strategies.MyOBStrategy.load_orderbook_features", fake_loader, raising=False
    )

    out = strat.populate_indicators(df.copy(), metadata={"pair": "BTC/USDT:USDT"})

    # Check prefixed columns exist
    for c in strat.FEAT_COLS:
        assert f"{strat.FEAT_PREFIX}{c}" in out.columns

    # no_trade column exists
    assert "no_trade" in out.columns
