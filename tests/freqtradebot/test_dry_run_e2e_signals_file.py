from __future__ import annotations

import time
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import rapidjson

from freqtrade.data.converter.converter import ohlcv_to_dataframe
from freqtrade.enums import CandleType
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.misc import dataframe_to_json
from freqtrade.persistence import Trade
from tests.conftest import EXMS, patch_exchange


def _write_signals_file(filename: Path, pair: str, timeframe: str, ohlcv_df) -> None:
    signals_df = ohlcv_df[["date"]].copy()
    signals_df["enter_long"] = 0
    signals_df.loc[signals_df.index[-1], "enter_long"] = 1
    signals_df["exit_long"] = 0
    signals_df["enter_short"] = 0
    signals_df["exit_short"] = 0
    signals_df["enter_tag"] = None
    signals_df["exit_tag"] = None

    messages = [
        {"type": "whitelist", "data": [pair]},
        {
            "type": "analyzed_df",
            "data": {
                "key": [pair, timeframe, CandleType.SPOT.value],
                "la": datetime.now(UTC).isoformat(),
                "df": {
                    "__type__": "dataframe",
                    "__value__": dataframe_to_json(signals_df),
                },
            },
        },
    ]

    with filename.open("w") as fp:
        rapidjson.dump(messages, fp)


def test_dry_run_e2e_signals_file(default_conf, mocker, tmp_path) -> None:
    pair = "ETH/BTC"
    timeframe = "5m"

    config = deepcopy(default_conf)
    config["strategy"] = "StrategyTestProducerSignals"
    config["exchange"]["pair_whitelist"] = [pair]

    # Provide upstream signals via signals.json replay
    signals_file = tmp_path / "signals.json"

    now = datetime.now(UTC).replace(second=0, microsecond=0)
    start = now - timedelta(minutes=10)
    ohlcv = [
        [int(start.timestamp() * 1000), 1.0, 1.0, 1.0, 1.0, 1.0],
        [int((start + timedelta(minutes=5)).timestamp() * 1000), 1.0, 1.0, 1.0, 1.0, 1.0],
        [int((start + timedelta(minutes=10)).timestamp() * 1000), 1.0, 1.0, 1.0, 1.0, 1.0],
    ]
    ohlcv_df = ohlcv_to_dataframe(
        ohlcv, timeframe, pair=pair, fill_missing=True, drop_incomplete=False
    )
    _write_signals_file(signals_file, pair, timeframe, ohlcv_df)

    config["external_message_consumer"] = {
        "enabled": True,
        "producers": [
            {
                "name": "default",
                "signals_file": str(signals_file),
            }
        ],
    }

    # Patch exchange + RPC, but keep the real ExternalMessageConsumer.
    mocker.patch("freqtrade.freqtradebot.RPCManager", MagicMock())
    mocker.patch("freqtrade.freqtradebot.RPCManager._init", MagicMock())
    mocker.patch("freqtrade.freqtradebot.RPCManager.send_msg", MagicMock())
    patch_exchange(mocker)
    mocker.patch(
        "freqtrade.freqtradebot.FreqtradeBot._refresh_active_whitelist",
        MagicMock(return_value=[pair]),
    )

    # Avoid waiting on exchange interaction.
    mocker.patch(f"{EXMS}.create_order", side_effect=lambda **kwargs: {
        "id": "dry-run-order",
        "status": "closed",
        "filled": kwargs["amount"],
        "amount": kwargs["amount"],
        "remaining": 0.0,
        "average": kwargs["rate"],
        "price": kwargs["rate"],
    })
    mocker.patch(f"{EXMS}.get_rate", return_value=1.0)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.0)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=1e9)
    mocker.patch(f"{EXMS}.get_pair_base_currency", return_value="ETH")
    mocker.patch(f"{EXMS}.get_pair_quote_currency", return_value="BTC")
    mocker.patch(f"{EXMS}.get_funding_fees", return_value=0.0)
    mocker.patch(f"{EXMS}.get_precision_amount", return_value=8)
    mocker.patch(f"{EXMS}.get_precision_price", return_value=8)
    mocker.patch(f"{EXMS}.get_contract_size", return_value=1)
    mocker.patch(f"{EXMS}.amount_to_precision", side_effect=lambda *a, **k: a[-1])
    mocker.patch(f"{EXMS}.price_to_precision", side_effect=lambda *a, **k: a[-1])

    freqtrade = FreqtradeBot(config)
    try:
        # Make OHLCV available for analysis.
        freqtrade.exchange._klines[(pair, timeframe, CandleType.SPOT)] = ohlcv_df

        # Wait for the signals file to be consumed.
        for _ in range(50):
            pdf, _ = freqtrade.dataprovider.get_producer_df(
                pair, timeframe=timeframe, candle_type=CandleType.SPOT, producer_name="default"
            )
            if not pdf.empty:
                break
            time.sleep(0.05)

        # Analyze candles (strategy merges upstream signals) and enter based on last candle.
        freqtrade.strategy.analyze_pair(pair)
        n = freqtrade.enter_positions()
        assert n == 1
        assert len(Trade.get_open_trades()) == 1
        assert Trade.get_open_trades()[0].pair == pair
    finally:
        if freqtrade.emc:
            freqtrade.emc.shutdown()
