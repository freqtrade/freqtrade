# Orderbook FeatureStore + FreqAI Pipeline (Fork Extensions)

This document explains how to collect orderbook data into a parquet feature store and
consume it from strategies and FreqAI.

## Components
- tools/ob_collector_ws.py — Bybit v5 official WS -> 1s last -> 1m parquet batches (free)
- freqtrade_ext/feature_store.py — load + embargo + shift(1) + derived features
- user_data/strategies/MyOBStrategy.py — join features, FreqAI handoff
- tools/dq_check.py — Daily DQ report (missing/latency/outliers)

## Requirements (extras)

pip install -r requirements-ext.txt

requirements-ext.txt (free stack)
- aiohttp, orjson, httpx
- pyarrow, pandas, numpy, scikit-learn

## Environment variables (.env example)

EXCHANGE=bybit
SYMBOL=BTCUSDT
DEPTH=200
ROOT_DIR=user_data/featurestore/bybit/BTCUSDT/1s
WS_URL=wss://stream.bybit.com/v5/public/linear
REST_URL=https://api.bybit.com
HEARTBEAT_TIMEOUT=15
BACKOFF_BASE=3.0

## Quickstart

1) Install extras

pip install -r requirements-ext.txt

2) Start collector (Bybit official WS)

python tools/ob_collector_ws.py

Data will be written under user_data/featurestore/<exchange>/<pair>/1s/ partitioned by year/month/day.

3) Backtesting

freqtrade backtesting --strategy MyOBStrategy --timeframe 1m --timerange=20250101-20250107

4) FreqAI training

freqtrade freqai-train --strategy MyOBStrategy --timeframe 1m --timerange=20250101-20250121

5) DQ check (optional)

python tools/dq_check.py user_data/featurestore/bybit/BTCUSDT/1s 2025-01-02

Notes:
- All timestamps are UTC.
- Embargo and shift(1) prevent look-ahead leakage.
- Keep exchange fees consistent to avoid double-counting when adding an execution model.
- No paid dependencies (no ccxtpro). Uses Bybit official WS + REST snapshot.
