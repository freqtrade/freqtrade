# Bot Factory Market State Schema

Status: draft schema for Increment 0.

Implementation status: documentation contract only. This file defines the
artifact shape for future local-only market-state tooling. It does not implement
data loading, classification, strategy matching, paper trading, dry-run trading,
live trading, process control, exchange access, or order placement.

## Scope

The market-state layer answers what local historical artifacts say as of an
explicit timestamp. It must not claim to be live exchange state.

Allowed inputs:

- local closed-candle OHLCV artifacts;
- local public market-structure artifacts that already exist in the workspace;
- local data quality, feature quality, and cost-model artifacts.

Disallowed inputs:

- live order book, spread, account, wallet, fill, or order endpoints;
- API keys, secrets, private environment values, or exchange credentials;
- paper, dry-run, live, canary, or exchange-facing process state.

## Artifacts

Future producers should write these files:

```text
data/market_state/<run_id>/market_state_snapshot.json
data/market_state/<run_id>/market_state_windows.jsonl
data/market_state/<run_id>/market_state_report.md
data/market_state/current/<run_id>/current_market_state.json
data/market_state/current/<run_id>/current_market_state_report.md
```

## Invariants

- `current` always means current as of `data_asof`, never live exchange state.
- State labels are derived before strategy performance is evaluated for the
  same holdout window.
- Market-state features are independent from strategy PnL, trade outcomes, and
  hindsight-best strategy labels.
- Unknown, stale, low-confidence, conflicting, or out-of-distribution states
  default to `no_trade` at later matching layers.
- Every artifact records the safety scope proving no startup or order path was
  used.

## State Labels

The initial label vocabulary is intentionally small:

```text
trend_up
trend_down
range
high_volatility
low_volatility
liquidity_stress
post_spike_reversion
mixed
transition
out_of_distribution
unknown
```

Labels describe market state only. They must not encode actions such as
`buy`, `sell`, `avoid_long`, or `trade_this_strategy`.

## State ID

`state_id` is a stable, auditable identifier for a state definition:

```text
<state_encoder_version>:<horizon>:<label>:<confidence_bucket>:<feature_version>
```

Example:

```text
deterministic_market_state_encoder_v1:1h:trend_up:high:ohlcv_state_features_v1
```

`state_id` must not include strategy returns, candidate ids, trade outcomes, or
post-window labels.

## Horizon Profile ID

`horizon_profile_id` identifies the multi-horizon combination used for matching:

```text
<state_encoder_version>:micro=<label>:intraday=<label>:swing=<label>
```

Example:

```text
deterministic_market_state_encoder_v1:micro=mixed:intraday=trend_up:swing=high_volatility
```

The profile is descriptive only. It is not proof that any strategy is eligible.

## `market_state_snapshot_v1`

The snapshot is one point-in-time, multi-horizon summary.

Required top-level fields:

| Field | Type | Notes |
| --- | --- | --- |
| `factory` | string | Must be `bot_factory`. |
| `schema_version` | string | Must be `market_state_snapshot_v1`. |
| `run_id` | string | Stable local run identifier. |
| `generated_at` | timestamp | Artifact creation time. |
| `data_asof` | timestamp | Latest trusted local source timestamp used by the snapshot. |
| `latest_local_candle_at` | timestamp | Latest closed candle timestamp in the base timeframe. |
| `git_commit` | string or null | Workspace commit when available. |
| `source_data_paths` | array[string] | Local paths only. |
| `source_data_hashes` | object | Path or logical source id to content hash. |
| `pair` | string | Pair covered by the snapshot. |
| `pair_group` | string | Example: `btc_major`, `eth_major`, `alt`, or `single_pair`. |
| `base_timeframe` | string | Base timeframe used for the primary candle series. |
| `horizons` | array[object] | One `market_state_window_v1` compatible object per horizon. |
| `state_encoder_version` | string | Version of the market-state encoder. |
| `regime_classifier_version` | string | Existing deterministic classifier version when used. |
| `feature_version` | string | Feature contract version. |
| `cost_model_id` | string | Cost context used for pressure features. |
| `data_quality_summary` | object | Summary of missingness, gaps, and freshness. |
| `feature_quality_summary` | object | Summary of feature availability and quality. |
| `state_confidence` | number | Aggregate confidence in `[0.0, 1.0]`. |
| `uncertainty` | number | Aggregate uncertainty in `[0.0, 1.0]`. |
| `unknown_reason` | string or null | Required when aggregate state is `unknown`. |
| `out_of_distribution_score` | number | Higher means less historical analog coverage. |
| `horizon_profile_id` | string | Stable id for the combined horizon profile. |
| `horizon_conflict` | object | Conflict status and reason codes. |
| `no_trade_default` | boolean | True when future matching must abstain by default. |
| `reason_codes` | array[string] | Stable audit reason codes. |
| `safety_scope` | object | Required no-startup, no-live, no-secret flags. |

Required `safety_scope` flags:

```json
{
  "local_artifacts_source_of_truth": true,
  "closed_candle_local_market_data_only": true,
  "live_data_used": false,
  "freqtrade_trade_started": false,
  "paper_trading_started": false,
  "dry_run_trading_started": false,
  "live_trading_started": false,
  "exchange_order_placement": false,
  "uses_api_keys_or_secrets": false,
  "metadata_contains_secrets": false,
  "process_control": false,
  "leverage_above_one": false,
  "shorting": false
}
```

## `market_state_window_v1`

Each JSONL row and each `horizons[]` entry uses this shape.

Required fields:

| Field | Type | Notes |
| --- | --- | --- |
| `schema_version` | string | Must be `market_state_window_v1`. |
| `run_id` | string | Parent snapshot run id. |
| `pair` | string | Pair covered by this row. |
| `timeframe` | string | Source candle timeframe. |
| `horizon` | string | One of `5m`, `15m`, `1h`, `4h`, `1d`, `1w` when local data exists. |
| `horizon_group` | string | `micro`, `intraday`, or `swing`. |
| `lookback_window` | object | Candle count and wall-clock duration. |
| `decision_window_start` | timestamp | Start of the as-of decision window. |
| `decision_window_end` | timestamp | End of the as-of decision window. |
| `label` | string | One state label from the vocabulary. |
| `state_id` | string | Stable state identifier. |
| `confidence` | number | Horizon confidence in `[0.0, 1.0]`. |
| `uncertainty` | number | Horizon uncertainty in `[0.0, 1.0]`. |
| `out_of_distribution_score` | number | Horizon OOD score. |
| `state_vector` | object | Numeric feature vector, independent from strategy returns. |
| `feature_cutoff_timestamp` | timestamp | Latest timestamp used for feature values. |
| `label_cutoff_timestamp` | timestamp | Latest timestamp used for the label. |
| `future_data_used` | boolean | Must be `false`. |
| `data_quality_flags` | array[string] | Missingness, gap, and freshness flags. |
| `feature_quality_flags` | array[string] | Feature validation flags. |
| `unknown_reason` | string or null | Required for `unknown`. |
| `reason_codes` | array[string] | Stable reasons for the label and confidence. |

Required `state_vector` keys:

| Field | Type | Notes |
| --- | --- | --- |
| `rolling_return_bps` | number | Return over the horizon lookback. |
| `realized_volatility_bps` | number | Realized volatility over the lookback. |
| `volatility_zscore` | number | Volatility relative to local history. |
| `trend_slope_bps_per_candle` | number | Predeclared trend slope feature. |
| `moving_average_distance_bps` | number | Close versus moving average distance. |
| `range_efficiency` | number | Directional return divided by traveled range. |
| `drawdown_from_local_high_bps` | number | Drawdown from the local lookback high. |
| `high_low_range_pct` | number | High-low range relative to close. |
| `candle_gap_proxy_bps` | number | Local candle gap proxy. |
| `volume_liquidity_zscore` | number or null | Volume/liquidity proxy from local candles. |
| `turnover_cost_pressure` | number or null | Local cost or turnover pressure when available. |
| `missing_candle_rate` | number | Missing candle fraction. |
| `freshness_age_minutes` | number | Age of the latest local candle at generation time. |

Optional context fields may be included only when local public artifacts already
exist:

```text
mark_price_context
funding_rate_context
open_interest_context
liquidation_context
orderbook_depth_context
```

## Anti-Leakage Rules

- `feature_cutoff_timestamp <= decision_window_end`.
- `label_cutoff_timestamp <= decision_window_end`.
- `future_data_used=false`.
- `decision_window_start` and `decision_window_end` are recorded for every row.
- Any future strategy scorecard that joins these rows must join only on state
  artifacts available before its own evaluation decision time.

## Staleness And Unknown Rules

Future producers must set `no_trade_default=true` and include a reason code when:

- local candles are stale for the horizon and base timeframe;
- required source paths are missing;
- data quality fails;
- feature quality fails;
- confidence is below the configured threshold;
- horizon labels conflict materially;
- OOD score exceeds the configured threshold.

Recommended reason codes:

```text
stale_local_data
missing_required_source
data_quality_failed
feature_quality_failed
low_state_confidence
horizon_conflict
out_of_distribution_state
unknown_state
```

## Current-State Report Contract

`current_market_state.json` should reference one snapshot and add report-level
review fields:

| Field | Type | Notes |
| --- | --- | --- |
| `schema_version` | string | Must be `current_market_state_v1`. |
| `snapshot_schema_version` | string | Must be `market_state_snapshot_v1`. |
| `snapshot_path` | string | Local path to the snapshot. |
| `data_asof` | timestamp | Repeated at the top of the report. |
| `stale_data` | boolean | Whether local data is stale. |
| `no_trade_default` | boolean | Abstention default for future matching. |
| `horizon_conflict` | object | Compact conflict summary. |
| `not_allowed_confirmation` | object | Confirms no paper, dry-run, live, bot, or order path started. |

## Example Snapshot

```json
{
  "factory": "bot_factory",
  "schema_version": "market_state_snapshot_v1",
  "run_id": "20260528T000000Z_btc_usdt_market_state",
  "generated_at": "2026-05-28T00:00:00+09:00",
  "data_asof": "2026-05-27T23:55:00+09:00",
  "latest_local_candle_at": "2026-05-27T23:55:00+09:00",
  "git_commit": null,
  "source_data_paths": [
    "user_data/data/bybit/futures/BTC_USDT_USDT-5m-futures.parquet"
  ],
  "source_data_hashes": {
    "ohlcv_5m": "sha256:example"
  },
  "pair": "BTC/USDT:USDT",
  "pair_group": "btc_major",
  "base_timeframe": "5m",
  "state_encoder_version": "deterministic_market_state_encoder_v1",
  "regime_classifier_version": "deterministic_regime_classifier_v1",
  "feature_version": "ohlcv_state_features_v1",
  "cost_model_id": "calibrated_cost_model_v1",
  "horizon_profile_id": "deterministic_market_state_encoder_v1:micro=mixed:intraday=trend_up:swing=unknown",
  "state_confidence": 0.62,
  "uncertainty": 0.38,
  "unknown_reason": null,
  "out_of_distribution_score": 0.21,
  "data_quality_summary": {
    "missing_candle_rate": 0.0,
    "stale_data": false
  },
  "feature_quality_summary": {
    "feature_quality_pass": true,
    "failed_features": []
  },
  "horizon_conflict": {
    "conflict_detected": true,
    "reason_codes": ["micro_intraday_label_conflict"]
  },
  "no_trade_default": true,
  "reason_codes": ["horizon_conflict"],
  "horizons": [
    {
      "schema_version": "market_state_window_v1",
      "run_id": "20260528T000000Z_btc_usdt_market_state",
      "pair": "BTC/USDT:USDT",
      "timeframe": "5m",
      "horizon": "1h",
      "horizon_group": "intraday",
      "lookback_window": {"candles": 12, "duration": "PT1H"},
      "decision_window_start": "2026-05-27T22:55:00+09:00",
      "decision_window_end": "2026-05-27T23:55:00+09:00",
      "label": "trend_up",
      "state_id": "deterministic_market_state_encoder_v1:1h:trend_up:medium:ohlcv_state_features_v1",
      "confidence": 0.7,
      "uncertainty": 0.3,
      "out_of_distribution_score": 0.18,
      "state_vector": {
        "rolling_return_bps": 42.0,
        "realized_volatility_bps": 18.0,
        "volatility_zscore": 0.4,
        "trend_slope_bps_per_candle": 3.1,
        "moving_average_distance_bps": 24.0,
        "range_efficiency": 0.62,
        "drawdown_from_local_high_bps": -8.0,
        "high_low_range_pct": 0.31,
        "candle_gap_proxy_bps": 0.0,
        "volume_liquidity_zscore": 0.2,
        "turnover_cost_pressure": null,
        "missing_candle_rate": 0.0,
        "freshness_age_minutes": 5.0
      },
      "feature_cutoff_timestamp": "2026-05-27T23:55:00+09:00",
      "label_cutoff_timestamp": "2026-05-27T23:55:00+09:00",
      "future_data_used": false,
      "data_quality_flags": [],
      "feature_quality_flags": [],
      "unknown_reason": null,
      "reason_codes": ["positive_return", "positive_trend_slope"]
    }
  ],
  "safety_scope": {
    "local_artifacts_source_of_truth": true,
    "closed_candle_local_market_data_only": true,
    "live_data_used": false,
    "freqtrade_trade_started": false,
    "paper_trading_started": false,
    "dry_run_trading_started": false,
    "live_trading_started": false,
    "exchange_order_placement": false,
    "uses_api_keys_or_secrets": false,
    "metadata_contains_secrets": false,
    "process_control": false,
    "leverage_above_one": false,
    "shorting": false
  }
}
```

## Example JSONL Row

```json
{"schema_version":"market_state_window_v1","run_id":"20260528T000000Z_btc_usdt_market_state","pair":"BTC/USDT:USDT","timeframe":"5m","horizon":"5m","horizon_group":"micro","lookback_window":{"candles":1,"duration":"PT5M"},"decision_window_start":"2026-05-27T23:50:00+09:00","decision_window_end":"2026-05-27T23:55:00+09:00","label":"mixed","state_id":"deterministic_market_state_encoder_v1:5m:mixed:low:ohlcv_state_features_v1","confidence":0.45,"uncertainty":0.55,"out_of_distribution_score":0.24,"state_vector":{"rolling_return_bps":-3.0,"realized_volatility_bps":11.0,"volatility_zscore":0.1,"trend_slope_bps_per_candle":-0.8,"moving_average_distance_bps":2.0,"range_efficiency":0.12,"drawdown_from_local_high_bps":-5.0,"high_low_range_pct":0.09,"candle_gap_proxy_bps":0.0,"volume_liquidity_zscore":-0.1,"turnover_cost_pressure":null,"missing_candle_rate":0.0,"freshness_age_minutes":5.0},"feature_cutoff_timestamp":"2026-05-27T23:55:00+09:00","label_cutoff_timestamp":"2026-05-27T23:55:00+09:00","future_data_used":false,"data_quality_flags":[],"feature_quality_flags":[],"unknown_reason":null,"reason_codes":["low_directional_efficiency","low_confidence"]}
```
