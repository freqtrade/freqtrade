# Bot Factory Calibrated Cost Table

Checked on 2026-05-10 JST from the first local cost calibration run.

Candidate generation result:

`no candidate generated`

This table is the current cost context that may be supplied to the next local
Edge Discovery screen. It is not a strategy candidate, thesis result, backtest
result, paper-trading approval, dry-run approval, or live-trading approval.

## Usable Contexts

Use only these completed contexts:

| pair | timeframe | order_type | liquidity_tier | volatility_regime labels run | source artifact |
| --- | --- | --- | --- | --- | --- |
| BTC/USDT:USDT | 5m | taker | liquid | normal, stress | `user_data\data\bybit\futures\BTC_USDT_USDT-5m-futures.parquet` |
| ETH/USDT:USDT | 5m | taker | liquid | normal, stress | `user_data\data\bybit\futures\ETH_USDT_USDT-5m-futures.parquet` |

Do not use maker, 1h, large-alt, spread-driven, order-book-driven, or fills-driven
contexts from this run; those remain blocked or unavailable.

## Calibrated Scenario Costs

BTC/USDT:USDT 5m taker:

| scenario | total_cost_bps | fee_entry | fee_exit | spread | slippage_entry | slippage_exit | adverse_selection | no_fill_rate | partial_fill_rate | exit_taker_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best | 5.285156 | 2.25 | 2.25 | 0.5 | 0.142578 | 0.142578 | 0.0 | 0.0 | 0.0 | 1.0 |
| normal | 9.469967 | 3.0 | 3.0 | 1.176384 | 1.01937 | 1.01937 | 0.254843 | 0.0 | 0.0 | 1.0 |
| stress | 16.900396 | 3.0 | 3.0 | 4.434224 | 2.639254 | 2.639254 | 1.187664 | 0.0 | 0.0 | 1.0 |

ETH/USDT:USDT 5m taker:

| scenario | total_cost_bps | fee_entry | fee_exit | spread | slippage_entry | slippage_exit | adverse_selection | no_fill_rate | partial_fill_rate | exit_taker_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best | 5.608752 | 2.25 | 2.25 | 0.698948 | 0.204902 | 0.204902 | 0.0 | 0.0 | 0.0 | 1.0 |
| normal | 10.856899 | 3.0 | 3.0 | 1.711201 | 1.398088 | 1.398088 | 0.349522 | 0.0 | 0.0 | 1.0 |
| stress | 20.874029 | 3.0 | 3.0 | 6.095478 | 3.583082 | 3.583082 | 1.612387 | 0.0 | 0.0 | 1.0 |

No accepted row has `stress` total cost below `normal` total cost.

## Provenance

| field | provenance |
| --- | --- |
| fee fields | Runner defaults: 3.0 bps entry, 3.0 bps exit; `best` uses the runner's 0.75 fee discount. |
| spread | `ohlcv_range_proxy` because no local spread or order-book artifact was present. |
| slippage | `ohlcv_abs_return_distribution` from closed-candle 5m OHLCV. |
| adverse selection | Taker proxy derived by runner from slippage. |
| no-fill and partial-fill | `0.0` for taker contexts only. Maker contexts are blocked. |
| exit taker rate | `1.0` for taker contexts. |

## Blocked Contexts

| context | status | blocker |
| --- | --- | --- |
| BTC/USDT:USDT 5m maker normal/stress | blocked | Missing maker no-fill, partial-fill, adverse-selection, and exit-taker evidence. |
| ETH/USDT:USDT 5m maker normal/stress | blocked | Missing maker no-fill, partial-fill, adverse-selection, and exit-taker evidence. |
| BTC/USDT:USDT 1h | blocked | No local 1h OHLCV artifact found. |
| ETH/USDT:USDT 1h | blocked | No local 1h OHLCV artifact found. |
| Large alt 1-2 symbols | blocked | No local large-alt OHLCV artifact found. |

Maker blocker names recorded in generated artifacts:

- `maker_no_fill_rate_missing`
- `maker_partial_fill_rate_missing`
- `maker_adverse_selection_bps_missing`
- `maker_exit_taker_rate_missing`

## Edge Discovery Use

Allowed next Edge Discovery cost context:

- BTC/USDT:USDT 5m taker with `all_in_cost_bps` mapped to the `normal`
  `total_cost_bps` of 9.469967.
- ETH/USDT:USDT 5m taker with `all_in_cost_bps` mapped to the `normal`
  `total_cost_bps` of 10.856899.
- Require the same thesis screen to survive the corresponding `stress`
  scenario: 16.900396 bps for BTC and 20.874029 bps for ETH.

Blocked next Edge Discovery cost context:

- any maker thesis or mixed thesis that depends on passive entry fills;
- any thesis requiring order-book, spread, or fills evidence;
- any 1h or large-alt thesis;
- any context whose edge only survives `best`;
- any context that would require a new strategy candidate, new research thesis
  exploration, backtesting, paper trading, dry-run trading, live trading, or an
  exchange order endpoint.

