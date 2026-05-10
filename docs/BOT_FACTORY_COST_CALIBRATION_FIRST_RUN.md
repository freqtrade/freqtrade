# Bot Factory Cost Calibration First Run

Checked on 2026-05-10 JST after fast-forwarding the current work branch to
`origin/develop` merge commit `6632cfd48`.

Candidate generation result:

`no candidate generated`

This first run did not generate a strategy candidate, explore a research thesis,
retry an existing failed thesis, run a backtest, start paper/dry-run/live
trading, call exchange order endpoints, use API keys or secrets, change
leverage, or enable shorting.

## Source Docs Reviewed

- `docs/BOT_FACTORY_COST_CALIBRATION_PLAN.md`
- `docs/BOT_FACTORY_EXECUTION_QUALITY_AUDIT.md`
- `docs/BOT_FACTORY_COST_CALIBRATION_REPORT.md`
- CLI help:
  `.\.venv\Scripts\python.exe scripts\bot_factory_calibrate_cost_model.py --help`

The CLI writes local-only artifacts under `data/cost_calibration` and always
records `candidate_generation_result=no candidate generated`.

## Local Artifact Inventory

Available local market data:

| artifact | status | rows | span |
| --- | --- | ---: | --- |
| `user_data\data\bybit\futures\BTC_USDT_USDT-5m-futures.parquet` | usable OHLCV | 246895 | 2024-01-01T00:00:00+00:00 to 2026-05-07T06:30:00+00:00 |
| `user_data\data\bybit\futures\ETH_USDT_USDT-5m-futures.parquet` | usable OHLCV | 246941 | 2024-01-01T00:00:00+00:00 to 2026-05-07T10:20:00+00:00 |

Quality checks passed for both OHLCV files with no duplicate timestamps and no
missing 5m intervals.

Missing local artifacts:

| context | blocker |
| --- | --- |
| BTC/USDT:USDT 1h | No local 1h OHLCV artifact found. |
| ETH/USDT:USDT 1h | No local 1h OHLCV artifact found. |
| Large-alt 5m/1h | No local large-alt OHLCV artifact found. |
| Spread artifact | No local spread artifact found under `data` or `user_data\data`. |
| Order-book/depth artifact | No local order-book artifact found under `data` or `user_data\data`. |
| Fills artifact | No local fills artifact found under `data` or `registry`. |

## Calibration Matrix

The target matrix was:

- BTC/USDT:USDT: 5m and 1h, taker and maker, normal and stress volatility.
- ETH/USDT:USDT: 5m and 1h, taker and maker, normal and stress volatility.
- Large alt 1-2 symbols: 5m and/or 1h if local artifacts exist.

Executed matrix, limited to local artifacts:

| pair | timeframe | order_type | volatility_regime | result |
| --- | --- | --- | --- | --- |
| BTC/USDT:USDT | 5m | taker | normal | completed |
| BTC/USDT:USDT | 5m | taker | stress | completed |
| ETH/USDT:USDT | 5m | taker | normal | completed |
| ETH/USDT:USDT | 5m | taker | stress | completed |
| BTC/USDT:USDT | 5m | maker | normal | blocked |
| BTC/USDT:USDT | 5m | maker | stress | blocked |
| ETH/USDT:USDT | 5m | maker | normal | blocked |
| ETH/USDT:USDT | 5m | maker | stress | blocked |

Blocked matrix entries:

| pair/context | reason |
| --- | --- |
| BTC/USDT:USDT 1h | No local 1h OHLCV input was present. |
| ETH/USDT:USDT 1h | No local 1h OHLCV input was present. |
| Large alt 1-2 symbols | No local large-alt OHLCV input was present. |
| Maker contexts | No local order-book/depth or fills artifact was present, so no-fill, partial-fill, adverse-selection, and exit-taker risk could not be estimated. |

## Commands And Results

Repository update:

```powershell
git status --short --untracked-files=all
git fetch origin develop
git merge --ff-only origin/develop
git status --short --untracked-files=all
```

Result: initial and post-merge working tree status were clean. The current
branch fast-forwarded from `ac96e4acb` to `6632cfd48`.

OHLCV validation:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_check_ohlcv.py user_data\data\bybit\futures\BTC_USDT_USDT-5m-futures.parquet --timeframe 5m
.\.venv\Scripts\python.exe scripts\bot_factory_check_ohlcv.py user_data\data\bybit\futures\ETH_USDT_USDT-5m-futures.parquet --timeframe 5m
```

Result: both commands returned `ok=true`. Local quality JSON reports were
written under `registry\strategies\checks` and remain generated artifacts.

Completed taker runs:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_calibrate_cost_model.py --ohlcv-path user_data\data\bybit\futures\BTC_USDT_USDT-5m-futures.parquet --pair BTC/USDT:USDT --timeframe 5m --order-type taker --liquidity-tier liquid --volatility-regime normal --cost-calibration-id first_run_btc_usdt_5m_taker_normal --output-root data\cost_calibration --reviewer-note "First local run; OHLCV-only calibration; no spread, order-book, or fills artifact was present." --created-at 2026-05-10T12:05:00+00:00
.\.venv\Scripts\python.exe scripts\bot_factory_calibrate_cost_model.py --ohlcv-path user_data\data\bybit\futures\BTC_USDT_USDT-5m-futures.parquet --pair BTC/USDT:USDT --timeframe 5m --order-type taker --liquidity-tier liquid --volatility-regime stress --cost-calibration-id first_run_btc_usdt_5m_taker_stress --output-root data\cost_calibration --reviewer-note "First local run; OHLCV-only calibration; no spread, order-book, or fills artifact was present." --created-at 2026-05-10T12:05:10+00:00
.\.venv\Scripts\python.exe scripts\bot_factory_calibrate_cost_model.py --ohlcv-path user_data\data\bybit\futures\ETH_USDT_USDT-5m-futures.parquet --pair ETH/USDT:USDT --timeframe 5m --order-type taker --liquidity-tier liquid --volatility-regime normal --cost-calibration-id first_run_eth_usdt_5m_taker_normal --output-root data\cost_calibration --reviewer-note "First local run; OHLCV-only calibration; no spread, order-book, or fills artifact was present." --created-at 2026-05-10T12:05:20+00:00
.\.venv\Scripts\python.exe scripts\bot_factory_calibrate_cost_model.py --ohlcv-path user_data\data\bybit\futures\ETH_USDT_USDT-5m-futures.parquet --pair ETH/USDT:USDT --timeframe 5m --order-type taker --liquidity-tier liquid --volatility-regime stress --cost-calibration-id first_run_eth_usdt_5m_taker_stress --output-root data\cost_calibration --reviewer-note "First local run; OHLCV-only calibration; no spread, order-book, or fills artifact was present." --created-at 2026-05-10T12:05:30+00:00
```

Result: all four taker commands completed with `blocker_count=0`.

Blocked maker runs:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_calibrate_cost_model.py --ohlcv-path user_data\data\bybit\futures\BTC_USDT_USDT-5m-futures.parquet --pair BTC/USDT:USDT --timeframe 5m --order-type maker --liquidity-tier liquid --volatility-regime normal --cost-calibration-id first_run_btc_usdt_5m_maker_normal --output-root data\cost_calibration --reviewer-note "First local run; OHLCV-only calibration; maker context intentionally blocked because no order-book/depth or fills artifact was present." --created-at 2026-05-10T12:05:40+00:00
.\.venv\Scripts\python.exe scripts\bot_factory_calibrate_cost_model.py --ohlcv-path user_data\data\bybit\futures\BTC_USDT_USDT-5m-futures.parquet --pair BTC/USDT:USDT --timeframe 5m --order-type maker --liquidity-tier liquid --volatility-regime stress --cost-calibration-id first_run_btc_usdt_5m_maker_stress --output-root data\cost_calibration --reviewer-note "First local run; OHLCV-only calibration; maker context intentionally blocked because no order-book/depth or fills artifact was present." --created-at 2026-05-10T12:05:50+00:00
.\.venv\Scripts\python.exe scripts\bot_factory_calibrate_cost_model.py --ohlcv-path user_data\data\bybit\futures\ETH_USDT_USDT-5m-futures.parquet --pair ETH/USDT:USDT --timeframe 5m --order-type maker --liquidity-tier liquid --volatility-regime normal --cost-calibration-id first_run_eth_usdt_5m_maker_normal --output-root data\cost_calibration --reviewer-note "First local run; OHLCV-only calibration; maker context intentionally blocked because no order-book/depth or fills artifact was present." --created-at 2026-05-10T12:06:00+00:00
.\.venv\Scripts\python.exe scripts\bot_factory_calibrate_cost_model.py --ohlcv-path user_data\data\bybit\futures\ETH_USDT_USDT-5m-futures.parquet --pair ETH/USDT:USDT --timeframe 5m --order-type maker --liquidity-tier liquid --volatility-regime stress --cost-calibration-id first_run_eth_usdt_5m_maker_stress --output-root data\cost_calibration --reviewer-note "First local run; OHLCV-only calibration; maker context intentionally blocked because no order-book/depth or fills artifact was present." --created-at 2026-05-10T12:06:10+00:00
```

Result: all four maker commands wrote structured blocked artifacts with
`blocker_count=7`.

## Completed Results

| cost_calibration_id | normal total bps | stress total bps | spread source | result |
| --- | ---: | ---: | --- | --- |
| `first_run_btc_usdt_5m_taker_normal` | 9.469967 | 16.900396 | `ohlcv_range_proxy` | completed |
| `first_run_btc_usdt_5m_taker_stress` | 9.469967 | 16.900396 | `ohlcv_range_proxy` | completed |
| `first_run_eth_usdt_5m_taker_normal` | 10.856899 | 20.874029 | `ohlcv_range_proxy` | completed |
| `first_run_eth_usdt_5m_taker_stress` | 10.856899 | 20.874029 | `ohlcv_range_proxy` | completed |

No completed context has `stress` cost below `normal` cost.

## Blocked Results

Maker runs blocked because maker context requires fill-risk fields for every
scenario and no local order-book/depth or fills evidence was present.

Blocker names for each maker run:

- `best_cost_missing`
- `normal_cost_missing`
- `stress_cost_missing`
- `maker_no_fill_rate_missing`
- `maker_partial_fill_rate_missing`
- `maker_adverse_selection_bps_missing`
- `maker_exit_taker_rate_missing`

This satisfies the maker execution-quality requirement as a blocker: no-fill,
partial-fill, adverse-selection, and exit-taker conversion were not silently
assumed to be zero.

## Generated Artifacts

Generated artifacts were written locally under `data\cost_calibration`:

- `first_run_btc_usdt_5m_taker_normal`
- `first_run_btc_usdt_5m_taker_stress`
- `first_run_eth_usdt_5m_taker_normal`
- `first_run_eth_usdt_5m_taker_stress`
- `first_run_btc_usdt_5m_maker_normal`
- `first_run_btc_usdt_5m_maker_stress`
- `first_run_eth_usdt_5m_maker_normal`
- `first_run_eth_usdt_5m_maker_stress`

These generated artifacts are local evidence for this run and must not be added
to Git. `.gitignore` now excludes `data/cost_calibration/**` while allowing a
future `.gitkeep` placeholder if the directory needs one.

## Edge Discovery Context

Use for the next Edge Discovery:

- BTC/USDT:USDT 5m taker, liquid context.
- ETH/USDT:USDT 5m taker, liquid context.
- Scenario-level `best`, `normal`, and `stress` costs from
  `docs/BOT_FACTORY_CALIBRATED_COST_TABLE.md`.
- `normal` as the primary research gate cost and `stress` as the robustness
  gate cost.

Do not use for the next Edge Discovery:

- maker or mixed maker/taker assumptions;
- BTC/USDT:USDT or ETH/USDT:USDT 1h;
- large-alt contexts;
- spread/order-book/fills-driven contexts;
- paper, dry-run, live, or exchange-facing execution.

The stress-volatility runs are labels over the same available 5m OHLCV input;
no separate high-volatility-only dataset was present. The usable stress context
is therefore the runner's `stress` scenario, not a distinct stress-only market
artifact.

## Verification

```powershell
.\.venv\Scripts\python.exe -m py_compile freqtrade_ext\bot_factory\cost_calibration.py scripts\bot_factory_calibrate_cost_model.py tests\test_bot_factory.py
.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q -k "cost_calibration or execution_quality or cost_model"
.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q
git diff --check
git check-ignore -v data\cost_calibration\first_run_btc_usdt_5m_taker_normal\cost_calibration.json data\cost_calibration\first_run_eth_usdt_5m_taker_normal\cost_calibration.json
```

Results: compile passed; focused selector passed 26 tests and reached `[100%]`;
full `tests\test_bot_factory.py` reached `[100%]`; `git diff --check` exited
`0` with existing LF-to-CRLF working-copy warnings for `.gitignore` and
`docs/BOT_FACTORY_MVP_TODO.md`; `git check-ignore -v` confirmed generated cost
calibration JSON artifacts are ignored by `data/cost_calibration/**`.
