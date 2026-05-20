# Historical Uptrend Selector Replay

- Run id: `historical_uptrend_20240202_20240304`
- Window: `2024-02-02T00:00:00Z` to `2024-03-04T23:55:00Z`
- Selector action: `select`
- Selected candidate: `strong-uptrend-historical-ohlcv-candidate`
- Selected logic: `strong_uptrend_momentum_v1`
- Scorecard decision: `REGIME_SCOPED_SELECTOR_ELIGIBLE`

## Pair Summary

| pair | return % | stress return % | max DD % | daily win rate | rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| BTC/USDT:USDT | 58.8072 | 58.6072 | 6.4605 | 0.6774 | 9216 |
| ETH/USDT:USDT | 57.7765 | 57.5765 | 6.8303 | 0.6774 | 9216 |

## Evaluated Candidates

| candidate | logic | selectable | reasons |
| --- | --- | --- | --- |
| strong-uptrend-historical-ohlcv-candidate | strong_uptrend_momentum_v1 | `True` | runtime_selection_passed |
| downtrend_defensive_rebound_v1-historical-companion | downtrend_defensive_rebound_v1 | `False` | runtime_regime_eligible, runtime_regime_not_blocked, scorecard_decision_selector_eligible |
| range_mean_reversion_v1-historical-companion | range_mean_reversion_v1 | `False` | runtime_regime_eligible, runtime_regime_not_blocked, scorecard_decision_selector_eligible |

## Limitations

- This is a close-to-close historical OHLCV proxy, not a generated Freqtrade strategy backtest.
- The replay relaxes calendar concentration to 1.0 because it intentionally tests one past uptrend window.
- The result tests selector adoption behavior, not profitability approval or paper/live readiness.

## Safety

No paper, dry-run, live, exchange order endpoint, API key, secret, leverage, shorting, or process-control action was performed.
