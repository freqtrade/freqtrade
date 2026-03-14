# Known Issues & Gotchas

> Living document of bugs, limitations, and design constraints discovered during benchmark v2 and production optimization.
> Updated: 2025-07-20

---

## Active Issues

### 1. Short Selling — Strategy Generator Cannot Produce Valid Short Signals
- **Severity**: Critical (feature broken)
- **Discovered**: Benchmark v2 R7
- **Symptom**: 0 trades across all pairs over 3 years of data
- **Root cause**: The strategy code generator (`FreqtradeStrategy` / `generate_strategy_code()`) does not emit valid `short_entry` / `short_exit` conditions. The GA generates indicator parameters but the code template doesn't wire them into short signal columns.
- **Impact**: `short_selling: enabled: true` is a no-op. R7 ran for ~40 min producing zero strategies.
- **Fix effort**: Medium — need to extend strategy template to map indicators → short signals (mirror of long signals with inverted thresholds).
- **Workaround**: Don't enable short selling until code gen is fixed.

### 2. NSGA-II — No Minimum Trade Count in Pareto Objectives
- **Severity**: High
- **Discovered**: Benchmark v2 R6
- **Symptom**: Top Pareto front dominated by 3-trade strategies with perfect metrics (100% win rate, 0% drawdown) but meaningless statistical significance. All 3 top strategies rated OVERFIT.
- **Root cause**: `mode: 'nsga2'` uses multi-objective Pareto ranking (profit, risk, robustness) without a hard floor on trade count. A strategy with 3 lucky trades easily dominates on Sharpe/Sortino/win_rate.
- **Impact**: NSGA-II mode produces degenerate solutions.
- **Fix options**:
  - Add `min_trades` as a constraint (not objective) that eliminates candidates below threshold
  - Add trade_count as an explicit Pareto objective (maximize)
  - Apply a steep penalty multiplier when trades < threshold (e.g., fitness *= trades/min_trades)
- **Workaround**: Use `mode: 'single_objective'` with `fitness_penalties.min_trades: 15+`

### 3. Island Model Specialists Overfit Narrow Regime Data
- **Severity**: Medium-High
- **Discovered**: Benchmark v2 R4v2 (post regime fix)
- **Symptom**: After regime detection fixes, segments are now balanced (16 bearish, 15 bullish, 3 sideways) but specialist islands still overfit. Sideways island: 0.6133 fitness → 77% holdout degradation. All 5 top strategies OVERFIT or UNKNOWN (0/5 SAFE).
- **Root cause**: Even with correct segmentation, each specialist island trains on only ~30% of the data. 3-year dataset split into 34 segments means each regime gets ~1 year of non-contiguous data. Small population (25) on limited data = overfitting.
- **Potential fixes**:
  - Increase population size substantially for island runs (pop=60+)
  - Use longer timerange (5+ years) so each regime gets more data
  - Reduce number of islands (just bull+bear, skip sideways)
  - Add stronger regularization (higher complexity_weight, lower max_indicators)
  - Master island should aggregate with more weight on cross-regime generalization
- **Status**: Not yet re-tested with larger population

### 4. Island Model Incompatible with Walk-Forward
- **Severity**: Design constraint (not a bug)
- **Location**: `island_model.py` ~L1264 — walk-forward is force-disabled
- **Reason**: Island model splits data by regime segments, which conflicts with walk-forward's temporal window splitting. The two approaches impose contradictory data partitioning.
- **Impact**: Cannot combine island model + walk-forward. Must choose one or the other.

### 5. Island Model Incompatible with Monte Carlo / CPCV
- **Severity**: Design constraint
- **Location**: Code prints "not supported yet" warnings
- **Impact**: `monte_carlo.enabled: true` and `cpcv.enabled: true` are silently ignored when island model is active.

### 6. Multi-Timeframe (MTF) — Untested at Scale
- **Severity**: Unknown risk
- **Note**: `multi_timeframe.enabled: true` was never tested in benchmark v2. Could multiply runtime significantly (each strategy backtested on multiple timeframes). Interaction with walk-forward and holdout unknown.
- **Recommendation**: Test MTF in isolation first with small pop/gen before combining with other features.

---

## Fixed Issues (Reference)

### F1. Regime Detection — Bearish-Only Segmentation (FIXED 2025-07-20)
- **Root cause**: 4 interacting bugs
  1. `_merge_short_segments()` never recomputed `avg_trend_score` during merges → stale score from first segment determined mega-segment regime
  2. SMA params (period=20, slope_window=5) designed for daily candles but applied to 1h data → noisy microsegments
  3. Dead zone: `merge_threshold_days=7` but `min_segment_days=max(14, ...)` → killed 7-14 day segments
  4. Asymmetric band defaults: bull≥0.44 vs bear≤-0.37
- **Fix**: Weighted-average recomputation on merge, auto-scaling SMA params by bar interval, aligned min_segment_days with merge_threshold, symmetric ±0.40 bands
- **Files changed**: `regime_detector.py`, `island_model.py`
- **Tests**: 8 regression tests added to `test_regime_detector.py`
- **Verified**: BTC 1h 3y → 69 segments (31 bull, 34 bear, 4 sideways). R4v2 startup: 34 segments balanced.

### F2. Population Constructor Kwarg (FIXED 2025-07-19)
- **Root cause**: `Population(max_size=self.population_size)` but constructor expects `size=`
- **Fix**: Changed to `Population(size=self.population_size)` in `evolution.py` L2481
- **Impact**: R1, R4 were failing silently with default population size

### F3. SMA Slope Dispatch Missing (FIXED 2025-07-19)
- **Root cause**: `_detect_regimes()` had no dispatch branch for `method='sma_slope'` in the discrete path
- **Fix**: Added `elif method == 'sma_slope': return self._detect_sma_slope(df)` in `regime_detector.py`
- **Impact**: SMA slope detection fell through to default method silently

---

## Configuration Gotchas

### G1. `fitness_sharing` should be OFF for NSGA-II
- NSGA-II has its own crowding distance mechanism. Adding fitness sharing on top distorts the Pareto front.
- R6 had both enabled — likely contributed to degenerate results.

### G2. `auto_download_data: false` requires pre-downloaded data
- All benchmark configs use `auto_download_data: false`. Data must exist in `user_data/data/binance/` for all pairs × timeframes.
- Missing data → silent 0-trade backtests (no error, just empty results).

### G3. Walk-forward `max_windows` silently caps
- If `train_days + validation_days` × `max_windows` > timerange, fewer windows are created silently.
- With 3y data and train=90/val=30/step=30: theoretical max ≈ 30 windows, but `max_windows: 10` caps it.

### G4. `timeout` in backtesting section is per-backtest
- A stuck backtest (infinite loop in indicator calc) will block for `timeout` seconds before being killed.
- Low timeout (60s) can kill legitimate slow strategies on 3y data.
- R5 used 180s which was adequate for 4-pair 3y runs.

### G5. XRP/USDT Has Less Historical Data
- Binance XRP/USDT 5m data may not extend back to 20230301. Verify data availability before adding to 5m runs.
- 15m and 1h data confirmed available for all 5 pairs across full timerange.
