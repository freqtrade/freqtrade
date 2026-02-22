# Market Regime-Aware Dataset Selection for GA Strategy Evolution

## Problem Statement

When a Genetic Algorithm (GA) optimizes trading strategies on a dataset that represents only **one market regime** (e.g., only bullish), the evolved strategies will perform poorly when the live market enters a different regime (bearish or sideways). This is a fundamental **distribution shift** problem — the training data does not represent the full distribution of market conditions.

This document collects, ranks, and proposes implementation plans for solving this problem in the context of a GA-based strategy evolution system built on top of Freqtrade.

---

## Table of Contents

1. [Concept Overview](#concept-overview)
2. [Approach 1: Regime-Balanced Period Selection](#approach-1-regime-balanced-period-selection)
3. [Approach 2: Island Model with Regime Specialization](#approach-2-island-model-with-regime-specialization)
4. [Approach 3: Multi-Regime Fitness Aggregation](#approach-3-multi-regime-fitness-aggregation)
5. [Approach 4: Regime-Aware Walk-Forward Validation](#approach-4-regime-aware-walk-forward-validation)
6. [Approach 5: Ensemble / Meta-Strategy Switching](#approach-5-ensemble--meta-strategy-switching)
7. [Approach 6: Adversarial Regime Training](#approach-6-adversarial-regime-training)
8. [Approach 7: Synthetic Regime Augmentation](#approach-7-synthetic-regime-augmentation)
9. [Approach Comparison & Ranking](#approach-comparison--ranking)
10. [Combined Implementation Plan](#combined-implementation-plan)
11. [Regime Detection Methods](#appendix-a-regime-detection-methods)
12. [Practical Considerations](#appendix-b-practical-considerations)

---

## Concept Overview

The core insight is that **financial markets cycle through distinct regimes**:

| Regime | Characteristics | Strategy Needs |
|--------|----------------|----------------|
| **Bullish (Uptrend)** | Rising prices, higher highs/lows | Trend-following, buy-the-dip |
| **Bearish (Downtrend)** | Falling prices, lower highs/lows | Short-selling, defensive exits |
| **Sideways (Range)** | Mean-reverting, no clear trend | Mean-reversion, tight stops |
| **High Volatility** | Large price swings, regime transitions | Wide stops, reduced position size |
| **Low Volatility** | Compressed ranges, breakout potential | Breakout strategies, patience |

A robust GA strategy must either:
- **A)** Perform well across ALL regimes (generalist)
- **B)** Specialize in one regime with a meta-system that switches strategies (specialist ensemble)

Both paths require **regime-aware dataset selection**.

---

## Approach 1: Regime-Balanced Period Selection

### Concept
Before evolution begins, **automatically classify historical data into regime periods**, then select a balanced training set containing equal representation from each regime type.

### How It Works
1. **Detect regimes** across the full historical dataset (see [Appendix A](#appendix-a-regime-detection-methods))
2. **Label each period** (e.g., 2-week blocks) as bullish, bearish, or sideways
3. **Select N periods from each regime** (e.g., 5 bullish + 5 bearish + 5 sideways = 15 periods)
4. **Backtest each candidate strategy** across ALL selected periods
5. **Aggregate fitness** across all periods (mean, harmonic mean, or worst-case)

### Example Configuration
```yaml
regime_balanced_selection:
  enabled: true
  regime_detector: 'sma_crossover'     # Detection method
  num_periods_per_regime: 5             # 5 from each type
  period_length_days: 14                # 2-week blocks
  regimes: ['bullish', 'bearish', 'sideways']
  fitness_aggregation: 'harmonic_mean'  # Penalizes inconsistency
```

### Advantages
- Simple to implement on top of existing backtester
- Ensures training data covers all market conditions
- No structural changes needed to the GA engine
- Works with current single-population model

### Disadvantages
- Increases backtesting time proportionally (3x for 3 regimes)
- Regime detection adds a preprocessing step
- Period boundaries can be noisy (transitions between regimes)

### Implementation Effort: ⭐⭐ (Medium)
### Impact on Live Performance: ⭐⭐⭐⭐ (High)

---

## Approach 2: Island Model with Regime Specialization

### Concept
Run **multiple isolated sub-populations (islands)**, each evolving on a **different market regime**. Periodically **migrate** the best individuals between islands to share genetic material.

### How It Works
1. **Create N islands** (e.g., 3 islands for bullish/bearish/sideways)
2. Each island **evolves independently** on its regime-specific data
3. Every K generations, **migrate top M individuals** between islands
4. At the end, each island produces **regime-specialist strategies**
5. Optionally, create a **master island** that evaluates migrants from all islands on balanced data

### Architecture
```
┌─────────────────────────────────────────────────────┐
│                    Master Island                     │
│          (balanced data, selects generalists)        │
│                                                      │
│    ┌──────────┐   ┌──────────┐   ┌──────────┐      │
│    │  migrate  │   │  migrate  │   │  migrate  │      │
│    └────▲─────┘   └────▲─────┘   └────▲─────┘      │
│         │              │              │              │
├─────────┼──────────────┼──────────────┼──────────────┤
│         │              │              │              │
│  ┌──────┴──────┐ ┌─────┴──────┐ ┌────┴───────┐     │
│  │  Island 1   │ │  Island 2   │ │  Island 3   │     │
│  │  Bullish    │ │  Bearish    │ │  Sideways   │     │
│  │  Data       │ │  Data       │ │  Data       │     │
│  │  pop=30     │ │  pop=30     │ │  pop=30     │     │
│  └─────────────┘ └─────────────┘ └─────────────┘     │
│                                                      │
│  Migration every 5 generations, top 3 individuals    │
└─────────────────────────────────────────────────────┘
```

### Example Configuration
```yaml
island_model:
  enabled: true
  islands:
    - name: 'bullish'
      population_size: 30
      data_regime: 'bullish'
    - name: 'bearish'
      population_size: 30
      data_regime: 'bearish'
    - name: 'sideways'
      population_size: 30
      data_regime: 'sideways'
    - name: 'master'
      population_size: 30
      data_regime: 'balanced'    # All regimes combined
  migration:
    frequency: 5                 # Every 5 generations
    count: 3                     # Top 3 individuals migrate
    topology: 'ring'             # ring | fully_connected | star
```

### Advantages
- Produces **specialist strategies** for each regime
- Migration enables **cross-pollination** of good genetic material
- Naturally parallelizable (each island runs independently)
- Master island can discover **generalist strategies** from specialist genes

### Disadvantages
- Significantly more complex implementation
- Requires N× population memory
- Need a live regime detection system to choose which strategy to deploy
- Migration policy tuning adds complexity

### Implementation Effort: ⭐⭐⭐⭐ (High)
### Impact on Live Performance: ⭐⭐⭐⭐⭐ (Very High — if combined with live regime switching)

---

## Approach 3: Multi-Regime Fitness Aggregation

### Concept
Keep the **single-population GA** but modify the **fitness function** to evaluate each strategy across **multiple regime-specific periods** and aggregate the scores in a way that penalizes regime-specific weakness.

### How It Works
1. **Detect regime periods** in the historical data
2. During fitness evaluation, **backtest on each regime** separately
3. Compute **regime-specific fitness scores** (profit, Sharpe, drawdown per regime)
4. **Aggregate** using a method that rewards consistency:
   - **Harmonic Mean**: Heavily penalizes poor performance in any regime
   - **Minimum Score**: Only as good as the worst regime (most conservative)
   - **Weighted Mean**: Assign importance weights to each regime
   - **CVaR-style**: Average of the bottom 30% of regime scores

### Fitness Aggregation Methods

```
Harmonic Mean:  F = n / (1/f₁ + 1/f₂ + ... + 1/fₙ)
   → Penalizes ANY weak regime heavily

Minimum:        F = min(f₁, f₂, ..., fₙ)
   → Ultra-conservative, evolves for worst-case

Weighted Mean:  F = w₁·f₁ + w₂·f₂ + ... + wₙ·fₙ
   → Allows prioritizing certain regimes

CVaR-style:     F = mean(bottom 30% of regime scores)
   → Focuses on avoiding catastrophic failure
```

### Example Configuration
```yaml
multi_regime_fitness:
  enabled: true
  regimes:
    bullish:
      weight: 0.33
      periods: 5
    bearish:
      weight: 0.33
      periods: 5
    sideways:
      weight: 0.34
      periods: 5
  aggregation: 'harmonic_mean'
  min_acceptable_score: 0.0      # Strategy must be profitable in all regimes
```

### Advantages
- **No structural changes** to the GA engine needed
- Directly evolves **generalist strategies**
- Simple to tune via aggregation method and weights
- Integrates naturally with existing fitness calculation

### Disadvantages
- 3× backtest time per evaluation (one per regime)
- Harder to evolve highly specialized strategies
- May lead to mediocre performance in all regimes (jack of all trades)

### Implementation Effort: ⭐⭐ (Medium)
### Impact on Live Performance: ⭐⭐⭐⭐ (High)

---

## Approach 4: Regime-Aware Walk-Forward Validation

### Concept
Enhance the existing walk-forward validation to ensure each **validation window** covers a **different market regime**, and adjust scoring to penalize strategies that only work in one regime.

### How It Works
1. Use existing walk-forward infrastructure
2. After generating windows, **classify each window's dominant regime**
3. **Ensure regime balance** by:
   - Adding/removing windows until all regimes are represented
   - Weighting window scores by regime representation
4. Report **per-regime performance** alongside aggregate fitness

### Enhancements to Current Walk-Forward
```
Current Walk-Forward:
  Window 1 (Jan-Feb train, Mar validate) → score 1
  Window 2 (Feb-Mar train, Apr validate) → score 2
  Window 3 (Mar-Apr train, May validate) → score 3
  Fitness = mean(score 1, score 2, score 3)

Enhanced Regime-Aware Walk-Forward:
  Window 1 (Bullish train period, Bearish validate) → score 1  [bearish]
  Window 2 (Bearish train period, Sideways validate) → score 2  [sideways]
  Window 3 (Sideways train period, Bullish validate) → score 3  [bullish]
  Fitness = harmonic_mean(score 1, score 2, score 3)  ← all regimes covered
```

### Advantages
- Builds on existing walk-forward infrastructure (minimal new code)
- Adds regime awareness without major refactoring
- Validation windows naturally test out-of-sample regime performance
- Compatible with all existing GA configurations

### Disadvantages
- Limited by available historical data (may not have enough of each regime)
- Window boundaries may not align with regime boundaries
- Still single-population approach

### Implementation Effort: ⭐ (Low)
### Impact on Live Performance: ⭐⭐⭐ (Medium-High)

---

## Approach 5: Ensemble / Meta-Strategy Switching

### Concept
Instead of finding **one strategy for all regimes**, evolve **multiple specialist strategies** and a **meta-strategy** that detects the current regime and switches to the appropriate specialist.

### How It Works
1. **Phase 1**: Evolve specialist strategies for each regime (using Island Model or separate GA runs)
2. **Phase 2**: Build a **regime classifier** that runs in real-time
3. **Phase 3**: Deploy an **ensemble strategy** that:
   - Monitors the current regime in real-time
   - Activates the specialist strategy matching the current regime
   - Manages transitions between specialists (position cleanup, etc.)

### Architecture
```
Live Market Data
      │
      ▼
┌─────────────────┐
│ Regime Detector  │  ← SMA crossover / ADX / HMM
│ Current: BEARISH │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Strategy Router │
│                  │
│  if BULLISH  → Strategy A (evolved on bullish data)
│  if BEARISH  → Strategy B (evolved on bearish data)
│  if SIDEWAYS → Strategy C (evolved on sideways data)
│  if UNCERTAIN→ Cash / Reduced position
└─────────────────┘
```

### Example Configuration
```yaml
ensemble_strategy:
  enabled: true
  regime_detector:
    method: 'adx_sma'
    params:
      adx_period: 14
      adx_trend_threshold: 25
      sma_fast: 50
      sma_slow: 200
      confirmation_bars: 3
  specialists:
    bullish: 'evolved_strategy_bullish_v1'
    bearish: 'evolved_strategy_bearish_v1'
    sideways: 'evolved_strategy_sideways_v1'
  transition:
    close_positions_on_switch: true
    cooldown_bars: 5           # Wait 5 bars after switch before trading
```

### Advantages
- **Best of both worlds**: Specialists outperform generalists in their regime
- Adapts dynamically to live market conditions
- Each specialist is simpler and easier to evolve
- Can evolve specialists in parallel (fast)

### Disadvantages
- Most complex overall system
- Regime detection errors cause wrong strategy activation
- Transition management adds complexity (open positions during switch)
- Need to maintain and update multiple strategies

### Implementation Effort: ⭐⭐⭐⭐⭐ (Very High)
### Impact on Live Performance: ⭐⭐⭐⭐⭐ (Very High — highest potential if regime detection is accurate)

---

## Approach 6: Adversarial Regime Training

### Concept
Inspired by adversarial training in machine learning: deliberately **evolve strategies against the hardest regime periods** to maximize robustness. The system dynamically selects the **worst-performing regime** and over-samples it in subsequent generations.

### How It Works
1. Start with balanced regime representation
2. After each generation, **identify which regime has the lowest average fitness**
3. In the next generation, **increase the weight** of the worst regime in fitness calculation
4. This creates **evolutionary pressure** to improve on weak regimes
5. Gradually converge to balanced performance

### Adaptive Weighting
```
Generation 1: weights = [0.33, 0.33, 0.34] (bull, bear, side)
              → avg fitness: [0.8, 0.3, 0.5]
              → bearish is worst

Generation 2: weights = [0.25, 0.50, 0.25] (boost bearish)
              → avg fitness: [0.7, 0.5, 0.5]
              → bearish improving

Generation 3: weights = [0.30, 0.40, 0.30] (rebalance)
              ...converges to balanced performance
```

### Advantages
- Automatically focuses evolution on weaknesses
- No need to pre-decide regime importance
- Creates maximally robust strategies
- Simple to implement on top of multi-regime fitness

### Disadvantages
- Can oscillate between regimes (one gets better, another gets worse)
- Needs careful damping/smoothing of weight updates
- Longer convergence time
- May not find specialized strategies

### Implementation Effort: ⭐⭐⭐ (Medium-High)
### Impact on Live Performance: ⭐⭐⭐⭐ (High)

---

## Approach 7: Synthetic Regime Augmentation

### Concept
When historical data doesn't contain enough examples of a certain regime, **generate synthetic data** that mimics that regime's characteristics, or **transform existing data** to simulate different regimes.

### How It Works
1. **Analyze regime statistics** from historical data (volatility, trend strength, mean return)
2. **Generate synthetic price series** matching each regime's statistical profile
3. **Augment training set** with synthetic data for under-represented regimes
4. Alternatively, **time-stretch or compress** existing data to create variations

### Synthetic Generation Methods
- **Bootstrap resampling**: Resample returns from regime-specific blocks
- **Monte Carlo simulation**: Generate paths with regime-specific drift and volatility
- **GAN-based generation**: Train a generative model on regime-specific data (advanced)
- **Regime transformation**: Flip bullish data to create synthetic bearish data (reverse returns)

### Advantages
- Unlimited training data for any regime
- Fills gaps when real historical data is insufficient
- Can create extreme scenarios for stress testing

### Disadvantages
- Synthetic data may not capture real market microstructure
- Risk of overfitting to synthetic patterns that don't exist in reality
- Complex to validate quality of synthetic data
- Adding another layer of complexity

### Implementation Effort: ⭐⭐⭐⭐ (High)
### Impact on Live Performance: ⭐⭐ (Medium — high risk of synthetic artifacts)

---

## Approach Comparison & Ranking

### Overall Ranking (Best Balance of Impact vs. Effort)

| Rank | Approach | Effort | Impact | ROI | Best For |
|------|----------|--------|--------|-----|----------|
| **1** | **Multi-Regime Fitness Aggregation** | ⭐⭐ | ⭐⭐⭐⭐ | 🏆 Highest | Immediate improvement, least disruption |
| **2** | **Regime-Balanced Period Selection** | ⭐⭐ | ⭐⭐⭐⭐ | 🥈 Very High | Smart data preprocessing |
| **3** | **Regime-Aware Walk-Forward** | ⭐ | ⭐⭐⭐ | 🥉 High | Quick enhancement to existing system |
| **4** | **Adversarial Regime Training** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Good | Automated robustness improvement |
| **5** | **Island Model + Regime Specialization** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Good | Research, maximum flexibility |
| **6** | **Ensemble / Meta-Strategy Switching** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Moderate | Production system, live trading |
| **7** | **Synthetic Regime Augmentation** | ⭐⭐⭐⭐ | ⭐⭐ | Low | Only when data is scarce |

### Combination Synergies

These approaches **combine naturally** in layers:

```
Layer 0 (Foundation):  Regime Detection Algorithm
                       └── Required by ALL approaches

Layer 1 (Quick Win):   Regime-Balanced Period Selection (Approach 1)
                       + Multi-Regime Fitness Aggregation (Approach 3)
                       └── Immediate improvement with minimal code changes

Layer 2 (Enhancement): Regime-Aware Walk-Forward (Approach 4)
                       + Adversarial Regime Training (Approach 6)
                       └── Smarter evolution dynamics

Layer 3 (Advanced):    Island Model (Approach 2)
                       + Ensemble Switching (Approach 5)
                       └── Full specialist ecosystem with live deployment

Layer 4 (Optional):    Synthetic Augmentation (Approach 7)
                       └── Only if historical data is insufficient
```

---

## Combined Implementation Plan

### Phase 1: Regime Detection Foundation (Week 1-2)
**Goal**: Build the regime classification engine that all other approaches depend on.

**Tasks**:
- [ ] Implement `RegimeDetector` class in `genetic_algorithm/utils/regime_detector.py`
- [ ] Support multiple detection methods (SMA crossover, ADX-based, volatility-based)
- [ ] Add `classify_periods()` method that labels time periods by regime
- [ ] Add unit tests for regime detection accuracy
- [ ] Integrate with existing data loading in `direct_backtester.py`

**Recommended Detection Method** (for initial implementation):
```python
# Note: SMA() and ADX() are placeholders for technical indicator functions
# from libraries like ta-lib or pandas-ta that would need to be imported.

def detect_regime(close_prices, sma_fast=50, sma_slow=200, adx_period=14, adx_threshold=25):
    """
    Classify market regime using SMA crossover + ADX trend strength.

    Args:
        close_prices: pandas Series of closing prices
        sma_fast: Fast SMA period (default 50)
        sma_slow: Slow SMA period (default 200)
        adx_period: ADX calculation period (default 14)
        adx_threshold: ADX threshold to distinguish trending vs. sideways (default 25)

    Returns: 'bullish' | 'bearish' | 'sideways'
    """
    sma_fast_val = SMA(close_prices, sma_fast)
    sma_slow_val = SMA(close_prices, sma_slow)
    adx_val = ADX(close_prices, adx_period)

    if adx_val < adx_threshold:
        return 'sideways'
    elif sma_fast_val > sma_slow_val:
        return 'bullish'
    else:
        return 'bearish'
```

### Phase 2: Regime-Balanced Selection + Multi-Regime Fitness (Week 3-4)
**Goal**: Ensure every strategy is evaluated across all market regimes.

**Tasks**:
- [ ] Add `RegimeBalancedSelector` that picks N periods per regime from historical data
- [ ] Modify `fitness.py` to evaluate on multiple regime periods
- [ ] Add harmonic mean / min / CVaR aggregation options
- [ ] Add `multi_regime_fitness` section to `ga_config.yaml`
- [ ] Add per-regime fitness reporting to evolution output
- [ ] Run comparative experiments: standard vs. regime-balanced fitness

### Phase 3: Enhanced Walk-Forward with Regime Awareness (Week 5-6)
**Goal**: Ensure walk-forward validation windows cover all regimes.

**Tasks**:
- [ ] Modify `timerange.py` to accept regime constraints
- [ ] Add regime classification to each walk-forward window
- [ ] Implement regime-balanced window selection (ensure all regimes in validation)
- [ ] Add regime coverage metrics to walk-forward reporting
- [ ] Add adversarial weight adjustment option (boost worst regime)

### Phase 4: Island Model (Week 7-10)
**Goal**: Evolve specialist strategies for each regime in parallel.

**Tasks**:
- [ ] Create `IslandPopulation` class (wraps multiple `Population` instances)
- [ ] Implement migration operators (ring, fully-connected, star topologies)
- [ ] Add per-island regime-specific data assignment
- [ ] Add master island for generalist evaluation
- [ ] Add migration scheduling and rate configuration
- [ ] Parallelize island evolution (one process per island)
- [ ] Add island model configuration to `ga_config.yaml`

### Phase 5: Ensemble Strategy Deployment (Week 11-14)
**Goal**: Deploy regime-switching ensemble for live trading.

**Tasks**:
- [ ] Create `RegimeEnsembleStrategy` Freqtrade strategy class
- [ ] Implement real-time regime detection in the strategy
- [ ] Add strategy switching logic with position management
- [ ] Add transition cooldown and safety mechanisms
- [ ] Backtest ensemble vs. individual strategies vs. generalist
- [ ] Add monitoring and logging for regime switches

---

## Appendix A: Regime Detection Methods

### Method 1: SMA Crossover (Simple, Recommended for Start)
```
Bullish:  SMA(50) > SMA(200) AND ADX > 25
Bearish:  SMA(50) < SMA(200) AND ADX > 25
Sideways: ADX < 25
```
- **Pros**: Simple, well-understood, robust
- **Cons**: Lagging (moving averages are slow), binary transitions

### Method 2: ADX + Directional Movement
```
Strong Trend:  ADX > 25
  → Bullish if +DI > -DI
  → Bearish if -DI > +DI
Sideways:      ADX < 20
Transition:    20 < ADX < 25 (uncertain)
```
- **Pros**: More nuanced, includes trend strength
- **Cons**: ADX itself is lagging

### Method 3: Rolling Return Distribution
```
Calculate rolling N-day returns:
  Bullish:  mean(returns) > +threshold AND std(returns) < vol_cap
  Bearish:  mean(returns) < -threshold AND std(returns) < vol_cap
  Sideways: |mean(returns)| < threshold
  Volatile: std(returns) > vol_cap
```
- **Pros**: Statistical, no indicator dependency, captures volatility
- **Cons**: Sensitive to lookback period and thresholds

### Method 4: Hidden Markov Model (HMM)
```
Train HMM with N states on return series:
  - Each state represents a regime
  - Transition probabilities capture regime persistence
  - Viterbi algorithm finds most likely state sequence
```
- **Pros**: Statistically rigorous, captures complex regime dynamics
- **Cons**: Complex to implement, requires model fitting, can overfit

### Method 5: Bollinger Band Position
```
Price relative to Bollinger Bands (20, 2):
  Bullish:  Price consistently above middle band
  Bearish:  Price consistently below middle band
  Sideways: Price oscillates around middle band
  Volatile: Band width expanding
```
- **Pros**: Adapts to volatility, visual intuition
- **Cons**: Needs "consistently" definition (lookback)

### Recommended Combination
For maximum robustness, **combine 2-3 methods with a voting system**:
```python
def detect_regime_ensemble(close_prices):
    vote_sma = sma_crossover_regime(close_prices)
    vote_adx = adx_regime(close_prices)
    vote_returns = return_distribution_regime(close_prices)

    votes = [vote_sma, vote_adx, vote_returns]
    # Majority vote, or 'uncertain' if no majority
    regime = majority(votes) or 'uncertain'
    confidence = votes.count(regime) / len(votes)

    return regime, confidence
```

---

## Appendix B: Practical Considerations

### Data Requirements
| Approach | Minimum Data Needed | Recommended |
|----------|-------------------|-------------|
| Regime-Balanced Selection | 1 year (to cover all regimes) | 3+ years |
| Island Model | 2 years per pair | 4+ years |
| Walk-Forward Regime-Aware | 1.5 years | 3+ years |
| Ensemble Strategy | 3+ years (need enough per regime) | 5+ years |

### Computational Cost
| Approach | Relative Backtesting Cost | Parallelizable? |
|----------|--------------------------|-----------------|
| Standard (no regime) | 1× | Yes |
| Regime-Balanced Selection | 3× (3 regimes) | Yes |
| Multi-Regime Fitness | 3× per evaluation | Yes |
| Island Model | N× islands + migration | Yes (per island) |
| Ensemble | 3× evolution + meta | Partially |
| Adversarial | 3× + overhead | Yes |

### Regime Transition Handling
One of the biggest challenges is handling **regime transitions**:
- Markets don't switch regimes instantly — there are **transition periods**
- Strategies evolved on "pure" regimes may fail during transitions
- **Solution**: Include "transition" or "uncertain" as a 4th regime category
- **Solution**: Add overlap between regime-labeled periods (fuzzy boundaries)
- **Solution**: Use a confirmation period before declaring a regime change (avoid whipsaws)

### Overfitting Risks
| Risk | Mitigation |
|------|-----------|
| Overfitting to regime boundaries | Use multiple detection methods, test boundary sensitivity |
| Overfitting to specific historical regimes | Use walk-forward with regime constraints |
| Island model: specialists too narrow | Migration between islands, master island validation |
| Ensemble: switching too frequently | Confirmation periods, cooldown after switches |
| Synthetic data: fake patterns | Validate synthetic stats match real data, use sparingly |

### Quick-Start Recommendation

For **immediate improvement** with **minimal code changes**:

1. **Implement regime detection** (SMA crossover + ADX, ~100 lines of code)
2. **Add regime-balanced period selection** to the existing backtester (~50 lines)
3. **Change fitness aggregation** to harmonic mean across regime periods (~20 lines)

This gives you **80% of the benefit with 20% of the effort** and requires no structural changes to the GA engine. The existing walk-forward validation, fitness sharing, and NSGA-II infrastructure all continue to work as-is.

After validating this quick-start approach works, proceed to Phase 4 (Island Model) and Phase 5 (Ensemble) for the full production system.
