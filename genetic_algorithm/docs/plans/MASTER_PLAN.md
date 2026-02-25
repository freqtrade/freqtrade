# Master Plan: GA Strategy Optimizer Roadmap

**Last Updated:** February 25, 2026  
**Current Focus:** Ready for Production Testing

---

## 📊 Status Dashboard

| Feature | Status | Impact | Effort |
|---------|--------|--------|--------|
| Walk-Forward Optimization | ✅ Complete | ⭐⭐⭐⭐⭐ | - |
| Multi-Timeframe Strategies | ✅ Complete | ⭐⭐⭐⭐⭐ | - |
| NSGA-II Multiobjective | ✅ Complete | ⭐⭐⭐⭐ | - |
| Parallel Evaluation | ✅ Complete | ⭐⭐⭐ | - |
| Market Regime Detection (Basic) | ✅ Complete | ⭐⭐⭐⭐ | - |
| **Regime Detection Accuracy** | ✅ Complete | ⭐⭐⭐⭐⭐ | - |
| Elite Fitness Caching | ✅ Complete | ⭐⭐⭐ | - |
| Island Model | 📋 Planned | ⭐⭐⭐ | 3-6 days |
| Strategy Grammar | 📋 Planned | ⭐⭐ | 5-10 days |

---

## 🎯 Current Priority: Regime Detection Accuracy

### Problem Statement

The current regime detection methods (SMA-based) are **too simple and inaccurate**:
- Classifies most BTC/crypto periods as "sideways" (~95%)
- Few bullish/bearish segments detected
- Lagging indicators miss regime transitions
- High noise in consolidation periods

**Evidence from testing:**
```
36 walk-forward windows created
Regime distribution:
  sideways: 36 windows (100%)
  bullish: 0 windows
  bearish: 0 windows
```

### Goal

Improve regime detection to achieve:
- **Balanced classification**: ~30-40% each for bullish/bearish/sideways
- **Higher confidence**: >70% confidence for detected regimes  
- **Faster detection**: Catch regime transitions within 1-2 weeks, not months
- **Lower noise**: Reduce false regime flips in consolidation

---

## 📋 Phase 6: Advanced Regime Detection

### Research Areas (To Investigate)

| Method | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| **Shorter SMA Periods** | SMA(20/50) instead of SMA(50/200) | Simple, fast | Still lagging | ⭐⭐ |
| **Hidden Markov Models (HMM)** | Probabilistic state transitions | Standard in finance, handles uncertainty | Requires training, complex | ⭐⭐⭐⭐⭐ |
| **Directional Movement (ADX/DI)** | ADX threshold + DI crossovers | Built-in to TA-Lib, trend strength | Can be noisy | ⭐⭐⭐ |
| **Price Action Patterns** | Higher highs/lows, swing detection | Intuitive, no lag | Subjective definitions | ⭐⭐⭐ |
| **Rolling Returns Distribution** | Mean/std of returns over windows | Simple statistics | Needs good window size | ⭐⭐⭐ |
| **Bayesian Change Point** | Detect structural breaks | Mathematically rigorous | Computationally expensive | ⭐⭐⭐⭐ |
| **ML Classifiers (RF/XGB)** | Train on labeled regime data | High accuracy if good labels | Needs labeled data, overfitting risk | ⭐⭐⭐⭐ |
| **ℓ1 Trend Filtering** | Piecewise linear trend detection (CVXPY) | Smooth, handles noise, interpretable slope | λ tuning, trend-only (no vol) | ⭐⭐⭐ |
| **Wavelet Decomposition** | Multi-scale trend analysis | Handles multiple timeframes | Complex to interpret | ⭐⭐ |
| **Ensemble/Voting** | Combine multiple methods | Robust, reduces false positives | Slower, complexity | ⭐⭐⭐⭐ |
| **HSMM (Semi-Markov)** | Duration-aware HMM with dwell time | Better regime persistence | Fewer libraries, custom code | ⭐⭐ |

### Recommended Libraries (Verified)

| Library | Install | Use Case | Notes |
|---------|---------|----------|-------|
| `statsmodels` | Built-in | `MarkovAutoregression`, `MarkovRegression` | **BEST for returns**. Mature, 3-state natural |
| `hmmlearn` | `pip install hmmlearn` | `GaussianHMM` baseline | Fast, simple API, but Gaussian sensitive to outliers |
| `pyro-ppl` | `pip install pyro-ppl torch` | `GammaGaussianHMM` | **Heavy-tail friendly!** Uses Student-t, ideal for crypto |
| `ruptures` | `pip install ruptures` | Change point detection (PELT, BinSeg) | Active, v1.1.10. Finds WHERE regimes change |
| `scikit-learn` | Built-in | `GaussianMixture`, `KMeans` clustering | Quick unsupervised baseline |
| `cvxpy` | `pip install cvxpy` | L1 trend filtering | Piecewise linear trends |
| `ta` / `ta-lib` | `pip install ta` | ADX, DI, MA slopes | Rule-based gating |

### Implementation Plan

#### Step 1: Baseline Evaluation (1-2 days)
- [ ] Create regime detection benchmark dataset
- [ ] Manually label 6-12 months of BTC/USDT 4h with correct regimes
- [ ] Define evaluation metrics (accuracy, dwell-time sanity, flip rate)
- [ ] Measure current SMA-based method performance

#### Step 2: Quick Wins (2-3 days)
- [ ] **Rolling Returns Classification** - Replace SMA with rolling mean returns
- [ ] **ADX + DI Gating** - Trend vs range with hysteresis (enter >25, exit <20)
- [ ] **Shorter Windows** - Try 20-day instead of 200-day
- [ ] Compare performance vs baseline

#### Step 3: HMM / Markov-Switching (3-4 days)
- [ ] **Option A: Statsmodels MarkovAutoregression** (recommended first)
  ```python
  from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
  model = MarkovAutoregression(returns, k_regimes=3, order=1)
  result = model.fit()
  regimes = result.smoothed_marginal_probabilities
  ```
- [ ] **Option B: Pyro GammaGaussianHMM** (if fat tails cause issues)
- [ ] Add hysteresis: require posterior > 0.7 to switch regime
- [ ] Map states to bull/bear/sideways by mean returns

#### Step 4: Change-Point Detection (2 days)
- [ ] Use ruptures to find macro regime boundaries
- [ ] Classify each segment by returns direction
- [ ] Combine: Ruptures (macro) + HMM (micro) + ADX (guardrail)

#### Step 5: Integration & Testing (2-3 days)
- [ ] Update RegimeDetector with best method(s)
- [ ] Persist labels: Parquet with timestamp, label, confidence, model_version
- [ ] Validate on multiple pairs (BTC, ETH, SOL)
- [ ] Validate on multiple timeframes (1h, 4h, 1d)
- [ ] Run Freqtrade lookahead-analysis to check for leakage

### Hysteresis & Stability Rules

To prevent label flickering:

| Method | Hysteresis Rule |
|--------|-----------------|
| **HMM/Markov** | Switch only if posterior_prob > 0.6-0.8 |
| **ADX Gating** | Enter trend when ADX > 25, exit when ADX < 20 |
| **Change-Point** | Require minimum segment length (e.g., 7 days) |
| **All methods** | Track flip_rate per timeframe, alert if too high |

### Key Questions to Answer

1. **What defines a "bullish" period in crypto?**
   - Option A: Mean daily returns > 0.1% over rolling window
   - Option B: HMM state with highest mean returns
   - Option C: Slope of L1 trend > threshold

2. **How to handle high-volatility sideways markets?**
   - Separate trend regimes from volatility regimes
   - Use 2D regime: (bull/bear/side) × (low vol/high vol)
   - ADX helps: high ADX + directionless = volatile sideways

3. **What's the minimum regime duration?**
   - Recommendation: 7-14 days for daily, 84-168 bars for 4h
   - Enforce via min_segment_length in ruptures or HSMM

4. **Crypto-specific adjustments needed?**
   - Yes: Higher volatility thresholds, 24/7 markets
   - Consider exchange events (liquidation cascades) as noise

---

## 📋 Phase 7: Island Model (After Regime Detection)

### Overview
Run N isolated populations ("islands") that evolve independently, with periodic migration of best individuals between islands.

### Benefits
- More diversity in exploration
- Parallel evolution of different strategy styles
- Natural parallelization

### Implementation Plan
- See [TODO_ga_improvements.md](TODO_ga_improvements.md) for detailed plan

---

## 📋 Phase 8: Future Features

### Strategy Grammar (Low Priority)
Type-safe genetic programming to prevent invalid conditions like `RSI > 70 AND RSI < 30`.

### Ensemble Strategies (Research)
Combine multiple evolved strategies with voting or stacking.

### Multi-Exchange Evolution (Research)
Evolve strategies that work across Binance, Kraken, Coinbase.

---

## 📚 Reference Documents

| Document | Purpose |
|----------|---------|
| [TODO_ga_improvements.md](TODO_ga_improvements.md) | Detailed technical TODOs |
| [MARKET_REGIME_DATASET_SELECTION.md](../features/MARKET_REGIME_DATASET_SELECTION.md) | Regime detection concepts |
| [REGIME_DETECTION_IMPLEMENTATION.md](../features/REGIME_DETECTION_IMPLEMENTATION.md) | Current implementation details |

---

## 🔬 Research Notes

### Literature & Resources

**Key Papers & Articles:**
- C. Truong, L. Oudre, N. Vayatis. "Selective review of offline change point detection methods." Signal Processing, 167:107299, 2020
- QuantStart: "Market Regime Detection using Hidden Markov Models in QSTrader"

**Production-Ready Libraries:**

| Library | Focus | Install | Notes |
|---------|-------|---------|-------|
| `hmmlearn` | Hidden Markov Models | `pip install hmmlearn` | Scikit-learn compatible API, GaussianHMM |
| `ruptures` | Change Point Detection | `pip install ruptures` | Multiple algorithms (PELT, BinSeg, BottomUp), well-maintained |
| `pomegranate` | Probabilistic Models | `pip install pomegranate` | HMM + Bayesian methods |
| `pymc` | Bayesian Inference | `pip install pymc` | Full Bayesian change point detection |

### Your Findings (User Research - Feb 2026)

#### Summary of Methods Researched

| Method | Best Library | Key Insight |
|--------|--------------|-------------|
| HMM / Markov-Switching | `statsmodels.MarkovAutoregression` | Handles returns directly, 3-state natural fit |
| Heavy-Tail HMM | `pyro.distributions.hmm.GammaGaussianHMM` | Uses Student-t, robust to crypto outliers! |
| Change-Point | `ruptures` (PELT, BinSeg, Dynp) | Macro regime boundaries, O(n) to O(n²) |
| L1 Trend Filtering | `cvxpy` | Piecewise linear, slope = direction |
| Clustering | `sklearn.GaussianMixture` | Soft clustering on feature vectors |
| Rule-Based | `ta-lib` ADX + DI | No training, interpretable, guardrail |

#### Critical Implementation Notes

**Hysteresis to Prevent Flickering:**
- HMM: require `posterior_prob > 0.6-0.8` to switch regime
- ADX: enter trend at `ADX > 25`, exit at `ADX < 20` (not same threshold)
- Change-point: enforce minimum segment length

**Warm-up Handling:**
- Rolling volatility: drop first `vol_window` bars
- ADX: drop documented "unstable period" and initial NaNs
- HMM: ensure training window is sufficient (200+ bars)

**Preprocessing:**
- Don't forward-fill OHLC gaps (creates fake low-vol regimes)
- If must impute: forward-fill volume only is safer
- Use log-returns for HMM input, not raw prices

**Ensemble Architecture (Recommended):**
1. **Ruptures on volatility** → finds major macro breaks
2. **HMM within segments** → smooth micro-regime probabilities  
3. **ADX gating** → decides "trend vs range" as guardrail
4. **ADWIN (optional)** → online drift detection in live trading

**Evaluation Metrics:**
- Dwell-time distribution (are regimes unrealistically short?)
- Transition matrix sanity (self-transition prob should be high)
- Flip rate per timeframe
- Conditional returns/vol per regime (should separate meaningfully)

**Time-Series Validation:**
- Use `sklearn.TimeSeriesSplit(gap=N)` - NOT random CV
- Holdout last 20-30% as locked OOS
- Walk-forward with embargo between train/test

#### Detailed Method Notes

**1. Statsmodels MarkovAutoregression (Recommended First)**
```python
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

# returns = daily log-returns
model = MarkovAutoregression(returns, k_regimes=3, order=1)
result = model.fit()

# Get smoothed probabilities for each regime
probs = result.smoothed_marginal_probabilities  # shape: (T, k_regimes)
regime_labels = probs.argmax(axis=1)

# Map states by mean return
state_means = result.filtered_marginal_probabilities.T @ returns
# Lowest mean = bearish, middle = sideways, highest = bullish
```

**2. Pyro GammaGaussianHMM (For Heavy Tails)**
- Uses MultivariateStudentT joint distribution
- Parallelized over time: O(log T) complexity
- Handles crypto fat tails better than Gaussian HMM
- More complex setup, requires PyTorch

**3. Ruptures + Classification**
```python
import ruptures as rpt
import numpy as np

# Detect change points in volatility
volatility = df["close"].pct_change().rolling(20).std().dropna().values
algo = rpt.Pelt(model="rbf", min_size=7).fit(volatility.reshape(-1, 1))
breakpoints = algo.predict(pen=5)

# Classify each segment
def classify_segment(returns):
    mean_ret = returns.mean()
    if mean_ret > 0.001:  # 0.1% daily
        return "bullish"
    elif mean_ret < -0.001:
        return "bearish"
    return "sideways"
```

**4. ADX + DI Gating Rules**
```python
# Using ta library
from ta.trend import ADXIndicator

adx_indicator = ADXIndicator(df["high"], df["low"], df["close"], window=14)
adx = adx_indicator.adx()
plus_di = adx_indicator.adx_pos()
minus_di = adx_indicator.adx_neg()

# Hysteresis-aware regime
def get_trend_regime(adx, plus_di, minus_di, prev_regime):
    if adx > 25:  # Strong trend
        if plus_di > minus_di:
            return "bullish"
        else:
            return "bearish"
    elif adx < 20:  # No trend
        return "sideways"
    else:  # Hysteresis zone: keep previous
        return prev_regime
```

### My (Claude's) Findings

#### 1. Hidden Markov Models (HMM) - ⭐⭐⭐⭐⭐ Recommended

**How it works:**
- Assumes "hidden" market states (regimes) influence observable returns
- Uses Expectation-Maximization to fit state transition probabilities
- Predicts most likely current state given historical returns

**Implementation (from QuantStart):**
```python
from hmmlearn.hmm import GaussianHMM
import numpy as np

# Prepare returns as column vector
returns = df["Close"].pct_change().dropna()
rets = np.column_stack([returns])

# Fit 2-state HMM (low vol / high vol)
hmm_model = GaussianHMM(
    n_components=2,      # Number of hidden states
    covariance_type="full", 
    n_iter=1000
).fit(rets)

# Predict current regime
hidden_states = hmm_model.predict(rets)
current_regime = hidden_states[-1]  # 0 or 1
```

**Key Insights:**
- 2 states = low/high volatility (works well for risk management)
- 3 states = bullish/bearish/sideways (what we want)
- Train on historical data, use out-of-sample for prediction
- Should periodically retrain as market structure changes

**Pros:**
- Industry standard for financial regime detection
- Handles uncertainty natively (probabilities, not hard labels)
- Can detect regime changes before they're visually obvious
- Returns probabilities for each state (confidence measure)

**Cons:**
- Requires sufficient training data (1+ years)
- Needs parameter tuning (n_components, covariance_type)
- Not designed for online/real-time detection

---

#### 2. Ruptures Library (Change Point Detection) - ⭐⭐⭐⭐

**How it works:**
- Finds "breakpoints" where signal statistics change
- Multiple algorithms: PELT (fast), Dynamic Programming (optimal), BinSeg (approximate)
- Multiple cost functions: L2 (mean shift), RBF (kernel), AR (autoregressive)

**Implementation:**
```python
import ruptures as rpt

# Use returns or prices
signal = df["Close"].values.reshape(-1, 1)

# PELT algorithm with RBF kernel
algo = rpt.Pelt(model="rbf").fit(signal)
breakpoints = algo.predict(pen=10)  # penalty controls sensitivity

# Breakpoints are indices where regime changes
# e.g., [100, 250, 400, 600] = 4 regime changes
```

**Key Insights:**
- Good for finding WHERE regimes change
- Need additional logic to CLASSIFY each segment (bullish/bearish/sideways)
- Penalty parameter controls sensitivity (high = fewer breakpoints)
- Very fast for large datasets

**Combination Strategy:**
1. Use ruptures to find breakpoints
2. For each segment between breakpoints, classify as bullish/bearish/sideways
3. Classification can be simple: positive returns = bullish, negative = bearish, mixed = sideways

---

#### 3. Combined Approach (Recommended)

**Best Strategy: HMM + Classification**

```python
# Step 1: Use 3-state HMM for regime detection
hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
hmm_model.fit(returns_data)

# Step 2: Identify which state is which
# After fitting, look at state means:
# - State with highest mean returns = bullish
# - State with lowest mean returns = bearish  
# - State in middle = sideways

state_means = hmm_model.means_.flatten()
sorted_states = np.argsort(state_means)
bearish_state = sorted_states[0]
sideways_state = sorted_states[1]
bullish_state = sorted_states[2]

# Step 3: Create regime labels
regime_map = {
    bearish_state: "bearish",
    sideways_state: "sideways", 
    bullish_state: "bullish"
}
```

**Why this works better than SMA:**
- SMA compares current price to historical average → always "where we've been"
- HMM looks at returns distribution → "what the market is doing now"
- HMM naturally handles volatility clustering in crypto

---

#### 4. Quick Win: Volatility-Based Classification

Before implementing HMM, try this simpler approach:

```python
def classify_regime_returns(df, window=20):
    """Classify based on rolling returns and volatility"""
    returns = df["close"].pct_change()
    
    # Rolling statistics
    roll_return = returns.rolling(window).mean()
    roll_vol = returns.rolling(window).std()
    
    # Thresholds (tune for crypto)
    bullish_threshold = 0.001   # 0.1% daily = ~35% annual
    bearish_threshold = -0.001
    vol_threshold = 0.03       # 3% daily volatility
    
    conditions = [
        roll_return > bullish_threshold,
        roll_return < bearish_threshold,
        True  # default
    ]
    choices = ["bullish", "bearish", "sideways"]
    
    return np.select(conditions, choices)
```

**Why this might work better:**
- Uses returns directly (not lagging price comparison)
- Crypto-appropriate thresholds
- Faster to detect regime changes (20-day window vs 200-day)

---

## 📅 Timeline

| Week | Focus | Deliverable |
|------|-------|-------------|
| Week 1 | Research + Baseline | Evaluation framework, labeled dataset |
| Week 2 | Quick Wins + HMM | Improved regime detection |
| Week 3 | Testing + Integration | Production-ready detection |
| Week 4 | Island Model | Start next feature |

---

*This is a living document. Update as we make progress.*
