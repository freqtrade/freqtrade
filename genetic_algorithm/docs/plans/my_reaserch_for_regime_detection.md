## HMM and Markov-switching models
Method description
Markov-switching (econometrics): assumes observations follow a model whose parameters change according to a hidden Markov chain (seminally formalized in regime-switching autoregressions). 
HMM (ML): assumes a discrete latent state sequence (regimes) drives emissions of observed features; you infer state posteriors and transitions using EM / Viterbi / smoothing. 
Strengths for crypto
Produces soft regime probabilities (confidence) and transition matrices, letting you implement hysteresis and avoid rapid strategy switching.
Naturally models volatility regimes (low/high vol) and trend/mean regimes if you include appropriate features.
Weaknesses for crypto
Standard Gaussian emissions are sensitive to fat tails/outliers and can split regimes to “explain” extreme candles rather than structural states. 
Can be non-identifiable (label switching) and unstable unless you constrain states, regularize, or use robust emissions. 
Typical inputs
Returns + realized volatility (+ volume/range). For MarkovAutoregression, you can pass a single series (e.g., returns) and let regimes switch mean/variance. 

Recommended libraries
Statsmodels Markov switching (very mature):

Install: pip install statsmodels
Key classes: statsmodels.tsa.regime_switching.markov_regression.MarkovRegression, MarkovAutoregression 
hmmlearn (fast baseline HMM):

Install: pip install hmmlearn
Key class: hmmlearn.hmm.GaussianHMM (also GMMHMM) 
Recent maintenance: active releases through 2024. 
Pyro HMM distributions (heavy-tail friendly option):

Install: pip install pyro-ppl torch
Notable: pyro.distributions.hmm.GammaGaussianHMM uses a MultivariateStudentT joint distribution and is parallelized over time. 
pomegranate (HMMs, but check version/API carefully):

Install: pip install pomegranate
The project has releases up to 2025; docs you’ll often find online may correspond to older versions, so verify APIs in your environment. 
Runtime/complexity
HMM inference is typically O(T·K²) per sequence for K states; EM adds iterations. (Practical: fast for daily/4h length series and small K.)
Markov-switching in statsmodels is MLE-based and can be slower than hmmlearn but remains practical for single-asset series sizes. 


## Bayesian change-point detection (offline/online) and classical change-point methods
Method description
Offline change-point detection: segment a series into contiguous regimes by optimizing a cost + penalty (or specifying number of breakpoints). 
Bayesian online change-point detection (BOCPD): maintains a posterior over “run length” (time since last change) and triggers when probability mass shifts to small run lengths. 
Strengths for crypto
Excellent for building balanced historical datasets: you can force the GA to evaluate across multiple segments representative of different market phases.
Offline methods can be very robust when applied to volatility proxies rather than raw returns.
Weaknesses for crypto
Change points can be triggered by short-lived shocks (exchange events, liquidation cascades), so you need minimum segment length and a thoughtful penalty.
Online Bayesian methods require specifying hazard / priors; poor choices create either too many or too few detected breaks.
Typical inputs
For bull/bear/side regime carving: log_close, trend slope, or rolling regression slope.
For risk regimes: rv (rolling vol), range/ATR, or absolute returns. 
Recommended libraries
ruptures (offline, well-documented):

Install: pip install ruptures
Key algorithms: rpt.Pelt, rpt.Binseg, rpt.Dynp
Documentation emphasizes offline segmentation and provides complexity guidance (e.g., dynamic programming costs). 
BOCPD packages (quality varies):

bayesian-changepoint-detection (PyPI name), last seen pre-release 2019 → treat as less maintained. 
bocd (PyPI), releases 2019 → also more “project” than production. 
If you want a more actively maintained “online CP in a package,” consider modern alternatives; river focuses on drift detection (e.g., ADWIN) rather than BOCPD, but is engineered for streaming. 
Runtime/complexity
ruptures dynamic programming has quadratic scaling (documented as ~O(C·K·n²)), while penalized approaches like PELT are often much faster in practice. 
BOCPD in its exact form scales with time unless truncated/approximated; many practical implementations cap run length.

## L1 trend filtering (piecewise-linear trend extraction)
Method description
L1 trend filtering is a robust alternative to Hodrick–Prescott style smoothing: it penalizes the absolute value of second differences, producing piecewise linear trends whose “kinks” can be interpreted as regime changes. 

Strengths for crypto
Works directly on log-price and produces interpretable “trend segments.”
Because the trend is piecewise linear, it maps cleanly to bull/bear/side: segment slope sign and magnitude.
Weaknesses for crypto
Choose λ too small → over-segmentation (reacts to noise); too large → misses meaningful shifts.
It is trend-centric; you still need a separate “volatility regime” concept if that matters for strategy behavior.
Typical inputs
log(close) (optionally log(hl2) or VWAP proxy). 

Recommended libraries
Install: pip install cvxpy
Implement via convex optimization in CVXPY. 
Runtime/complexity
The original work highlights near-linear scaling with specialized solvers; in CVXPY you’ll generally rely on installed solvers and should expect slower performance for very long series unless you downsample or solve in chunks. 

## Regime clustering on engineered features (unsupervised ML)
Method description
Treat regime detection as unsupervised clustering on a feature vector per candle (or per rolling window). This is often simpler than state-space models and easy to integrate with GA pipelines.

Common choices:

KMeans (fast)
Gaussian Mixture Models (soft clustering + probabilistic scores)
Bayesian Gaussian Mixture (variational mixture; can infer “effective” components) 
Strengths for crypto
Very easy to operationalize and retrain (rolling monthly/weekly).
Lets you include “context features” (trend strength, volatility, volume, funding, etc.) without heavy model design.
Weaknesses for crypto
Pure clustering ignores time dependence, so labels can flicker unless you add smoothing/hysteresis.
Euclidean KMeans is sensitive to outliers and heavy tails; robust scaling helps but does not fully solve distributional issues. 
Typical inputs
Multi-feature vectors like [ret, rv, range, log_vol, MA_slope, ADX].

Recommended libraries (very mature)
Install: pip install scikit-learn
sklearn.cluster.KMeans for hard clustering 
sklearn.mixture.GaussianMixture for soft clustering and likelihoods 
sklearn.preprocessing.StandardScaler (or RobustScaler) with Pipeline to avoid leakage 
Runtime/complexity
KMeans average complexity is documented as roughly O(k·n·T) (k clusters, n samples, T iterations).

## Regime rules (ADX + volatility + moving averages)
Method description
Rule-based regime detection is often underestimated: it is extremely interpretable, easy to debug, and can be used either as a standalone regime labeler or as a gating layer for probabilistic models.

Common rule families:

Trend vs range: ADX threshold (e.g., ADX > 20–25 indicates trend strength) plus directional cues (MA slope, +DI/-DI).
Risk regime: rolling volatility quantiles (low/medium/high).
Breakout regimes: ATR-normalized range expansions.
Strengths for crypto
Robust operationally; no training required; works well as a stability anchor for ML/HMM labels.
Easy to align with strategy logic (trend-following vs mean reversion).
Weaknesses for crypto
Thresholds are heuristic and can drift by timeframe/market structure.
Rule-only labels can miss “structural breaks” that do not manifest as strong ADX trends.
Recommended libraries
ta (pure pandas/numpy): pip install ta 
TA-Lib wrapper: pip install TA-Lib (requires TA-Lib system lib unless wheels are used) 
TA-Lib docs explicitly flag that some indicators (including ADX) have an “unstable period,” so you must ignore the warm-up region. 


## Hidden semi-Markov models (HSMM) and duration-aware regimes
Method description
HSMMs extend HMMs by explicitly modeling state durations (dwell time), which is often desirable for regimes: bull markets and bear markets have characteristic persistence patterns, and HSMMs encode that directly.

Strengths for crypto
Better “dwell-time realism” than standard HMMs, which can produce too-frequent switching unless transition probabilities are extreme.
Useful when you need stable regime blocks for downstream strategy selection.
Weaknesses for crypto
Fewer “turnkey” Python libraries; often requires custom probabilistic programming.
More parameters: duration distributions can overfit, especially on short histories.
Sources and practical tooling reality in Python
HSMM theory and applications are commonly summarized in the HSMM literature; practical implementations often rely on custom inference routines. 

If you want a pragmatic route in Python today:

Use HMM + explicit hysteresis/dwell-time constraints (post-processing) as an approximation; or
Use probabilistic programming (Pyro / PyMC) to encode duration structure, at higher implementation cost. 

## Neural HMMs and unsupervised sequence models
Method description
These methods keep the “discrete regimes” idea but use neural nets for emissions / transitions or structured inference. They are most useful when regimes depend on nonlinear feature interactions and you have enough data to justify model complexity.

A representative example class is recurrent semi-Markov / neural HSMM variants designed for segmentation and labeling. 

Strengths for crypto
Can model nonlinear dependencies and heterogeneous emissions (e.g., mixture-of-experts style) that are hard for linear HMMs.
Potentially more robust to complex regime structure across multi-feature inputs.
Weaknesses for crypto
Significant engineering and validation burden; increased risk of overfitting.
Harder interpretability and debugging, especially when regimes are used to select real capital-allocation behavior.
Pragmatic recommendation: treat neural HMMs as a “phase 2” once you have a strong baseline and a clear failure mode that simpler models cannot handle.

### Practical engineering notes for automation in Python
Timeframes and warm-up
For BTC/USDT, a default benchmark of daily + 4h is reasonable: it reduces microstructure noise compared with 1m/5m while still capturing regime transitions at subweekly horizons. Evidence exists that predictability can appear at sub-daily horizons (up to hours), so 4h is a sensible compromise. 

Warm-up rules of thumb:

Rolling volatility: drop at least vol_window bars.
ADX (and many TA indicators): drop the documented unstable period and initial NaNs. 
Preprocessing and missing data
Ensure OHLCV has consistent spacing; if gaps occur, do not blindly forward-fill price bars as that can create fake low-volatility regimes.
If you must impute: forward-fill volume is usually less damaging than forward-filling OHLC. For OHLC gaps, consider dropping periods or resampling from trades (if you have them).
Hyperparameters you should expect to tune
HMM/Markov-switching: number of regimes K, whether variance switches, covariance type, regularization, initialization stability.
Ruptures: cost model (l2, rbf), penalty pen, minimum segment length / jump grid. 
L1 trend filtering: λ (and whether you solve on log-price or detrended series). 
Rule-based: ADX threshold, MA windows, volatility quantile cutoffs.
Confidence thresholds and hysteresis (to stop flickering)
HMM/Markov-switching: require posterior max probability > 0.6–0.8 to switch; otherwise keep prior regime (state “sticky mode”).
Change-point: require minimum segment length and require the change to persist for X bars before acting.
Rule-based: use two thresholds (enter trend when ADX > 25, exit trend when ADX < 20) to create hysteresis.
Persisting labels for backtesting and GA integration
Persisting regimes is best treated as a first-class artifact:

Store regime_label, confidence, and model_version keyed by timestamp.
Save as Parquet in your user_data area (fast, columnar).
Regenerate labels on a schedule (weekly/monthly) and version the outputs.
If you run regimes inside Freqtrade, remember backtesting loads full dataframes at once; you should validate you didn’t introduce lookahead effects through your labeling or features. Freqtrade provides lookahead-analysis to detect lookahead bias in strategies. 

Complementary ensembles and a recommended default pipeline for BTC/USDT daily and 4h
Ensemble patterns that work well in practice
A robust architecture for crypto regimes is “segmentation + probabilistic labeling + rule gating”:

Ruptures on volatility (rv) finds major breaks (macro conditions).
Within each segment, HMM/Markov-switching produces smooth posteriors (micro regimes).
ADX gating decides “trend vs range,” acting as a guardrail.
Online drift detection (e.g., ADWIN) raises a “regime-break alarm” in live trading, triggering a re-label/retrain. 


### Evaluation metrics and validation recipes for regime labels
Why evaluation is tricky
Regime labels are usually unsupervised. That means you validate:

internal fit (likelihood/cost),
stability (do labels persist under small perturbations?),
usefulness (does conditional strategy performance improve without leakage?).
Also, in finance/crypto, repeated selection and backtesting can heavily inflate performance without careful validation; the backtest overfitting literature emphasizes this risk. 

Metrics that are directly actionable
Label stability and sanity checks

Dwell-time distribution: Are regimes unrealistically short? HSMM motivation is precisely duration realism. 
Transition matrix sanity (HMM/Markov-switching): Is self-transition probability high enough to reflect plausible persistence?
Flip rate: fraction of timestamps where regime changes vs total bars; track by timeframe.
Segment separation diagnostics Without using future returns as labels, you can compare:

Mean return and volatility per regime (descriptive).
Distributional differences (e.g., regime A has higher conditional volatility). Heavy tails and volatility clustering are expected in crypto; your regimes should typically separate at least one of these dimensions. 
Change-point evaluation

On synthetic series with known breakpoints (including Student‑t noise), compute precision/recall of detected change points within a tolerance window.
Use ruptures evaluation tooling and/or your own hit-window scoring.
Time-series validation (anti-leakage) Use blocked or gap time series splits:

TimeSeriesSplit(gap=...) exists specifically to separate training and test by a temporal buffer. 
Cross-validation for dependent/time series data has a substantial literature (blocked/hv-block approaches), reinforcing that naive random CV is invalid under dependence. 
If you want engineered splitters: the tscv package provides “gap” splitters designed to mitigate temporal dependence leakage. 
Backtest protocol that fits regime labeling
A practical recipe:

Holdout by contiguous era, not random samples: e.g., last 20–30% of history as a locked OOS slice.
Within the training era, use walk-forward / expanding windows with a gap/embargo between train/test.
For GA evaluation, compute fitness as a weighted aggregate across:
bullish segments,
bearish segments,
sideways segments,
and ideally both low-vol and high-vol segments, with caps to prevent one regime from dominating.
Run Freqtrade lookahead checks on the final strategy