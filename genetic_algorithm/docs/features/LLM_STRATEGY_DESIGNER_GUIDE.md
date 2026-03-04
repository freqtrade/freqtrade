# LLM Strategy Designer Guide

**Status:** ✅ Available (Phase 1A)  
**Requires:** API key for a cloud provider, or a running local LLM server

---

## Table of Contents

1. [What the LLM integration does](#1-what-the-llm-integration-does)
2. [Quick-start: enable LLM in 3 steps](#2-quick-start-enable-llm-in-3-steps)
3. [Supported providers](#3-supported-providers)
4. [API key setup](#4-api-key-setup)
5. [All configuration options](#5-all-configuration-options)
6. [How it fits into the GA evolution loop](#6-how-it-fits-into-the-ga-evolution-loop)
7. [What the LLM actually generates](#7-what-the-llm-actually-generates)
8. [Limitations and known constraints](#8-limitations-and-known-constraints)
9. [Costs and rate limits](#9-costs-and-rate-limits)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What the LLM integration does

Without LLMs the GA starts every run from a **purely random population**: each strategy is assembled by picking indicators, conditions, and risk parameters at random.  Random initialisation works, but wastes many early generations discovering obvious patterns that an experienced trader already knows.

With LLMs enabled, the GA uses a language model as a **knowledge source** to bootstrap better starting strategies and inject guided diversity throughout evolution:

| Role | When | How many | Controlled by |
|------|------|----------|---------------|
| **Seed strategies** | Generation 0 (initial population) | `seed_ratio` × remaining slots | `seed_ratio` |
| **Immigrants** | Every generation, alongside random immigrants | `immigrants_per_generation` (fixed count) | `immigrants_per_generation` |

**Seeding** means the LLM fills part of the first population with strategies it designs from scratch, using domain knowledge about trend-following, mean-reversion, momentum, etc.

**Immigrant injection** means that every generation, instead of only adding purely random new individuals, the GA also asks the LLM for new strategies that *complement* what is already in the population.  The LLM receives a summary of the current top performers and a list of identified weaknesses (e.g. "No strategies use CMF indicator") and is asked to generate something different.

The result: **faster early convergence** and **better diversity** than fully random evolution.

---

## 2. Quick-start: enable LLM in 3 steps

**Step 1 — Install the HTTP client library** (only needed once):

```bash
pip install httpx
```

**Step 2 — Add your API key** to `ga_config.yaml` (or set an environment variable, see [§4](#4-api-key-setup)):

```yaml
advanced:
  llm:
    enabled: true          # ← flip this to true
    provider: "grok"       # grok | openai | anthropic | local
    api_key: "xai-..."     # ← paste your key here
```

**Step 3 — Run the GA as normal**:

```bash
python genetic_algorithm/run_ga.py --config genetic_algorithm/config/ga_config.yaml
```

You will see a banner in the log output when LLM mode is active:

```
======================================================================
LLM STRATEGY DESIGNER ENABLED
  Provider: grok
  Model: grok-3-mini
  Seed ratio: 20%
  Immigrant ratio: 50%
======================================================================
```

---

## 3. Supported providers

| Provider name(s) | Service | Default model | Default API base URL |
|-----------------|---------|---------------|----------------------|
| `grok`, `xai` | xAI Grok | `grok-3-mini` | `https://api.x.ai/v1` |
| `openai` | OpenAI | `gpt-4o-mini` | `https://api.openai.com/v1` |
| `anthropic`, `claude` | Anthropic Claude | `claude-sonnet-4-20250514` | `https://api.anthropic.com/v1` |
| `local`, `ollama` | Any local server (Ollama, llama.cpp, vLLM) | `llama3` | `http://localhost:11434/v1` |

All cloud providers use their standard REST APIs (OpenAI-compatible chat completions for Grok/OpenAI/local; Anthropic Messages API for Claude).

### Choosing a provider

- **Grok (recommended default):** Fast, cheap, instruction-following is strong for structured JSON.
- **OpenAI GPT-4o-mini:** Reliable JSON output, widely available, reasonable cost.
- **Anthropic Claude:** Excellent reasoning, slightly higher latency and cost.
- **Local (Ollama / llama.cpp):** Free, fully private, no API key needed — but requires a running local server and JSON quality varies by model.  Good choice for experimentation.

---

## 4. API key setup

### Option A — Paste directly into the config file

```yaml
advanced:
  llm:
    api_key: "xai-your-key-here"
```

⚠️ **Do not commit API keys to git.** Add `ga_config.yaml` (or your custom config) to `.gitignore` if it contains a real key.

### Option B — Environment variable (recommended for shared/CI environments)

The config comment lists the conventional variable names.  You can export them in your shell before running the GA:

| Provider | Environment variable |
|----------|---------------------|
| Grok / xAI | `GROK_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |

Then reference it in the config with an empty `api_key` field and load it yourself, or pass it at runtime:

```bash
export GROK_API_KEY="xai-your-key"
# Then in your launch script or Python code:
import os, yaml
config = yaml.safe_load(open("ga_config.yaml"))
config['advanced']['llm']['api_key'] = os.environ['GROK_API_KEY']
```

Alternatively, leave `api_key: ""` in the YAML and override it at the Python entry point before constructing `GeneticAlgorithm`.

### Option C — Local server (no API key)

For `local` / `ollama`, no API key is required.  The code automatically sets `api_key: "not-needed"` if you leave it blank:

```yaml
advanced:
  llm:
    enabled: true
    provider: "local"
    base_url: "http://localhost:11434/v1"   # Ollama default
    model: "llama3"
    api_key: ""   # leave blank — set automatically
```

Make sure Ollama (or your server) is already running and the model is pulled:

```bash
ollama serve           # start server
ollama pull llama3     # download model (first time only)
```

---

## 5. All configuration options

All LLM options live under `advanced.llm` in your `ga_config.yaml`:

```yaml
advanced:
  llm:
    # ── Activation ─────────────────────────────────────────────────────
    enabled: false                  # Set to true to activate LLM features

    # ── Provider ───────────────────────────────────────────────────────
    provider: "grok"                # grok | openai | anthropic | local
    api_key: ""                     # Your API key (or leave blank for local)
    model: ""                       # Leave blank for provider default:
                                    #   grok     → grok-3-mini
                                    #   openai   → gpt-4o-mini
                                    #   anthropic → claude-sonnet-4-20250514
                                    #   local    → llama3
    base_url: ""                    # Leave blank for provider default URL.
                                    # Override for custom endpoints or local servers.

    # ── Generation quality ─────────────────────────────────────────────
    temperature: 0.7                # 0.0 = deterministic, 1.0 = very creative.
                                    # 0.7–0.9 gives good variety without nonsense.
    max_tokens: 4096                # Max tokens in the LLM response.
                                    # 4096 is ample for one strategy JSON.

    # ── Reliability / retries ──────────────────────────────────────────
    timeout: 60                     # HTTP timeout in seconds per API call.
    max_retries: 3                  # Retry attempts per strategy request.
                                    # On JSON parse failure the error message is
                                    # fed back to the LLM for self-correction.
    retry_delay: 2.0                # Seconds between retries (exponential for 429s).
    min_call_interval: 1.0          # Minimum seconds between API calls (rate limiting).

    # ── Population injection ratios ────────────────────────────────────
    seed_ratio: 0.20                # Fraction of the initial population filled by LLM.
                                    # E.g. 0.20 with population_size=50 → 10 LLM seeds.
                                    # The rest are random or seeded from archetypes.
    immigrants_per_generation: 2    # Fixed number of LLM immigrants injected each
                                    # generation (on top of random immigrants).
    immigrant_ratio: 0.50           # Alternative ratio-based control used by the
                                    # legacy StrategyDesigner path (50% of immigrant
                                    # slots come from LLM when this path is active).

    # ── Reference ──────────────────────────────────────────────────────
    providers: [grok, openai, anthropic, local]   # Informational — lists all valid names.
```

### Key parameters explained

#### `seed_ratio`

Controls how many strategies in **generation 0** come from the LLM versus being generated randomly.

```
population_size = 50
seed_ratio      = 0.20
→ ~10 LLM-designed strategies + 40 random strategies at startup
```

Set `seed_ratio: 0.0` to disable LLM seeding while keeping per-generation immigrants.  
Set `seed_ratio: 0.5` for aggressive seeding (50% LLM at startup).  
The actual number may be slightly less if the LLM fails some requests.

#### `immigrants_per_generation`

Fixed number of LLM-generated individuals added every generation *in addition to* the `random_immigrants` count from the main GA config.  These "immigrants" replace strategies that underperform in the selection phase.

#### `temperature`

- `0.3–0.5`: Conservative, consistent strategies.  Good for production runs where you want reliable JSON output.
- `0.7–0.9`: Creative variety.  Recommended default; LLM explores diverse indicator combinations.
- `1.0+`: Very random.  Likely produces more parse errors.

#### `max_retries`

When the LLM returns malformed JSON or a strategy that fails validation (e.g. no valid indicators), the parser generates an error message and appends it to the next attempt's prompt so the LLM can self-correct.  The total number of attempts per strategy is `max_retries`.

---

## 6. How it fits into the GA evolution loop

```
Generation 0 (initialize_population)
├─ Hall-of-fame injection (best from previous runs, if any)
├─ Archetype seeding (hand-crafted example strategies)
├─ [LLM] generate_seed_strategies(count = remaining × seed_ratio)
└─ Random fill (remaining slots)

Generation N (create_next_generation)
├─ Elite preservation
├─ Tournament selection + crossover + mutation
├─ [LLM] generate_immigrants(count = immigrants_per_generation)
│         └─ Context: top-5 performers + population weakness list
└─ Random immigrants (random_immigrants from GA config)
```

The LLM immigrants receive the following context so the model can generate *complementary* strategies:

- **Top performers summary**: fitness score, profit %, drawdown, indicator types used
- **Population weaknesses**: e.g. "No strategies use CMF indicator", "Population skews toward high-frequency — need fewer, higher-quality trades"

This guided injection is designed to fill gaps the purely random search would take many generations to discover.

---

## 7. What the LLM actually generates

The LLM is asked to return a JSON object that maps directly to a `StrategyGene`.  A typical response looks like:

```json
{
  "indicators": [
    {
      "type": "RSI",
      "instance_id": "RSI_0",
      "parameters": {"period": 14},
      "weight": 1.0,
      "timeframe": null
    },
    {
      "type": "EMA",
      "instance_id": "EMA_0",
      "parameters": {"period": 20},
      "weight": 1.0,
      "timeframe": null
    }
  ],
  "entry_conditions": [
    {"indicator": "RSI_0", "operator": "<", "threshold": 35, "logic": "AND"},
    {"indicator": "EMA_0", "operator": "cross_above", "threshold": 0, "logic": "AND"}
  ],
  "exit_conditions": [
    {"indicator": "RSI_0", "operator": ">", "threshold": 65, "logic": "AND"}
  ],
  "timeframe": "15m",
  "stoploss": -0.08,
  "minimal_roi": {"0": 0.04, "30": 0.02, "60": 0.01},
  "max_open_trades": 3,
  "trailing_stop": false
}
```

### Available indicators the LLM can use

| Indicator | Description |
|-----------|-------------|
| `RSI` | Relative Strength Index (momentum oscillator, 0–100) |
| `MACD` | Moving Average Convergence/Divergence |
| `BBANDS` | Bollinger Bands (volatility) |
| `EMA` | Exponential Moving Average |
| `SMA` | Simple Moving Average |
| `ADX` | Average Directional Index (trend strength) |
| `SUPERTREND` | ATR-based trend-following bands |
| `ICHIMOKU` | Ichimoku Cloud |
| `DONCHIAN` | Donchian Channel (breakout) |
| `PSAR` | Parabolic SAR (trend reversal) |
| `CMF` | Chaikin Money Flow (volume) |
| `VROC` | Volume Rate of Change |
| `CDL_ENGULFING` | Engulfing candlestick pattern |
| `CDL_HAMMER` | Hammer candlestick pattern |
| `CDL_MORNINGSTAR` | Morning Star pattern |
| `CDL_EVENINGSTAR` | Evening Star pattern |
| `CDL_DOJI` | Doji candlestick pattern |

The available set is controlled by the `indicators.available` list in your config.  The LLM prompt is built from exactly those indicators — it cannot use indicators that are not in your config.

### Available condition operators

| Operator | Meaning |
|----------|---------|
| `>` | Value is above threshold |
| `<` | Value is below threshold |
| `cross_above` | Value crosses above threshold (momentum signal) |
| `cross_below` | Value crosses below threshold |
| `increasing` | Value has been increasing over `lookback` periods |
| `decreasing` | Value has been decreasing over `lookback` periods |
| `between` | Value is between `threshold` and `threshold_upper` |
| `value_above_ago` | Current value is above value from `lookback` periods ago |

### Strategy styles cycled for seed generation

For the initial population the LLM is prompted with a different style hint each time, cycling through: `trend_following`, `mean_reversion`, `breakout`, `momentum`, `volatility`.  This ensures the seed population covers diverse trading philosophies.

---

## 8. Limitations and known constraints

### 8.1 The LLM does not backtest anything

The LLM designs strategies based on **general trading knowledge**, not on your specific data.  It has no access to your price data, your historical fitness scores, or what worked in previous runs.  Every strategy the LLM proposes still needs to be backtested by the GA's fitness evaluator before it can influence evolution.

Think of the LLM as a knowledgeable human trader suggesting starting points — the GA's selection pressure then decides which suggestions are actually worth keeping.

### 8.2 Parse failures are expected and handled

LLMs occasionally produce malformed JSON or reference indicators not in the allowed set.  The parser handles this automatically:

- Unknown indicator types are silently removed
- Duplicate or missing `instance_id` values are auto-assigned
- Conditions referencing non-existent indicators are dropped
- If the minimum number of entry/exit conditions is not met, defaults are added
- Stoploss is clamped to the configured range
- On complete parse failure, the error message is fed back to the LLM for a retry (up to `max_retries` times)

If all retries fail for a given slot, that slot is simply filled by a random strategy instead.  **LLM failures never crash the GA.**

### 8.3 The LLM cannot guarantee profitable strategies

The LLM produces *plausible* trading logic based on widely documented patterns.  It does not guarantee profitability.  Treat LLM seeds exactly like random seeds: most will be mediocre, some will be good starting points, and the GA's evolution process will refine them.

### 8.4 API cost grows with population size and generations

Each LLM call costs tokens and real money (for cloud providers).  Cost scales as:

```
total_calls ≈ (population_size × seed_ratio) + (generations × immigrants_per_generation)
```

**Example** with defaults (`population_size=30`, `seed_ratio=0.20`, `generations=12`, `immigrants_per_generation=2`):

```
seed calls      = 30 × 0.20 = 6
immigrant calls = 12 × 2    = 24
total           = 30 API calls per full run
```

With `grok-3-mini` or `gpt-4o-mini`, each call uses roughly 1,000–2,000 tokens (prompt + response).  At typical prices this is **a few cents per run** — negligible for small runs, but check your provider's pricing before running with `population_size=200`.

### 8.5 Local models require sufficient hardware

Running a local model (e.g. Llama 3 8B via Ollama) requires at minimum 8 GB of RAM (or VRAM).  JSON output quality varies significantly between models — some smaller models produce structurally invalid JSON more often.  If you see many parse failures with a local model, try:

1. A larger model (e.g. 13B or 70B)
2. A model fine-tuned for instruction following (e.g. `llama3-instruct`)
3. Reducing `temperature` to 0.3–0.5

### 8.6 Not a replacement for the full genetic algorithm

The LLM is an *optional accelerator*, not a replacement for the GA.  Even with `seed_ratio: 1.0` (100% LLM seeding), you still need the GA's crossover, mutation, and selection to refine strategies.  The LLM has no way to know what will actually perform well on your specific dataset and time range.

---

## 9. Costs and rate limits

### Rate limiting

The `min_call_interval` setting (default: 1.0 second) enforces a minimum gap between consecutive API calls.  This prevents hitting per-minute rate limits on free tiers.

For stricter limits (e.g. OpenAI free tier: 3 requests/minute), set:

```yaml
min_call_interval: 20.0   # one call every 20 seconds
```

### Retry back-off

HTTP 429 (rate-limit) responses trigger **exponential back-off**: the delay doubles each retry:

```
attempt 1 → wait retry_delay × 2⁰ = 2 s
attempt 2 → wait retry_delay × 2¹ = 4 s
attempt 3 → wait retry_delay × 2² = 8 s
```

HTTP 5xx server errors use a fixed `retry_delay` wait.

### Estimated costs per run (cloud)

| Provider | Model | ~Cost per 1K tokens | ~Cost per full run (30 calls) |
|----------|-------|--------------------|-----------------------------|
| xAI Grok | grok-3-mini | ~$0.001 | < $0.10 |
| OpenAI | gpt-4o-mini | ~$0.0002 | < $0.05 |
| Anthropic | claude-sonnet-4 | ~$0.003 | < $0.30 |
| Local | any | free | free |

Costs are approximate and depend on actual prompt/response lengths.

---

## 10. Troubleshooting

### "ModuleNotFoundError: No module named 'httpx'"

Install the required HTTP client:

```bash
pip install httpx
```

### "ValueError: API key required for GrokProvider"

You have `enabled: true` but `api_key` is empty.  Either:
- Paste your key into `api_key:` in the YAML
- Set the environment variable and load it in your launch script (see [§4](#4-api-key-setup))

### "LLM disabled, skipping seed generation"

Check that **both** conditions are true:

1. `advanced.llm.enabled: true`
2. `advanced.llm.provider` is set to a valid name AND the provider was successfully initialised (check the log for errors)

### "HTTP 401 from GrokProvider"

Wrong or expired API key.  Double-check the key on your provider's dashboard.

### "HTTP 429 from GrokProvider (attempt 1/3)"

Rate limit hit.  Increase `min_call_interval` or `retry_delay`.  The code will retry automatically with exponential back-off.

### Many parse failures / "No valid indicators found"

Likely causes:
1. **Temperature too high** — lower it to 0.5–0.7
2. **Model too small** (local) — try a larger or instruction-tuned model
3. **Indicators list mismatch** — verify `indicators.available` in your config matches what you expect.  The LLM is only told about indicators in that list.

### "Failed to create LLM client ... Injector disabled"

The provider constructor raised an exception (wrong API key format, network unreachable, etc.).  The GA continues with random-only strategies.  Check the log line before this for the specific error.

### Log shows "LLM generated strategy … 0 indicators"

Should not happen after validation fixes, but if it does:  the LLM response was parsed as valid JSON but contained no recognisable indicator types.  Check that `indicators.available` is populated in your config (the config value defaults to the full list if empty).

---

## See also

- [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md) — full reference for all GA configuration options
- [`WALK_FORWARD_GUIDE.md`](WALK_FORWARD_GUIDE.md) — preventing overfitting with time-based validation splits
- [`TIER3_ROBUSTNESS_FEATURES.md`](TIER3_ROBUSTNESS_FEATURES.md) — Monte-Carlo robustness, parsimony, Pareto archive

---

*Last updated: March 2026 (Phase 1A)*
