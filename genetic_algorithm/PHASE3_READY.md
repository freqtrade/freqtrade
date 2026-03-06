# Phase 3 Overnight Run - READY TO START

## Test Results (2026-03-04 22:11:XX)

### LLM Provider Status
| Provider | Status | Notes |
|----------|--------|-------|
| **Groq** | ✅ OK | Llama-3.3-70B, <500ms response |
| **Anthropic** | ✅ OK | Claude-3-Haiku, <1sec response |
| **OpenAI** | ❌ FAIL | 429 Quota - key/billing mismatch |

### Configuration
- **Population**: 100
- **Generations**: 35
- **LLM Providers**: Groq (primary) → Anthropic (fallback)
- **LLM Budget**: 500 calls/run
- **Timeframe**: 3 years (2023-01-01 to 2026-02-28)
- **Pairs**: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT

### Expected Features Active
- ✅ Regime-aware evaluation (ensemble)
- ✅ Holdout validation (20%)
- ✅ Walk-forward optimization (180/60 days)
- ✅ Hall of fame (50 strategies)
- ✅ Adaptive mutation
- ✅ Fitness sharing
- ✅ LLM strategy design (Groq + Anthropic)
- ✅ Checkpointing every 3 gens

### Expected Runtime
**4-6 hours** on 8-core machine
~3,500 strategy evaluations total

### Start Command
```bash
cd /home/kali/trading/freqtradeForkGA
source .venv/bin/activate
nohup python genetic_algorithm/run_ga.py \
  --config genetic_algorithm/config/ga_config_phase3_overnight.yaml \
  --visualize --no-interactive --yes \
  > genetic_algorithm/logs/phase3_overnight_stdout.log 2>&1 &
echo $! > genetic_algorithm/logs/phase3_overnight_pid.txt
```

### Monitor
```bash
# Watch logs in real-time
tail -f genetic_algorithm/logs/ga_phase3_overnight.log

# Check generation progress
grep "GENERATION\|STATS\|HALL" genetic_algorithm/logs/ga_phase3_overnight.log | tail -20

# Get PID if needed
cat genetic_algorithm/logs/phase3_overnight_pid.txt
```

### What To Expect
- **Gen 1-10**: Random population + initial LLM seeding
- **Gen 11-25**: Fitness landscape exploration, regime detection finding patterns
- **Gen 26-35**: Fine-tuning top performers, LLM mutation on stagnating areas
- **Final**: Hall of Fame report with top 50 strategies

### Success Metrics
- Best fitness > 0.65 (Phase 2 achieved 0.79 peak)
- LLM contribution measurable in top 25%
- > 8 trades average per strategy
- Positive Sharpe ratio on > 70% of hall of fame
- All 5 pairs showing positive cumulative returns
