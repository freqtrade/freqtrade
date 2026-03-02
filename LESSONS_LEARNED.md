# TrendRider — Lessons Learned

Every mistake we've made and what we did to fix it. Reference this before making strategy changes.

---

## Trade Losses

### 1. SUI -1.34% (Feb 27)
**Problem:** Entered late in the move. Price only went up +0.26% then reversed.
**Root cause:** No check for how far price had already risen from recent low.
**Fix:** Added "freshness filter" — only enters if price hasn't already risen >0.8% from recent low.
**Result:** Would have blocked this trade.

### 2. SOL -0.51% (Mar 2)
**Problem:** Same as SUI — entered after price already bounced +0.30%, no room left to run.
**Root cause:** Same late-entry issue.
**Fix:** Same freshness filter fixes this.
**Result:** Would have blocked this trade.

---

## Strategy Mistakes

### 3. Breakeven profit lock was too aggressive
**Problem:** Moving stop to breakeven at +0.2-0.3% profit caused hundreds of false trailing stop exits from normal 5m price noise.
**Evidence:** Backtest went from -$0.04 to -$4.17 when we added tight profit locking.
**Fix:** Only lock profits above +0.5%. Small gains (+0.2%) aren't worth protecting — the exit cost from noise exceeds the saved profit.
**Rule:** Don't try to lock profits smaller than 2x the fee (0.2% round trip).

### 4. Loosening entries without testing = disaster
**Problem:** Removed 4 entry conditions at once to get more trades. Got 600 trades but -$4.17 loss.
**Fix:** Add back conditions one at a time, test each. The winning combination was: volume filter + MACD rising + double green + trend age 5.
**Rule:** Only change ONE thing at a time and backtest.

### 5. Custom stoploss caused more harm than good
**Problem:** Complex smart stoploss logic reported as "trailing_stop_loss" by Freqtrade, creating 260+ false exits.
**Fix:** Disabled custom_stoploss entirely. Simple -0.9% fixed stop works better.
**Rule:** Simple beats complex for stoploss. Custom stoploss only if you have a very specific reason.

---

## Infrastructure Mistakes

### 6. Two bots fighting over Telegram
**Problem:** Both 1h and 5m bots polling the same Telegram token → conflicts → API timeouts → watchdog alerts.
**Fix:** 1h bot owns Telegram (polling + commands). 5m bot uses webhooks (send-only, no polling).
**Rule:** Only ONE bot per Telegram token can poll. Others must use webhooks.

### 7. Stale pair locks blocking all trading
**Problem:** Protections (CooldownPeriod, StoplossGuard) locked ALL pairs for 48h after a single stop loss. Missed opportunities for days.
**Fix:** Removed all protections. The strategy's entry conditions are selective enough.
**Rule:** Don't add cooldowns unless you're taking too many rapid-fire losses (>5 per hour).

### 8. CoinGecko rate limit errors
**Problem:** Fiat conversion hitting CoinGecko API limit, causing error spam.
**Fix:** Disabled fiat_display_currency. Values show in USDT which is close enough to USD.
**Rule:** Don't use external APIs for non-essential features in a trading bot.

### 9. BTC/ETH "not active" on MEXC
**Problem:** MEXC API marks BTC/USDT and ETH/USDT as active=False even though they trade fine.
**Fix:** Added `"allow_inactive": true` to StaticPairList config.
**Rule:** Always check pair availability with allow_inactive on MEXC.

---

## Strategy Evolution

| Version | Change | Result | Lesson |
|---------|--------|--------|--------|
| v1 strict | 16 conditions, 1% stop | -$0.04 (27 trades) | Too few trades |
| v2 loose | Removed 4 conditions | -$4.17 (600 trades) | Too loose = disaster |
| v2 + volume | Added volume filter back | -$0.55 | Volume filters bad entries |
| v2 + MACD | Added MACD rising back | -$0.31 | Momentum direction matters |
| v2 + green | Added double green back | +$0.07 (205 trades) | Confirmation candles work |
| v2 + fresh | Added freshness filter | **+$0.36 (129 trades)** | Entry timing is everything |

---

## Golden Rules

1. **Entry timing > exit timing.** Getting in early matters more than optimizing take-profit.
2. **Simple stoploss > complex stoploss.** Fixed -0.9% beats any smart logic we tried.
3. **Test ONE change at a time.** Multiple changes = can't tell what helped/hurt.
4. **Fewer better trades > many marginal trades.** 129 trades at +$0.36 beats 600 trades at -$4.17.
5. **The strategy protects capital by NOT trading.** No trades on a -22% market day is the correct output.
6. **Fees eat small profits.** Minimum viable ROI after MEXC fees is 0.3%.
7. **Backtest before deploying.** Every change gets tested on 2 months of data first.
