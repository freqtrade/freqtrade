"""
TrendRider Watchdog — independent health monitor on Railway.
Runs as a SEPARATE service so it stays alive even if the bot crashes.

Checks every 5 minutes:
- Is the bot process responding? (API health check)
- Is MEXC reachable?
- Are there stuck trades?
- Is balance dropping unexpectedly?

Sends Telegram alerts on any issue.
Sends a daily summary at 8AM UTC.
"""

import os
import time
import logging
import requests
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("watchdog")

TG_TOKEN = os.environ.get("TG_TOKEN", "8272000103:AAErikRTrml-LzGype0LM4eY_Vi634ZHMi8")
TG_CHAT = os.environ.get("TG_CHAT", "5216799062")
BOT_API = os.environ.get("BOT_API", "http://trendrider-bot.railway.internal:8080")
DB_URL = os.environ.get("DATABASE_URL", "")

last_alert = {}
last_daily = ""
last_balance = None
consecutive_failures = 0


def send_tg(text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        logger.info(f"TG sent: {text[:60]}")
    except Exception as e:
        logger.error(f"TG failed: {e}")


def alert_once(key, msg, cooldown=3600):
    """Send alert max once per cooldown period."""
    now = time.time()
    if key in last_alert and now - last_alert[key] < cooldown:
        return
    last_alert[key] = now
    send_tg(msg)


def get_bot_token():
    try:
        r = requests.post(f"{BOT_API}/api/v1/token/login",
                          auth=("freqtrader", "SuperSecurePassword"), timeout=15)
        return r.json().get("access_token", "")
    except Exception:
        return ""


def check_bot():
    """Check bot health via API."""
    token = get_bot_token()
    if not token:
        return None

    headers = {"Authorization": f"Bearer {token}"}
    try:
        config = requests.get(f"{BOT_API}/api/v1/show_config", headers=headers, timeout=15).json()
        balance = requests.get(f"{BOT_API}/api/v1/balance", headers=headers, timeout=15).json()
        trades = requests.get(f"{BOT_API}/api/v1/status", headers=headers, timeout=15).json()
        profit = requests.get(f"{BOT_API}/api/v1/profit", headers=headers, timeout=15).json()
        return {
            "state": config.get("state"),
            "balance": balance.get("total", 0),
            "open_trades": trades if isinstance(trades, list) else [],
            "profit": profit,
        }
    except Exception as e:
        logger.warning(f"API check failed: {e}")
        return None


def check_mexc():
    """Check if MEXC is reachable."""
    try:
        r = requests.get("https://api.mexc.com/api/v3/ping", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def log_to_pg(status, balance, open_trades, total_trades, profit, pct):
    if not DB_URL:
        return
    try:
        import psycopg2
        conn = psycopg2.connect(DB_URL.replace("postgres://", "postgresql://"))
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO bot_status (status, balance, open_trades, total_trades, total_profit, total_profit_pct)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, (status, round(balance, 4), open_trades, total_trades, round(profit, 6), round(pct, 2)))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"PG write failed: {e}")


def get_market_analysis() -> str:
    """Check all pairs against actual 5m strategy conditions."""
    try:
        import ccxt
        ex = ccxt.mexc({'enableRateLimit': True})

        pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT',
                 'DOGE/USDT', 'ADA/USDT', 'SUI/USDT', 'BNB/USDT',
                 'DOT/USDT', 'AVAX/USDT', 'LINK/USDT', 'PEPE/USDT']

        results = []

        def ema(data, period):
            k = 2 / (period + 1)
            result = [data[0]]
            for i in range(1, len(data)):
                result.append(data[i] * k + result[-1] * (1 - k))
            return result

        for pair in pairs:
            try:
                ohlcv = ex.fetch_ohlcv(pair, '5m', limit=65)
                if len(ohlcv) < 60:
                    continue

                closes = [c[4] for c in ohlcv]
                highs = [c[2] for c in ohlcv]
                lows = [c[3] for c in ohlcv]
                opens = [c[1] for c in ohlcv]
                volumes = [c[5] for c in ohlcv]

                e9 = ema(closes, 9)
                e21 = ema(closes, 21)
                e55 = ema(closes, 55)

                # RSI
                deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                gains = [d if d > 0 else 0 for d in deltas[-14:]]
                losses_r = [-d if d < 0 else 0 for d in deltas[-14:]]
                avg_gain = sum(gains) / 14
                avg_loss = sum(losses_r) / 14
                rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 100

                ticker = ex.fetch_ticker(pair)
                chg = ticker.get('percentage', 0) or 0
                price = closes[-1]

                vol_avg = sum(volumes[-20:]) / 20

                # Check actual strategy conditions
                checks = {
                    'trend': e21[-1] > e55[-1],
                    'ema_align': e9[-1] > e21[-1],
                    'above_ema': price > e9[-1],
                    'ema_rising': e21[-1] > e21[-4],
                    'rsi_ok': 45 < rsi < 65,
                    'rsi_up': rsi > 50,
                    'volume': volumes[-1] > vol_avg,
                    'green': closes[-1] > opens[-1],
                }

                passed = sum(1 for v in checks.values() if v)
                total = len(checks)
                name = pair.split('/')[0]

                # What's blocking
                blockers = []
                if not checks['trend']:
                    blockers.append("bearish")
                if not checks['ema_align']:
                    blockers.append("no momentum")
                if not checks['above_ema']:
                    blockers.append("below EMA")
                if not checks['rsi_ok']:
                    if rsi >= 65:
                        blockers.append("overbought")
                    elif rsi <= 45:
                        blockers.append("oversold")
                if not checks['volume']:
                    blockers.append("low vol")

                results.append({
                    'name': name,
                    'score': passed,
                    'max': total,
                    'chg': chg,
                    'price': price,
                    'rsi': rsi,
                    'blockers': blockers,
                    'trend': checks['trend'],
                    'ready': passed >= 7,
                })
            except Exception:
                continue

        if not results:
            return "⚠️ Cannot check market data"

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)

        msg = "*Market Scan (5m):*\n"

        # Show top 6 pairs
        for r in results[:6]:
            filled = r['score']
            empty = r['max'] - filled
            bar = "🟩" * filled + "⬜" * empty

            if r['ready']:
                status = "🔥"
            elif filled >= 6:
                status = "👀"
            elif r['trend']:
                status = "📈"
            else:
                status = "📉"

            bl = ", ".join(r['blockers'][:2]) if r['blockers'] else "✅ ready"
            msg += f"{status} `{r['name']:5}` {bar} RSI:{r['rsi']:.0f} {r['chg']:+.1f}% _{bl}_\n"

        # Bottom line
        ready = [r for r in results if r['ready']]
        close = [r for r in results if r['score'] >= 6 and not r['ready']]
        bullish = [r for r in results if r['trend']]

        msg += "\n"
        if ready:
            names = ", ".join(r['name'] for r in ready)
            msg += f"🔥 *{names} ready to trade!*"
        elif close:
            names = ", ".join(r['name'] for r in close)
            msg += f"👀 *{names} almost ready* — watching closely"
        elif bullish:
            msg += f"📈 {len(bullish)} pairs bullish but entry conditions not met yet"
        else:
            msg += "📉 Market bearish — protecting capital, waiting for reversal"

        return msg

    except Exception as e:
        return f"⚠️ Market scan error: {str(e)[:50]}"


def main():
    global consecutive_failures, last_daily, last_balance

    logger.info("Watchdog started")
    send_tg("🐕 *Watchdog online* — monitoring TrendRider 24/7")

    cycle = 0
    while True:
        try:
            now = datetime.now(timezone.utc)
            hour = now.strftime("%H")
            today = now.strftime("%Y-%m-%d")

            # 1. Check bot
            status = check_bot()

            if status is None:
                consecutive_failures += 1
                logger.warning(f"Bot unreachable ({consecutive_failures})")

                if consecutive_failures == 3:
                    alert_once("bot_down",
                        "🔴 *ALERT: Bot unreachable*\n"
                        "Cannot connect to TrendRider API.\n"
                        "Check Railway dashboard.",
                        cooldown=1800)
            else:
                if consecutive_failures >= 3:
                    send_tg("🟢 *Bot recovered* — back online")
                consecutive_failures = 0

                bal = status["balance"]
                state = status["state"]
                open_count = len(status["open_trades"])
                profit_data = status.get("profit", {})
                total_trades = profit_data.get("trade_count", 0)
                total_profit = profit_data.get("profit_all_coin", 0)

                # Log to DB every check
                log_to_pg(state, bal, open_count, total_trades, total_profit, 0)

                # 2. Check if bot stopped
                if state != "running":
                    alert_once("bot_stopped",
                        f"⚠️ *Bot state: {state}*\nNot actively trading.",
                        cooldown=3600)

                # 3. Check balance drop
                if last_balance is not None and bal < last_balance * 0.95:
                    alert_once("balance_drop",
                        f"⚠️ *Balance dropped*\n"
                        f"Was: ${last_balance:.2f} → Now: ${bal:.2f}",
                        cooldown=3600)
                last_balance = bal

                # 4. Check stuck trades (open > 48h)
                for t in status["open_trades"]:
                    open_date = t.get("open_date", "")
                    if open_date:
                        try:
                            od = datetime.fromisoformat(open_date.replace("Z", "+00:00"))
                            hours_open = (now - od).total_seconds() / 3600
                            if hours_open > 48:
                                alert_once(f"stuck_{t['pair']}",
                                    f"⚠️ *Stuck trade:* {t['pair']}\n"
                                    f"Open {hours_open:.0f}h | P&L: {t.get('profit_pct', 0):.2f}%",
                                    cooldown=21600)
                        except Exception:
                            pass

                # 5. Hourly market analysis + status
                if now.minute < 31 and int(hour) != getattr(main, '_last_status_hour', -1):
                    main._last_status_hour = int(hour)

                    # Get market analysis
                    market_msg = get_market_analysis()

                    trades_str = ""
                    if status["open_trades"]:
                        trades_str = "\n".join(
                            f"  {'📈' if t.get('profit_pct',0)>=0 else '📉'} {t['pair']}: {t.get('profit_pct',0):+.2f}%"
                            for t in status["open_trades"])
                        trades_str = f"\n*Open trades:*\n{trades_str}"

                    send_tg(
                        f"📊 *Hourly Update*\n"
                        f"💰 Balance: *${bal:.2f}*\n"
                        f"📂 Trades: {open_count} open | {total_trades} closed | P&L: {total_profit:+.4f}"
                        f"{trades_str}\n"
                        f"\n{market_msg}"
                    )

                # 6. Daily report at 8AM
                if hour == "08" and today != last_daily:
                    last_daily = today
                    send_tg(
                        f"📊 *Daily Report — {today}*\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"💰 Balance: *${bal:.2f}*\n"
                        f"📈 Trades: {total_trades} closed\n"
                        f"💵 P&L: {total_profit:+.4f} USDT\n"
                        f"📂 Open: {open_count}\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"🐕 Watchdog: all systems normal"
                    )

            # 7. Check MEXC
            if not check_mexc():
                alert_once("mexc_down",
                    "⚠️ *MEXC API unreachable*\nExchange may be down.",
                    cooldown=1800)

        except Exception as e:
            logger.error(f"Watchdog error: {e}")

        logger.info(f"Check #{cycle} done | bot={'OK' if status else 'FAIL'} | mexc={'OK' if check_mexc() else 'FAIL'}")
        cycle += 1
        time.sleep(1800)  # Check every 30 minutes


if __name__ == "__main__":
    main()
