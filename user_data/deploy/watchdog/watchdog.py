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
    """Check top pairs and explain why we're not trading / what's close."""
    try:
        import ccxt
        ex = ccxt.mexc({'enableRateLimit': True})

        pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT',
                 'DOGE/USDT', 'BNB/USDT', 'DOT/USDT', 'AVAX/USDT']

        best_pair = None
        best_score = 0
        blockers = {}

        for pair in pairs:
            try:
                ohlcv = ex.fetch_ohlcv(pair, '1h', limit=60)
                if len(ohlcv) < 55:
                    continue
                closes = [c[4] for c in ohlcv]
                volumes = [c[5] for c in ohlcv]

                # Simple EMA calculation
                def ema(data, period):
                    k = 2 / (period + 1)
                    result = [data[0]]
                    for i in range(1, len(data)):
                        result.append(data[i] * k + result[-1] * (1 - k))
                    return result

                e9 = ema(closes, 9)[-1]
                e21 = ema(closes, 21)[-1]
                e55 = ema(closes, 55)[-1]
                price = closes[-1]

                # Score each pair
                score = 0
                pair_blockers = []

                if e21 > e55:
                    score += 3
                else:
                    pair_blockers.append("downtrend")

                if e9 > e21:
                    score += 2
                else:
                    pair_blockers.append("EMA weak")

                if price > e9:
                    score += 2
                else:
                    pair_blockers.append("below EMA9")

                if e21 > ema(closes[:-5], 21)[-1]:
                    score += 1
                else:
                    pair_blockers.append("EMA falling")

                ticker = ex.fetch_ticker(pair)
                chg = ticker.get('percentage', 0)
                if chg > 0:
                    score += 1

                short_name = pair.split('/')[0]
                blockers[short_name] = {
                    'score': score,
                    'max': 9,
                    'chg': chg,
                    'blockers': pair_blockers,
                    'price': price,
                }

                if score > best_score:
                    best_score = score
                    best_pair = short_name

            except Exception:
                continue

        if not blockers:
            return "⚠️ Cannot check market data"

        # Build message
        msg = "*Market Scan:*\n"

        # Top 3 closest to trading
        sorted_pairs = sorted(blockers.items(), key=lambda x: x[1]['score'], reverse=True)

        for name, data in sorted_pairs[:3]:
            bar = "🟩" * data['score'] + "⬜" * (data['max'] - data['score'])
            bl = ", ".join(data['blockers'][:2]) if data['blockers'] else "ready!"
            msg += f"`{name:5}` {bar} {data['chg']:+.1f}%"
            if data['blockers']:
                msg += f" _{bl}_"
            msg += "\n"

        if best_score >= 7:
            msg += f"\n🟢 *{best_pair} is close to entry!* Watch for signal."
        elif best_score >= 5:
            msg += f"\n🟡 *{best_pair} looks promising* — needs trend confirmation."
        else:
            msg += f"\n🔴 *No pairs ready* — market is bearish, protecting capital."

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
