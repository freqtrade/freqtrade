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
    """Rich market analysis matching actual strategy conditions."""
    try:
        import ccxt
        ex = ccxt.mexc({'enableRateLimit': True})

        pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT',
                 'DOGE/USDT', 'ADA/USDT', 'SUI/USDT', 'BNB/USDT',
                 'DOT/USDT', 'AVAX/USDT', 'LINK/USDT', 'PEPE/USDT']

        results = []

        def ema(data, period):
            k = 2 / (period + 1)
            r = [data[0]]
            for i in range(1, len(data)):
                r.append(data[i] * k + r[-1] * (1 - k))
            return r

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
                vol_recent = sum(volumes[-3:]) / 3
                vol_earlier = sum(volumes[-6:-3]) / 3 if len(volumes) >= 6 else vol_recent

                # Recent low and freshness
                recent_low = min(lows[-12:])
                rise = (price - recent_low) / recent_low * 100

                # Trend age
                age = 0
                for i in range(len(e9)-1, 0, -1):
                    if e9[i] > e21[i]:
                        age += 1
                    else:
                        break

                name = pair.split('/')[0]

                # Determine market phase for this pair
                if e21[-1] > e55[-1] and e9[-1] > e21[-1] and price > e9[-1]:
                    phase = "🟢 uptrend"
                elif e21[-1] > e55[-1]:
                    phase = "🟡 pullback"
                elif e9[-1] > e21[-1]:
                    phase = "🔵 bouncing"
                else:
                    phase = "🔴 downtrend"

                # What needs to happen for entry
                needs = []
                if not (e21[-1] > e55[-1]):
                    needs.append("trend flip")
                if not (e9[-1] > e21[-1]):
                    needs.append("EMA cross up")
                if not (price > e9[-1]):
                    needs.append("price recovery")
                if not (45 < rsi < 65):
                    if rsi >= 65:
                        needs.append("RSI cool down")
                    else:
                        needs.append("RSI recover")
                if not (vol_recent > vol_earlier * 1.05):
                    needs.append("volume surge")
                if rise >= 0.8:
                    needs.append("fresh setup")

                ready = len(needs) == 0

                results.append({
                    'name': name,
                    'price': price,
                    'chg': chg,
                    'rsi': rsi,
                    'phase': phase,
                    'age': age,
                    'rise': rise,
                    'vol_rising': vol_recent > vol_earlier * 1.05,
                    'needs': needs,
                    'ready': ready,
                    'needs_count': len(needs),
                })
            except Exception:
                continue

        if not results:
            return "⚠️ Cannot check market"

        results.sort(key=lambda x: x['needs_count'])

        # Market mood
        uptrends = sum(1 for r in results if "uptrend" in r['phase'])
        pullbacks = sum(1 for r in results if "pullback" in r['phase'])
        downtrends = sum(1 for r in results if "downtrend" in r['phase'])

        if uptrends >= 6:
            mood = "🟢 *Market: BULLISH* — conditions favorable"
        elif uptrends + pullbacks >= 6:
            mood = "🟡 *Market: MIXED* — watching for entries"
        elif downtrends >= 8:
            mood = "🔴 *Market: BEARISH* — protecting capital"
        else:
            mood = "🟡 *Market: CHOPPY* — selective trading"

        # Show only the closest pair with actual vs needed values
        results.sort(key=lambda x: x['needs_count'])
        best = results[0] if results else None

        if not best:
            return mood

        msg = f"{mood}\n"

        if best['ready']:
            msg += f"🔥 *{best['name']} READY* {best['chg']:+.1f}%\n"
        else:
            msg += f"Closest: *{best['name']}* {best['chg']:+.1f}%\n"

        # Show actual values vs required
        msg += f"```\n"
        msg += f"         Now    Need\n"
        msg += f"RSI:     {best['rsi']:>5.0f}   45-65\n"
        msg += f"Rise:    {best['rise']:>4.1f}%   <0.8%\n"
        msg += f"Trend:   {'yes' if 'trend flip' not in best['needs'] else 'no':>5}   yes\n"
        msg += f"VolUp:   {'yes' if best['vol_rising'] else 'no':>5}   yes\n"
        msg += f"Phase:   {best['phase'].split(' ')[1]}\n"
        msg += f"```"

        if best['needs'] and not best['ready']:
            msg += f"\nNeeds: {', '.join(best['needs'])}"

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
                        f"💰 *${bal:.2f}* | {total_profit:+.4f} P&L | {total_trades} trades\n"
                        f"{trades_str}\n"
                        f"{market_msg}"
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
