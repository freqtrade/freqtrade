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
                          auth=("freqtrader", "SuperSecurePassword"), timeout=5)
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
        config = requests.get(f"{BOT_API}/api/v1/show_config", headers=headers, timeout=5).json()
        balance = requests.get(f"{BOT_API}/api/v1/balance", headers=headers, timeout=5).json()
        trades = requests.get(f"{BOT_API}/api/v1/status", headers=headers, timeout=5).json()
        profit = requests.get(f"{BOT_API}/api/v1/profit", headers=headers, timeout=5).json()
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

                # 5. Hourly status
                if now.minute < 6:
                    trades_str = "None" if not status["open_trades"] else \
                        "\n".join(f"  {'📈' if t.get('profit_pct',0)>=0 else '📉'} {t['pair']}: {t.get('profit_pct',0):+.2f}%"
                                  for t in status["open_trades"])
                    send_tg(
                        f"📊 *Status Update*\n"
                        f"Balance: *${bal:.2f}*\n"
                        f"Open: {open_count} trades\n"
                        f"{trades_str}\n"
                        f"Total closed: {total_trades} | P&L: {total_profit:+.4f}"
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
        time.sleep(300)  # Check every 5 minutes


if __name__ == "__main__":
    main()
