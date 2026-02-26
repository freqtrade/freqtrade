"""
TrendRider Monitor — sends periodic status updates to Telegram.
Runs alongside the bot as a lightweight background thread.

Sends:
- Hourly heartbeat (silent, just logs)
- Every 6h: market summary + bot status to Telegram
- Daily 8AM UTC: full daily report with P&L
"""

import os
import time
import logging
import threading
from datetime import datetime, timezone

import requests

logger = logging.getLogger("monitor")

TG_TOKEN = os.environ.get("TG_TOKEN", "8272000103:AAErikRTrml-LzGype0LM4eY_Vi634ZHMi8")
TG_CHAT = os.environ.get("TG_CHAT", "5216799062")
BOT_API = os.environ.get("BOT_API", "http://localhost:8080")
BOT_USER = os.environ.get("BOT_USER", "freqtrader")
BOT_PASS = os.environ.get("BOT_PASS", "SuperSecurePassword")
DB_URL = os.environ.get("DATABASE_URL", "")


def send_tg(text: str):
    """Send message to Telegram."""
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"TG send failed: {e}")


def get_bot_token() -> str:
    """Get JWT token from bot API."""
    try:
        r = requests.post(
            f"{BOT_API}/api/v1/token/login",
            auth=(BOT_USER, BOT_PASS),
            timeout=5,
        )
        return r.json().get("access_token", "")
    except Exception:
        return ""


def get_bot_status() -> dict:
    """Get full bot status from API."""
    token = get_bot_token()
    if not token:
        return {}
    headers = {"Authorization": f"Bearer {token}"}
    try:
        config = requests.get(f"{BOT_API}/api/v1/show_config", headers=headers, timeout=5).json()
        balance = requests.get(f"{BOT_API}/api/v1/balance", headers=headers, timeout=5).json()
        trades = requests.get(f"{BOT_API}/api/v1/status", headers=headers, timeout=5).json()
        profit = requests.get(f"{BOT_API}/api/v1/profit", headers=headers, timeout=5).json()
        return {
            "state": config.get("state", "unknown"),
            "balance": balance.get("total", 0),
            "currency": balance.get("stake", "USDT"),
            "open_trades": trades if isinstance(trades, list) else [],
            "profit": profit,
        }
    except Exception as e:
        logger.warning(f"API call failed: {e}")
        return {}


def log_status_to_pg(status: dict):
    """Write status snapshot to Postgres."""
    if not DB_URL:
        return
    try:
        import psycopg2
        conn = psycopg2.connect(DB_URL.replace("postgres://", "postgresql://"))
        cur = conn.cursor()

        profit = status.get("profit", {})
        cur.execute("""
            INSERT INTO bot_status (status, balance, open_trades, total_trades, total_profit, total_profit_pct)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            status.get("state", "unknown"),
            status.get("balance", 0),
            len(status.get("open_trades", [])),
            profit.get("trade_count", 0),
            profit.get("profit_all_coin", 0),
            profit.get("profit_all_ratio_mean", 0) * 100 if profit.get("profit_all_ratio_mean") else 0,
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"PG status write failed: {e}")


def format_status_msg(status: dict) -> str:
    """Format a clean status message."""
    bal = status.get("balance", 0)
    state = status.get("state", "unknown")
    trades = status.get("open_trades", [])
    profit = status.get("profit", {})

    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    emoji = "🟢" if state == "running" else "🔴"

    msg = f"{emoji} *TrendRider Status* ({now})\n"
    msg += f"Balance: *{bal:.2f} USDT*\n"

    if trades:
        msg += f"Open: {len(trades)} trade(s)\n"
        for t in trades:
            p = t.get("profit_pct", 0)
            e = "📈" if p >= 0 else "📉"
            msg += f"  {e} {t['pair']}: {p:+.2f}%\n"
    else:
        msg += "Open: No trades (waiting for signals)\n"

    total_profit = profit.get("profit_all_coin", 0)
    total_count = profit.get("trade_count", 0)
    if total_count > 0:
        msg += f"Closed: {total_count} trades | P&L: {total_profit:+.4f} USDT"

    return msg


def format_daily_report(status: dict) -> str:
    """Format the daily summary report."""
    bal = status.get("balance", 0)
    profit = status.get("profit", {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    msg = f"📊 *Daily Report* — {now}\n"
    msg += f"━━━━━━━━━━━━━━━━━\n"
    msg += f"💰 Balance: *{bal:.2f} USDT*\n"

    total_count = profit.get("trade_count", 0)
    total_profit = profit.get("profit_all_coin", 0)
    today_profit = profit.get("profit_closed_coin", 0)

    if total_count > 0:
        msg += f"📈 Today P&L: {today_profit:+.4f} USDT\n"
        msg += f"📊 All-time: {total_profit:+.4f} USDT ({total_count} trades)\n"
    else:
        msg += f"No trades yet\n"

    msg += f"━━━━━━━━━━━━━━━━━\n"
    msg += f"Strategy: TrendRider v8 | MEXC Spot"

    return msg


def monitor_loop():
    """Main monitoring loop."""
    logger.info("Monitor started")

    cycle = 0
    last_daily = ""

    while True:
        try:
            cycle += 1
            now = datetime.now(timezone.utc)
            hour = now.strftime("%H")
            today = now.strftime("%Y-%m-%d")

            status = get_bot_status()

            if not status:
                if cycle % 6 == 0:
                    send_tg("⚠️ *TrendRider*: Cannot reach bot API. Bot may be restarting.")
                time.sleep(600)
                continue

            # Log to Postgres every hour
            log_status_to_pg(status)

            # Every 6 hours: send status to Telegram
            if cycle % 6 == 0:
                send_tg(format_status_msg(status))

            # Daily report at 8AM UTC
            if hour == "08" and today != last_daily:
                send_tg(format_daily_report(status))
                last_daily = today

            # Warn if bot stopped
            if status.get("state") != "running":
                send_tg(f"🔴 *TrendRider STOPPED* — state: {status.get('state')}")

        except Exception as e:
            logger.error(f"Monitor error: {e}")

        time.sleep(3600)  # Check every hour


def start_monitor():
    """Start monitor in background thread."""
    t = threading.Thread(target=monitor_loop, daemon=True, name="monitor")
    t.start()
    logger.info("Monitor thread started")
    return t
