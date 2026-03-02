"""
TrendRider Strategy Analyst — learns from every trade.

Runs every 30 minutes:
1. Checks all closed trades
2. For each trade: what SHOULD have happened vs what DID happen
3. Identifies patterns in losses
4. Suggests strategy improvements
5. Sends daily learning report to Telegram
6. Tracks prediction accuracy over time
"""

import os
import time
import logging
import requests
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("analyst")

TG_TOKEN = os.environ.get("TG_TOKEN", "8272000103:AAErikRTrml-LzGype0LM4eY_Vi634ZHMi8")
TG_CHAT = os.environ.get("TG_CHAT", "5216799062")
DB_URL = os.environ.get("DATABASE_URL", "")


def send_tg(text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"TG failed: {e}")


def get_db():
    if not DB_URL:
        return None
    try:
        import psycopg2
        return psycopg2.connect(DB_URL.replace("postgres://", "postgresql://"))
    except Exception:
        return None


def ensure_tables():
    """Create analyst tables if they don't exist."""
    conn = get_db()
    if not conn:
        return
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trade_analysis (
            id SERIAL PRIMARY KEY,
            trade_id INTEGER,
            pair VARCHAR(20),
            open_date TIMESTAMPTZ,
            close_date TIMESTAMPTZ,
            entry_price NUMERIC(18,8),
            exit_price NUMERIC(18,8),
            profit_pct NUMERIC(8,4),
            exit_reason VARCHAR(50),
            peak_profit_pct NUMERIC(8,4),
            worst_drawdown_pct NUMERIC(8,4),
            price_after_1h_pct NUMERIC(8,4),
            entry_timing VARCHAR(20),
            exit_quality VARCHAR(20),
            should_have_entered BOOLEAN,
            should_have_held BOOLEAN,
            lesson TEXT,
            analyzed_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS strategy_score (
            id SERIAL PRIMARY KEY,
            date DATE,
            total_trades INTEGER,
            wins INTEGER,
            losses INTEGER,
            win_rate NUMERIC(5,2),
            avg_win_pct NUMERIC(8,4),
            avg_loss_pct NUMERIC(8,4),
            profit_factor NUMERIC(8,4),
            entry_timing_score NUMERIC(5,2),
            exit_quality_score NUMERIC(5,2),
            improvements TEXT,
            scored_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()


def analyze_trade(trade, ex):
    """Deep analysis of a single trade — what happened and what should have happened."""
    pair = trade['pair']
    entry_price = float(trade['open_rate'])
    exit_price = float(trade['close_rate'] or 0)
    profit_pct = float(trade['close_profit'] or 0) * 100
    open_date = trade['open_date']
    close_date = trade['close_date']
    exit_reason = trade['exit_reason'] or ''

    analysis = {
        'trade_id': trade['id'],
        'pair': pair,
        'open_date': open_date,
        'close_date': close_date,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'profit_pct': profit_pct,
        'exit_reason': exit_reason,
        'peak_profit_pct': 0,
        'worst_drawdown_pct': 0,
        'price_after_1h_pct': 0,
        'entry_timing': 'unknown',
        'exit_quality': 'unknown',
        'should_have_entered': True,
        'should_have_held': False,
        'lesson': '',
    }

    try:
        since = int((open_date - timedelta(minutes=30)).timestamp() * 1000)
        ohlcv = ex.fetch_ohlcv(pair, '5m', since=since, limit=200)

        entry_ts = open_date.timestamp()
        exit_ts = close_date.timestamp() if close_date else time.time()

        # Track peak and worst during trade
        peak = 0
        worst = 0
        for candle in ohlcv:
            ts = candle[0] / 1000
            if entry_ts <= ts <= exit_ts:
                high_pct = (candle[2] - entry_price) / entry_price * 100
                low_pct = (candle[3] - entry_price) / entry_price * 100
                peak = max(peak, high_pct)
                worst = min(worst, low_pct)

        # Track price 1 hour after exit
        price_after = 0
        for candle in ohlcv:
            ts = candle[0] / 1000
            if exit_ts < ts <= exit_ts + 3600:
                pct = (candle[4] - entry_price) / entry_price * 100
                price_after = pct

        analysis['peak_profit_pct'] = round(peak, 2)
        analysis['worst_drawdown_pct'] = round(worst, 2)
        analysis['price_after_1h_pct'] = round(price_after, 2)

        # Entry timing assessment
        if peak > 1.0:
            analysis['entry_timing'] = 'excellent'
        elif peak > 0.5:
            analysis['entry_timing'] = 'good'
        elif peak > 0.2:
            analysis['entry_timing'] = 'late'
        else:
            analysis['entry_timing'] = 'bad'

        # Exit quality assessment
        if profit_pct > 0:
            if peak > 0 and profit_pct / peak > 0.5:
                analysis['exit_quality'] = 'good'
            elif peak > profit_pct * 2:
                analysis['exit_quality'] = 'too_early'
            else:
                analysis['exit_quality'] = 'ok'
        else:
            if price_after > 0.5:
                analysis['exit_quality'] = 'premature'
                analysis['should_have_held'] = True
            else:
                analysis['exit_quality'] = 'correct'

        # Should we have entered?
        if peak < 0.1:
            analysis['should_have_entered'] = False

        # Generate lesson
        lessons = []
        if analysis['entry_timing'] == 'late':
            lessons.append(f"Entered late — only {peak:.2f}% room")
        if analysis['entry_timing'] == 'bad':
            lessons.append(f"Bad entry — price barely moved ({peak:.2f}%)")
        if analysis['exit_quality'] == 'premature':
            lessons.append(f"Exited too early — price went to {price_after:+.2f}% after")
        if analysis['exit_quality'] == 'too_early':
            lessons.append(f"Left money on table — peaked at {peak:.2f}% but took {profit_pct:.2f}%")
        if profit_pct > 0 and analysis['exit_quality'] == 'good':
            lessons.append(f"Clean trade — good entry, good exit")
        if not analysis['should_have_entered']:
            lessons.append(f"Should have skipped — no real move")

        analysis['lesson'] = "; ".join(lessons) if lessons else "Normal trade"

    except Exception as e:
        analysis['lesson'] = f"Analysis error: {str(e)[:50]}"

    return analysis


def save_analysis(analysis):
    """Save trade analysis to database."""
    conn = get_db()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trade_analysis
                (trade_id, pair, open_date, close_date, entry_price, exit_price,
                 profit_pct, exit_reason, peak_profit_pct, worst_drawdown_pct,
                 price_after_1h_pct, entry_timing, exit_quality,
                 should_have_entered, should_have_held, lesson)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
        """, (
            analysis['trade_id'], analysis['pair'],
            analysis['open_date'], analysis['close_date'],
            analysis['entry_price'], analysis['exit_price'],
            analysis['profit_pct'], analysis['exit_reason'],
            analysis['peak_profit_pct'], analysis['worst_drawdown_pct'],
            analysis['price_after_1h_pct'], analysis['entry_timing'],
            analysis['exit_quality'], analysis['should_have_entered'],
            analysis['should_have_held'], analysis['lesson'],
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"Save analysis failed: {e}")
        try:
            conn.close()
        except:
            pass


def generate_report():
    """Generate learning report from all analyzed trades."""
    conn = get_db()
    if not conn:
        return None
    try:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM trade_analysis")
        total = cur.fetchone()[0]
        if total == 0:
            cur.close()
            conn.close()
            return None

        cur.execute("SELECT COUNT(*) FROM trade_analysis WHERE profit_pct > 0")
        wins = cur.fetchone()[0]

        cur.execute("SELECT AVG(profit_pct) FROM trade_analysis WHERE profit_pct > 0")
        avg_win = cur.fetchone()[0] or 0

        cur.execute("SELECT AVG(profit_pct) FROM trade_analysis WHERE profit_pct <= 0")
        avg_loss = cur.fetchone()[0] or 0

        cur.execute("SELECT AVG(peak_profit_pct) FROM trade_analysis WHERE profit_pct <= 0")
        avg_loss_peak = cur.fetchone()[0] or 0

        cur.execute("SELECT COUNT(*) FROM trade_analysis WHERE should_have_entered = false")
        bad_entries = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM trade_analysis WHERE should_have_held = true")
        premature_exits = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM trade_analysis WHERE entry_timing = 'late' OR entry_timing = 'bad'")
        late_entries = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM trade_analysis WHERE entry_timing = 'excellent'")
        excellent_entries = cur.fetchone()[0]

        cur.execute("""
            SELECT pair, COUNT(*) as trades,
                   SUM(CASE WHEN profit_pct > 0 THEN 1 ELSE 0 END) as wins,
                   ROUND(AVG(profit_pct)::numeric, 2) as avg_pnl
            FROM trade_analysis
            GROUP BY pair ORDER BY avg_pnl DESC
        """)
        pair_stats = cur.fetchall()

        cur.close()
        conn.close()

        win_rate = (wins / total * 100) if total > 0 else 0
        losses = total - wins

        msg = f"🧠 *Strategy Learning Report*\n"
        msg += f"━━━━━━━━━━━━━━━━━\n"
        msg += f"📊 {total} trades analyzed\n"
        msg += f"Win rate: {win_rate:.0f}% ({wins}W {losses}L)\n"
        msg += f"Avg win: {avg_win:+.2f}% | Avg loss: {avg_loss:+.2f}%\n\n"

        msg += f"*Entry Quality:*\n"
        msg += f"  🎯 Excellent entries: {excellent_entries}\n"
        msg += f"  ⏰ Late entries: {late_entries}\n"
        msg += f"  🚫 Should've skipped: {bad_entries}\n\n"

        msg += f"*Exit Quality:*\n"
        msg += f"  😤 Premature exits: {premature_exits}\n"
        msg += f"  📈 Avg peak before loss: {avg_loss_peak:+.2f}%\n\n"

        msg += f"*Best/Worst Pairs:*\n"
        for p in pair_stats[:3]:
            emoji = "🟢" if p[3] >= 0 else "🔴"
            msg += f"  {emoji} {p[0]}: {p[1]} trades, {p[2]}W, avg {p[3]:+.2f}%\n"
        if len(pair_stats) > 3:
            for p in pair_stats[-2:]:
                emoji = "🟢" if p[3] >= 0 else "🔴"
                msg += f"  {emoji} {p[0]}: {p[1]} trades, {p[2]}W, avg {p[3]:+.2f}%\n"

        # Improvement suggestions
        msg += f"\n*Suggestions:*\n"
        if late_entries > total * 0.3:
            msg += f"  ⚡ {late_entries} late entries — tighten freshness filter\n"
        if bad_entries > total * 0.2:
            msg += f"  🚫 {bad_entries} bad entries — add more filters\n"
        if premature_exits > total * 0.1:
            msg += f"  ⏳ {premature_exits} premature exits — widen stop or be patient\n"
        if avg_loss and abs(avg_loss) > avg_win * 1.5:
            msg += f"  ✂️ Losses ({avg_loss:.2f}%) too big vs wins ({avg_win:.2f}%) — tighten stop\n"
        if win_rate > 60 and avg_win > 0:
            msg += f"  ✅ Strategy is working — keep running\n"

        return msg

    except Exception as e:
        logger.warning(f"Report generation failed: {e}")
        return None


def main():
    ensure_tables()
    logger.info("Analyst started")
    send_tg("🧠 *Strategy Analyst online* — learning from every trade")

    import ccxt
    ex = ccxt.mexc({'enableRateLimit': True})

    analyzed_ids = set()
    last_report = ""

    while True:
        try:
            conn = get_db()
            if conn:
                cur = conn.cursor()

                # Get unanalyzed closed trades
                cur.execute("""
                    SELECT id, pair, open_rate, close_rate, close_profit,
                           exit_reason, open_date, close_date, stake_amount
                    FROM trades
                    WHERE close_date IS NOT NULL
                    ORDER BY close_date DESC LIMIT 20
                """)
                cols = [d[0] for d in cur.description]
                trades = [dict(zip(cols, row)) for row in cur.fetchall()]
                cur.close()
                conn.close()

                for trade in trades:
                    if trade['id'] not in analyzed_ids:
                        analysis = analyze_trade(trade, ex)
                        save_analysis(analysis)
                        analyzed_ids.add(trade['id'])

                        # Send immediate analysis for new trades
                        emoji = "✅" if analysis['profit_pct'] > 0 else "❌"
                        msg = (
                            f"{emoji} *Trade Analysis: {analysis['pair']}*\n"
                            f"P&L: {analysis['profit_pct']:+.2f}% | Exit: {analysis['exit_reason']}\n"
                            f"📈 Peak: {analysis['peak_profit_pct']:+.2f}% | 📉 Worst: {analysis['worst_drawdown_pct']:+.2f}%\n"
                            f"After exit: {analysis['price_after_1h_pct']:+.2f}%\n"
                            f"Entry: {analysis['entry_timing']} | Exit: {analysis['exit_quality']}\n"
                            f"💡 {analysis['lesson']}"
                        )
                        send_tg(msg)
                        logger.info(f"Analyzed trade {trade['id']}: {analysis['lesson']}")

                # Daily report at 9AM UTC
                now = datetime.now(timezone.utc)
                today = now.strftime("%Y-%m-%d")
                if now.hour == 9 and now.minute < 31 and today != last_report:
                    report = generate_report()
                    if report:
                        send_tg(report)
                        last_report = today

        except Exception as e:
            logger.error(f"Analyst error: {e}")

        time.sleep(1800)  # Check every 30 minutes


if __name__ == "__main__":
    main()
