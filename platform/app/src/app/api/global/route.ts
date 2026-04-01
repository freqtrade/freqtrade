import { NextResponse } from "next/server";
import { query } from "@/utils/db";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    // Latest bot status snapshot
    const statusResult = await query(
      `SELECT status, balance, open_trades, total_trades, total_profit, total_profit_pct, created_at
       FROM bot_status ORDER BY created_at DESC LIMIT 1`
    );

    // Recent closed trades (last 50)
    const tradesResult = await query(
      `SELECT trade_id, pair, side, open_time, close_time, open_rate, close_rate,
              stake_amount, profit_amount, profit_pct, exit_reason, entry_tag, duration_minutes, balance_after
       FROM trade_log ORDER BY close_time DESC LIMIT 50`
    );

    // Summary stats
    const summaryResult = await query(
      `SELECT COUNT(*) as total_trades,
              SUM(profit_amount) as total_profit,
              AVG(profit_pct) as avg_profit_pct,
              SUM(CASE WHEN profit_amount > 0 THEN 1 ELSE 0 END) as winning_trades,
              SUM(CASE WHEN profit_amount <= 0 THEN 1 ELSE 0 END) as losing_trades,
              AVG(duration_minutes) as avg_duration,
              MAX(close_time) as last_trade_time
       FROM trade_log`
    );

    // Today's trades
    const todayResult = await query(
      `SELECT COUNT(*) as trades_today,
              SUM(profit_amount) as profit_today,
              AVG(profit_pct) as avg_pct_today
       FROM trade_log WHERE close_time >= NOW() - INTERVAL '24 hours'`
    );

    const botStatus = statusResult.rows[0] || null;
    const trades = tradesResult.rows;
    const summary = summaryResult.rows[0] || {};
    const today = todayResult.rows[0] || {};

    const winRate = summary.total_trades > 0
      ? ((summary.winning_trades / summary.total_trades) * 100).toFixed(1)
      : "0";

    return NextResponse.json({
      bot: botStatus ? {
        status: botStatus.status,
        balance: parseFloat(botStatus.balance) || 0,
        openTrades: botStatus.open_trades || 0,
        totalTrades: botStatus.total_trades || 0,
        totalProfit: parseFloat(botStatus.total_profit) || 0,
        totalProfitPct: parseFloat(botStatus.total_profit_pct) || 0,
        lastUpdate: botStatus.created_at,
      } : null,
      summary: {
        totalTrades: parseInt(summary.total_trades) || 0,
        totalProfit: parseFloat(summary.total_profit) || 0,
        avgProfitPct: parseFloat(summary.avg_profit_pct) || 0,
        winRate: parseFloat(winRate),
        winningTrades: parseInt(summary.winning_trades) || 0,
        losingTrades: parseInt(summary.losing_trades) || 0,
        avgDuration: Math.round(parseFloat(summary.avg_duration) || 0),
        lastTradeTime: summary.last_trade_time || null,
      },
      today: {
        trades: parseInt(today.trades_today) || 0,
        profit: parseFloat(today.profit_today) || 0,
        avgPct: parseFloat(today.avg_pct_today) || 0,
      },
      recentTrades: trades.map((t: any) => ({
        id: t.trade_id,
        pair: t.pair,
        side: t.side,
        openTime: t.open_time,
        closeTime: t.close_time,
        openRate: parseFloat(t.open_rate),
        closeRate: parseFloat(t.close_rate),
        stake: parseFloat(t.stake_amount),
        profit: parseFloat(t.profit_amount),
        profitPct: parseFloat(t.profit_pct),
        exitReason: t.exit_reason,
        entryTag: t.entry_tag,
        duration: t.duration_minutes,
        balanceAfter: parseFloat(t.balance_after),
      })),
    });
  } catch (err: any) {
    console.error("Global stats error:", err);
    return NextResponse.json({
      bot: null,
      summary: { totalTrades: 0, totalProfit: 0, avgProfitPct: 0, winRate: 0, winningTrades: 0, losingTrades: 0, avgDuration: 0, lastTradeTime: null },
      today: { trades: 0, profit: 0, avgPct: 0 },
      recentTrades: [],
    });
  }
}
