import { NextResponse } from "next/server";
import { query } from "@/utils/db";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    // Open trades (live positions)
    const openResult = await query(
      `SELECT id, pair, stake_amount, open_rate, open_date, is_short, strategy, enter_tag
       FROM trades WHERE is_open = true ORDER BY open_date DESC`
    );

    // Recent closed trades (last 50)
    const tradesResult = await query(
      `SELECT id, pair, stake_amount, open_rate, close_rate, close_profit, close_profit_abs,
              open_date, close_date, exit_reason, enter_tag, is_short, strategy,
              EXTRACT(EPOCH FROM (close_date - open_date))/60 as duration_minutes
       FROM trades WHERE is_open = false ORDER BY close_date DESC LIMIT 50`
    );

    // Summary stats (all closed trades)
    const summaryResult = await query(
      `SELECT COUNT(*) as total_trades,
              SUM(close_profit_abs) as total_profit,
              AVG(close_profit) * 100 as avg_profit_pct,
              SUM(CASE WHEN close_profit_abs > 0 THEN 1 ELSE 0 END) as winning_trades,
              SUM(CASE WHEN close_profit_abs <= 0 THEN 1 ELSE 0 END) as losing_trades,
              AVG(EXTRACT(EPOCH FROM (close_date - open_date))/60) as avg_duration,
              MAX(close_date) as last_trade_time
       FROM trades WHERE is_open = false`
    );

    // Last 24h trades
    const todayResult = await query(
      `SELECT COUNT(*) as trades_today,
              SUM(close_profit_abs) as profit_today,
              AVG(close_profit) * 100 as avg_pct_today
       FROM trades WHERE is_open = false AND close_date >= NOW() - INTERVAL '24 hours'`
    );

    const trades = tradesResult.rows;
    const summary = summaryResult.rows[0] || {};
    const today = todayResult.rows[0] || {};
    const openTrades = openResult.rows;

    const totalTrades = parseInt(summary.total_trades) || 0;
    const winningTrades = parseInt(summary.winning_trades) || 0;
    const winRate = totalTrades > 0
      ? ((winningTrades / totalTrades) * 100).toFixed(1)
      : "0";

    return NextResponse.json({
      bot: {
        status: openTrades.length > 0 ? "trading" : "idle",
        openTrades: openTrades.length,
        strategy: trades[0]?.strategy || openTrades[0]?.strategy || "TrendRider5m",
        lastTradeTime: summary.last_trade_time || null,
      },
      openPositions: openTrades.map((t: any) => ({
        id: t.id,
        pair: t.pair,
        stake: parseFloat(t.stake_amount),
        openRate: parseFloat(t.open_rate),
        openTime: t.open_date,
        side: t.is_short ? "short" : "long",
        entryTag: t.enter_tag,
      })),
      summary: {
        totalTrades,
        totalProfit: parseFloat(summary.total_profit) || 0,
        avgProfitPct: parseFloat(summary.avg_profit_pct) || 0,
        winRate: parseFloat(winRate),
        winningTrades,
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
        id: t.id,
        pair: t.pair,
        side: t.is_short ? "short" : "long",
        openTime: t.open_date,
        closeTime: t.close_date,
        openRate: parseFloat(t.open_rate),
        closeRate: parseFloat(t.close_rate),
        stake: parseFloat(t.stake_amount),
        profit: parseFloat(t.close_profit_abs),
        profitPct: (parseFloat(t.close_profit) * 100),
        exitReason: t.exit_reason,
        entryTag: t.enter_tag,
        duration: Math.round(parseFloat(t.duration_minutes)),
      })),
    });
  } catch (err: any) {
    console.error("Global stats error:", err);
    return NextResponse.json({
      bot: null,
      openPositions: [],
      summary: { totalTrades: 0, totalProfit: 0, avgProfitPct: 0, winRate: 0, winningTrades: 0, losingTrades: 0, avgDuration: 0, lastTradeTime: null },
      today: { trades: 0, profit: 0, avgPct: 0 },
      recentTrades: [],
    });
  }
}
