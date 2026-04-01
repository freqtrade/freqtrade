"use client";

import { ConnectButton, useWallet } from "@/components/ConnectButton";
import { useEffect, useState } from "react";

function timeAgo(date: string | null) {
  if (!date) return "—";
  const s = Math.floor((Date.now() - new Date(date).getTime()) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function formatDuration(mins: number) {
  if (mins < 60) return `${mins}m`;
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

export default function GlobalDashboard() {
  const { connected } = useWallet();
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/global")
      .then((r) => r.json())
      .then((d) => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));

    // Auto-refresh every 60s
    const interval = setInterval(() => {
      fetch("/api/global").then((r) => r.json()).then(setData).catch(() => {});
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  const bot = data?.bot;
  const summary = data?.summary;
  const today = data?.today;
  const trades = data?.recentTrades || [];

  return (
    <div className="min-h-screen">
      <nav className="border-b border-dark-700 px-6 py-4 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span className="text-2xl">📈</span>
          <span className="text-xl font-bold text-white">TrendRider</span>
        </div>
        <div className="flex items-center gap-4">
          <a href="/dashboard" className="text-accent font-medium">Global</a>
          {connected && <a href="/my-bot" className="text-gray-400 hover:text-white transition">My Bot</a>}
          {connected && <a href="/config" className="text-gray-400 hover:text-white transition">Config</a>}
          <ConnectButton />
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white">Global Dashboard</h1>
            <p className="text-gray-500 text-sm mt-1">Live TrendRider bot performance — auto-refreshes every 60s</p>
          </div>
          {bot && (
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${bot.status === "running" ? "bg-accent/10 text-accent" : "bg-gray-500/10 text-gray-400"}`}>
              <span className={`w-2.5 h-2.5 rounded-full ${bot.status === "running" ? "bg-accent animate-pulse" : "bg-gray-500"}`} />
              {bot.status === "running" ? "Bot Running" : "Bot Offline"}
            </div>
          )}
        </div>

        {loading ? (
          <div className="text-center py-20 text-gray-500">Loading bot data...</div>
        ) : (
          <>
            {/* Bot Status Bar */}
            {bot && (
              <div className="bg-dark-800 rounded-xl p-6 flex flex-wrap items-center justify-between gap-4 border border-dark-700">
                <div>
                  <div className="text-sm text-gray-500">Balance</div>
                  <div className="text-2xl font-bold text-white">{bot.balance.toFixed(2)} USDT</div>
                </div>
                <div>
                  <div className="text-sm text-gray-500">Open Trades</div>
                  <div className="text-2xl font-bold text-white">{bot.openTrades}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-500">All-Time P&L</div>
                  <div className={`text-2xl font-bold ${bot.totalProfit >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {bot.totalProfit >= 0 ? "+" : ""}{bot.totalProfit.toFixed(4)} USDT
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-500">Last Update</div>
                  <div className="text-lg font-medium text-gray-300">{timeAgo(bot.lastUpdate)}</div>
                </div>
              </div>
            )}

            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-dark-800 rounded-xl p-5 border border-dark-700">
                <div className="text-sm text-gray-500">Total Trades</div>
                <div className="text-2xl font-bold text-white">{summary?.totalTrades || 0}</div>
              </div>
              <div className="bg-dark-800 rounded-xl p-5 border border-dark-700">
                <div className="text-sm text-gray-500">Win Rate</div>
                <div className="text-2xl font-bold text-accent">{summary?.winRate || 0}%</div>
              </div>
              <div className="bg-dark-800 rounded-xl p-5 border border-dark-700">
                <div className="text-sm text-gray-500">Avg Profit/Trade</div>
                <div className={`text-2xl font-bold ${(summary?.avgProfitPct || 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                  {(summary?.avgProfitPct || 0) >= 0 ? "+" : ""}{(summary?.avgProfitPct || 0).toFixed(2)}%
                </div>
              </div>
              <div className="bg-dark-800 rounded-xl p-5 border border-dark-700">
                <div className="text-sm text-gray-500">Avg Duration</div>
                <div className="text-2xl font-bold text-white">{formatDuration(summary?.avgDuration || 0)}</div>
              </div>
            </div>

            {/* Today's Stats */}
            <div className="bg-dark-800 rounded-xl p-6 border border-dark-700">
              <h2 className="text-lg font-semibold text-white mb-3">Last 24 Hours</h2>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <div className="text-sm text-gray-500">Trades</div>
                  <div className="text-xl font-bold text-white">{today?.trades || 0}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-500">P&L</div>
                  <div className={`text-xl font-bold ${(today?.profit || 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {(today?.profit || 0) >= 0 ? "+" : ""}{(today?.profit || 0).toFixed(4)} USDT
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-500">Avg Return</div>
                  <div className={`text-xl font-bold ${(today?.avgPct || 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {(today?.avgPct || 0) >= 0 ? "+" : ""}{(today?.avgPct || 0).toFixed(2)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Recent Trades Table */}
            <div className="bg-dark-800 rounded-xl p-6 border border-dark-700">
              <h2 className="text-lg font-semibold text-white mb-4">Recent Trades</h2>
              {trades.length === 0 ? (
                <div className="text-gray-500 text-center py-8">No trades recorded yet — bot will log here when trades close</div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-gray-500 text-left border-b border-dark-700">
                        <th className="pb-3 pr-4">Pair</th>
                        <th className="pb-3 pr-4">Opened</th>
                        <th className="pb-3 pr-4">Duration</th>
                        <th className="pb-3 pr-4">Stake</th>
                        <th className="pb-3 pr-4">Profit</th>
                        <th className="pb-3 pr-4">P&L %</th>
                        <th className="pb-3">Exit</th>
                      </tr>
                    </thead>
                    <tbody>
                      {trades.map((t: any, i: number) => (
                        <tr key={i} className="border-b border-dark-700/50 hover:bg-dark-700/30">
                          <td className="py-3 pr-4 font-medium text-white">{t.pair}</td>
                          <td className="py-3 pr-4 text-gray-400">{timeAgo(t.openTime)}</td>
                          <td className="py-3 pr-4 text-gray-400">{formatDuration(t.duration)}</td>
                          <td className="py-3 pr-4 text-gray-300">{t.stake.toFixed(2)}</td>
                          <td className={`py-3 pr-4 font-medium ${t.profit >= 0 ? "text-green-400" : "text-red-400"}`}>
                            {t.profit >= 0 ? "+" : ""}{t.profit.toFixed(4)}
                          </td>
                          <td className={`py-3 pr-4 font-medium ${t.profitPct >= 0 ? "text-green-400" : "text-red-400"}`}>
                            {t.profitPct >= 0 ? "+" : ""}{t.profitPct.toFixed(2)}%
                          </td>
                          <td className="py-3 text-gray-500 text-xs">{t.exitReason}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* CTA for non-connected users */}
            {!connected && (
              <div className="bg-dark-800 rounded-xl p-8 border border-accent/20 text-center space-y-4">
                <h2 className="text-xl font-bold text-white">Want your own bot?</h2>
                <p className="text-gray-400">Connect your wallet to set up and run TrendRider on your own MEXC account.</p>
                <ConnectButton />
              </div>
            )}
          </>
        )}
      </div>

      <footer className="border-t border-dark-700 px-6 py-6 text-center text-sm text-gray-600 mt-8">
        <p>TrendRider — Automated Crypto Trading | <a href="/terms" className="text-accent underline">Terms</a></p>
      </footer>
    </div>
  );
}
