"use client";

import { ConnectButton, useWallet } from "@/components/ConnectButton";
import { useEffect, useMemo, useState } from "react";

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

function signalTone(label: string) {
  const value = label.toLowerCase();
  if (value.includes("profit") || value.includes("entry")) return "text-green-300 bg-green-400/10 border-green-400/30";
  if (value.includes("stop") || value.includes("loss")) return "text-red-300 bg-red-400/10 border-red-400/30";
  return "text-cyan-300 bg-cyan-400/10 border-cyan-400/30";
}

export default function GlobalDashboard() {
  const { connected } = useWallet();
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"signals" | "openTrades">("signals");

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
  const openPositions = data?.openPositions || [];

  const signals = useMemo(() => {
    const openSignals = openPositions.map((p: any) => ({
      id: `open-${p.id}`,
      pair: p.pair,
      side: p.side,
      signal: p.entryTag || "ENTRY_SIGNAL",
      stage: "LIVE_ENTRY",
      detail: `Opened ${timeAgo(p.openTime)} at ${p.openRate.toFixed(6)}`,
      time: p.openTime,
    }));

    const closeSignals = trades.slice(0, 20).map((t: any) => {
      const stage = t.profit >= 0 ? "CLOSE_PROFIT" : "CLOSE_STOP";
      const signal =
        t.exitReason?.replaceAll("_", " ") || (t.profit >= 0 ? "TAKE PROFIT" : "STOP LOSS");
      return {
        id: `close-${t.id}`,
        pair: t.pair,
        side: t.side,
        signal,
        stage,
        detail: `${t.profit >= 0 ? "+" : ""}${t.profit.toFixed(4)} USDT (${t.profitPct.toFixed(2)}%)`,
        time: t.closeTime,
      };
    });

    return [...openSignals, ...closeSignals]
      .sort((a: any, b: any) => new Date(b.time).getTime() - new Date(a.time).getTime())
      .slice(0, 30);
  }, [openPositions, trades]);

  const signalBreakdown = useMemo(() => {
    const stats: Record<string, number> = {};
    for (const sig of signals) {
      stats[sig.stage] = (stats[sig.stage] || 0) + 1;
    }
    return Object.entries(stats).sort((a, b) => b[1] - a[1]);
  }, [signals]);

  const cumulativeSeries = useMemo(() => {
    if (!trades.length) return [0];
    const points = trades.slice(0, 20).reverse();
    let running = 0;
    return points.map((t: any) => {
      running += Number(t.profit || 0);
      return running;
    });
  }, [trades]);

  const winCount = summary?.winningTrades || 0;
  const lossCount = summary?.losingTrades || 0;
  const totalForPie = Math.max(1, winCount + lossCount);
  const winPercent = Math.round((winCount / totalForPie) * 100);
  const pieStyle = {
    background: `conic-gradient(#10b981 0 ${winPercent}%, #f87171 ${winPercent}% 100%)`,
  };

  const chartWidth = 460;
  const chartHeight = 140;
  const maxY = Math.max(...cumulativeSeries, 1);
  const minY = Math.min(...cumulativeSeries, 0);
  const spanY = Math.max(1, maxY - minY);
  const points = cumulativeSeries
    .map((value, index) => {
      const x = (index / Math.max(1, cumulativeSeries.length - 1)) * chartWidth;
      const y = chartHeight - ((value - minY) / spanY) * chartHeight;
      return `${x},${y}`;
    })
    .join(" ");

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
            <p className="text-gray-500 text-sm mt-1">Live global bot monitor with auto-refresh every 60s</p>
          </div>
          {bot && (
            <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${bot.openTrades > 0 ? "bg-accent/10 text-accent" : "bg-gray-500/10 text-gray-400"}`}>
              <span className={`w-2.5 h-2.5 rounded-full ${bot.openTrades > 0 ? "bg-accent animate-pulse" : "bg-gray-500"}`} />
              {bot.openTrades > 0 ? `Trading (${bot.openTrades} open)` : "Idle"}
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
                  <div className="text-sm text-gray-500">Strategy</div>
                  <div className="text-2xl font-bold text-white">{bot.strategy}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-500">Open Trades</div>
                  <div className="text-2xl font-bold text-white">{bot.openTrades}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-500">All-Time P&L</div>
                  <div className={`text-2xl font-bold ${(summary?.totalProfit || 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {(summary?.totalProfit || 0) >= 0 ? "+" : ""}{(summary?.totalProfit || 0).toFixed(4)} USDT
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-500">Last Trade</div>
                  <div className="text-lg font-medium text-gray-300">{timeAgo(bot.lastTradeTime)}</div>
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

            {/* Interactive panel */}
            <div className="rounded-2xl border border-dark-700 bg-gradient-to-br from-dark-800 via-dark-800 to-dark-700 p-2">
              <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
                <div className="rounded-xl border border-dark-700/80 bg-dark-900/70 p-5">
                  <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
                    <h2 className="text-lg font-semibold text-white">Global Feed</h2>
                    <div className="flex rounded-xl bg-dark-700 p-1">
                      <button
                        className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
                          activeTab === "signals"
                            ? "bg-accent text-dark-900 shadow-[0_0_20px_rgba(0,212,170,0.35)]"
                            : "text-gray-300 hover:text-white"
                        }`}
                        onClick={() => setActiveTab("signals")}
                      >
                        Signals ({signals.length})
                      </button>
                      <button
                        className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
                          activeTab === "openTrades"
                            ? "bg-accent text-dark-900 shadow-[0_0_20px_rgba(0,212,170,0.35)]"
                            : "text-gray-300 hover:text-white"
                        }`}
                        onClick={() => setActiveTab("openTrades")}
                      >
                        Open Trades ({openPositions.length})
                      </button>
                    </div>
                  </div>

                  {activeTab === "signals" && (
                    <div className="space-y-4">
                      {signals.length === 0 ? (
                        <div className="rounded-xl border border-dark-700 p-8 text-center text-gray-500">No global signals yet.</div>
                      ) : (
                        <div className="grid gap-3">
                          {signals.map((sig: any) => (
                            <div
                              key={sig.id}
                              className="group rounded-xl border border-dark-700 bg-dark-800/60 p-4 transition hover:translate-x-1 hover:border-accent/30 hover:bg-dark-700/60"
                            >
                              <div className="flex flex-wrap items-center justify-between gap-3">
                                <div className="flex items-center gap-2">
                                  <span className="rounded-full bg-accent/10 px-2 py-1 text-xs font-semibold text-accent">{sig.pair}</span>
                                  <span className={`rounded-full border px-2 py-1 text-xs font-medium ${signalTone(sig.signal)}`}>
                                    {sig.signal}
                                  </span>
                                </div>
                                <div className="text-xs text-gray-500">{timeAgo(sig.time)}</div>
                              </div>
                              <div className="mt-2 text-sm font-medium text-white">{sig.stage}</div>
                              <div className="mt-1 text-sm text-gray-400">{sig.detail}</div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {activeTab === "openTrades" && (
                    <div>
                      {openPositions.length === 0 ? (
                        <div className="rounded-xl border border-dark-700 p-8 text-center text-gray-500">No open global trades right now.</div>
                      ) : (
                        <div className="overflow-x-auto">
                          <table className="w-full text-sm">
                            <thead>
                              <tr className="text-left text-gray-500 border-b border-dark-700">
                                <th className="pb-3 pr-4">Pair</th>
                                <th className="pb-3 pr-4">Side</th>
                                <th className="pb-3 pr-4">Stake</th>
                                <th className="pb-3 pr-4">Entry Price</th>
                                <th className="pb-3 pr-4">Opened</th>
                                <th className="pb-3">Signal</th>
                              </tr>
                            </thead>
                            <tbody>
                              {openPositions.map((p: any) => (
                                <tr key={p.id} className="border-b border-dark-700/50 hover:bg-dark-700/30 transition">
                                  <td className="py-3 pr-4 font-medium text-white">{p.pair}</td>
                                  <td className="py-3 pr-4">
                                    <span
                                      className={`px-2 py-0.5 rounded text-xs font-medium ${
                                        p.side === "long" ? "bg-green-400/10 text-green-400" : "bg-red-400/10 text-red-400"
                                      }`}
                                    >
                                      {p.side}
                                    </span>
                                  </td>
                                  <td className="py-3 pr-4 text-gray-300">{p.stake.toFixed(2)} USDT</td>
                                  <td className="py-3 pr-4 text-gray-300">{p.openRate.toFixed(6)}</td>
                                  <td className="py-3 pr-4 text-gray-400">{timeAgo(p.openTime)}</td>
                                  <td className="py-3 text-xs text-gray-500">{p.entryTag || "ENTRY_SIGNAL"}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <div className="space-y-4">
                  <div className="rounded-xl border border-dark-700 bg-dark-900/70 p-5">
                    <h3 className="text-sm font-semibold uppercase tracking-wide text-gray-400">Win Split</h3>
                    <div className="mt-4 flex items-center gap-4">
                      <div className="relative h-24 w-24 rounded-full" style={pieStyle}>
                        <div className="absolute inset-3 rounded-full bg-dark-900" />
                      </div>
                      <div className="space-y-2 text-sm">
                        <div className="text-green-300">Wins: {winCount}</div>
                        <div className="text-red-300">Losses: {lossCount}</div>
                        <div className="text-gray-400">Win rate: {summary?.winRate || 0}%</div>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-xl border border-dark-700 bg-dark-900/70 p-5">
                    <h3 className="text-sm font-semibold uppercase tracking-wide text-gray-400">Signal Mix</h3>
                    <div className="mt-4 space-y-3">
                      {signalBreakdown.length === 0 && <div className="text-sm text-gray-500">No signals to classify.</div>}
                      {signalBreakdown.map(([label, count]) => {
                        const width = Math.round((count / Math.max(1, signals.length)) * 100);
                        return (
                          <div key={label}>
                            <div className="mb-1 flex justify-between text-xs text-gray-400">
                              <span>{label}</span>
                              <span>{count}</span>
                            </div>
                            <div className="h-2 rounded-full bg-dark-700">
                              <div className="h-2 rounded-full bg-gradient-to-r from-cyan-400 via-accent to-green-300 transition-all" style={{ width: `${width}%` }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
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

            {/* Performance Graph */}
            <div className="bg-dark-800 rounded-xl p-6 border border-dark-700">
              <h2 className="text-lg font-semibold text-white mb-4">Recent Performance Curve</h2>
              <div className="w-full overflow-x-auto">
                <svg viewBox={`0 0 ${chartWidth} ${chartHeight + 20}`} className="h-40 w-full min-w-[360px]">
                  <line x1="0" y1={chartHeight} x2={chartWidth} y2={chartHeight} stroke="#2c2d3d" strokeWidth="1" />
                  <polyline fill="none" stroke="#00d4aa" strokeWidth="3" points={points} />
                </svg>
              </div>
              <p className="mt-2 text-xs text-gray-500">Cumulative P&L from the most recent closed trades.</p>
            </div>

            {/* Recent Trades Table */}
            <div className="bg-dark-800 rounded-xl p-6 border border-dark-700">
              <h2 className="text-lg font-semibold text-white mb-4">Recent Trades</h2>
              {trades.length === 0 ? (
                <div className="text-gray-500 text-center py-8">No trades recorded yet - bot will log here when trades close.</div>
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
