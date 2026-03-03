/**
 * BacktestResultPage — displays a single backtest's results.
 *
 * Route: /backtest/:backtestId
 *
 * Features:
 * - Polls for completion if still running
 * - Summary metric cards (profit, win rate, max DD, Sharpe, trade count)
 * - Equity curve chart
 * - Interactive candlestick chart with trade markers
 * - Paginated trade list table
 */

import { useEffect, useState, useCallback, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  ArrowLeft,
  Loader2,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  BarChart3,
  CandlestickChart as CandlestickIcon,
  TrendingUp,
} from 'lucide-react';
import { api } from '../api/client';
import { MetricsCard } from '../components/MetricsCard';
import { EquityCurve } from '../components/EquityCurve';
import { CandlestickChart, parseOHLCVCandles } from '../components/CandlestickChart';
import type {
  BacktestResult,
  BacktestTrade,
  BacktestTradesResponse,
  PairInfo,
  OHLCVResponse,
} from '../types';
import type { Candle } from '../components/CandlestickChart';

const POLL_INTERVAL = 2000;
const TRADES_PER_PAGE = 50;

export function BacktestResultPage() {
  const { backtestId } = useParams<{ backtestId: string }>();
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Trades
  const [allTrades, setAllTrades] = useState<BacktestTrade[]>([]);
  const [tradesLoading, setTradesLoading] = useState(false);
  const [tradePage, setTradePage] = useState(0);
  const [totalTrades, setTotalTrades] = useState(0);

  // Chart data
  const [candles, setCandles] = useState<Candle[]>([]);
  const [chartPair, setChartPair] = useState<string>('');
  const [chartLoading, setChartLoading] = useState(false);
  const [availablePairs, setAvailablePairs] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<'equity' | 'chart' | 'trades'>('equity');

  // Fetch backtest result (with polling for running state)
  const fetchResult = useCallback(async () => {
    if (!backtestId) return;
    try {
      const r = await api.getBacktestResult(backtestId);
      setResult(r);
      setError(null);

      // Stop polling when done
      if (r.status === 'completed' || r.status === 'failed') {
        if (pollRef.current) {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [backtestId]);

  useEffect(() => {
    fetchResult();
    // Start polling
    pollRef.current = setInterval(fetchResult, POLL_INTERVAL);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [fetchResult]);

  // Load all trades once backtest is completed
  useEffect(() => {
    if (!backtestId || result?.status !== 'completed') return;

    async function loadTrades() {
      setTradesLoading(true);
      try {
        // Load all trades in batches
        const allT: BacktestTrade[] = [];
        let offset = 0;
        const batchSize = 500;
        let hasMore = true;

        while (hasMore) {
          const resp: BacktestTradesResponse = await api.getBacktestTrades(backtestId!, {
            offset,
            limit: batchSize,
          });
          allT.push(...resp.trades);
          setTotalTrades(resp.total);
          offset += batchSize;
          hasMore = offset < resp.total;
        }

        setAllTrades(allT);

        // Determine available pairs from trades
        const pairs = Array.from(new Set(allT.map((t) => t.pair))).sort();
        setAvailablePairs(pairs);
        if (pairs.length > 0 && !chartPair) {
          setChartPair(pairs[0]);
        }
      } catch (err) {
        console.error('Failed to load trades:', err);
      } finally {
        setTradesLoading(false);
      }
    }

    loadTrades();
  }, [backtestId, result?.status]);

  // Load OHLCV candles when chart pair changes
  useEffect(() => {
    if (!chartPair || !result?.result) return;

    async function loadCandles() {
      setChartLoading(true);
      try {
        const timeframe = (result!.result as Record<string, unknown>)?.timeframe as string || '5m';
        const exchange = (result!.result as Record<string, unknown>)?.exchange as string || 'binance';
        const resp: OHLCVResponse = await api.getOHLCV({
          pair: chartPair,
          timeframe,
          exchange,
          limit: 10000,
        });
        setCandles(parseOHLCVCandles(resp.candles));
      } catch (err) {
        console.error('Failed to load OHLCV data:', err);
        setCandles([]);
      } finally {
        setChartLoading(false);
      }
    }

    loadCandles();
  }, [chartPair, result]);

  // Paginated trades for the table
  const pagedTrades = allTrades.slice(
    tradePage * TRADES_PER_PAGE,
    (tradePage + 1) * TRADES_PER_PAGE
  );
  const totalPages = Math.ceil(allTrades.length / TRADES_PER_PAGE);

  // Trades filtered for the current chart pair
  const chartTrades = chartPair
    ? allTrades.filter((t) => t.pair === chartPair)
    : allTrades;

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16 text-gray-500 gap-2">
        <Loader2 className="w-5 h-5 animate-spin" /> Loading backtest...
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="card text-center py-16">
        <AlertCircle className="w-8 h-8 text-loss mx-auto mb-2" />
        <p className="text-loss mb-2">Failed to load backtest</p>
        <p className="text-xs text-gray-500">{error || 'Not found'}</p>
      </div>
    );
  }

  // Running / Pending state
  if (result.status === 'pending' || result.status === 'running') {
    return (
      <div className="space-y-6">
        <div>
          <Link to="/" className="text-sm text-gray-500 hover:text-gray-300 flex items-center gap-1 mb-1">
            <ArrowLeft className="w-3 h-3" /> Back
          </Link>
          <h1 className="text-xl font-bold text-gray-100 font-mono">{backtestId}</h1>
        </div>
        <div className="card text-center py-16">
          <Loader2 className="w-8 h-8 animate-spin text-accent mx-auto mb-3" />
          <p className="text-gray-300 mb-1">Backtest is {result.status}...</p>
          <div className="w-48 mx-auto bg-surface-2 rounded-full h-2 mt-3">
            <div
              className="bg-accent h-2 rounded-full transition-all duration-500"
              style={{ width: `${(result.progress || 0) * 100}%` }}
            />
          </div>
          <p className="text-xs text-gray-500 mt-2">
            {((result.progress || 0) * 100).toFixed(0)}% complete
          </p>
        </div>
      </div>
    );
  }

  // Failed state
  if (result.status === 'failed') {
    return (
      <div className="space-y-6">
        <div>
          <Link to="/" className="text-sm text-gray-500 hover:text-gray-300 flex items-center gap-1 mb-1">
            <ArrowLeft className="w-3 h-3" /> Back
          </Link>
          <h1 className="text-xl font-bold text-gray-100 font-mono">{backtestId}</h1>
        </div>
        <div className="card text-center py-16">
          <AlertCircle className="w-8 h-8 text-loss mx-auto mb-2" />
          <p className="text-loss mb-2">Backtest Failed</p>
          <p className="text-xs text-gray-400 font-mono">{result.error}</p>
        </div>
      </div>
    );
  }

  // Completed — extract result metrics
  const r = result.result || {};
  const totalProfit = (r.total_profit as number) ?? 0;
  const profitPct = (r.profit_percent as number) ?? 0;
  const numTrades = (r.total_trades as number) ?? 0;
  const winRate = (r.win_rate as number) ?? 0;
  const maxDrawdown = (r.max_drawdown as number) ?? 0;
  const sharpe = (r.sharpe_ratio as number) ?? 0;
  const sortino = (r.sortino_ratio as number) ?? 0;
  const profitFactor = (r.profit_factor as number) ?? 0;
  const avgProfit = (r.avg_profit as number) ?? 0;
  const avgDuration = (r.avg_duration as string) ?? '—';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <Link
          to="/"
          className="text-sm text-gray-500 hover:text-gray-300 flex items-center gap-1 mb-1"
        >
          <ArrowLeft className="w-3 h-3" /> Back
        </Link>
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold text-gray-100 font-mono">{backtestId}</h1>
          <span className="text-xs bg-profit/20 text-profit px-2 py-0.5 rounded-full">
            Completed
          </span>
        </div>
      </div>

      {/* Summary Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        <MetricsCard
          label="Total Profit"
          value={`${totalProfit > 0 ? '+' : ''}${totalProfit.toFixed(2)}`}
          trend={totalProfit >= 0 ? 'up' : 'down'}
        />
        <MetricsCard
          label="Profit %"
          value={`${profitPct > 0 ? '+' : ''}${profitPct.toFixed(1)}%`}
          trend={profitPct >= 0 ? 'up' : 'down'}
        />
        <MetricsCard
          label="Win Rate"
          value={`${(winRate * 100).toFixed(1)}%`}
        />
        <MetricsCard
          label="Max Drawdown"
          value={`${(maxDrawdown * 100).toFixed(1)}%`}
          trend="down"
        />
        <MetricsCard
          label="Trades"
          value={String(numTrades)}
        />
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricsCard label="Sharpe" value={sharpe.toFixed(2)} />
        <MetricsCard label="Sortino" value={sortino.toFixed(2)} />
        <MetricsCard label="Profit Factor" value={profitFactor.toFixed(2)} />
        <MetricsCard label="Avg Profit" value={`${avgProfit.toFixed(2)}%`} subtitle={`Avg Duration: ${avgDuration}`} />
      </div>

      {/* Tab navigation */}
      <div className="flex gap-1 border-b border-white/10 pb-px">
        {[
          { key: 'equity' as const, label: 'Equity Curve', icon: TrendingUp },
          { key: 'chart' as const, label: 'Price Chart', icon: CandlestickIcon },
          { key: 'trades' as const, label: 'Trades', icon: BarChart3 },
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`flex items-center gap-1.5 px-4 py-2 text-xs font-medium rounded-t-lg transition-colors ${
              activeTab === tab.key
                ? 'bg-surface-1 text-accent border-b-2 border-accent'
                : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            <tab.icon className="w-3.5 h-3.5" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'equity' && (
        <div className="card">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Equity Curve</h3>
          {tradesLoading ? (
            <div className="flex items-center justify-center py-8 text-gray-500 gap-2">
              <Loader2 className="w-4 h-4 animate-spin" /> Loading trades...
            </div>
          ) : (
            <EquityCurve trades={allTrades} height={350} />
          )}
        </div>
      )}

      {activeTab === 'chart' && (
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-300">Price Chart with Trades</h3>
            {availablePairs.length > 1 && (
              <select
                value={chartPair}
                onChange={(e) => setChartPair(e.target.value)}
                className="bg-surface-2 border border-white/10 rounded-lg px-2 py-1 text-xs text-gray-200 focus:outline-none focus:ring-1 focus:ring-accent/50"
              >
                {availablePairs.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            )}
          </div>
          {chartLoading ? (
            <div className="flex items-center justify-center py-16 text-gray-500 gap-2">
              <Loader2 className="w-4 h-4 animate-spin" /> Loading chart data...
            </div>
          ) : candles.length > 0 ? (
            <CandlestickChart
              candles={candles}
              trades={chartTrades}
              height={500}
            />
          ) : (
            <div className="text-center text-gray-500 text-xs py-16">
              No OHLCV data available for {chartPair}.
              <br />
              <span className="text-gray-600">Download data with: freqtrade download-data --pairs {chartPair}</span>
            </div>
          )}
        </div>
      )}

      {activeTab === 'trades' && (
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-300">
              Trade List <span className="text-gray-500 font-normal">({totalTrades || allTrades.length} trades)</span>
            </h3>
            {totalPages > 1 && (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setTradePage(Math.max(0, tradePage - 1))}
                  disabled={tradePage === 0}
                  className="p-1 text-gray-400 hover:text-gray-200 disabled:opacity-30"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <span className="text-xs text-gray-500">
                  {tradePage + 1} / {totalPages}
                </span>
                <button
                  onClick={() => setTradePage(Math.min(totalPages - 1, tradePage + 1))}
                  disabled={tradePage >= totalPages - 1}
                  className="p-1 text-gray-400 hover:text-gray-200 disabled:opacity-30"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>

          {tradesLoading ? (
            <div className="flex items-center justify-center py-8 text-gray-500 gap-2">
              <Loader2 className="w-4 h-4 animate-spin" /> Loading trades...
            </div>
          ) : allTrades.length === 0 ? (
            <p className="text-center text-gray-500 text-xs py-8">No trades</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-500 uppercase tracking-wider border-b border-white/5">
                    <th className="text-left py-2 px-2 font-medium">#</th>
                    <th className="text-left py-2 px-2 font-medium">Pair</th>
                    <th className="text-left py-2 px-2 font-medium">Type</th>
                    <th className="text-left py-2 px-2 font-medium">Open</th>
                    <th className="text-left py-2 px-2 font-medium">Close</th>
                    <th className="text-right py-2 px-2 font-medium">Profit %</th>
                    <th className="text-right py-2 px-2 font-medium">Profit $</th>
                    <th className="text-right py-2 px-2 font-medium">Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {pagedTrades.map((t, i) => {
                    const idx = tradePage * TRADES_PER_PAGE + i + 1;
                    const profitable = t.profit_ratio > 0;
                    return (
                      <tr key={idx} className="table-row border-b border-white/[0.03]">
                        <td className="py-1.5 px-2 text-gray-500">{idx}</td>
                        <td className="py-1.5 px-2 text-gray-300 font-mono">{t.pair}</td>
                        <td className="py-1.5 px-2">
                          <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                            t.is_short ? 'bg-violet-500/20 text-violet-400' : 'bg-blue-500/20 text-blue-400'
                          }`}>
                            {t.is_short ? 'SHORT' : 'LONG'}
                          </span>
                        </td>
                        <td className="py-1.5 px-2 text-gray-400 font-mono text-[10px]">
                          {formatTradeDate(t.open_date)}
                        </td>
                        <td className="py-1.5 px-2 text-gray-400 font-mono text-[10px]">
                          {formatTradeDate(t.close_date)}
                        </td>
                        <td className={`py-1.5 px-2 text-right font-mono ${profitable ? 'text-profit' : 'text-loss'}`}>
                          {(t.profit_ratio * 100).toFixed(2)}%
                        </td>
                        <td className={`py-1.5 px-2 text-right font-mono ${profitable ? 'text-profit' : 'text-loss'}`}>
                          {t.profit_abs.toFixed(2)}
                        </td>
                        <td className="py-1.5 px-2 text-right text-gray-400">
                          {formatDuration(t.trade_duration)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function formatTradeDate(dateStr: string): string {
  try {
    const d = new Date(dateStr);
    return d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return dateStr;
  }
}

function formatDuration(minutes: number): string {
  if (minutes < 60) return `${minutes}m`;
  if (minutes < 1440) return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
  return `${Math.floor(minutes / 1440)}d ${Math.floor((minutes % 1440) / 60)}h`;
}
