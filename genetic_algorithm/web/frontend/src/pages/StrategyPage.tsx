import { useEffect, useState, useRef, useCallback } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { ArrowLeft, Code, Copy, Check, Shield, AlertTriangle, Syringe, Download, GitBranch, Play, Loader2, BarChart3, ExternalLink, TrendingUp, Zap, Square } from 'lucide-react';
import { api } from '../api/client';
import { useStore } from '../store/useStore';
import { LoadingState, ErrorState } from '../components/StateDisplays';
import { MetricsCard } from '../components/MetricsCard';
import { StrategyGeneTree } from '../components/StrategyGeneTree';
import { CandlestickChart, parseOHLCVCandles } from '../components/CandlestickChart';
import type { StrategyDetail, RunSummary, PairInfo, OHLCVResponse, BacktestTrade, BacktestTradesResponse, LineageNode } from '../types';
import type { Candle, IndicatorLine } from '../components/CandlestickChart';
import { LineageChart } from '../components/LineageChart';

export function StrategyPage() {
  const { runId, strategyId } = useParams<{ runId: string; strategyId: string }>();
  const [strategy, setStrategy] = useState<StrategyDetail | null>(null);
  const [code, setCode] = useState<string | null>(null);
  const [showCode, setShowCode] = useState(false);
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [injecting, setInjecting] = useState(false);
  const [injectResult, setInjectResult] = useState<string | null>(null);
  const [showInjectMenu, setShowInjectMenu] = useState(false);
  // Backtest launch state
  const [showBacktest, setShowBacktest] = useState(false);
  const [btTimerange, setBtTimerange] = useState('20250101-20250401');
  const [btPairs, setBtPairs] = useState('');
  const [btTimeframe, setBtTimeframe] = useState('');
  const [btExchange, setBtExchange] = useState('binance');
  const [btRunning, setBtRunning] = useState(false);
  const [btError, setBtError] = useState<string | null>(null);

  // OHLCV Price Data chart state
  const [availablePairs, setAvailablePairs] = useState<PairInfo[]>([]);
  const [selectedPair, setSelectedPair] = useState('');
  const [selectedTimeframe, setSelectedTimeframe] = useState('');
  const [candles, setCandles] = useState<Candle[]>([]);
  const [chartLoading, setChartLoading] = useState(false);

  // Inline backtest result + trade markers
  const [lastBacktestId, setLastBacktestId] = useState<string | null>(null);
  const [backtestTrades, setBacktestTrades] = useState<BacktestTrade[]>([]);
  const [btStatus, setBtStatus] = useState<string | null>(null);
  const [btResult, setBtResult] = useState<Record<string, unknown> | null>(null);
  const btPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Lineage timeline
  const [lineage, setLineage] = useState<LineageNode[] | null>(null);

  // Indicator overlays for the chart
  const [indicatorLines, setIndicatorLines] = useState<IndicatorLine[]>([]);
  const [indicatorsLoading, setIndicatorsLoading] = useState(false);

  // Dry-run state
  const [showDryRun, setShowDryRun] = useState(false);
  const [drExchange, setDrExchange] = useState('binance');
  const [drPairs, setDrPairs] = useState('');
  const [drStake, setDrStake] = useState('100');
  const [drTimeframe, setDrTimeframe] = useState('');
  const [drRunning, setDrRunning] = useState(false);
  const [drError, setDrError] = useState<string | null>(null);
  const navigate = useNavigate();
  const runsMap = useStore((s) => s.runs);
  const activeRuns = Array.from(runsMap.values()).filter(
    (r) => r.status === 'running' || r.status === 'paused',
  );

  useEffect(() => {
    if (!runId || !strategyId) return;
    setLoading(true);
    api
      .getStrategy(runId, strategyId)
      .then((s) => { setStrategy(s); setError(null); })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [runId, strategyId]);

  // Load available trading pairs on mount
  useEffect(() => {
    api.listPairs().then((resp) => {
      setAvailablePairs(resp.pairs);
    }).catch(() => {});
  }, []);

  // Load lineage on mount
  useEffect(() => {
    if (!runId || !strategyId) return;
    api.getLineage(runId, strategyId).then((resp) => {
      setLineage(resp.chain);
    }).catch(() => {});
  }, [runId, strategyId]);

  // Load OHLCV when pair + timeframe selected
  useEffect(() => {
    if (!selectedPair || !selectedTimeframe) return;
    setChartLoading(true);
    api.getOHLCV({
      pair: selectedPair,
      timeframe: selectedTimeframe,
      exchange: btExchange,
      limit: 10000,
    }).then((resp) => {
      setCandles(parseOHLCVCandles(resp.candles));
    }).catch(() => {
      setCandles([]);
    }).finally(() => setChartLoading(false));
  }, [selectedPair, selectedTimeframe, btExchange]);

  // Load indicator overlays when chart data + strategy gene are available
  useEffect(() => {
    if (!selectedPair || !selectedTimeframe || candles.length === 0 || !strategy?.gene) return;
    const indicators = (strategy.gene.indicators as Array<{ type: string; parameters: Record<string, number> }>) ?? [];
    if (indicators.length === 0) { setIndicatorLines([]); return; }

    // Build indicator request strings: e.g. "EMA_20", "RSI_14"
    const indKeys = indicators.map((ind) => {
      const period = ind.parameters?.period ?? ind.parameters?.timeperiod ?? '';
      return period ? `${ind.type}_${period}` : ind.type;
    });
    // Deduplicate
    const unique = [...new Set(indKeys)];

    setIndicatorsLoading(true);
    api.getIndicators({
      pair: selectedPair,
      timeframe: selectedTimeframe,
      indicators: unique.join(','),
      exchange: btExchange,
    }).then((resp) => {
      const lines: IndicatorLine[] = Object.entries(resp.indicators).map(([name, data]) => ({
        name,
        data: data.values.map(([ts, val]: [number, number]) => ({
          time: Math.floor(ts / 1000),
          value: val,
        })),
        pane: data.pane as 'price' | 'separate',
      }));
      setIndicatorLines(lines);
    }).catch(() => {
      setIndicatorLines([]);
    }).finally(() => setIndicatorsLoading(false));
  }, [selectedPair, selectedTimeframe, btExchange, candles.length, strategy?.gene]);

  // Poll for backtest completion and load trades
  useEffect(() => {
    if (!lastBacktestId) return;
    const poll = async () => {
      try {
        const r = await api.getBacktestResult(lastBacktestId);
        setBtStatus(r.status);
        if (r.status === 'completed' || r.status === 'failed') {
          if (btPollRef.current) { clearInterval(btPollRef.current); btPollRef.current = null; }
          if (r.status === 'completed') {
            setBtResult(r.result);
            // Load all trades
            const allT: BacktestTrade[] = [];
            let offset = 0;
            let hasMore = true;
            while (hasMore) {
              const resp: BacktestTradesResponse = await api.getBacktestTrades(lastBacktestId, { offset, limit: 500 });
              allT.push(...resp.trades);
              offset += 500;
              hasMore = offset < resp.total;
            }
            setBacktestTrades(allT);
          } else if (r.error) {
            setBtResult({ error_message: r.error });
          }
        }
      } catch { /* ignore */ }
    };
    poll();
    btPollRef.current = setInterval(poll, 2000);
    return () => { if (btPollRef.current) clearInterval(btPollRef.current); };
  }, [lastBacktestId]);

  // Auto-select first traded pair on chart when backtest trades arrive
  const chartDetailsRef = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    if (backtestTrades.length === 0) return;
    // Auto-select the first traded pair if chart pair is not set
    const firstPair = backtestTrades[0]?.pair;
    if (firstPair && !selectedPair) {
      setSelectedPair(firstPair);
    }
    // Auto-select timeframe from strategy gene if not set
    if (!selectedTimeframe && strategy?.gene?.timeframe) {
      setSelectedTimeframe(strategy.gene.timeframe);
    }
    // Auto-open the chart details section
    if (chartDetailsRef.current && !chartDetailsRef.current.open) {
      chartDetailsRef.current.open = true;
    }
  }, [backtestTrades]);

  const loadCode = async () => {
    if (!runId || !strategyId) return;
    try {
      const { code: c } = await api.getStrategyCode(runId, strategyId);
      setCode(c);
      setShowCode(true);
    } catch (err) {
      console.error(err);
    }
  };

  const copyCode = () => {
    if (code) {
      navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const downloadCode = () => {
    if (!code) return;
    const blob = new Blob([code], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `strategy_${strategyId}.py`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleInject = async (targetRunId: string) => {
    if (!strategy?.gene) return;
    setInjecting(true);
    try {
      await api.injectStrategy(targetRunId, {
        strategy_gene: strategy.gene as unknown as Record<string, unknown>,
        source_description: `Strategy ${strategy.id} from run ${runId}`,
      });
      setInjectResult(`Injected into ${targetRunId}`);
      setShowInjectMenu(false);
      setTimeout(() => setInjectResult(null), 3000);
    } catch (err) {
      setInjectResult(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setInjecting(false);
    }
  };

  if (loading) return <LoadingState message="Loading strategy..." />;
  if (error || !strategy) {
    return <ErrorState title="Failed to load strategy" message={error || 'Not found'} />;
  }

  const q = strategy.quality;

  // Helper to safely read numeric metrics from Record<string, unknown>
  const m = (key: string): number | undefined => {
    const v = strategy.metrics[key];
    return typeof v === 'number' ? v : undefined;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <Link
          to={`/runs/${runId}`}
          className="text-sm text-gray-500 hover:text-gray-300 flex items-center gap-1 mb-1"
        >
          <ArrowLeft className="w-3 h-3" /> Back to run
        </Link>
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold text-gray-100 font-mono">{strategy.id}</h1>
          <span className="text-xs text-gray-500">Gen {strategy.generation}</span>
        </div>
        {/* Action buttons */}
        <div className="flex items-center gap-2 mt-2 flex-wrap">
          {activeRuns.length > 0 && (
            <div className="relative">
              <button
                onClick={() => setShowInjectMenu(!showInjectMenu)}
                disabled={injecting}
                className="flex items-center gap-1.5 text-xs bg-accent/10 text-accent border border-accent/20 px-3 py-1.5 rounded-lg hover:bg-accent/20 transition-colors disabled:opacity-50"
              >
                <Syringe className="w-3 h-3" />
                {injecting ? 'Injecting...' : 'Inject into Run'}
              </button>
              {showInjectMenu && (
                <div className="absolute top-full left-0 mt-1 bg-surface-1 border border-white/10 rounded-lg shadow-xl z-10 min-w-[200px]">
                  {activeRuns.map((run) => (
                    <button
                      key={run.run_id}
                      onClick={() => handleInject(run.run_id)}
                      className="w-full text-left px-3 py-2 text-xs text-gray-300 hover:bg-white/[0.05] transition-colors first:rounded-t-lg last:rounded-b-lg"
                    >
                      <span className="font-mono">{run.run_id}</span>
                      <span className="text-gray-500 ml-2">Gen {run.current_generation}/{run.total_generations}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          {injectResult && (
            <span className={`text-xs ${injectResult.startsWith('Error') ? 'text-loss' : 'text-profit'}`}>
              {injectResult}
            </span>
          )}
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricsCard
          label="Fitness"
          value={strategy.fitness?.toFixed(4) ?? '—'}
          trend={strategy.fitness !== null ? 'up' : undefined}
        />
        <MetricsCard
          label="Profit"
          value={
            m('profit') !== undefined
              ? `${m('profit')! > 0 ? '+' : ''}${m('profit')!.toFixed(1)}%`
              : '—'
          }
          trend={
            m('profit') !== undefined
              ? m('profit')! >= 0
                ? 'up'
                : 'down'
              : undefined
          }
        />
        <MetricsCard
          label="Sharpe Ratio"
          value={m('sharpe_ratio')?.toFixed(2) ?? '—'}
        />
        <MetricsCard
          label="Win Rate"
          value={
            m('win_rate') !== undefined
              ? `${(m('win_rate')! * 100).toFixed(1)}%`
              : '—'
          }
        />
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricsCard
          label="Trades"
          value={m('num_trades')?.toFixed(0) ?? '—'}
        />
        <MetricsCard
          label="Max Drawdown"
          value={
            m('max_drawdown') !== undefined
              ? `${(m('max_drawdown')! * 100).toFixed(1)}%`
              : '—'
          }
          trend={m('max_drawdown') !== undefined ? 'down' : undefined}
        />
        <MetricsCard
          label="Profit Factor"
          value={m('profit_factor')?.toFixed(2) ?? '—'}
        />
        <MetricsCard
          label="Sortino"
          value={m('sortino_ratio')?.toFixed(2) ?? '—'}
        />
      </div>

      {/* Quality Assessment */}
      {q && (
        <div className="card">
          <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
            <Shield className="w-4 h-4" /> Quality Assessment
          </h3>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <QualityItem
              label="Holdout"
              value={q.holdout_degradation !== null ? `${(q.holdout_degradation * 100).toFixed(1)}%` : '—'}
              rating={q.holdout_label}
            />
            <QualityItem
              label="Walk-Forward"
              value={q.wf_gap !== null ? `${(q.wf_gap * 100).toFixed(1)}%` : '—'}
              rating={q.wf_label}
            />
            <QualityItem
              label="Monte Carlo"
              value={q.mc_robustness !== null ? `${(q.mc_robustness * 100).toFixed(0)}%` : '—'}
              rating={q.mc_label}
            />
            <QualityItem
              label="Overall"
              value={q.composite_score !== null ? q.composite_score.toFixed(2) : '—'}
              rating={q.overall_label}
              highlight
            />
          </div>
        </div>
      )}

      {/* Lineage */}
      {(strategy.parent_ids.length > 0 || strategy.mutations.length > 0) && (
        <div className="card">
          <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
            <GitBranch className="w-4 h-4" /> Lineage
          </h3>
          <div className="space-y-2">
            {strategy.parent_ids.length > 0 && (
              <div>
                <span className="text-[10px] text-gray-500 uppercase">Parents</span>
                <div className="flex flex-wrap gap-1.5 mt-1">
                  {strategy.parent_ids.map((pid) => (
                    <Link
                      key={pid}
                      to={`/runs/${runId}/strategies/${pid}`}
                      className="text-xs font-mono text-accent hover:underline bg-accent/10 px-2 py-0.5 rounded"
                    >
                      {pid}
                    </Link>
                  ))}
                </div>
              </div>
            )}
            {strategy.mutations.length > 0 && (
              <div>
                <span className="text-[10px] text-gray-500 uppercase">Mutations Applied</span>
                <div className="flex flex-wrap gap-1.5 mt-1">
                  {strategy.mutations.map((m, i) => (
                    <span key={i} className="text-xs bg-yellow-500/10 text-yellow-400 px-2 py-0.5 rounded">
                      {m}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Walk-Forward Detail */}

      {/* Fitness Evolution Timeline */}
      {lineage && lineage.length > 1 && (
        <div className="card">
          <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
            <TrendingUp className="w-4 h-4" /> Fitness Evolution
          </h3>
          <p className="text-[10px] text-gray-500 mb-2">
            Tracing lineage back through {lineage.length} generations via parent chain
          </p>
          <LineageChart chain={lineage} />
        </div>
      )}

      {/* Walk-Forward Detail */}
      {strategy.walk_forward_windows && strategy.walk_forward_windows.length > 0 && (
        <details className="card group">
          <summary className="text-sm font-medium text-gray-300 cursor-pointer select-none flex items-center gap-2">
            Walk-Forward Windows
            <span className="text-xs text-gray-500 group-open:hidden">
              ({strategy.walk_forward_windows.length} windows — click to expand)
            </span>
          </summary>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-500 uppercase tracking-wider border-b border-white/5">
                  <th className="text-left py-1.5 px-2 font-medium">#</th>
                  <th className="text-left py-1.5 px-2 font-medium">Period</th>
                  <th className="text-right py-1.5 px-2 font-medium">Train Profit</th>
                  <th className="text-right py-1.5 px-2 font-medium">Test Profit</th>
                  <th className="text-right py-1.5 px-2 font-medium">Degradation</th>
                </tr>
              </thead>
              <tbody>
                {strategy.walk_forward_windows.map((w, i) => {
                  const trainProfit = (w.train_profit ?? w.train_result ?? 0) as number;
                  const testProfit = (w.test_profit ?? w.test_result ?? 0) as number;
                  const degradation = trainProfit !== 0 ? ((trainProfit - testProfit) / Math.abs(trainProfit)) : 0;
                  return (
                    <tr key={i} className="table-row">
                      <td className="py-1.5 px-2 text-gray-500">{i + 1}</td>
                      <td className="py-1.5 px-2 text-gray-400 font-mono">
                        {(w.period as string) || (w.start as string) || `Window ${i + 1}`}
                      </td>
                      <td className={`py-1.5 px-2 text-right font-mono ${trainProfit >= 0 ? 'text-profit' : 'text-loss'}`}>
                        {trainProfit.toFixed(1)}%
                      </td>
                      <td className={`py-1.5 px-2 text-right font-mono ${testProfit >= 0 ? 'text-profit' : 'text-loss'}`}>
                        {testProfit.toFixed(1)}%
                      </td>
                      <td className={`py-1.5 px-2 text-right font-mono ${degradation > 0.3 ? 'text-loss' : 'text-gray-400'}`}>
                        {(degradation * 100).toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </details>
      )}

      {/* Monte Carlo Detail */}
      {strategy.monte_carlo && Object.keys(strategy.monte_carlo).length > 0 && (
        <details className="card group">
          <summary className="text-sm font-medium text-gray-300 cursor-pointer select-none flex items-center gap-2">
            Monte Carlo Simulation
            <span className="text-xs text-gray-500 group-open:hidden">(click to expand)</span>
          </summary>
          <div className="mt-3 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
            {Object.entries(strategy.monte_carlo)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([key, val]) => (
                <div key={key} className="bg-surface-2 rounded-lg px-3 py-2">
                  <div className="text-[10px] text-gray-500 truncate">{key}</div>
                  <div className="text-sm font-mono text-gray-300">
                    {typeof val === 'number' ? val.toFixed(4) : String(val)}
                  </div>
                </div>
              ))}
          </div>
        </details>
      )}

      {/* Strategy Gene Tree */}
      {strategy.gene && <StrategyGeneTree gene={strategy.gene} />}

      {/* Price Data / OHLCV Chart */}
      <details ref={chartDetailsRef} className="card group">
        <summary className="text-sm font-medium text-gray-300 cursor-pointer select-none flex items-center gap-2">
          <BarChart3 className="w-4 h-4" /> Price Data
          <span className="text-xs text-gray-500 group-open:hidden">(click to expand chart)</span>
        </summary>
        <div className="mt-3 space-y-3">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div>
              <label className="text-[10px] text-gray-500 uppercase block mb-1">Pair</label>
              <select
                value={selectedPair}
                onChange={(e) => setSelectedPair(e.target.value)}
                className="w-full bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
              >
                <option value="">Select pair...</option>
                {[...new Set(availablePairs.map((p) => p.pair))].sort().map((pair) => (
                  <option key={pair} value={pair}>{pair}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-[10px] text-gray-500 uppercase block mb-1">Timeframe</label>
              <select
                value={selectedTimeframe}
                onChange={(e) => setSelectedTimeframe(e.target.value)}
                className="w-full bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
              >
                <option value="">Select timeframe...</option>
                {[...new Set(
                  availablePairs
                    .filter((p) => !selectedPair || p.pair === selectedPair)
                    .map((p) => p.timeframe)
                )].sort().map((tf) => (
                  <option key={tf} value={tf}>{tf}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-[10px] text-gray-500 uppercase block mb-1">Exchange</label>
              <select
                value={btExchange}
                onChange={(e) => setBtExchange(e.target.value)}
                className="w-full bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
              >
                {[...new Set(availablePairs.map((p) => p.exchange))].sort().map((ex) => (
                  <option key={ex} value={ex}>{ex}</option>
                ))}
                {availablePairs.length === 0 && <option value="binance">binance</option>}
              </select>
            </div>
          </div>

          {chartLoading && (
            <div className="flex items-center justify-center py-8 text-gray-500 gap-2">
              <Loader2 className="w-4 h-4 animate-spin" /> Loading chart...
            </div>
          )}

          {!chartLoading && candles.length > 0 && (
            <>
              <CandlestickChart
                candles={candles}
                trades={backtestTrades.filter((t) => t.pair === selectedPair)}
                indicators={indicatorLines}
                height={400}
                onTimeRangeSelect={(start, end) => {
                  setBtTimerange(`${start}-${end}`);
                  setShowBacktest(true);
                }}
              />
              {btStatus === 'completed' && backtestTrades.length > 0 && (
                <div className="flex items-center gap-2 text-xs text-gray-400">
                  <Check className="w-3 h-3 text-profit" />
                  Showing {backtestTrades.filter((t) => t.pair === selectedPair).length} trade markers for {selectedPair}
                  {lastBacktestId && (
                    <Link to={`/backtest/${lastBacktestId}`} className="text-accent hover:underline flex items-center gap-1">
                      View full results <ExternalLink className="w-3 h-3" />
                    </Link>
                  )}
                </div>
              )}
              {btStatus === 'running' && (
                <div className="flex items-center gap-2 text-xs text-yellow-400">
                  <Loader2 className="w-3 h-3 animate-spin" /> Backtest running — markers will appear when complete
                </div>
              )}
            </>
          )}

          {!chartLoading && candles.length === 0 && selectedPair && selectedTimeframe && (
            <p className="text-xs text-gray-500 py-4 text-center">
              No data available for {selectedPair} {selectedTimeframe} on {btExchange}
            </p>
          )}
        </div>
      </details>

      {/* Code View */}
      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
            <Code className="w-4 h-4" /> Generated Code
          </h3>
          <div className="flex items-center gap-2">
            {showCode && code && (
              <>
                <button
                  onClick={copyCode}
                  className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200 transition-colors"
                >
                  {copied ? <Check className="w-3 h-3 text-profit" /> : <Copy className="w-3 h-3" />}
                  {copied ? 'Copied!' : 'Copy'}
                </button>
                <button
                  onClick={downloadCode}
                  className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200 transition-colors"
                >
                  <Download className="w-3 h-3" /> Download .py
                </button>
              </>
            )}
            <button
              onClick={loadCode}
              className="text-xs text-accent hover:underline"
            >
              {showCode ? 'Reload' : 'View Code'}
            </button>
          </div>
        </div>

        {showCode && code ? (
          <pre className="text-xs text-gray-300 font-mono overflow-x-auto bg-surface-0 p-4 rounded-lg max-h-[600px] overflow-y-auto leading-relaxed">
            {code}
          </pre>
        ) : (
          <p className="text-xs text-gray-500 py-4 text-center">
            Click "View Code" to generate and display the strategy code
          </p>
        )}
      </div>

      {/* Backtest Launch */}
      {strategy.gene && (
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <Play className="w-4 h-4" /> Backtest This Strategy
            </h3>
            <button
              onClick={() => setShowBacktest(!showBacktest)}
              className="text-xs text-accent hover:underline"
            >
              {showBacktest ? 'Hide' : 'Configure & Run'}
            </button>
          </div>

          {showBacktest && (
            <div className="space-y-3">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div>
                  <label className="text-[10px] text-gray-500 uppercase block mb-1">Time Range</label>
                  <input
                    type="text"
                    value={btTimerange}
                    onChange={(e) => setBtTimerange(e.target.value)}
                    placeholder="20250101-20250401"
                    className="w-full bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-gray-500 uppercase block mb-1">Pairs (comma-separated)</label>
                  <input
                    type="text"
                    value={btPairs}
                    onChange={(e) => setBtPairs(e.target.value)}
                    placeholder="BTC/USDT, ETH/USDT"
                    className="w-full bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-gray-500 uppercase block mb-1">Timeframe</label>
                  <input
                    type="text"
                    value={btTimeframe}
                    onChange={(e) => setBtTimeframe(e.target.value)}
                    placeholder={strategy.gene.timeframe || '5m'}
                    className="w-full bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-gray-500 uppercase block mb-1">Exchange</label>
                  <select
                    value={btExchange}
                    onChange={(e) => setBtExchange(e.target.value)}
                    className="w-full bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
                  >
                    <option value="binance">binance</option>
                    <option value="kraken">kraken</option>
                    <option value="bybit">bybit</option>
                    <option value="okx">okx</option>
                  </select>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={async () => {
                    if (!strategy.gene) return;
                    setBtRunning(true);
                    setBtError(null);
                    setBacktestTrades([]);
                    setBtStatus(null);
                    setBtResult(null);
                    try {
                      const pairs = btPairs.split(',').map(p => p.trim()).filter(Boolean);
                      const result = await api.startBacktest({
                        strategy_gene: strategy.gene as unknown as Record<string, unknown>,
                        timerange: btTimerange,
                        exchange: btExchange,
                        ...(pairs.length > 0 && { pairs }),
                        ...(btTimeframe && { timeframe: btTimeframe }),
                      });
                      setLastBacktestId(result.backtest_id);
                      setBtStatus('running');
                    } catch (err) {
                      setBtError(err instanceof Error ? err.message : String(err));
                    } finally {
                      setBtRunning(false);
                    }
                  }}
                  disabled={btRunning || !btTimerange}
                  className="flex items-center gap-1.5 text-xs bg-accent text-white px-4 py-2 rounded-lg hover:bg-accent/90 transition-colors disabled:opacity-50"
                >
                  {btRunning ? (
                    <><Loader2 className="w-3 h-3 animate-spin" /> Submitting...</>
                  ) : (
                    <><Play className="w-3 h-3" /> Start Backtest</>
                  )}
                </button>
                {lastBacktestId && (
                  <Link to={`/backtest/${lastBacktestId}`} className="text-xs text-accent hover:underline flex items-center gap-1">
                    View full results <ExternalLink className="w-3 h-3" />
                  </Link>
                )}
                {btError && (
                  <span className="text-xs text-loss">{btError}</span>
                )}
                {btStatus === 'running' && (
                  <span className="text-xs text-yellow-400 flex items-center gap-1">
                    <Loader2 className="w-3 h-3 animate-spin" /> Running...
                  </span>
                )}
                {btStatus === 'completed' && (
                  <span className="text-xs text-profit flex items-center gap-1">
                    <Check className="w-3 h-3" /> Complete — {backtestTrades.length} trades
                  </span>
                )}
                {btStatus === 'failed' && (
                  <span className="text-xs text-loss">Backtest failed</span>
                )}
              </div>

              {/* Inline backtest results summary */}
              {btStatus === 'completed' && btResult != null && (
                <div className="mt-4 border border-white/10 rounded-lg bg-surface-0 p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-medium text-gray-200 flex items-center gap-2">
                      <BarChart3 className="w-4 h-4 text-accent" /> Backtest Results
                    </h4>
                    {lastBacktestId && (
                      <Link to={`/backtest/${lastBacktestId}`} className="text-xs text-accent hover:underline flex items-center gap-1">
                        Full details <ExternalLink className="w-3 h-3" />
                      </Link>
                    )}
                  </div>

                  {btResult.error_message != null && (
                    <div className="flex items-center gap-2 text-xs text-yellow-400 bg-yellow-500/10 rounded px-3 py-1.5">
                      <AlertTriangle className="w-3 h-3 flex-shrink-0" />
                      {String(btResult.error_message)}
                    </div>
                  )}

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    <InlineStat label="Total Trades" value={String(btResult.total_trades ?? 0)} />
                    <InlineStat
                      label="Total Profit"
                      value={`${Number(btResult.profit_percent ?? 0) > 0 ? '+' : ''}${Number(btResult.profit_percent ?? 0).toFixed(2)}%`}
                      color={Number(btResult.profit_percent ?? 0) >= 0 ? 'text-profit' : 'text-loss'}
                    />
                    <InlineStat
                      label="Win Rate"
                      value={btResult.total_trades ? `${((Number(btResult.wins ?? 0) / Number(btResult.total_trades)) * 100).toFixed(1)}%` : '—'}
                    />
                    <InlineStat label="Max Drawdown" value={`${(Number(btResult.max_drawdown ?? 0) * 100).toFixed(1)}%`} color="text-loss" />
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    <InlineStat label="Sharpe" value={Number(btResult.sharpe_ratio ?? 0).toFixed(2)} />
                    <InlineStat label="Sortino" value={Number(btResult.sortino_ratio ?? 0).toFixed(2)} />
                    <InlineStat label="Profit Factor" value={Number(btResult.profit_factor ?? 0).toFixed(2)} />
                    <InlineStat label="Avg Duration" value={String(btResult.avg_duration ?? '—')} />
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-2 gap-2">
                    <InlineStat label="Wins" value={String(btResult.wins ?? 0)} color="text-profit" />
                    <InlineStat label="Losses" value={String(btResult.losses ?? 0)} color="text-loss" />
                  </div>
                </div>
              )}

              {btStatus === 'failed' && btResult?.error_message && (
                <div className="mt-3 flex items-center gap-2 text-xs text-loss bg-loss/10 rounded-lg px-3 py-2">
                  <AlertTriangle className="w-3 h-3 flex-shrink-0" />
                  {String(btResult.error_message)}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Dry Run Launch */}
      {strategy.gene && (
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <Zap className="w-4 h-4" /> Start Dry Run
            </h3>
            <button
              onClick={() => setShowDryRun(!showDryRun)}
              className="text-xs text-accent hover:underline"
            >
              {showDryRun ? 'Hide' : 'Configure'}
            </button>
          </div>

          {showDryRun && (
            <div className="space-y-3">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div>
                  <label className="text-[10px] text-gray-500 uppercase block mb-1">Exchange</label>
                  <select
                    value={drExchange}
                    onChange={(e) => setDrExchange(e.target.value)}
                    className="w-full bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
                  >
                    <option value="binance">binance</option>
                    <option value="kraken">kraken</option>
                    <option value="bybit">bybit</option>
                    <option value="okx">okx</option>
                  </select>
                </div>
                <div>
                  <label className="text-[10px] text-gray-500 uppercase block mb-1">Pairs</label>
                  <input
                    type="text"
                    value={drPairs}
                    onChange={(e) => setDrPairs(e.target.value)}
                    placeholder="BTC/USDT, ETH/USDT"
                    className="w-full bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-gray-500 uppercase block mb-1">Stake Amount</label>
                  <input
                    type="text"
                    value={drStake}
                    onChange={(e) => setDrStake(e.target.value)}
                    placeholder="100"
                    className="w-full bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-gray-500 uppercase block mb-1">Timeframe</label>
                  <input
                    type="text"
                    value={drTimeframe}
                    onChange={(e) => setDrTimeframe(e.target.value)}
                    placeholder={strategy.gene?.timeframe || '5m'}
                    className="w-full bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
                  />
                </div>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={async () => {
                    if (!strategy.gene) return;
                    setDrRunning(true);
                    setDrError(null);
                    try {
                      const pairs = drPairs.split(',').map(p => p.trim()).filter(Boolean);
                      const result = await api.startDryRun({
                        strategy_gene: strategy.gene as unknown as Record<string, unknown>,
                        exchange: drExchange,
                        ...(pairs.length > 0 && { pairs }),
                        stake_amount: parseFloat(drStake) || 100,
                        ...(drTimeframe && { timeframe: drTimeframe }),
                      });
                      navigate(`/dry-run/${result.dry_run_id}`);
                    } catch (err) {
                      setDrError(err instanceof Error ? err.message : String(err));
                    } finally {
                      setDrRunning(false);
                    }
                  }}
                  disabled={drRunning}
                  className="flex items-center gap-1.5 text-xs bg-green-500/20 text-green-400 border border-green-500/30 px-4 py-2 rounded-lg hover:bg-green-500/30 transition-colors disabled:opacity-50"
                >
                  {drRunning ? (
                    <><Loader2 className="w-3 h-3 animate-spin" /> Starting...</>
                  ) : (
                    <><Zap className="w-3 h-3" /> Start Dry Run</>
                  )}
                </button>
                {drError && (
                  <span className="text-xs text-loss">{drError}</span>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* All Metrics (raw) */}
      <details className="card group">
        <summary className="text-sm font-medium text-gray-300 cursor-pointer select-none">
          All Metrics
          <span className="text-xs text-gray-500 ml-2 group-open:hidden">Click to expand</span>
        </summary>
        <div className="mt-3 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
          {Object.entries(strategy.metrics)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([key, val]) => (
              <div key={key} className="bg-surface-2 rounded-lg px-3 py-2">
                <div className="text-[10px] text-gray-500 truncate">{key}</div>
                <div className="text-sm font-mono text-gray-300">
                  {typeof val === 'number'
                    ? val.toFixed(4)
                    : typeof val === 'boolean'
                    ? String(val)
                    : typeof val === 'string'
                    ? val
                    : JSON.stringify(val)}
                </div>
              </div>
            ))}
        </div>
      </details>
    </div>
  );
}

function QualityItem({
  label,
  value,
  rating,
  highlight,
}: {
  label: string;
  value: string;
  rating: string;
  highlight?: boolean;
}) {
  const ratingColor = getRatingColor(rating);
  return (
    <div className={`bg-surface-2 rounded-lg px-3 py-2 ${highlight ? 'ring-1 ring-accent/30' : ''}`}>
      <div className="text-[10px] text-gray-500 uppercase">{label}</div>
      <div className="text-sm font-mono text-gray-200">{value}</div>
      <div className={`text-[10px] font-medium uppercase mt-0.5 ${ratingColor}`}>
        {rating}
      </div>
    </div>
  );
}

function InlineStat({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="bg-surface-2 rounded-lg px-3 py-2">
      <div className="text-[10px] text-gray-500 uppercase">{label}</div>
      <div className={`text-sm font-mono ${color || 'text-gray-200'}`}>{value}</div>
    </div>
  );
}

function getRatingColor(rating: string): string {
  const r = rating.toUpperCase();
  if (r.includes('EXCELLENT') || r.includes('GOOD') || r === 'A' || r === 'B') return 'text-profit';
  if (r.includes('MODERATE') || r.includes('FAIR') || r === 'C') return 'text-warn';
  if (r.includes('POOR') || r.includes('BAD') || r === 'D' || r === 'F') return 'text-loss';
  return 'text-gray-500';
}
