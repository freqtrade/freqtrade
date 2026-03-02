/**
 * AnalyticsPage — Feature importance & overfitting dashboards.
 *
 * Tab 1: Feature Importance — which indicators appear most in top strategies
 * Tab 2: Overfitting Detection — holdout degradation and diversity trends
 */

import { useEffect, useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
  ScatterChart,
  Scatter,
  ZAxis,
} from 'recharts';
import { Brain, ShieldAlert, Loader2 } from 'lucide-react';
import { api } from '../api/client';
import { LoadingState, EmptyState } from '../components/StateDisplays';
import type { RunSummary, RunDetail, HoFEntry, GenerationStats } from '../types';
import { useStore } from '../store/useStore';

type Tab = 'importance' | 'overfitting';

export function AnalyticsPage() {
  const [tab, setTab] = useState<Tab>('importance');

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100 flex items-center gap-2">
        <Brain className="w-5 h-5 text-accent" /> Analytics
      </h1>

      {/* Tab switcher */}
      <div className="flex gap-1 bg-surface-1 rounded-lg p-1 w-fit">
        <TabButton active={tab === 'importance'} onClick={() => setTab('importance')}>
          <Brain className="w-3.5 h-3.5" /> Feature Importance
        </TabButton>
        <TabButton active={tab === 'overfitting'} onClick={() => setTab('overfitting')}>
          <ShieldAlert className="w-3.5 h-3.5" /> Overfitting Detection
        </TabButton>
      </div>

      {tab === 'importance' && <FeatureImportance />}
      {tab === 'overfitting' && <OverfittingDashboard />}
    </div>
  );
}

function TabButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-colors ${
        active ? 'bg-accent/20 text-accent' : 'text-gray-400 hover:text-gray-200'
      }`}
    >
      {children}
    </button>
  );
}

// ── Feature Importance ────────────────────────────────────────

function FeatureImportance() {
  const [hofEntries, setHofEntries] = useState<HoFEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.getHallOfFame()
      .then(setHofEntries)
      .catch(() => setHofEntries([]))
      .finally(() => setLoading(false));
  }, []);

  // Aggregate indicator frequency across all HoF strategies
  const indicatorFrequency = useMemo(() => {
    const freq: Record<string, { count: number; totalFitness: number; avgFitness: number }> = {};
    for (const entry of hofEntries) {
      const gene = entry.strategy_gene;
      const indicators = (gene?.indicators as Array<{ type: string }>) ?? [];
      for (const ind of indicators) {
        if (!freq[ind.type]) {
          freq[ind.type] = { count: 0, totalFitness: 0, avgFitness: 0 };
        }
        freq[ind.type].count++;
        freq[ind.type].totalFitness += entry.fitness ?? 0;
      }
    }
    // Compute averages
    for (const key of Object.keys(freq)) {
      freq[key].avgFitness = freq[key].count > 0 ? freq[key].totalFitness / freq[key].count : 0;
    }
    return Object.entries(freq)
      .map(([type, data]) => ({ type, ...data }))
      .sort((a, b) => b.count - a.count);
  }, [hofEntries]);

  // Condition operator frequency
  const conditionFrequency = useMemo(() => {
    const freq: Record<string, number> = {};
    for (const entry of hofEntries) {
      const gene = entry.strategy_gene;
      const allConds = [
        ...((gene?.entry_conditions as Array<{ operator: string }>) ?? []),
        ...((gene?.exit_conditions as Array<{ operator: string }>) ?? []),
      ];
      for (const cond of allConds) {
        const key = cond.operator;
        freq[key] = (freq[key] || 0) + 1;
      }
    }
    return Object.entries(freq)
      .map(([operator, count]) => ({ operator, count }))
      .sort((a, b) => b.count - a.count);
  }, [hofEntries]);

  // Parameter heatmap: for top indicators, show parameter distribution
  const parameterRanges = useMemo(() => {
    const ranges: Record<string, Record<string, number[]>> = {};
    for (const entry of hofEntries) {
      const gene = entry.strategy_gene;
      const indicators = (gene?.indicators as Array<{ type: string; parameters: Record<string, number> }>) ?? [];
      for (const ind of indicators) {
        if (!ranges[ind.type]) ranges[ind.type] = {};
        for (const [param, val] of Object.entries(ind.parameters ?? {})) {
          if (typeof val !== 'number') continue;
          if (!ranges[ind.type][param]) ranges[ind.type][param] = [];
          ranges[ind.type][param].push(val);
        }
      }
    }
    // Compute stats for top 5 indicators
    const top5 = indicatorFrequency.slice(0, 5).map(i => i.type);
    return top5.map(type => ({
      type,
      params: Object.entries(ranges[type] ?? {}).map(([param, values]) => ({
        param,
        min: Math.min(...values),
        max: Math.max(...values),
        avg: values.reduce((a, b) => a + b, 0) / values.length,
        count: values.length,
      })),
    }));
  }, [hofEntries, indicatorFrequency]);

  if (loading) return <LoadingState message="Loading Hall of Fame data..." />;
  if (hofEntries.length === 0) {
    return <EmptyState title="No data" message="Run evolutions and build a Hall of Fame to see analytics." />;
  }

  return (
    <div className="space-y-6">
      {/* Indicator frequency bar chart */}
      <div className="card p-4">
        <h2 className="text-sm font-medium text-gray-400 mb-3">Indicator Frequency in Top Strategies</h2>
        <p className="text-[10px] text-gray-500 mb-4">
          How often each indicator appears across {hofEntries.length} Hall of Fame strategies
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={indicatorFrequency} margin={{ left: 0, right: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="type" tick={{ fill: '#9ca3af', fontSize: 11 }} />
            <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }}
              formatter={(v: number, name: string) => [v, name === 'count' ? 'Appearances' : 'Avg Fitness']}
            />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Appearances" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Avg fitness per indicator */}
      <div className="card p-4">
        <h2 className="text-sm font-medium text-gray-400 mb-3">Average Fitness by Indicator</h2>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={indicatorFrequency} margin={{ left: 0, right: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="type" tick={{ fill: '#9ca3af', fontSize: 11 }} />
            <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} tickFormatter={(v: number) => v.toFixed(2)} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }}
              formatter={(v: number) => [v.toFixed(4), 'Avg Fitness']}
            />
            <Bar dataKey="avgFitness" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Avg Fitness" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Condition operators */}
      <div className="card p-4">
        <h2 className="text-sm font-medium text-gray-400 mb-3">Condition Operators</h2>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={conditionFrequency} layout="vertical" margin={{ left: 80, right: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis type="number" tick={{ fill: '#9ca3af', fontSize: 11 }} />
            <YAxis dataKey="operator" type="category" tick={{ fill: '#9ca3af', fontSize: 11 }} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }}
            />
            <Bar dataKey="count" fill="#34d399" radius={[0, 4, 4, 0]} name="Count" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Top indicator parameter ranges */}
      <div className="card p-4">
        <h2 className="text-sm font-medium text-gray-400 mb-3">Parameter Ranges (Top Indicators)</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {parameterRanges.map(({ type, params }) => (
            <div key={type} className="bg-surface-0 rounded-lg p-3">
              <h3 className="text-xs font-medium text-accent mb-2">{type}</h3>
              {params.length === 0 ? (
                <p className="text-[10px] text-gray-600">No parameters</p>
              ) : (
                <div className="space-y-1.5">
                  {params.map(({ param, min, max, avg, count }) => (
                    <div key={param} className="flex items-center justify-between text-[10px]">
                      <span className="text-gray-400">{param}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-gray-500">{min.toFixed(1)}</span>
                        <div className="w-16 h-1 bg-surface-2 rounded-full relative">
                          <div
                            className="absolute h-full bg-accent/50 rounded-full"
                            style={{
                              left: `${max > min ? 0 : 0}%`,
                              width: '100%',
                            }}
                          />
                          <div
                            className="absolute w-1.5 h-1.5 bg-accent rounded-full -top-0.5"
                            style={{
                              left: `${max > min ? ((avg - min) / (max - min)) * 100 : 50}%`,
                            }}
                          />
                        </div>
                        <span className="text-gray-500">{max.toFixed(1)}</span>
                        <span className="text-gray-600 ml-1">({count})</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Overfitting Detection ─────────────────────────────────────

function OverfittingDashboard() {
  const runsMap = useStore((s) => s.runs);
  const [selectedRunId, setSelectedRunId] = useState<string>('');
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(false);

  const runs = useMemo(() => {
    return Array.from(runsMap.values()).sort((a, b) => (b.started_at || 0) - (a.started_at || 0));
  }, [runsMap]);

  // Load runs on mount
  useEffect(() => {
    api.listRuns().then((r) => {
      const store = useStore.getState();
      store.setRuns(r);
      // Auto-select first run
      if (r.length > 0 && !selectedRunId) {
        setSelectedRunId(r[0].run_id);
      }
    }).catch(() => {});
  }, []);

  // Fetch run detail when selection changes
  useEffect(() => {
    if (!selectedRunId) return;
    setLoading(true);
    api.getRun(selectedRunId)
      .then(setRunDetail)
      .catch(() => setRunDetail(null))
      .finally(() => setLoading(false));
  }, [selectedRunId]);

  const stats = runDetail?.generation_stats ?? [];

  // Prepare overfitting chart data
  const overfitData = useMemo(() => {
    return stats
      .filter((s) => s.generation != null)
      .map((s) => ({
        generation: s.generation,
        holdout_degradation: s.holdout_avg_degradation != null ? s.holdout_avg_degradation * 100 : null,
        best_holdout_degradation: s.holdout_best_degradation != null ? s.holdout_best_degradation * 100 : null,
        genetic_diversity: s.genetic_diversity,
        avg_unused_indicators: s.avg_unused_indicators,
      }));
  }, [stats]);

  // In-sample vs out-of-sample scatter (generation-level)
  const scatterData = useMemo(() => {
    return stats
      .filter((s) => s.best_fitness != null && s.holdout_avg_degradation != null)
      .map((s) => ({
        fitness: s.best_fitness,
        degradation: (s.holdout_avg_degradation ?? 0) * 100,
        generation: s.generation,
      }));
  }, [stats]);

  return (
    <div className="space-y-6">
      {/* Run selector */}
      <div className="card p-4">
        <div className="flex items-center gap-3">
          <label className="text-xs text-gray-400">Select Run:</label>
          <select
            value={selectedRunId}
            onChange={(e) => setSelectedRunId(e.target.value)}
            className="bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:ring-1 focus:ring-accent/50"
          >
            {runs.length === 0 && <option value="">No runs available</option>}
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} — Gen {r.current_generation}/{r.total_generations} ({r.status})
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading && <LoadingState message="Loading run data..." />}

      {!loading && stats.length === 0 && (
        <EmptyState title="No generation data" message="Select a run with generation data to analyze overfitting trends." />
      )}

      {!loading && overfitData.length > 0 && (
        <>
          {/* Holdout degradation over generations */}
          <div className="card p-4">
            <h2 className="text-sm font-medium text-gray-400 mb-1">Holdout Degradation Over Generations</h2>
            <p className="text-[10px] text-gray-500 mb-4">
              Rising degradation suggests the population is overfitting to the training data
            </p>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={overfitData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="generation" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} tickFormatter={(v: number) => `${v.toFixed(0)}%`} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }}
                  formatter={(v: any) => [v != null ? `${Number(v).toFixed(1)}%` : '—', '']}
                />
                <Legend wrapperStyle={{ fontSize: 11, color: '#9ca3af' }} />
                <Line type="monotone" dataKey="holdout_degradation" name="Avg Degradation" stroke="#f59e0b" strokeWidth={2} dot={false} connectNulls />
                <Line type="monotone" dataKey="best_holdout_degradation" name="Best Degradation" stroke="#ef4444" strokeWidth={1.5} dot={false} strokeDasharray="4 4" connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Genetic diversity */}
          <div className="card p-4">
            <h2 className="text-sm font-medium text-gray-400 mb-1">Genetic Diversity</h2>
            <p className="text-[10px] text-gray-500 mb-4">
              Low diversity may indicate premature convergence. Healthy evolution maintains diversity.
            </p>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={overfitData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="generation" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} domain={[0, 1]} tickFormatter={(v: number) => v.toFixed(1)} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }}
                  formatter={(v: any) => [v != null ? Number(v).toFixed(3) : '—', 'Diversity']}
                />
                <Line type="monotone" dataKey="genetic_diversity" name="Diversity" stroke="#34d399" strokeWidth={2} dot={false} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Fitness vs Degradation scatter */}
          {scatterData.length > 0 && (
            <div className="card p-4">
              <h2 className="text-sm font-medium text-gray-400 mb-1">Fitness vs Holdout Degradation</h2>
              <p className="text-[10px] text-gray-500 mb-4">
                Points in the upper-right indicate high fitness but also high degradation (likely overfitting)
              </p>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis
                    dataKey="fitness" type="number" name="Fitness"
                    tick={{ fill: '#9ca3af', fontSize: 11 }}
                    label={{ value: 'Best Fitness', position: 'bottom', fill: '#6b7280', fontSize: 10 }}
                  />
                  <YAxis
                    dataKey="degradation" type="number" name="Degradation"
                    tick={{ fill: '#9ca3af', fontSize: 11 }}
                    tickFormatter={(v: number) => `${v.toFixed(0)}%`}
                    label={{ value: 'Degradation %', angle: -90, position: 'left', fill: '#6b7280', fontSize: 10 }}
                  />
                  <ZAxis dataKey="generation" range={[30, 200]} name="Generation" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }}
                    formatter={(v: number, name: string) => {
                      if (name === 'Degradation') return [`${v.toFixed(1)}%`, name];
                      if (name === 'Fitness') return [v.toFixed(4), name];
                      return [v, name];
                    }}
                  />
                  <Scatter data={scatterData} fill="#60a5fa" fillOpacity={0.7} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Unused indicators trend */}
          {overfitData.some(d => d.avg_unused_indicators != null) && (
            <div className="card p-4">
              <h2 className="text-sm font-medium text-gray-400 mb-1">Average Unused Indicators</h2>
              <p className="text-[10px] text-gray-500 mb-4">
                Strategies with many unused indicators are adding noise, not signal
              </p>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={overfitData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="generation" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }}
                  />
                  <Line type="monotone" dataKey="avg_unused_indicators" name="Avg Unused" stroke="#f97316" strokeWidth={2} dot={false} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}
    </div>
  );
}
