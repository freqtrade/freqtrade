/**
 * RunComparePage — Overlay fitness curves from multiple runs.
 *
 * Accessed via /runs/compare?ids=runA,runB,runC
 * Allows selecting runs, overlaying their fitness data, and comparing key metrics.
 */

import { useEffect, useState, useMemo } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { ArrowLeft, Layers, X, Plus, Loader2 } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { api } from '../api/client';
import { useStore } from '../store/useStore';
import type { RunDetail, RunSummary, GenerationStats } from '../types';
import { LoadingState, EmptyState } from '../components/StateDisplays';

const COLORS = ['#3b82f6', '#34d399', '#f472b6', '#facc15', '#a78bfa', '#fb923c', '#ef4444', '#06b6d4'];

interface RunCompareData {
  runId: string;
  detail: RunDetail | null;
  loading: boolean;
  error: string | null;
}

export function RunComparePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const runsMap = useStore((s) => s.runs);
  const allRuns = useMemo(
    () => Array.from(runsMap.values()).sort((a, b) => (b.started_at || 0) - (a.started_at || 0)),
    [runsMap],
  );

  const [compareData, setCompareData] = useState<RunCompareData[]>([]);
  const [showAddMenu, setShowAddMenu] = useState(false);

  // Selected run ids from query params
  const selectedIds = useMemo(() => {
    const raw = searchParams.get('ids') || '';
    return raw.split(',').filter(Boolean);
  }, [searchParams]);

  // Load run list on mount
  useEffect(() => {
    api.listRuns().then((r) => useStore.getState().setRuns(r)).catch(() => {});
  }, []);

  // Fetch details for each selected run
  useEffect(() => {
    if (selectedIds.length === 0) {
      setCompareData([]);
      return;
    }

    const newData: RunCompareData[] = selectedIds.map((id) => ({
      runId: id,
      detail: null,
      loading: true,
      error: null,
    }));
    setCompareData(newData);

    selectedIds.forEach((id, i) => {
      api.getRun(id)
        .then((d) => {
          setCompareData((prev) =>
            prev.map((e, idx) => (idx === i ? { ...e, detail: d, loading: false } : e)),
          );
        })
        .catch((err) => {
          setCompareData((prev) =>
            prev.map((e, idx) => (idx === i ? { ...e, error: err.message, loading: false } : e)),
          );
        });
    });
  }, [selectedIds]);

  const addRun = (runId: string) => {
    if (selectedIds.includes(runId)) return;
    const newIds = [...selectedIds, runId];
    setSearchParams({ ids: newIds.join(',') });
    setShowAddMenu(false);
  };

  const removeRun = (runId: string) => {
    const newIds = selectedIds.filter((id) => id !== runId);
    setSearchParams(newIds.length > 0 ? { ids: newIds.join(',') } : {});
  };

  const anyLoading = compareData.some((d) => d.loading);
  const loaded = compareData.filter((d) => d.detail !== null);

  // Build unified chart data: align by generation number across runs
  const chartData = useMemo(() => {
    const maxGen = Math.max(
      0,
      ...loaded.map((d) => d.detail!.generation_stats.length),
    );
    const rows: Array<Record<string, number | null>> = [];
    for (let g = 0; g < maxGen; g++) {
      const row: Record<string, number | null> = { generation: g + 1 };
      for (const d of loaded) {
        const stats = d.detail!.generation_stats;
        const s = stats.find((ss) => ss.generation === g) ?? stats[g];
        row[`${d.runId}_best`] = s?.best_fitness ?? null;
        row[`${d.runId}_avg`] = s?.avg_fitness ?? null;
      }
      rows.push(row);
    }
    return rows;
  }, [loaded]);

  // Available runs not yet selected
  const availableRuns = useMemo(
    () => allRuns.filter((r) => !selectedIds.includes(r.run_id)),
    [allRuns, selectedIds],
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Link to="/runs" className="text-gray-500 hover:text-gray-300 transition-colors">
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <h1 className="text-2xl font-bold text-gray-100 flex items-center gap-2">
          <Layers className="w-5 h-5 text-accent" /> Compare Runs
        </h1>
      </div>

      {/* Selected runs chips */}
      <div className="flex flex-wrap items-center gap-2">
        {selectedIds.map((id, i) => (
          <div
            key={id}
            className="flex items-center gap-2 bg-surface-1 border border-white/10 rounded-lg px-3 py-1.5"
          >
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: COLORS[i % COLORS.length] }}
            />
            <span className="text-xs font-mono text-gray-300">{id}</span>
            <button
              onClick={() => removeRun(id)}
              className="text-gray-500 hover:text-loss transition-colors"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        ))}

        {/* Add run button */}
        <div className="relative">
          <button
            onClick={() => setShowAddMenu(!showAddMenu)}
            className="flex items-center gap-1 text-xs text-gray-400 hover:text-accent bg-surface-1 border border-dashed border-white/10 rounded-lg px-3 py-1.5 transition-colors"
          >
            <Plus className="w-3 h-3" /> Add Run
          </button>

          {showAddMenu && availableRuns.length > 0 && (
            <div className="absolute top-full left-0 mt-1 bg-surface-2 border border-white/10 rounded-lg shadow-lg z-20 w-64 max-h-60 overflow-y-auto">
              {availableRuns.map((r) => (
                <button
                  key={r.run_id}
                  onClick={() => addRun(r.run_id)}
                  className="w-full text-left px-3 py-2 text-xs hover:bg-white/5 transition-colors"
                >
                  <span className="font-mono text-gray-300">{r.run_id}</span>
                  <span className="text-gray-500 ml-2">
                    Gen {r.current_generation}/{r.total_generations}
                  </span>
                  {r.best_fitness != null && (
                    <span className="text-accent ml-2">{r.best_fitness.toFixed(4)}</span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {selectedIds.length === 0 && (
        <EmptyState
          title="Select runs to compare"
          message="Use the Add Run button above, or visit this page from the Runs list."
        />
      )}

      {anyLoading && <LoadingState message="Loading run data..." />}

      {/* Fitness overlay chart */}
      {!anyLoading && chartData.length > 0 && (
        <div className="card p-4">
          <h2 className="text-sm font-medium text-gray-400 mb-1">Best Fitness Overlay</h2>
          <p className="text-[10px] text-gray-500 mb-4">
            Solid lines = best fitness, dashed lines = average fitness
          </p>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="generation"
                tick={{ fill: '#9ca3af', fontSize: 11 }}
                label={{ value: 'Generation', position: 'bottom', fill: '#6b7280', fontSize: 10, dy: 10 }}
              />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: 8,
                  fontSize: 12,
                }}
                formatter={(v: any) => [v != null ? Number(v).toFixed(4) : '—', '']}
              />
              <Legend wrapperStyle={{ fontSize: 11, color: '#9ca3af' }} />
              {loaded.map((d, i) => (
                <Line
                  key={`${d.runId}_best`}
                  type="monotone"
                  dataKey={`${d.runId}_best`}
                  name={`${d.runId} (best)`}
                  stroke={COLORS[i % COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                  connectNulls
                />
              ))}
              {loaded.map((d, i) => (
                <Line
                  key={`${d.runId}_avg`}
                  type="monotone"
                  dataKey={`${d.runId}_avg`}
                  name={`${d.runId} (avg)`}
                  stroke={COLORS[i % COLORS.length]}
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  dot={false}
                  connectNulls
                  strokeOpacity={0.5}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Summary table */}
      {!anyLoading && loaded.length > 0 && (
        <div className="card p-4">
          <h2 className="text-sm font-medium text-gray-400 mb-3">Run Summary Comparison</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left text-gray-500 py-2 pr-4">Metric</th>
                  {loaded.map((d, i) => (
                    <th key={d.runId} className="text-right text-gray-400 py-2 px-3">
                      <div className="flex items-center justify-end gap-1.5">
                        <div
                          className="w-2 h-2 rounded-full"
                          style={{ backgroundColor: COLORS[i % COLORS.length] }}
                        />
                        <span className="font-mono">{d.runId.slice(0, 12)}</span>
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <SummaryRow label="Status" values={loaded.map((d) => d.detail!.status)} />
                <SummaryRow
                  label="Generations"
                  values={loaded.map((d) => `${d.detail!.current_generation}/${d.detail!.total_generations}`)}
                />
                <SummaryRow
                  label="Best Fitness"
                  values={loaded.map((d) => d.detail!.best_fitness?.toFixed(4) ?? '—')}
                  highlight="max"
                  numValues={loaded.map((d) => d.detail!.best_fitness)}
                />
                <SummaryRow
                  label="Best Profit %"
                  values={loaded.map((d) =>
                    d.detail!.best_profit != null ? `${d.detail!.best_profit.toFixed(1)}%` : '—',
                  )}
                  highlight="max"
                  numValues={loaded.map((d) => d.detail!.best_profit)}
                />
                <SummaryRow
                  label="Population Size"
                  values={loaded.map((d) => String(d.detail!.population_size))}
                />
                <SummaryRow
                  label="Pairs"
                  values={loaded.map((d) => (d.detail!.pairs || []).join(', ') || '—')}
                />
                {/* Last gen diversity */}
                <SummaryRow
                  label="Final Diversity"
                  values={loaded.map((d) => {
                    const stats = d.detail!.generation_stats;
                    const last = stats[stats.length - 1];
                    return last?.genetic_diversity?.toFixed(3) ?? '—';
                  })}
                />
                {/* Last gen holdout degradation */}
                <SummaryRow
                  label="Final Holdout Deg."
                  values={loaded.map((d) => {
                    const stats = d.detail!.generation_stats;
                    const last = stats[stats.length - 1];
                    return last?.holdout_avg_degradation != null
                      ? `${(last.holdout_avg_degradation * 100).toFixed(1)}%`
                      : '—';
                  })}
                />
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function SummaryRow({
  label,
  values,
  highlight,
  numValues,
}: {
  label: string;
  values: string[];
  highlight?: 'max' | 'min';
  numValues?: (number | null | undefined)[];
}) {
  // Find best index
  let bestIdx = -1;
  if (highlight && numValues) {
    const valid = numValues.map((v, i) => (v != null ? { v, i } : null)).filter(Boolean) as { v: number; i: number }[];
    if (valid.length > 0) {
      if (highlight === 'max') bestIdx = valid.reduce((a, b) => (b.v > a.v ? b : a)).i;
      else bestIdx = valid.reduce((a, b) => (b.v < a.v ? b : a)).i;
    }
  }

  return (
    <tr className="border-b border-white/5">
      <td className="text-gray-500 py-1.5 pr-4">{label}</td>
      {values.map((v, i) => (
        <td
          key={i}
          className={`text-right py-1.5 px-3 font-mono ${
            i === bestIdx ? 'text-accent font-semibold' : 'text-gray-300'
          }`}
        >
          {v}
        </td>
      ))}
    </tr>
  );
}
