import { useEffect, useState, useMemo } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { GitCompare, ArrowLeft, X } from 'lucide-react';
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { api } from '../api/client';
import type { StrategyDetail } from '../types';
import { LoadingState, ErrorState, EmptyState } from '../components/StateDisplays';

const COLORS = ['#34d399', '#60a5fa', '#f472b6', '#facc15', '#a78bfa', '#fb923c'];

interface CompareEntry {
  runId: string;
  strategyId: string;
  label: string;
  strategy: StrategyDetail | null;
  loading: boolean;
  error: string | null;
}

function normalize(val: number | null | undefined, min: number, max: number): number {
  if (val == null || max === min) return 0;
  return Math.max(0, Math.min(1, (val - min) / (max - min)));
}

export function ComparePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [entries, setEntries] = useState<CompareEntry[]>([]);

  // Parse ids from query string: runId1:stratId1,runId2:stratId2,...
  const idPairs = useMemo(() => {
    const raw = searchParams.get('ids') || '';
    return raw
      .split(',')
      .filter(Boolean)
      .map((pair) => {
        const [runId, stratId] = pair.split(':');
        return { runId, stratId };
      })
      .filter((p) => p.runId && p.stratId);
  }, [searchParams]);

  // Fetch strategies
  useEffect(() => {
    if (idPairs.length === 0) {
      setEntries([]);
      return;
    }

    const newEntries: CompareEntry[] = idPairs.map((p, i) => ({
      runId: p.runId,
      strategyId: p.stratId,
      label: `Strategy ${i + 1}`,
      strategy: null,
      loading: true,
      error: null,
    }));
    setEntries(newEntries);

    idPairs.forEach((p, i) => {
      api
        .getStrategy(p.runId, p.stratId)
        .then((s) => {
          setEntries((prev) =>
            prev.map((e, idx) =>
              idx === i ? { ...e, strategy: s, loading: false } : e,
            ),
          );
        })
        .catch((err) => {
          setEntries((prev) =>
            prev.map((e, idx) =>
              idx === i ? { ...e, error: err.message, loading: false } : e,
            ),
          );
        });
    });
  }, [idPairs]);

  const removeEntry = (index: number) => {
    const newPairs = idPairs.filter((_, i) => i !== index);
    if (newPairs.length === 0) {
      setSearchParams({});
    } else {
      setSearchParams({ ids: newPairs.map((p) => `${p.runId}:${p.stratId}`).join(',') });
    }
  };

  const anyLoading = entries.some((e) => e.loading);
  const loaded = entries.filter((e) => e.strategy !== null);

  // Build radar chart data
  const radarData = useMemo(() => {
    if (loaded.length === 0) return [];

    const metrics: { key: string; label: string; getter: (s: StrategyDetail) => number | null }[] = [
      { key: 'fitness', label: 'Fitness', getter: (s) => s.fitness },
      { key: 'profit', label: 'Profit', getter: (s) => s.metrics?.total_profit ?? null },
      { key: 'sharpe', label: 'Sharpe', getter: (s) => s.metrics?.sharpe_ratio ?? null },
      { key: 'win_rate', label: 'Win Rate', getter: (s) => s.metrics?.win_rate ?? null },
      { key: 'trades', label: 'Trades', getter: (s) => s.metrics?.num_trades ?? null },
      { key: 'drawdown', label: 'Low DD', getter: (s) => {
        const dd = s.metrics?.max_drawdown ?? null;
        return dd != null ? -dd : null; // invert: lower DD = better
      }},
    ];

    // Compute min/max for each metric
    const ranges = metrics.map((m) => {
      const vals = loaded.map((e) => m.getter(e.strategy!)).filter((v): v is number => v != null);
      return { min: Math.min(...vals, 0), max: Math.max(...vals, 1) };
    });

    return metrics.map((m, mi) => {
      const point: Record<string, string | number> = { metric: m.label };
      loaded.forEach((e, ei) => {
        const val = m.getter(e.strategy!);
        point[`s${ei}`] = Math.round(normalize(val, ranges[mi].min, ranges[mi].max) * 100);
      });
      return point;
    });
  }, [loaded]);

  if (idPairs.length === 0) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold text-gray-100 flex items-center gap-2">
          <GitCompare className="w-5 h-5 text-accent" /> Compare Strategies
        </h1>
        <EmptyState
          title="No strategies selected"
          message="Select strategies from the Generation page or Hall of Fame page using the compare checkboxes, then click 'Compare Selected'."
        />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <button
          onClick={() => window.history.back()}
          className="text-sm text-gray-500 hover:text-gray-300 flex items-center gap-1 mb-1"
        >
          <ArrowLeft className="w-3 h-3" /> Back
        </button>
        <h1 className="text-2xl font-bold text-gray-100 flex items-center gap-2">
          <GitCompare className="w-5 h-5 text-accent" /> Compare Strategies
          <span className="text-sm font-normal text-gray-500">({entries.length} strategies)</span>
        </h1>
      </div>

      {anyLoading && <LoadingState message="Loading strategies..." compact />}

      {/* Radar Chart */}
      {radarData.length > 0 && (
        <div className="card p-4">
          <h2 className="text-sm font-medium text-gray-400 mb-3">Performance Radar</h2>
          <ResponsiveContainer width="100%" height={350}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#374151" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: '#9ca3af', fontSize: 12 }} />
              <PolarRadiusAxis tick={false} axisLine={false} domain={[0, 100]} />
              {loaded.map((_, i) => (
                <Radar
                  key={i}
                  name={loaded[i].label}
                  dataKey={`s${i}`}
                  stroke={COLORS[i % COLORS.length]}
                  fill={COLORS[i % COLORS.length]}
                  fillOpacity={0.15}
                  strokeWidth={2}
                />
              ))}
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Metrics Table */}
      {loaded.length > 0 && (
        <div className="card overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                <th className="text-left py-2 px-3 text-gray-500 font-medium">Metric</th>
                {loaded.map((e, i) => (
                  <th key={i} className="text-right py-2 px-3 font-medium" style={{ color: COLORS[i % COLORS.length] }}>
                    <div className="flex items-center justify-end gap-2">
                      <Link
                        to={`/runs/${e.runId}/strategies/${e.strategyId}`}
                        className="hover:underline truncate max-w-[140px]"
                        title={e.strategyId}
                      >
                        {e.strategyId.slice(0, 10)}…
                      </Link>
                      <button
                        onClick={() => removeEntry(entries.indexOf(e))}
                        className="text-gray-600 hover:text-gray-300"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {renderMetricRow('Fitness', loaded, (s) => s.fitness, (v) => v?.toFixed(4))}
              {renderMetricRow('Profit %', loaded, (s) => s.metrics?.total_profit, (v) => v != null ? `${(v * 100).toFixed(2)}%` : '—')}
              {renderMetricRow('Sharpe Ratio', loaded, (s) => s.metrics?.sharpe_ratio, (v) => v?.toFixed(3))}
              {renderMetricRow('Sortino Ratio', loaded, (s) => s.metrics?.sortino_ratio, (v) => v?.toFixed(3))}
              {renderMetricRow('Win Rate', loaded, (s) => s.metrics?.win_rate, (v) => v != null ? `${(v * 100).toFixed(1)}%` : '—')}
              {renderMetricRow('Num Trades', loaded, (s) => s.metrics?.num_trades, (v) => v?.toFixed(0))}
              {renderMetricRow('Max Drawdown', loaded, (s) => s.metrics?.max_drawdown, (v) => v != null ? `${(v * 100).toFixed(2)}%` : '—')}
              {renderMetricRow('Profit Factor', loaded, (s) => s.metrics?.profit_factor, (v) => v?.toFixed(2))}
              {renderMetricRow('Complexity', loaded, (s) => {
                const g = s.gene;
                return g ? (g.indicators?.length ?? 0) + (g.entry_conditions?.length ?? 0) + (g.exit_conditions?.length ?? 0) : null;
              }, (v) => v?.toFixed(0))}
              {renderMetricRow('Holdout Label', loaded, () => null, () => null, (s) => s.quality?.holdout_label ?? '—')}
              {renderMetricRow('WF Label', loaded, () => null, () => null, (s) => s.quality?.wf_label ?? '—')}
              {renderMetricRow('MC Label', loaded, () => null, () => null, (s) => s.quality?.mc_label ?? '—')}
              {renderMetricRow('Overall Quality', loaded, () => null, () => null, (s) => s.quality?.overall_label ?? '—')}
              {renderMetricRow('Generation', loaded, (s) => s.generation, (v) => v?.toFixed(0))}
              {renderMetricRow('Timeframe', loaded, () => null, () => null, (s) => s.gene?.timeframe ?? '—')}
            </tbody>
          </table>
        </div>
      )}

      {/* Indicator Comparison */}
      {loaded.length >= 2 && (
        <div className="card p-4">
          <h2 className="text-sm font-medium text-gray-400 mb-3">Indicator Comparison</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {loaded.map((e, i) => (
              <div key={i} className="space-y-1">
                <h3 className="text-xs font-medium" style={{ color: COLORS[i % COLORS.length] }}>
                  {e.strategyId.slice(0, 10)}…
                </h3>
                <div className="flex flex-wrap gap-1">
                  {e.strategy!.gene?.indicators?.map((ind, j) => (
                    <span
                      key={j}
                      className="text-xs px-2 py-0.5 rounded-full bg-surface-1 text-gray-300"
                    >
                      {ind.type}({Object.values(ind.parameters).join(',')})
                    </span>
                  )) ?? <span className="text-xs text-gray-600">No indicators</span>}
                </div>
              </div>
            ))}
          </div>

          {/* Shared indicators */}
          {loaded.length >= 2 && (() => {
            const allSets = loaded.map((e) =>
              new Set(e.strategy!.gene?.indicators?.map((i) => i.type) ?? []),
            );
            const shared = [...allSets[0]].filter((t) => allSets.every((s) => s.has(t)));
            const unique = allSets.map((s, i) => ({
              entry: loaded[i],
              types: [...s].filter((t) => !allSets.every((os) => os.has(t))),
            }));

            return (
              <div className="mt-4 pt-3 border-t border-gray-800 space-y-2">
                {shared.length > 0 && (
                  <div>
                    <span className="text-xs text-gray-500">Shared: </span>
                    {shared.map((t) => (
                      <span key={t} className="text-xs px-2 py-0.5 rounded-full bg-accent/20 text-accent mr-1">
                        {t}
                      </span>
                    ))}
                  </div>
                )}
                {unique.map((u, i) =>
                  u.types.length > 0 ? (
                    <div key={i}>
                      <span className="text-xs text-gray-500">
                        Unique to {u.entry.strategyId.slice(0, 10)}…:{' '}
                      </span>
                      {u.types.map((t) => (
                        <span
                          key={t}
                          className="text-xs px-2 py-0.5 rounded-full mr-1"
                          style={{ backgroundColor: `${COLORS[i % COLORS.length]}20`, color: COLORS[i % COLORS.length] }}
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  ) : null,
                )}
              </div>
            );
          })()}
        </div>
      )}

      {/* Error entries */}
      {entries.filter((e) => e.error).map((e, i) => (
        <div key={i} className="card p-3 border border-loss/30">
          <ErrorState
            title={`Failed to load ${e.strategyId}`}
            message={e.error || 'Unknown error'}
            compact
          />
        </div>
      ))}
    </div>
  );
}

function renderMetricRow(
  label: string,
  entries: CompareEntry[],
  getter: (s: StrategyDetail) => number | null | undefined,
  formatter: (v: number | null | undefined) => string | null | undefined,
  stringGetter?: (s: StrategyDetail) => string | undefined,
) {
  const values = entries.map((e) => (stringGetter ? null : getter(e.strategy!)));
  const numericVals = values.filter((v): v is number => v != null);
  const best = numericVals.length > 0 ? Math.max(...numericVals) : null;

  return (
    <tr className="border-b border-gray-800/50 hover:bg-surface-1/50">
      <td className="py-2 px-3 text-gray-400">{label}</td>
      {entries.map((e, i) => {
        if (stringGetter) {
          return (
            <td key={i} className="py-2 px-3 text-right text-gray-300">
              {stringGetter(e.strategy!)}
            </td>
          );
        }
        const val = getter(e.strategy!);
        const formatted = formatter(val) ?? '—';
        const isBest = val != null && val === best && numericVals.length > 1;
        return (
          <td key={i} className={`py-2 px-3 text-right ${isBest ? 'text-profit font-medium' : 'text-gray-300'}`}>
            {formatted}
          </td>
        );
      })}
    </tr>
  );
}
