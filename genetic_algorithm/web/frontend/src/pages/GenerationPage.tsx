import { useEffect, useState, useMemo } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { ArrowLeft, TrendingUp, TrendingDown, Download, GitCompare } from 'lucide-react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  BarChart,
  Bar,
} from 'recharts';
import { api } from '../api/client';
import type { GenerationDetail, IndividualSummary } from '../types';
import { exportToCsv } from '../utils/csv';
import { LoadingState, ErrorState } from '../components/StateDisplays';

type SortKey = 'fitness' | 'profit' | 'sharpe_ratio' | 'num_trades' | 'max_drawdown' | 'win_rate' | 'complexity';

export function GenerationPage() {
  const { runId, gen } = useParams<{ runId: string; gen: string }>();
  const [data, setData] = useState<GenerationDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>('fitness');
  const [sortAsc, setSortAsc] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const navigate = useNavigate();

  useEffect(() => {
    if (!runId || !gen) return;
    setLoading(true);
    api
      .getGeneration(runId, parseInt(gen))
      .then((d) => { setData(d); setError(null); })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [runId, gen]);

  if (loading) return <LoadingState message="Loading generation..." />;
  if (error || !data) {
    return <ErrorState title="Failed to load generation" message={error || 'Not found'} />;
  }

  const sorted = [...data.individuals].sort((a, b) => {
    const av = getVal(a, sortKey);
    const bv = getVal(b, sortKey);
    if (av === null && bv === null) return 0;
    if (av === null) return 1;
    if (bv === null) return -1;
    return sortAsc ? av - bv : bv - av;
  });

  const handleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  // Scatter data: profit vs fitness
  const scatterData = data.individuals
    .filter((ind) => ind.fitness !== null && ind.profit !== null)
    .map((ind) => ({
      x: ind.profit!,
      y: ind.fitness!,
      id: ind.id,
      sharpe: ind.sharpe_ratio,
    }));

  // Indicator frequency data
  const indicatorFreq = useMemo(() => {
    const counts = new Map<string, number>();
    for (const ind of data.individuals) {
      for (const name of ind.indicators || []) {
        counts.set(name, (counts.get(name) || 0) + 1);
      }
    }
    return Array.from(counts.entries())
      .map(([name, count]) => ({ name, count, pct: Math.round((count / data.individuals.length) * 100) }))
      .sort((a, b) => b.count - a.count);
  }, [data.individuals]);

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
        <h1 className="text-xl font-bold text-gray-100">
          Generation {gen}
          <span className="text-sm font-normal text-gray-500 ml-2">
            {data.individuals.length} individuals
          </span>
        </h1>
        <button
          onClick={() => {
            const rows = data.individuals.map((ind) => ({
              id: ind.id,
              rank: ind.rank,
              fitness: ind.fitness,
              raw_fitness: ind.raw_fitness,
              profit: ind.profit,
              sharpe_ratio: ind.sharpe_ratio,
              sortino_ratio: ind.sortino_ratio,
              win_rate: ind.win_rate,
              num_trades: ind.num_trades,
              max_drawdown: ind.max_drawdown,
              profit_factor: ind.profit_factor,
              complexity: ind.complexity,
              indicators: ind.indicators?.join('; ') ?? '',
            }));
            exportToCsv(`generation_${gen}_run_${runId}.csv`, rows);
          }}
          className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200 transition-colors mt-1"
        >
          <Download className="w-3 h-3" /> Export CSV
        </button>
        {selected.size >= 2 && (
          <button
            onClick={() => {
              const ids = [...selected].map((id) => `${runId}:${id}`).join(',');
              navigate(`/compare?ids=${ids}`);
            }}
            className="flex items-center gap-1 text-xs text-accent hover:text-accent/80 transition-colors mt-1"
          >
            <GitCompare className="w-3 h-3" /> Compare {selected.size} Selected
          </button>
        )}
      </div>

      {/* Scatter Plot */}
      {scatterData.length > 0 && (
        <div className="card">
          <h2 className="text-sm font-medium text-gray-300 mb-3">Profit vs Fitness</h2>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                type="number"
                dataKey="x"
                name="Profit %"
                stroke="#6b7280"
                tick={{ fill: '#9ca3af', fontSize: 11 }}
                label={{ value: 'Profit %', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 11 }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name="Fitness"
                stroke="#6b7280"
                tick={{ fill: '#9ca3af', fontSize: 11 }}
                label={{ value: 'Fitness', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#e5e7eb',
                  fontSize: 12,
                }}
                formatter={(value: number, name: string) => [value.toFixed(4), name]}
              />
              <Scatter data={scatterData}>
                {scatterData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={entry.x >= 0 ? '#10b981' : '#ef4444'}
                    fillOpacity={0.7}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Indicator Frequency Chart */}
      {indicatorFreq.length > 0 && (
        <div className="card">
          <h2 className="text-sm font-medium text-gray-300 mb-3">
            Indicator Frequency
            <span className="text-xs text-gray-500 font-normal ml-2">
              across {data.individuals.length} individuals
            </span>
          </h2>
          <ResponsiveContainer width="100%" height={Math.max(200, indicatorFreq.length * 28)}>
            <BarChart
              data={indicatorFreq.slice(0, 25)}
              layout="vertical"
              margin={{ top: 5, right: 40, left: 10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" horizontal={false} />
              <XAxis
                type="number"
                stroke="#6b7280"
                tick={{ fill: '#9ca3af', fontSize: 11 }}
                label={{ value: 'Count', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 11 }}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={120}
                stroke="#6b7280"
                tick={{ fill: '#9ca3af', fontSize: 10 }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#e5e7eb',
                  fontSize: 12,
                }}
                formatter={(value: number, _name: string, props: { payload?: { pct: number } }) => [
                  `${value} (${props.payload?.pct ?? 0}%)`,
                  'Count',
                ]}
              />
              <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Individuals Table */}
      <div className="card">
        <h2 className="text-sm font-medium text-gray-300 mb-3">Individuals</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-500 uppercase tracking-wider border-b border-white/5">
                <th className="text-left py-2 px-1 font-medium w-6">
                  <input
                    type="checkbox"
                    className="accent-accent"
                    checked={selected.size === sorted.length && sorted.length > 0}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelected(new Set(sorted.map((s) => s.id)));
                      } else {
                        setSelected(new Set());
                      }
                    }}
                  />
                </th>
                <th className="text-left py-2 px-2 font-medium">#</th>
                <th className="text-left py-2 px-2 font-medium">ID</th>
                <SortHeader label="Fitness" k="fitness" current={sortKey} asc={sortAsc} onClick={handleSort} />
                <SortHeader label="Profit" k="profit" current={sortKey} asc={sortAsc} onClick={handleSort} />
                <SortHeader label="Sharpe" k="sharpe_ratio" current={sortKey} asc={sortAsc} onClick={handleSort} />
                <SortHeader label="Win Rate" k="win_rate" current={sortKey} asc={sortAsc} onClick={handleSort} />
                <SortHeader label="Trades" k="num_trades" current={sortKey} asc={sortAsc} onClick={handleSort} />
                <SortHeader label="Max DD" k="max_drawdown" current={sortKey} asc={sortAsc} onClick={handleSort} />
                <SortHeader label="Complexity" k="complexity" current={sortKey} asc={sortAsc} onClick={handleSort} />
              </tr>
            </thead>
            <tbody>
              {sorted.map((ind, i) => (
                <tr key={ind.id} className="table-row">
                  <td className="py-2 px-1">
                    <input
                      type="checkbox"
                      className="accent-accent"
                      checked={selected.has(ind.id)}
                      onChange={(e) => {
                        const next = new Set(selected);
                        if (e.target.checked) next.add(ind.id);
                        else next.delete(ind.id);
                        setSelected(next);
                      }}
                    />
                  </td>
                  <td className="py-2 px-2 text-gray-600">{i + 1}</td>
                  <td className="py-2 px-2">
                    <Link
                      to={`/runs/${runId}/strategies/${ind.id}`}
                      className="text-accent hover:underline font-mono"
                    >
                      {ind.id.slice(0, 12)}
                    </Link>
                  </td>
                  <td className="py-2 px-2 text-right font-mono text-gray-200">
                    {ind.fitness?.toFixed(4) ?? '—'}
                  </td>
                  <td className={`py-2 px-2 text-right font-mono ${pnlColor(ind.profit)}`}>
                    {ind.profit !== null ? `${ind.profit > 0 ? '+' : ''}${ind.profit.toFixed(1)}%` : '—'}
                  </td>
                  <td className="py-2 px-2 text-right font-mono text-gray-300">
                    {ind.sharpe_ratio?.toFixed(2) ?? '—'}
                  </td>
                  <td className="py-2 px-2 text-right font-mono text-gray-300">
                    {ind.win_rate !== null ? `${(ind.win_rate * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td className="py-2 px-2 text-right font-mono text-gray-400">
                    {ind.num_trades ?? '—'}
                  </td>
                  <td className="py-2 px-2 text-right font-mono text-loss">
                    {ind.max_drawdown !== null ? `${(ind.max_drawdown * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td className="py-2 px-2 text-right font-mono text-gray-500">
                    {ind.complexity ?? '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function SortHeader({
  label,
  k,
  current,
  asc,
  onClick,
}: {
  label: string;
  k: SortKey;
  current: SortKey;
  asc: boolean;
  onClick: (k: SortKey) => void;
}) {
  return (
    <th
      className="text-right py-2 px-2 font-medium cursor-pointer hover:text-gray-300 transition-colors select-none"
      onClick={() => onClick(k)}
    >
      <span className="inline-flex items-center gap-0.5">
        {label}
        {k === current && (
          asc
            ? <TrendingUp className="w-2.5 h-2.5" />
            : <TrendingDown className="w-2.5 h-2.5" />
        )}
      </span>
    </th>
  );
}

function getVal(ind: IndividualSummary, key: SortKey): number | null {
  switch (key) {
    case 'fitness': return ind.fitness;
    case 'profit': return ind.profit;
    case 'sharpe_ratio': return ind.sharpe_ratio;
    case 'num_trades': return ind.num_trades;
    case 'max_drawdown': return ind.max_drawdown;
    case 'win_rate': return ind.win_rate;
    case 'complexity': return ind.complexity;
    default: return ind.fitness;
  }
}

function pnlColor(v: number | null): string {
  if (v === null) return 'text-gray-500';
  if (v > 0) return 'text-profit';
  if (v < 0) return 'text-loss';
  return 'text-gray-400';
}
