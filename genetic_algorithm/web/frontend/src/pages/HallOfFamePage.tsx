import { useEffect, useState, useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Trophy, TrendingUp, BarChart3, ArrowUpDown, Search, Syringe, X, Download, GitCompare } from 'lucide-react';
import { api } from '../api/client';
import { useStore } from '../store/useStore';
import type { HoFEntry, RunSummary } from '../types';
import { exportToCsv } from '../utils/csv';
import { LoadingState } from '../components/StateDisplays';

type SortField = 'fitness' | 'profit' | 'sharpe_ratio' | 'win_rate' | 'num_trades' | 'max_drawdown' | 'complexity';

export function HallOfFamePage() {
  const [entries, setEntries] = useState<HoFEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortField, setSortField] = useState<SortField>('fitness');
  const [sortAsc, setSortAsc] = useState(false);
  const [filter, setFilter] = useState('');
  const [injectTarget, setInjectTarget] = useState<{ entry: HoFEntry; showModal: boolean } | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const runsMap = useStore((s) => s.runs);
  const navigate = useNavigate();

  useEffect(() => {
    api
      .getHallOfFame()
      .then((data) => {
        if (Array.isArray(data)) setEntries(data);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const filtered = useMemo(() => {
    let result = [...entries];
    if (filter) {
      const f = filter.toLowerCase();
      result = result.filter(
        (e) =>
          (e.config_name || '').toLowerCase().includes(f) ||
          (e.timeframe || '').toLowerCase().includes(f) ||
          String(e.id).toLowerCase().includes(f),
      );
    }
    result.sort((a, b) => {
      const av = (a[sortField] as number) ?? -Infinity;
      const bv = (b[sortField] as number) ?? -Infinity;
      return sortAsc ? av - bv : bv - av;
    });
    return result;
  }, [entries, sortField, sortAsc, filter]);

  const activeRuns = Array.from(runsMap.values()).filter(
    (r) => r.status === 'running' || r.status === 'paused',
  );

  const SortHeader = ({ field, label, className }: { field: SortField; label: string; className?: string }) => (
    <th
      className={`py-2 px-3 font-medium cursor-pointer hover:text-gray-300 transition-colors select-none ${className || ''}`}
      onClick={() => handleSort(field)}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {sortField === field && (
          <ArrowUpDown className="w-3 h-3 text-accent" />
        )}
      </span>
    </th>
  );

  if (loading) return <LoadingState message="Loading Hall of Fame..." />;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Trophy className="w-5 h-5 text-yellow-500" />
          <h1 className="text-2xl font-bold text-gray-100">Hall of Fame</h1>
          <span className="text-sm text-gray-500 ml-2">{entries.length} strategies</span>
          {entries.length > 0 && (
            <button
              onClick={() => {
                const rows = entries.map((e) => ({
                  id: e.id,
                  fitness: e.fitness,
                  profit: e.profit,
                  sharpe_ratio: e.sharpe_ratio,
                  num_trades: e.num_trades,
                  max_drawdown: e.max_drawdown,
                  win_rate: e.win_rate,
                  complexity: e.complexity,
                  timeframe: e.timeframe,
                  config_name: e.config_name,
                  added_at: e.added_at,
                }));
                exportToCsv('hall_of_fame.csv', rows);
              }}
              className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200 transition-colors ml-2"
            >
              <Download className="w-3 h-3" /> Export CSV
            </button>
          )}
          {selected.size >= 2 && (
            <button
              onClick={() => {
                const ids = [...selected]
                  .map((id) => {
                    const e = entries.find((x) => String(x.id) === id);
                    const runId = e?.run_id || 'unknown';
                    return `${runId}:${id}`;
                  })
                  .join(',');
                navigate(`/compare?ids=${ids}`);
              }}
              className="flex items-center gap-1 text-xs text-accent hover:text-accent/80 transition-colors ml-2"
            >
              <GitCompare className="w-3 h-3" /> Compare {selected.size}
            </button>
          )}
        </div>
        <div className="relative">
          <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
          <input
            type="text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter by config, timeframe, id..."
            className="bg-surface-0 border border-white/10 rounded-lg pl-9 pr-3 py-1.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-accent/50 w-64"
          />
        </div>
      </div>

      {entries.length === 0 ? (
        <div className="card text-center py-16">
          <Trophy className="w-10 h-10 mx-auto mb-3 text-gray-600" />
          <p className="text-gray-500">No strategies in the Hall of Fame yet</p>
          <p className="text-xs text-gray-600 mt-1">
            Top strategies from completed evolution runs will appear here
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-500 uppercase tracking-wider border-b border-white/5">
                <th className="text-left py-2 px-1 font-medium w-6">
                  <input
                    type="checkbox"
                    className="accent-accent"
                    checked={selected.size === filtered.length && filtered.length > 0}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelected(new Set(filtered.map((x) => String(x.id))));
                      } else {
                        setSelected(new Set());
                      }
                    }}
                  />
                </th>
                <th className="text-left py-2 px-3 font-medium">#</th>
                <th className="text-left py-2 px-3 font-medium">ID</th>
                <SortHeader field="fitness" label="Fitness" className="text-right" />
                <SortHeader field="profit" label="Profit" className="text-right" />
                <SortHeader field="sharpe_ratio" label="Sharpe" className="text-right" />
                <SortHeader field="win_rate" label="Win Rate" className="text-right" />
                <SortHeader field="num_trades" label="Trades" className="text-right" />
                <SortHeader field="max_drawdown" label="Max DD" className="text-right" />
                <SortHeader field="complexity" label="Complexity" className="text-right" />
                <th className="text-right py-2 px-3 font-medium">Timeframe</th>
                <th className="text-right py-2 px-3 font-medium">Config</th>
                {activeRuns.length > 0 && (
                  <th className="text-center py-2 px-3 font-medium">Actions</th>
                )}
              </tr>
            </thead>
            <tbody>
              {filtered.map((entry, i) => {
                const hasRunId = !!entry.run_id;
                const runId = entry.run_id;
                const idStr = String(entry.id).slice(0, 12);

                return (
                  <tr key={entry.id || i} className="table-row">
                    <td className="py-2 px-1">
                      <input
                        type="checkbox"
                        className="accent-accent"
                        checked={selected.has(String(entry.id))}
                        onChange={(e) => {
                          const next = new Set(selected);
                          const k = String(entry.id);
                          if (e.target.checked) next.add(k);
                          else next.delete(k);
                          setSelected(next);
                        }}
                      />
                    </td>
                    <td className="py-2 px-3 text-gray-600">{i + 1}</td>
                    <td className="py-2 px-3 font-mono">
                      {hasRunId ? (
                        <Link
                          to={`/runs/${runId}/strategies/${entry.id}`}
                          className="text-accent hover:underline"
                        >
                          {idStr}
                        </Link>
                      ) : (
                        <span className="text-accent">{idStr}</span>
                      )}
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-gray-200">
                      {entry.fitness?.toFixed(4) ?? '—'}
                    </td>
                    <td className={`py-2 px-3 text-right font-mono ${entry.profit >= 0 ? 'text-profit' : 'text-loss'}`}>
                      {entry.profit !== undefined
                        ? `${entry.profit > 0 ? '+' : ''}${entry.profit.toFixed(1)}%`
                        : '—'}
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-gray-300">
                      {entry.sharpe_ratio?.toFixed(2) ?? '—'}
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-gray-300">
                      {entry.win_rate !== undefined ? `${(entry.win_rate * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-gray-400">
                      {entry.num_trades ?? '—'}
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-loss">
                      {entry.max_drawdown !== undefined ? `${(entry.max_drawdown * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-gray-500">
                      {entry.complexity ?? '—'}
                    </td>
                    <td className="py-2 px-3 text-right text-gray-400">
                      {entry.timeframe || '—'}
                    </td>
                    <td className="py-2 px-3 text-right text-gray-500 truncate max-w-[120px]">
                      {entry.config_name || '—'}
                    </td>
                    {activeRuns.length > 0 && (
                      <td className="py-2 px-3 text-center">
                        <button
                          onClick={() => setInjectTarget({ entry, showModal: true })}
                          className="inline-flex items-center gap-1 text-[10px] text-accent hover:underline"
                          title="Inject into active run"
                        >
                          <Syringe className="w-3 h-3" /> Inject
                        </button>
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Inject Modal */}
      {injectTarget?.showModal && (
        <InjectModal
          entry={injectTarget.entry}
          activeRuns={activeRuns}
          onClose={() => setInjectTarget(null)}
        />
      )}
    </div>
  );
}

function InjectModal({
  entry,
  activeRuns,
  onClose,
}: {
  entry: HoFEntry;
  activeRuns: RunSummary[];
  onClose: () => void;
}) {
  const [injecting, setInjecting] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleInject = async (runId: string) => {
    setInjecting(true);
    try {
      // The HoF entry has strategy_gene included by the backend
      const gene = entry.strategy_gene || entry;
      await api.injectStrategy(runId, {
        strategy_gene: gene as Record<string, unknown>,
        source_description: `Hall of Fame: ${entry.id}`,
      });
      setResult(`Injected into ${runId}`);
      setTimeout(onClose, 1500);
    } catch (err) {
      setResult(`Error: ${err instanceof Error ? err.message : String(err)}`);
      setInjecting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-surface-1 border border-white/10 rounded-xl shadow-2xl w-full max-w-sm p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-gray-100">Inject Strategy</h3>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300">
            <X className="w-4 h-4" />
          </button>
        </div>
        <p className="text-xs text-gray-400 mb-3">
          Inject <span className="text-accent font-mono">{String(entry.id).slice(0, 12)}</span> into:
        </p>
        {result ? (
          <p className={`text-sm ${result.startsWith('Error') ? 'text-loss' : 'text-profit'}`}>{result}</p>
        ) : (
          <div className="space-y-1.5">
            {activeRuns.map((run) => (
              <button
                key={run.run_id}
                onClick={() => handleInject(run.run_id)}
                disabled={injecting}
                className="w-full text-left px-3 py-2 rounded-lg text-sm text-gray-300 hover:bg-white/[0.05] transition-colors disabled:opacity-50"
              >
                <span className="font-mono">{run.run_id}</span>
                <span className="text-xs text-gray-500 ml-2">Gen {run.current_generation}/{run.total_generations}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
