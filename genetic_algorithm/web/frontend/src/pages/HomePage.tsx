import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Activity, Dna, Trophy, Zap, Clock, TrendingUp, Plus } from 'lucide-react';
import { api } from '../api/client';
import { useStore } from '../store/useStore';
import { MetricsCard } from '../components/MetricsCard';
import { StatusBadge } from '../components/StatusBadge';
import { FitnessSparkline } from '../components/FitnessChart';
import { StartRunDialog } from '../components/StartRunDialog';
import type { RunSummary, HoFEntry, GenerationStats } from '../types';

export function HomePage() {
  const runsMap = useStore((s) => s.runs);
  const genStats = useStore((s) => s.generationStats);
  const connected = useStore((s) => s.connected);
  const setRuns = useStore((s) => s.setRuns);
  const events = useStore((s) => s.events);
  const [hofEntries, setHofEntries] = useState<HoFEntry[]>([]);
  const [showStartDialog, setShowStartDialog] = useState(false);

  // Initial fetch
  useEffect(() => {
    api.listRuns().then((runs) => setRuns(runs)).catch(() => {});
    api.getHallOfFame().then((hof) => {
      if (Array.isArray(hof)) setHofEntries(hof);
    }).catch(() => {});
  }, [setRuns]);

  const runs = Array.from(runsMap.values());
  const activeRuns = runs.filter((r) => r.status === 'running' || r.status === 'paused');
  const bestFitness = runs.reduce(
    (best, r) => (r.best_fitness !== null && (best === null || r.best_fitness > best) ? r.best_fitness : best),
    null as number | null,
  );
  const bestProfit = runs.reduce(
    (best, r) => (r.best_profit !== null && (best === null || r.best_profit > best) ? r.best_profit : best),
    null as number | null,
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">Dashboard</h1>
          <p className="text-sm text-gray-500 mt-0.5">GA Evolution Monitoring</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowStartDialog(true)}
            className="flex items-center gap-2 bg-profit hover:bg-profit/80 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
          >
            <Plus className="w-4 h-4" /> New Evolution
          </button>
          <Link
            to="/runs"
            className="bg-accent hover:bg-accent-hover text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
          >
            View All Runs
          </Link>
        </div>
      </div>

      <StartRunDialog open={showStartDialog} onClose={() => setShowStartDialog(false)} />

      {/* Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricsCard
          label="Active Runs"
          value={activeRuns.length}
          icon={<Activity className="w-4 h-4" />}
        />
        <MetricsCard
          label="Total Runs"
          value={runs.length}
          icon={<Dna className="w-4 h-4" />}
        />
        <MetricsCard
          label="Best Fitness"
          value={bestFitness?.toFixed(4) ?? '—'}
          trend={bestFitness !== null ? 'up' : undefined}
          icon={<TrendingUp className="w-4 h-4" />}
        />
        <MetricsCard
          label="Hall of Fame"
          value={hofEntries.length}
          icon={<Trophy className="w-4 h-4" />}
        />
      </div>

      {/* Active Runs */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-medium text-gray-300">Active Runs</h2>
          {connected && (
            <span className="flex items-center gap-1.5 text-[10px] text-profit">
              <span className="w-1.5 h-1.5 rounded-full bg-profit pulse-dot" />
              LIVE
            </span>
          )}
        </div>

        {activeRuns.length === 0 ? (
          <div className="text-center py-8 text-gray-500 text-sm">
            <Zap className="w-8 h-8 mx-auto mb-2 opacity-50" />
            No active evolution runs
            <div className="mt-2">
              <span className="text-xs text-gray-600">
                Start a run from the Runs page or via CLI with <code className="text-gray-500">--dashboard</code>
              </span>
            </div>
          </div>
        ) : (
          <div className="space-y-2">
            {activeRuns.map((run) => (
              <RunRow key={run.run_id} run={run} stats={genStats.get(run.run_id) || []} />
            ))}
          </div>
        )}
      </div>

      {/* Recent Events */}
      <div className="card">
        <h2 className="text-sm font-medium text-gray-300 mb-3">Recent Events</h2>
        <div className="space-y-1 max-h-64 overflow-y-auto">
          {events.length === 0 ? (
            <p className="text-gray-500 text-sm py-4 text-center">No events yet</p>
          ) : (
            events
              .slice(-20)
              .reverse()
              .map((ev, i) => (
                <div key={i} className="flex items-center gap-3 text-xs py-1.5 table-row px-2">
                  <span className="text-gray-600 font-mono w-20">
                    {new Date(ev.timestamp * 1000).toLocaleTimeString()}
                  </span>
                  <span className="text-accent font-medium w-32 truncate">{ev.type}</span>
                  <span className="text-gray-500 truncate">{ev.run_id}</span>
                  <span className="text-gray-600 ml-auto truncate max-w-xs">
                    {summarizeEventData(ev.data)}
                  </span>
                </div>
              ))
          )}
        </div>
      </div>
    </div>
  );
}

function RunRow({ run, stats }: { run: RunSummary; stats: GenerationStats[] }) {
  const progress = run.total_generations > 0
    ? (run.current_generation / run.total_generations) * 100
    : 0;

  return (
    <Link
      to={`/runs/${run.run_id}`}
      className="flex items-center gap-4 p-3 rounded-lg bg-surface-2/50 hover:bg-surface-2 transition-colors"
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-200 truncate">{run.run_id}</span>
          <StatusBadge status={run.status} />
        </div>
        <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
          <span>Gen {run.current_generation}/{run.total_generations}</span>
          <span>Pop {run.population_size}</span>
          {run.pairs.length > 0 && <span>{run.pairs.join(', ')}</span>}
        </div>
      </div>

      <FitnessSparkline data={stats} />

      <div className="text-right min-w-[80px]">
        {run.best_fitness !== null && (
          <div className="text-sm font-mono text-accent">{run.best_fitness.toFixed(4)}</div>
        )}
        {run.best_profit !== null && (
          <div className="text-xs font-mono text-profit">
            {run.best_profit > 0 ? '+' : ''}{run.best_profit.toFixed(1)}%
          </div>
        )}
      </div>

      {/* Progress bar */}
      <div className="w-24">
        <div className="h-1.5 bg-surface-3 rounded-full overflow-hidden">
          <div
            className="h-full bg-accent rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="text-[10px] text-gray-500 mt-0.5 text-right">{progress.toFixed(0)}%</div>
      </div>
    </Link>
  );
}

function summarizeEventData(data: Record<string, unknown>): string {
  const parts: string[] = [];
  if ('generation' in data) parts.push(`gen=${(data.generation as number) + 1}`);
  if ('best_fitness' in data) parts.push(`fit=${(data.best_fitness as number)?.toFixed(4)}`);
  if ('profit' in data) parts.push(`pnl=${(data.profit as number)?.toFixed(1)}%`);
  if ('phase' in data) parts.push(`${data.phase}`);
  if ('progress' in data) parts.push(`${((data.progress as number) * 100).toFixed(0)}%`);
  return parts.join(' · ') || JSON.stringify(data).slice(0, 50);
}
