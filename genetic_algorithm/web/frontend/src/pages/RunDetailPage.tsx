import { useEffect, useState, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  Pause,
  Play,
  Square,
  Save,
  ArrowLeft,
  Clock,
  Users,
  TrendingUp,
  BarChart3,
  Activity,
} from 'lucide-react';
import { api } from '../api/client';
import { useStore } from '../store/useStore';
import { StatusBadge } from '../components/StatusBadge';
import { MetricsCard } from '../components/MetricsCard';
import { FitnessChart } from '../components/FitnessChart';
import type { RunDetail, GenerationStats } from '../types';
import { LoadingState, ErrorState } from '../components/StateDisplays';

const EMPTY_GEN_STATS: GenerationStats[] = [];

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const [run, setRun] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  // Live gen stats from WS — use stable fallback to avoid infinite re-renders
  const wsGenStats = useStore((s) => s.generationStats.get(runId || '')) ?? EMPTY_GEN_STATS;
  const wsRun = useStore((s) => (runId ? s.runs.get(runId) : undefined));
  const currentPhase = useStore((s) => s.runPhases.get(runId || '') || null);
  const evalProgress = useStore((s) => s.runEvalProgress.get(runId || '') || null);

  useEffect(() => {
    if (!runId) return;
    setLoading(true);
    api
      .getRun(runId)
      .then((detail) => {
        setRun(detail);
        setError(null);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [runId]);

  // Merge API generation stats with live WS stats
  const allGenStats = useMemo(() => {
    const apiStats = run?.generation_stats || [];
    const merged = new Map<number, GenerationStats>();
    for (const s of apiStats) merged.set(s.generation, s);
    for (const s of wsGenStats) merged.set(s.generation, s);
    return Array.from(merged.values()).sort((a, b) => a.generation - b.generation);
  }, [run?.generation_stats, wsGenStats]);

  // Live-updating values from WS
  const currentGen = wsRun?.current_generation ?? run?.current_generation ?? 0;
  const totalGen = wsRun?.total_generations ?? run?.total_generations ?? 0;
  const status = wsRun?.status ?? run?.status ?? 'pending';
  const bestFitness = wsRun?.best_fitness ?? run?.best_fitness ?? null;
  const bestProfit = wsRun?.best_profit ?? run?.best_profit ?? null;

  const runAction = async (action: string, fn: () => Promise<unknown>) => {
    setActionLoading(action);
    try {
      await fn();
    } catch (err) {
      console.error(`${action} failed:`, err);
    } finally {
      setActionLoading(null);
    }
  };

  if (loading) {
    return <LoadingState message="Loading run details..." />;
  }

  if (error || !run) {
    return (
      <ErrorState
        title="Failed to load run"
        message={error || 'Run not found'}
        onRetry={() => window.location.reload()}
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <Link to="/runs" className="text-sm text-gray-500 hover:text-gray-300 flex items-center gap-1 mb-1">
            <ArrowLeft className="w-3 h-3" /> Back to runs
          </Link>
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold text-gray-100 font-mono">{run.run_id}</h1>
            <StatusBadge status={status} />
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          {status === 'running' && (
            <>
              <button
                onClick={() => runAction('pause', () => api.pauseRun(run.run_id))}
                disabled={actionLoading !== null}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-yellow-500/10 text-yellow-500 text-sm hover:bg-yellow-500/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Pause className="w-3.5 h-3.5" />
                {actionLoading === 'pause' ? 'Pausing...' : 'Pause'}
              </button>
              <button
                onClick={() => runAction('checkpoint', () => api.checkpointRun(run.run_id))}
                disabled={actionLoading !== null}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-500/10 text-blue-400 text-sm hover:bg-blue-500/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Save className="w-3.5 h-3.5" />
                {actionLoading === 'checkpoint' ? 'Saving...' : 'Checkpoint'}
              </button>
              <button
                onClick={() => runAction('stop', () => api.stopRun(run.run_id))}
                disabled={actionLoading !== null}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/10 text-red-500 text-sm hover:bg-red-500/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Square className="w-3.5 h-3.5" />
                {actionLoading === 'stop' ? 'Stopping...' : 'Stop'}
              </button>
            </>
          )}
          {status === 'paused' && (
            <>
              <button
                onClick={() => runAction('resume', () => api.resumeRun(run.run_id))}
                disabled={actionLoading !== null}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-green-500/10 text-green-500 text-sm hover:bg-green-500/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Play className="w-3.5 h-3.5" />
                {actionLoading === 'resume' ? 'Resuming...' : 'Resume'}
              </button>
              <button
                onClick={() => runAction('stop', () => api.stopRun(run.run_id))}
                disabled={actionLoading !== null}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/10 text-red-500 text-sm hover:bg-red-500/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Square className="w-3.5 h-3.5" />
                {actionLoading === 'stop' ? 'Stopping...' : 'Stop'}
              </button>
            </>
          )}
        </div>
      </div>

      {/* Phase indicator + eval progress */}
      {status === 'running' && (currentPhase || evalProgress) && (
        <div className="flex items-center gap-4">
          {currentPhase && (
            <div className="flex items-center gap-2 text-sm">
              <Activity className="w-3.5 h-3.5 text-accent animate-pulse" />
              <span className="text-gray-400">Phase:</span>
              <span className="text-gray-200 font-medium capitalize">{currentPhase.phase}</span>
            </div>
          )}
          {evalProgress && evalProgress.total > 0 && (
            <div className="flex items-center gap-2 flex-1">
              <span className="text-xs text-gray-500 whitespace-nowrap">
                Eval {evalProgress.completed}/{evalProgress.total}
              </span>
              <div className="flex-1 h-1.5 bg-surface-0 rounded-full overflow-hidden max-w-xs">
                <div
                  className="h-full bg-accent rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(100, (evalProgress.completed / evalProgress.total) * 100)}%` }}
                />
              </div>
              <span className="text-xs text-gray-500">
                {Math.round((evalProgress.completed / evalProgress.total) * 100)}%
              </span>
            </div>
          )}
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        <MetricsCard
          label="Generation"
          value={`${currentGen} / ${totalGen}`}
          subtitle={totalGen > 0 ? `${((currentGen / totalGen) * 100).toFixed(0)}% complete` : undefined}
          icon={<BarChart3 className="w-4 h-4" />}
        />
        <MetricsCard
          label="Best Fitness"
          value={bestFitness?.toFixed(4) ?? '—'}
          trend={bestFitness !== null ? 'up' : undefined}
          icon={<TrendingUp className="w-4 h-4" />}
        />
        <MetricsCard
          label="Best Profit"
          value={bestProfit !== null ? `${bestProfit > 0 ? '+' : ''}${bestProfit.toFixed(1)}%` : '—'}
          trend={bestProfit !== null ? (bestProfit >= 0 ? 'up' : 'down') : undefined}
        />
        <MetricsCard
          label="Population"
          value={run.population_size}
          icon={<Users className="w-4 h-4" />}
        />
        <MetricsCard
          label="Elapsed"
          value={run.elapsed_seconds !== null ? formatDuration(run.elapsed_seconds!) : '—'}
          icon={<Clock className="w-4 h-4" />}
        />
      </div>

      {/* Fitness Chart */}
      <FitnessChart data={allGenStats} height={350} showDiversity />

      {/* Generation Table */}
      <div className="card">
        <h2 className="text-sm font-medium text-gray-300 mb-3">Generations</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-500 uppercase tracking-wider border-b border-white/5">
                <th className="text-left py-2 px-3 font-medium">Gen</th>
                <th className="text-right py-2 px-3 font-medium">Best</th>
                <th className="text-right py-2 px-3 font-medium">Avg</th>
                <th className="text-right py-2 px-3 font-medium">Worst</th>
                <th className="text-right py-2 px-3 font-medium">Diversity</th>
                <th className="text-right py-2 px-3 font-medium">Mut Rate</th>
                <th className="text-right py-2 px-3 font-medium">Eval (s)</th>
                <th className="text-right py-2 px-3 font-medium">HO Deg</th>
              </tr>
            </thead>
            <tbody>
              {allGenStats.map((gs) => (
                <tr key={gs.generation} className="table-row">
                  <td className="py-2 px-3">
                    <Link
                      to={`/runs/${run.run_id}/generations/${gs.generation}`}
                      className="text-accent hover:underline font-mono"
                    >
                      {gs.generation}
                    </Link>
                  </td>
                  <td className="text-right py-2 px-3 font-mono text-profit">
                    {gs.best_fitness?.toFixed(4) ?? '—'}
                  </td>
                  <td className="text-right py-2 px-3 font-mono text-gray-300">
                    {gs.avg_fitness?.toFixed(4) ?? '—'}
                  </td>
                  <td className="text-right py-2 px-3 font-mono text-gray-500">
                    {gs.worst_fitness?.toFixed(4) ?? '—'}
                  </td>
                  <td className="text-right py-2 px-3 font-mono text-yellow-500">
                    {gs.genetic_diversity?.toFixed(3) ?? '—'}
                  </td>
                  <td className="text-right py-2 px-3 font-mono text-gray-400">
                    {gs.mutation_rate?.toFixed(3) ?? '—'}
                  </td>
                  <td className="text-right py-2 px-3 font-mono text-gray-500">
                    {gs.eval_seconds?.toFixed(1) ?? '—'}
                  </td>
                  <td className="text-right py-2 px-3 font-mono text-gray-500">
                    {gs.holdout_best_degradation !== null
                      ? `${(gs.holdout_best_degradation * 100).toFixed(1)}%`
                      : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Config (collapsible) */}
      <details className="card group">
        <summary className="text-sm font-medium text-gray-300 cursor-pointer select-none">
          Configuration
          <span className="text-xs text-gray-500 ml-2 group-open:hidden">Click to expand</span>
        </summary>
        <pre className="mt-3 text-xs text-gray-400 font-mono overflow-x-auto bg-surface-0 p-3 rounded-lg max-h-96 overflow-y-auto">
          {JSON.stringify(run.config, null, 2)}
        </pre>
      </details>
    </div>
  );
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}
