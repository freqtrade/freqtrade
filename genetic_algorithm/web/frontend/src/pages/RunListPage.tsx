import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Plus, Play, Square, Pause, RefreshCw } from 'lucide-react';
import { api } from '../api/client';
import { useStore } from '../store/useStore';
import { StatusBadge } from '../components/StatusBadge';
import { StartRunDialog } from '../components/StartRunDialog';
import type { RunSummary } from '../types';

export function RunListPage() {
  const runsMap = useStore((s) => s.runs);
  const setRuns = useStore((s) => s.setRuns);
  const [loading, setLoading] = useState(true);
  const [showStartDialog, setShowStartDialog] = useState(false);

  useEffect(() => {
    setLoading(true);
    api.listRuns()
      .then((runs) => setRuns(runs))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [setRuns]);

  const runs = Array.from(runsMap.values())
    .sort((a, b) => (b.started_at || 0) - (a.started_at || 0));

  const handleStop = async (e: React.MouseEvent, runId: string) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      await api.stopRun(runId);
    } catch (err) {
      console.error('Failed to stop run', err);
    }
  };

  const handlePause = async (e: React.MouseEvent, runId: string) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      await api.pauseRun(runId);
    } catch (err) {
      console.error('Failed to pause run', err);
    }
  };

  const handleResume = async (e: React.MouseEvent, runId: string) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      await api.resumeRun(runId);
    } catch (err) {
      console.error('Failed to resume run', err);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-100">Evolution Runs</h1>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowStartDialog(true)}
            className="flex items-center gap-2 bg-profit hover:bg-profit/80 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
          >
            <Plus className="w-4 h-4" /> New Evolution
          </button>
          <button
            onClick={() => api.listRuns().then(setRuns)}
            className="flex items-center gap-1.5 text-sm text-gray-400 hover:text-gray-200 transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Refresh
          </button>
        </div>
      </div>

      <StartRunDialog open={showStartDialog} onClose={() => setShowStartDialog(false)} />

      {loading ? (
        <div className="text-center py-16 text-gray-500">Loading runs...</div>
      ) : runs.length === 0 ? (
        <div className="card text-center py-16">
          <p className="text-gray-500 mb-4">No evolution runs found</p>
          <p className="text-xs text-gray-600">
            Start a run with <code className="text-gray-400">python run_ga.py --dashboard</code>
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {runs.map((run) => (
            <Link
              key={run.run_id}
              to={`/runs/${run.run_id}`}
              className="card-hover flex items-center gap-4"
            >
              {/* Status & ID */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-gray-200 truncate font-mono">
                    {run.run_id}
                  </span>
                  <StatusBadge status={run.status} />
                </div>
                <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
                  {run.config_name && <span>{run.config_name}</span>}
                  <span>Pop: {run.population_size}</span>
                  {run.pairs.length > 0 && (
                    <span className="truncate max-w-xs">{run.pairs.join(', ')}</span>
                  )}
                </div>
              </div>

              {/* Progress */}
              <div className="text-center min-w-[100px]">
                <div className="text-sm font-mono text-gray-300">
                  {run.current_generation} / {run.total_generations}
                </div>
                <div className="text-[10px] text-gray-500">generations</div>
                <div className="h-1 bg-surface-3 rounded-full mt-1.5 overflow-hidden">
                  <div
                    className="h-full bg-accent rounded-full transition-all"
                    style={{
                      width: `${run.total_generations > 0
                        ? (run.current_generation / run.total_generations) * 100
                        : 0}%`,
                    }}
                  />
                </div>
              </div>

              {/* Metrics */}
              <div className="text-right min-w-[100px]">
                {run.best_fitness !== null && (
                  <div className="text-sm font-mono text-accent">
                    {run.best_fitness.toFixed(4)}
                  </div>
                )}
                {run.best_profit !== null && (
                  <div className={`text-xs font-mono ${run.best_profit >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {run.best_profit > 0 ? '+' : ''}{run.best_profit.toFixed(1)}%
                  </div>
                )}
                {run.elapsed_seconds != null && (
                  <div className="text-[10px] text-gray-600 mt-0.5">
                    {formatDuration(run.elapsed_seconds)}
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="flex items-center gap-1">
                {run.status === 'running' && (
                  <>
                    <button
                      onClick={(e) => handlePause(e, run.run_id)}
                      className="p-1.5 rounded-lg hover:bg-yellow-500/10 text-yellow-500 transition-colors"
                      title="Pause"
                    >
                      <Pause className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={(e) => handleStop(e, run.run_id)}
                      className="p-1.5 rounded-lg hover:bg-red-500/10 text-red-500 transition-colors"
                      title="Stop"
                    >
                      <Square className="w-3.5 h-3.5" />
                    </button>
                  </>
                )}
                {run.status === 'paused' && (
                  <>
                    <button
                      onClick={(e) => handleResume(e, run.run_id)}
                      className="p-1.5 rounded-lg hover:bg-green-500/10 text-green-500 transition-colors"
                      title="Resume"
                    >
                      <Play className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={(e) => handleStop(e, run.run_id)}
                      className="p-1.5 rounded-lg hover:bg-red-500/10 text-red-500 transition-colors"
                      title="Stop"
                    >
                      <Square className="w-3.5 h-3.5" />
                    </button>
                  </>
                )}
              </div>
            </Link>
          ))}
        </div>
      )}
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
