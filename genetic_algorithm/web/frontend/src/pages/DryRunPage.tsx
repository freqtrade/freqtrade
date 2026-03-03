/**
 * DryRunPage — shows status and live logs for a dry-run trading session.
 *
 * Route: /dry-run/:dryRunId
 * Polls the backend every 3 seconds for updated status and log output.
 */

import { useEffect, useState, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Square, Loader2, AlertCircle, CheckCircle, Zap } from 'lucide-react';
import { api } from '../api/client';
import type { DryRunStatus } from '../types';

const POLL_INTERVAL = 3000;

export function DryRunPage() {
  const { dryRunId } = useParams<{ dryRunId: string }>();
  const [status, setStatus] = useState<DryRunStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (!dryRunId) return;

    const poll = async () => {
      try {
        const s = await api.getDryRunStatus(dryRunId);
        setStatus(s);
        setError(null);
        // Stop polling if terminal state
        if (s.status === 'stopped' || s.status === 'failed') {
          if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoading(false);
      }
    };

    poll();
    pollRef.current = setInterval(poll, POLL_INTERVAL);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [dryRunId]);

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [status?.log_tail]);

  const handleStop = async () => {
    if (!dryRunId) return;
    setStopping(true);
    try {
      const s = await api.stopDryRun(dryRunId);
      setStatus(s);
    } catch (err) {
      console.error('Failed to stop:', err);
    } finally {
      setStopping(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16 text-gray-500 gap-2">
        <Loader2 className="w-5 h-5 animate-spin" /> Loading dry run...
      </div>
    );
  }

  if (error || !status) {
    return (
      <div className="card text-center py-16">
        <AlertCircle className="w-8 h-8 text-loss mx-auto mb-2" />
        <p className="text-loss mb-2">Failed to load dry run</p>
        <p className="text-xs text-gray-500">{error || 'Not found'}</p>
      </div>
    );
  }

  const statusColor =
    status.status === 'running' ? 'text-green-400' :
    status.status === 'starting' ? 'text-yellow-400' :
    status.status === 'stopped' ? 'text-gray-400' :
    status.status === 'failed' ? 'text-loss' : 'text-gray-400';

  const StatusIcon =
    status.status === 'running' ? Zap :
    status.status === 'starting' ? Loader2 :
    status.status === 'stopped' ? CheckCircle :
    AlertCircle;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <Link to="/" className="text-sm text-gray-500 hover:text-gray-300 flex items-center gap-1 mb-1">
          <ArrowLeft className="w-3 h-3" /> Back
        </Link>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold text-gray-100 font-mono">{dryRunId}</h1>
            <span className={`text-xs font-medium uppercase flex items-center gap-1 ${statusColor}`}>
              <StatusIcon className={`w-3 h-3 ${status.status === 'starting' ? 'animate-spin' : ''}`} />
              {status.status}
            </span>
          </div>
          {(status.status === 'running' || status.status === 'starting') && (
            <button
              onClick={handleStop}
              disabled={stopping}
              className="flex items-center gap-1.5 text-xs bg-red-500/10 text-red-500 border border-red-500/20 px-3 py-1.5 rounded-lg hover:bg-red-500/20 transition-colors disabled:opacity-50"
            >
              <Square className="w-3 h-3" />
              {stopping ? 'Stopping...' : 'Stop'}
            </button>
          )}
        </div>
      </div>

      {/* Info Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="card">
          <div className="text-[10px] text-gray-500 uppercase">Strategy</div>
          <div className="text-sm font-mono text-gray-200 truncate">{status.strategy_name}</div>
        </div>
        <div className="card">
          <div className="text-[10px] text-gray-500 uppercase">PID</div>
          <div className="text-sm font-mono text-gray-200">{status.pid ?? '—'}</div>
        </div>
        <div className="card">
          <div className="text-[10px] text-gray-500 uppercase">Started</div>
          <div className="text-sm font-mono text-gray-200">
            {status.started_at ? new Date(status.started_at * 1000).toLocaleString() : '—'}
          </div>
        </div>
        <div className="card">
          <div className="text-[10px] text-gray-500 uppercase">Status</div>
          <div className={`text-sm font-mono uppercase ${statusColor}`}>{status.status}</div>
        </div>
      </div>

      {/* Error */}
      {status.error && (
        <div className="card border border-red-500/20 bg-red-500/5">
          <div className="text-xs text-red-400 font-medium mb-1">Error</div>
          <pre className="text-xs text-red-300 font-mono whitespace-pre-wrap">{status.error}</pre>
        </div>
      )}

      {/* Log Output */}
      <div className="card">
        <h2 className="text-sm font-medium text-gray-300 mb-3">Log Output</h2>
        <pre
          ref={logRef}
          className="text-xs text-gray-400 font-mono bg-surface-0 p-4 rounded-lg max-h-[500px] overflow-y-auto overflow-x-auto leading-relaxed"
        >
          {status.log_tail.length > 0
            ? status.log_tail.join('\n')
            : 'Waiting for output...'}
        </pre>
      </div>
    </div>
  );
}
