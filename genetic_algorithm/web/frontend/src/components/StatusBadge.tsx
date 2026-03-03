import type { RunStatus } from '../types';
import { clsx } from 'clsx';

const statusConfig: Record<RunStatus, { label: string; class: string; dot: string }> = {
  running:   { label: 'Running',   class: 'badge-running',   dot: 'bg-blue-400 pulse-dot' },
  paused:    { label: 'Paused',    class: 'badge-paused',    dot: 'bg-yellow-400' },
  completed: { label: 'Completed', class: 'badge-completed', dot: 'bg-green-400' },
  failed:    { label: 'Failed',    class: 'badge-failed',    dot: 'bg-red-400' },
  pending:   { label: 'Pending',   class: 'badge-pending',   dot: 'bg-gray-400' },
  stopping:  { label: 'Stopping',  class: 'badge-stopping',  dot: 'bg-orange-400' },
};

export function StatusBadge({ status }: { status: RunStatus }) {
  const cfg = statusConfig[status] || statusConfig.pending;
  return (
    <span className={cfg.class}>
      <span className={clsx('w-1.5 h-1.5 rounded-full', cfg.dot)} />
      {cfg.label}
    </span>
  );
}
