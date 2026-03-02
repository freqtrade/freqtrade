import { clsx } from 'clsx';

interface MetricsCardProps {
  label: string;
  value: string | number | null;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  icon?: React.ReactNode;
}

export function MetricsCard({ label, value, subtitle, trend, icon }: MetricsCardProps) {
  return (
    <div className="card flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-400 uppercase tracking-wider">{label}</span>
        {icon && <span className="text-gray-500">{icon}</span>}
      </div>
      <div
        className={clsx(
          'text-2xl font-semibold font-mono',
          trend === 'up' && 'text-profit',
          trend === 'down' && 'text-loss',
          !trend && 'text-gray-100',
        )}
      >
        {value ?? '—'}
      </div>
      {subtitle && <span className="text-xs text-gray-500">{subtitle}</span>}
    </div>
  );
}
