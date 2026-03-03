/**
 * Toast notification component — displays brief messages for key events.
 *
 * Renders a stack of toasts in the bottom-right corner.
 * Auto-dismissed after a configurable duration.
 */

import { X, CheckCircle2, AlertCircle, Info, AlertTriangle } from 'lucide-react';
import { useStore } from '../store/useStore';
import type { Toast as ToastType } from '../store/useStore';

const iconMap = {
  success: CheckCircle2,
  error: AlertCircle,
  info: Info,
  warning: AlertTriangle,
};

const colorMap = {
  success: 'border-green-500/40 bg-green-500/10',
  error: 'border-red-500/40 bg-red-500/10',
  info: 'border-blue-500/40 bg-blue-500/10',
  warning: 'border-yellow-500/40 bg-yellow-500/10',
};

const iconColorMap = {
  success: 'text-green-400',
  error: 'text-red-400',
  info: 'text-blue-400',
  warning: 'text-yellow-400',
};

function ToastItem({ toast }: { toast: ToastType }) {
  const removeToast = useStore((s) => s.removeToast);
  const Icon = iconMap[toast.type];

  return (
    <div
      className={`
        flex items-start gap-2.5 px-3.5 py-2.5 rounded-lg border shadow-lg
        backdrop-blur-sm max-w-sm animate-slide-in
        ${colorMap[toast.type]}
      `}
    >
      <Icon className={`w-4 h-4 mt-0.5 flex-shrink-0 ${iconColorMap[toast.type]}`} />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-200">{toast.title}</p>
        {toast.message && (
          <p className="text-xs text-gray-400 mt-0.5 truncate">{toast.message}</p>
        )}
      </div>
      <button
        onClick={() => removeToast(toast.id)}
        className="text-gray-500 hover:text-gray-300 transition-colors flex-shrink-0"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

export function ToastContainer() {
  const toasts = useStore((s) => s.toasts);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} />
      ))}
    </div>
  );
}
