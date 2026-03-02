/**
 * Reusable loading & error state components.
 *
 * Standardizes the appearance of loading spinners and error displays
 * across all pages, replacing ad-hoc inline patterns.
 */

import { Loader2, AlertCircle, RefreshCw, Inbox } from 'lucide-react';

// ── Loading ───────────────────────────────────────────────

interface LoadingStateProps {
  message?: string;
  /** Compact inline version vs. full-page centered */
  compact?: boolean;
}

export function LoadingState({ message = 'Loading...', compact = false }: LoadingStateProps) {
  if (compact) {
    return (
      <div className="flex items-center gap-2 text-gray-500 text-sm py-4">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span>{message}</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center py-16 text-gray-500">
      <Loader2 className="w-8 h-8 animate-spin mb-3 text-accent/60" />
      <p className="text-sm">{message}</p>
    </div>
  );
}

// ── Error ─────────────────────────────────────────────────

interface ErrorStateProps {
  title?: string;
  message?: string;
  onRetry?: () => void;
  compact?: boolean;
}

export function ErrorState({
  title = 'Something went wrong',
  message,
  onRetry,
  compact = false,
}: ErrorStateProps) {
  if (compact) {
    return (
      <div className="flex items-center gap-2 text-loss text-sm py-4">
        <AlertCircle className="w-4 h-4 flex-shrink-0" />
        <span>{message || title}</span>
        {onRetry && (
          <button
            onClick={onRetry}
            className="text-accent hover:underline text-xs ml-2"
          >
            Retry
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="card text-center py-16 space-y-3">
      <AlertCircle className="w-10 h-10 text-loss mx-auto" />
      <h3 className="text-sm font-medium text-gray-200">{title}</h3>
      {message && (
        <p className="text-xs text-gray-500 max-w-md mx-auto">{message}</p>
      )}
      {onRetry && (
        <button
          onClick={onRetry}
          className="inline-flex items-center gap-1.5 text-xs text-accent hover:underline mt-2"
        >
          <RefreshCw className="w-3 h-3" />
          Try again
        </button>
      )}
    </div>
  );
}

// ── Empty ─────────────────────────────────────────────────

interface EmptyStateProps {
  title?: string;
  message?: string;
  icon?: React.ReactNode;
}

export function EmptyState({
  title = 'No data',
  message,
  icon,
}: EmptyStateProps) {
  return (
    <div className="text-center py-12 space-y-2">
      {icon || <Inbox className="w-8 h-8 text-gray-600 mx-auto" />}
      <h3 className="text-sm font-medium text-gray-400">{title}</h3>
      {message && (
        <p className="text-xs text-gray-600 max-w-sm mx-auto">{message}</p>
      )}
    </div>
  );
}
