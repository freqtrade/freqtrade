/**
 * Error boundary — catches React render errors and shows recovery UI.
 *
 * Wraps the main content area so a crash on one page doesn't blank the
 * entire app (the sidebar and navigation remain functional).
 */

import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null, errorInfo: null };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ errorInfo });
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center min-h-[50vh]">
          <div className="card max-w-lg w-full text-center space-y-4">
            <AlertCircle className="w-12 h-12 text-loss mx-auto" />
            <h2 className="text-lg font-semibold text-gray-200">Something went wrong</h2>
            <p className="text-sm text-gray-400">
              This page encountered an error. You can try again or navigate to a different page.
            </p>

            {this.state.error && (
              <details className="text-left">
                <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
                  Technical details
                </summary>
                <pre className="mt-2 text-[10px] text-gray-500 font-mono bg-surface-0 rounded-lg p-3 overflow-x-auto max-h-40 overflow-y-auto">
                  {this.state.error.message}
                  {this.state.errorInfo?.componentStack}
                </pre>
              </details>
            )}

            <button
              onClick={this.handleReset}
              className="inline-flex items-center gap-2 px-4 py-2 text-sm bg-accent text-white rounded-lg hover:bg-accent/90 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
