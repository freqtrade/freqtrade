import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Component, type ReactNode } from 'react';
import { Layout } from './components/Layout';
import { useWebSocket } from './hooks/useWebSocket';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { HomePage } from './pages/HomePage';
import { RunListPage } from './pages/RunListPage';
import { RunDetailPage } from './pages/RunDetailPage';
import { GenerationPage } from './pages/GenerationPage';
import { StrategyPage } from './pages/StrategyPage';
import { HallOfFamePage } from './pages/HallOfFamePage';
import { ConfigPage } from './pages/ConfigPage';
import { BacktestResultPage } from './pages/BacktestResultPage';
import { ComparePage } from './pages/ComparePage';
import { DryRunPage } from './pages/DryRunPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { RunComparePage } from './pages/RunComparePage';

class ErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-surface-0 flex items-center justify-center p-8">
          <div className="max-w-lg w-full bg-surface-1 border border-white/10 rounded-xl p-6 space-y-4">
            <h2 className="text-lg font-bold text-loss">Something went wrong</h2>
            <pre className="text-xs text-gray-400 bg-surface-0 rounded-lg p-3 overflow-auto max-h-48">
              {this.state.error?.message}
            </pre>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: null });
                window.location.href = '/';
              }}
              className="text-xs bg-accent text-white px-4 py-2 rounded-lg hover:bg-accent/90 transition-colors"
            >
              Go Home
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function AppInner() {
  // Establish WebSocket connection on mount
  useWebSocket();
  // Register keyboard shortcuts
  useKeyboardShortcuts();

  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/runs" element={<RunListPage />} />
        <Route path="/runs/compare" element={<RunComparePage />} />
        <Route path="/runs/:runId" element={<RunDetailPage />} />
        <Route path="/runs/:runId/generations/:gen" element={<GenerationPage />} />
        <Route path="/runs/:runId/strategies/:strategyId" element={<StrategyPage />} />
        <Route path="/hall-of-fame" element={<HallOfFamePage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/config" element={<ConfigPage />} />
        <Route path="/backtest/:backtestId" element={<BacktestResultPage />} />
        <Route path="/dry-run/:dryRunId" element={<DryRunPage />} />
        <Route path="/compare" element={<ComparePage />} />
      </Route>
    </Routes>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <AppInner />
      </BrowserRouter>
    </ErrorBoundary>
  );
}
