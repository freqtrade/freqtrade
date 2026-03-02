/**
 * Lightweight fetch wrapper for the GA dashboard REST API.
 *
 * In dev, Vite proxies /api → http://127.0.0.1:8501/api.
 * In prod, the React build is served by FastAPI on the same origin.
 */

const BASE = '';  // same origin via proxy or static mount

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

// ── Runs ──────────────────────────────────────────────
import type {
  RunSummary,
  RunDetail,
  GenerationDetail,
  StrategyDetail,
  ConfigTemplate,
  HoFEntry,
  BacktestResult,
  BacktestTradesResponse,
  PairInfo,
  OHLCVResponse,
  LineageResponse,
  DryRunStatus,
} from '../types';

export const api = {
  // Health
  health: () => request<{ status: string; version: string; active_runs: number }>('/api/health'),

  // Runs
  listRuns: () => request<RunSummary[]>('/api/runs'),
  getRun: (id: string) => request<RunDetail>(`/api/runs/${id}`),
  startRun: (config: Record<string, unknown>, runId?: string) =>
    request<RunSummary>('/api/runs', {
      method: 'POST',
      body: JSON.stringify({ config, run_id: runId }),
    }),
  stopRun: (id: string) =>
    request<{ status: string }>(`/api/runs/${id}/stop`, { method: 'POST' }),
  pauseRun: (id: string) =>
    request<{ status: string }>(`/api/runs/${id}/pause`, { method: 'POST' }),
  resumeRun: (id: string) =>
    request<{ status: string }>(`/api/runs/${id}/resume`, { method: 'POST' }),
  checkpointRun: (id: string) =>
    request<{ status: string }>(`/api/runs/${id}/checkpoint`, { method: 'POST' }),

  // Generations
  getGeneration: (runId: string, gen: number) =>
    request<GenerationDetail>(`/api/runs/${runId}/generations/${gen}`),

  // Strategies
  getStrategy: (runId: string, strategyId: string) =>
    request<StrategyDetail>(`/api/runs/${runId}/strategies/${strategyId}`),
  getStrategyCode: (runId: string, strategyId: string) =>
    request<{ strategy_id: string; code: string }>(`/api/runs/${runId}/strategies/${strategyId}/code`),

  // Hall of Fame
  getHallOfFame: () => request<HoFEntry[]>('/api/hall-of-fame'),

  // Config
  getConfigTemplates: () => request<ConfigTemplate[]>('/api/config/templates'),
  getConfigTemplate: (name: string) => request<Record<string, unknown>>(`/api/config/templates/${name}`),
  validateConfig: (config: Record<string, unknown>) =>
    request<{ valid: boolean; errors: string[]; warnings: string[] }>('/api/config/validate', {
      method: 'POST',
      body: JSON.stringify(config),
    }),

  // Backtest
  startBacktest: (body: {
    strategy_gene: Record<string, unknown>;
    timerange: string;
    pairs?: string[];
    timeframe?: string;
    exchange?: string;
  }) =>
    request<BacktestResult>('/api/backtest', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  getBacktestResult: (id: string) =>
    request<BacktestResult>(`/api/backtest/${id}`),
  getBacktestTrades: (id: string, opts?: { offset?: number; limit?: number; pair?: string }) => {
    const params = new URLSearchParams();
    if (opts?.offset) params.set('offset', String(opts.offset));
    if (opts?.limit) params.set('limit', String(opts.limit));
    if (opts?.pair) params.set('pair', opts.pair);
    const qs = params.toString();
    return request<BacktestTradesResponse>(`/api/backtest/${id}/trades${qs ? `?${qs}` : ''}`);
  },

  // Data (OHLCV)
  listPairs: (exchange?: string) => {
    const qs = exchange ? `?exchange=${encodeURIComponent(exchange)}` : '';
    return request<{ pairs: PairInfo[]; message?: string }>(`/api/data/pairs${qs}`);
  },
  getOHLCV: (opts: {
    pair: string;
    timeframe: string;
    exchange?: string;
    start?: string;
    end?: string;
    limit?: number;
  }) => {
    const params = new URLSearchParams();
    params.set('pair', opts.pair);
    params.set('timeframe', opts.timeframe);
    if (opts.exchange) params.set('exchange', opts.exchange);
    if (opts.start) params.set('start', opts.start);
    if (opts.end) params.set('end', opts.end);
    if (opts.limit) params.set('limit', String(opts.limit));
    return request<OHLCVResponse>(`/api/data/ohlcv?${params.toString()}`);
  },

  // Indicators
  getIndicators: (opts: {
    pair: string;
    timeframe: string;
    indicators: string;
    exchange?: string;
    start?: string;
    end?: string;
    limit?: number;
  }) => {
    const params = new URLSearchParams();
    params.set('pair', opts.pair);
    params.set('timeframe', opts.timeframe);
    params.set('indicators', opts.indicators);
    if (opts.exchange) params.set('exchange', opts.exchange);
    if (opts.start) params.set('start', opts.start);
    if (opts.end) params.set('end', opts.end);
    if (opts.limit) params.set('limit', String(opts.limit));
    return request<import('../types').IndicatorsResponse>(`/api/data/indicators?${params.toString()}`);
  },

  // Inject
  injectStrategy: (runId: string, body: { strategy_gene: Record<string, unknown>; source_description?: string }) =>
    request<{ status: string }>(`/api/runs/${runId}/inject`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  // Run config
  getRunConfig: (runId: string) =>
    request<Record<string, unknown>>(`/api/runs/${runId}/config`),

  // Lineage
  getLineage: (runId: string, strategyId: string) =>
    request<LineageResponse>(`/api/runs/${runId}/lineage/${strategyId}`),

  // Dry Run
  startDryRun: (body: {
    strategy_gene: Record<string, unknown>;
    exchange?: string;
    pairs?: string[];
    stake_amount?: number;
    timeframe?: string;
  }) =>
    request<DryRunStatus>('/api/dry-run', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  getDryRunStatus: (id: string) =>
    request<DryRunStatus>(`/api/dry-run/${id}`),
  stopDryRun: (id: string) =>
    request<DryRunStatus>(`/api/dry-run/${id}/stop`, { method: 'POST' }),
  listDryRuns: () =>
    request<DryRunStatus[]>('/api/dry-run'),
};
