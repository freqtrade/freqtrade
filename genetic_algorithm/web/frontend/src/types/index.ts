/* ─── TypeScript interfaces matching backend Pydantic models ─── */

// ── Run ────────────────────────────────────────────────────────

export type RunStatus =
  | 'pending'
  | 'running'
  | 'paused'
  | 'stopping'
  | 'completed'
  | 'failed';

export interface GenerationStats {
  generation: number;
  size: number;
  best_fitness: number | null;
  avg_fitness: number | null;
  worst_fitness: number | null;
  median_fitness: number | null;
  best_raw_fitness: number | null;
  avg_raw_fitness: number | null;
  genetic_diversity: number | null;
  holdout_avg_degradation: number | null;
  holdout_best_degradation: number | null;
  holdout_num_evaluated: number | null;
  holdout_num_profitable: number | null;
  mutation_rate: number | null;
  holdout_penalties_applied: number | null;
  avg_holdout_penalty: number | null;
  avg_unused_indicators: number | null;
  eval_seconds: number | null;
}

export interface RunSummary {
  run_id: string;
  status: RunStatus;
  config_name: string;
  current_generation: number;
  total_generations: number;
  best_fitness: number | null;
  best_profit: number | null;
  population_size: number;
  started_at: number | null;
  elapsed_seconds: number | null;
  pairs: string[];
}

export interface RunDetail extends RunSummary {
  config: Record<string, unknown>;
  generation_stats: GenerationStats[];
  best_individual_id: string | null;
  mode: string;
}

// ── Metrics ────────────────────────────────────────────────────

export interface StrategyMetrics {
  total_profit?: number | null;
  profit_total?: number | null;
  total_profit_pct?: number | null;
  sharpe_ratio?: number | null;
  sortino_ratio?: number | null;
  win_rate?: number | null;
  num_trades?: number | null;
  max_drawdown?: number | null;
  profit_factor?: number | null;
  avg_profit?: number | null;
  avg_duration?: string | null;
  complexity?: number | null;
  monthly_stability?: number | null;
  cross_pair_score?: number | null;
  holdout_degradation?: number | null;
  mc_robustness?: number | null;
  train_val_gap?: number | null;
  [key: string]: unknown;
}

// ── Generation ─────────────────────────────────────────────────

export interface IndividualSummary {
  id: string;
  fitness: number | null;
  raw_fitness: number | null;
  rank: number;
  crowding_distance: number;
  evaluated: boolean;
  metrics: StrategyMetrics;
  profit: number | null;
  sharpe_ratio: number | null;
  sortino_ratio: number | null;
  win_rate: number | null;
  num_trades: number | null;
  max_drawdown: number | null;
  profit_factor: number | null;
  complexity: number | null;
  indicators: string[];
}

export interface GenerationDetail {
  run_id: string;
  generation: number;
  individuals: IndividualSummary[];
  stats: Record<string, unknown> | null;
}

// ── Strategy ───────────────────────────────────────────────────

export interface IndicatorModel {
  type: string;
  parameters: Record<string, unknown>;
  weight: number;
  instance_id: string | null;
  timeframe: string | null;
}

export interface ConditionModel {
  indicator: string;
  operator: string;
  threshold: unknown;
  logic: string;
  threshold_upper: unknown;
  lookback: number | null;
}

export interface StrategyGene {
  generation: number;
  individual_id: number;
  indicators: IndicatorModel[];
  entry_conditions: ConditionModel[];
  exit_conditions: ConditionModel[];
  timeframe: string;
  stoploss: number;
  minimal_roi: Record<string, number>;
  max_open_trades: number;
  informative_timeframes: string[];
  trailing_stop: boolean;
  trailing_stop_positive: number | null;
  trailing_stop_positive_offset: number | null;
  can_short: boolean;
}

export interface QualityAssessment {
  holdout_degradation: number | null;
  holdout_label: string;
  wf_gap: number | null;
  wf_label: string;
  mc_robustness: number | null;
  mc_label: string;
  composite_score: number | null;
  overall_label: string;
}

export interface StrategyDetail {
  id: string;
  run_id: string;
  generation: number;
  fitness: number | null;
  raw_fitness: number | null;
  metrics: StrategyMetrics;
  gene: StrategyGene | null;
  quality: QualityAssessment | null;
  parent_ids: string[];
  mutations: string[];
  walk_forward_windows: Record<string, unknown>[] | null;
  monte_carlo: Record<string, unknown> | null;
}

// ── Config ─────────────────────────────────────────────────────

export interface ConfigTemplate {
  name: string;
  path: string;
  pairs: string[];
  generations: number;
  population_size: number;
}

// ── WebSocket Events ───────────────────────────────────────────

export type EventType =
  | 'run.created'
  | 'run.started'
  | 'run.stopped'
  | 'run.paused'
  | 'run.resumed'
  | 'run.completed'
  | 'run.error'
  | 'generation.start'
  | 'generation.end'
  | 'phase.start'
  | 'phase.end'
  | 'eval.progress'
  | 'new_best'
  | 'convergence.warning'
  | 'checkpoint.saved'
  | 'evolution.complete'
  | 'strategy.injected'
  | 'log'
  | 'error'
  | 'heartbeat';

export interface WSEvent {
  type: EventType;
  run_id: string;
  data: Record<string, unknown>;
  timestamp: number;
}

// ── Backtest ───────────────────────────────────────────────────

export interface BacktestResult {
  backtest_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  result: Record<string, unknown> | null;
  error: string | null;
}

export interface BacktestTrade {
  pair: string;
  open_date: string;
  close_date: string;
  profit_ratio: number;
  profit_abs: number;
  trade_duration: number;
  is_short: boolean;
}

export interface BacktestTradesResponse {
  backtest_id: string;
  total: number;
  offset: number;
  limit: number;
  trades: BacktestTrade[];
}

// ── Data (OHLCV) ──────────────────────────────────────────────

export interface PairInfo {
  exchange: string;
  pair: string;
  timeframe: string;
  format: string;
  trading_mode?: string;
}

export interface OHLCVResponse {
  pair: string;
  timeframe: string;
  exchange: string;
  count: number;
  candles: number[][];  // [timestamp_ms, open, high, low, close, volume]
}

export interface IndicatorLineData {
  name: string;
  indicator_type: string;
  pane: 'price' | 'separate';
  data: number[][];  // [timestamp_ms, value]
}

export interface IndicatorsResponse {
  pair: string;
  timeframe: string;
  exchange: string;
  indicators: IndicatorLineData[];
}

// ── Hall of Fame ───────────────────────────────────────────────

export interface HoFEntry {
  id: string;
  fitness: number;
  profit: number;
  sharpe_ratio: number;
  num_trades: number;
  max_drawdown: number;
  win_rate: number;
  complexity: number;
  timeframe: string;
  added_at: string;
  config_name: string;
  run_id: string;
  generation_found: number;
  strategy_gene: Record<string, unknown>;
  [key: string]: unknown;
}

// ── Lineage ────────────────────────────────────────────────────

export interface LineageNode {
  id: string;
  generation: number;
  fitness: number | null;
  raw_fitness: number | null;
  profit: number | null;
  parent_ids: string[];
  mutations: unknown[];
}

export interface LineageResponse {
  strategy_id: string;
  run_id: string;
  chain: LineageNode[];
}

// ── Dry Run ────────────────────────────────────────────────────

export interface DryRunRequest {
  strategy_gene: Record<string, unknown>;
  exchange?: string;
  pairs?: string[];
  stake_amount?: number;
  timeframe?: string;
}

export interface DryRunStatus {
  dry_run_id: string;
  status: string;
  strategy_name: string;
  pid: number | null;
  started_at: number | null;
  error: string | null;
  log_tail: string[];
}
