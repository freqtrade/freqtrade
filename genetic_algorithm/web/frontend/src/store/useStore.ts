/**
 * Global Zustand store — single source of truth for the dashboard.
 *
 * Manages:
 *  - WebSocket connection state
 *  - Event stream (ring buffer)
 *  - Active run summaries (updated from WS events)
 */

import { create } from 'zustand';
import type { WSEvent, RunSummary, GenerationStats } from '../types';

const MAX_EVENTS = 2000;
const MAX_TOASTS = 5;

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning';
  title: string;
  message?: string;
  duration?: number;  // ms, 0 = sticky
}

interface DashboardState {
  // Connection
  connected: boolean;
  setConnected: (c: boolean) => void;

  // Events
  events: WSEvent[];
  pushEvent: (e: WSEvent) => void;
  clearEvents: () => void;

  // Runs (kept in sync from API + WS events)
  runs: Map<string, RunSummary>;
  setRuns: (runs: RunSummary[]) => void;
  updateRun: (runId: string, patch: Partial<RunSummary>) => void;
  removeRun: (runId: string) => void;

  // Per-run generation stats (accumulated from WS events)
  generationStats: Map<string, GenerationStats[]>;
  pushGenerationStats: (runId: string, stats: GenerationStats) => void;

  // Per-run phase tracking (from phase.start / phase.end events)
  runPhases: Map<string, { phase: string; startedAt: number } | null>;

  // Per-run eval progress (from eval.progress events)
  runEvalProgress: Map<string, { completed: number; total: number } | null>;

  // Toast notifications
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
}

export const useStore = create<DashboardState>((set, get) => ({
  // ── Connection ──────────────────────────────────────────
  connected: false,
  setConnected: (c) => set({ connected: c }),

  // ── Events ──────────────────────────────────────────────
  events: [],
  pushEvent: (e) => {
    const state = get();

    // Buffer event
    const events = [...state.events, e].slice(-MAX_EVENTS);

    // Auto-update run state from certain event types
    const runs = new Map(state.runs);
    const genStats = new Map(state.generationStats);

    if (e.type === 'run.started') {
      const data = e.data as Record<string, unknown>;
      runs.set(e.run_id, {
        run_id: e.run_id,
        status: 'running',
        config_name: (data.config_name as string) || '',
        current_generation: 0,
        total_generations: (data.generations as number) || (data.total_generations as number) || 0,
        best_fitness: null,
        best_profit: null,
        population_size: (data.population_size as number) || 0,
        started_at: e.timestamp,
        elapsed_seconds: 0,
        pairs: (data.pairs as string[]) || [],
      });
    } else if (e.type === 'generation.end') {
      const data = e.data as Record<string, unknown>;
      const existing = runs.get(e.run_id);
      if (existing) {
        runs.set(e.run_id, {
          ...existing,
          current_generation: (data.generation as number) ?? existing.current_generation,
          best_fitness: (data.best_fitness as number) ?? existing.best_fitness,
          best_profit: (data.best_profit as number) ?? existing.best_profit,
          elapsed_seconds: (e.timestamp - (existing.started_at || e.timestamp)),
        });
      }

      // Accumulate generation stats
      const stats = genStats.get(e.run_id) || [];
      const genStat: GenerationStats = {
        generation: (data.generation as number) ?? 0,
        size: (data.size as number) ?? 0,
        best_fitness: (data.best_fitness as number) ?? null,
        avg_fitness: (data.avg_fitness as number) ?? null,
        worst_fitness: (data.worst_fitness as number) ?? null,
        median_fitness: (data.median_fitness as number) ?? null,
        best_raw_fitness: (data.best_raw_fitness as number) ?? null,
        avg_raw_fitness: (data.avg_raw_fitness as number) ?? null,
        genetic_diversity: (data.genetic_diversity as number) ?? null,
        holdout_avg_degradation: (data.holdout_avg_degradation as number) ?? null,
        holdout_best_degradation: (data.holdout_best_degradation as number) ?? null,
        holdout_num_evaluated: (data.holdout_num_evaluated as number) ?? null,
        holdout_num_profitable: (data.holdout_num_profitable as number) ?? null,
        mutation_rate: (data.mutation_rate as number) ?? null,
        holdout_penalties_applied: (data.holdout_penalties_applied as number) ?? null,
        avg_holdout_penalty: (data.avg_holdout_penalty as number) ?? null,
        avg_unused_indicators: (data.avg_unused_indicators as number) ?? null,
        eval_seconds: (data.eval_seconds as number) ?? null,
      };
      genStats.set(e.run_id, [...stats, genStat]);
    } else if (e.type === 'new_best') {
      const data = e.data as Record<string, unknown>;
      const ind = (data.individual as Record<string, unknown>) || data;
      const metrics = (ind.metrics as Record<string, unknown>) || {};
      const existing = runs.get(e.run_id);
      if (existing) {
        runs.set(e.run_id, {
          ...existing,
          best_fitness: (ind.fitness as number) ?? existing.best_fitness,
          best_profit: (metrics.profit as number) ?? existing.best_profit,
        });
      }
    } else if (e.type === 'run.stopped' || e.type === 'evolution.complete') {
      const existing = runs.get(e.run_id);
      if (existing) {
        runs.set(e.run_id, {
          ...existing,
          status: 'completed',
        });
      }
    }

    // Phase tracking
    const phases = new Map(state.runPhases);
    if (e.type === 'phase.start') {
      const data = e.data as Record<string, unknown>;
      phases.set(e.run_id, {
        phase: (data.phase as string) || 'unknown',
        startedAt: e.timestamp,
      });
    } else if (e.type === 'phase.end') {
      phases.set(e.run_id, null);
    } else if (e.type === 'generation.end') {
      // Clear phase on new gen end
      phases.set(e.run_id, null);
    }

    // Eval progress tracking
    const evalProgress = new Map(state.runEvalProgress);
    if (e.type === 'eval.progress') {
      const data = e.data as Record<string, unknown>;
      evalProgress.set(e.run_id, {
        completed: (data.completed as number) ?? 0,
        total: (data.total as number) ?? 0,
      });
    } else if (e.type === 'generation.end' || e.type === 'phase.end') {
      // Clear eval progress after gen completes or phase ends
      evalProgress.set(e.run_id, null);
    }

    set({ events, runs, generationStats: genStats, runPhases: phases, runEvalProgress: evalProgress });

    // Auto-toast for notable events
    const addToast = get().addToast;
    if (e.type === 'new_best') {
      const data = e.data as Record<string, unknown>;
      const ind = (data.individual as Record<string, unknown>) || data;
      const fitness = (ind.fitness as number)?.toFixed(4) ?? '?';
      addToast({ type: 'success', title: 'New Best Found', message: `Fitness: ${fitness} (${e.run_id})` });
    } else if (e.type === 'evolution.complete' || e.type === 'run.completed') {
      addToast({ type: 'info', title: 'Run Completed', message: e.run_id });
    } else if (e.type === 'run.error') {
      const data = e.data as Record<string, unknown>;
      addToast({ type: 'error', title: 'Run Error', message: (data.error as string) || e.run_id, duration: 10000 });
    } else if (e.type === 'run.stopped') {
      addToast({ type: 'warning', title: 'Run Stopped', message: e.run_id });
    }
  },
  clearEvents: () => set({ events: [] }),

  // ── Runs ────────────────────────────────────────────────
  runs: new Map(),
  setRuns: (runs) => {
    const m = new Map<string, RunSummary>();
    for (const r of runs) m.set(r.run_id, r);
    set({ runs: m });
  },
  updateRun: (runId, patch) => {
    const runs = new Map(get().runs);
    const existing = runs.get(runId);
    if (existing) {
      runs.set(runId, { ...existing, ...patch });
      set({ runs });
    }
  },
  removeRun: (runId) => {
    const runs = new Map(get().runs);
    runs.delete(runId);
    set({ runs });
  },

  // ── Generation Stats ───────────────────────────────────
  generationStats: new Map(),
  pushGenerationStats: (runId, stats) => {
    const m = new Map(get().generationStats);
    const existing = m.get(runId) || [];
    m.set(runId, [...existing, stats]);
    set({ generationStats: m });
  },

  // ── Phase & Progress ──────────────────────────────────
  runPhases: new Map(),
  runEvalProgress: new Map(),

  // ── Toasts ──────────────────────────────────────────────
  toasts: [],
  addToast: (toast) => {
    const id = `toast_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    const full: Toast = { ...toast, id };
    set((state) => ({
      toasts: [...state.toasts, full].slice(-MAX_TOASTS),
    }));
    // Auto-dismiss after duration (default 5s)
    const dur = toast.duration ?? 5000;
    if (dur > 0) {
      setTimeout(() => {
        get().removeToast(id);
      }, dur);
    }
  },
  removeToast: (id) =>
    set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
}));
