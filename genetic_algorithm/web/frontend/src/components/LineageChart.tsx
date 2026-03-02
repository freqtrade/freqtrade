/**
 * LineageChart — shows a strategy's fitness evolution across generations.
 *
 * Traces the parent chain backwards and plots fitness/profit over generations.
 * Uses recharts (same library as FitnessChart).
 */

import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';
import type { LineageNode } from '../types';

interface LineageChartProps {
  chain: LineageNode[];
  height?: number;
}

export function LineageChart({ chain, height = 250 }: LineageChartProps) {
  if (chain.length === 0) return null;

  const data = chain.map((node) => ({
    generation: node.generation,
    fitness: node.fitness,
    profit: node.profit,
    id: node.id,
    mutations: Array.isArray(node.mutations) ? node.mutations.length : 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
        <XAxis
          dataKey="generation"
          stroke="#6b7280"
          tick={{ fontSize: 10, fill: '#9ca3af' }}
          label={{ value: 'Generation', position: 'insideBottom', offset: -2, style: { fontSize: 10, fill: '#6b7280' } }}
        />
        <YAxis
          yAxisId="fitness"
          stroke="#6b7280"
          tick={{ fontSize: 10, fill: '#9ca3af' }}
          width={55}
          label={{ value: 'Fitness', angle: -90, position: 'insideLeft', style: { fontSize: 10, fill: '#6b7280' } }}
        />
        <YAxis
          yAxisId="profit"
          orientation="right"
          stroke="#6b7280"
          tick={{ fontSize: 10, fill: '#9ca3af' }}
          width={55}
          label={{ value: 'Profit %', angle: 90, position: 'insideRight', style: { fontSize: 10, fill: '#6b7280' } }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1e2030',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '8px',
            fontSize: 11,
          }}
          labelFormatter={(gen) => `Generation ${gen}`}
          formatter={(value: number, name: string) => {
            if (name === 'fitness') return [value?.toFixed(4) ?? '—', 'Fitness'];
            if (name === 'profit') return [value !== null ? `${value > 0 ? '+' : ''}${value.toFixed(1)}%` : '—', 'Profit'];
            return [value, name];
          }}
        />
        <Legend wrapperStyle={{ fontSize: 10 }} />
        <Line
          yAxisId="fitness"
          type="monotone"
          dataKey="fitness"
          stroke="#6366f1"
          strokeWidth={2}
          dot={{ r: 3, fill: '#6366f1' }}
          activeDot={{ r: 5 }}
          name="fitness"
          connectNulls
        />
        <Line
          yAxisId="profit"
          type="monotone"
          dataKey="profit"
          stroke="#22c55e"
          strokeWidth={2}
          dot={{ r: 3, fill: '#22c55e' }}
          activeDot={{ r: 5 }}
          name="profit"
          connectNulls
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
