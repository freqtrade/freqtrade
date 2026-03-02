import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import type { GenerationStats } from '../types';

interface FitnessChartProps {
  data: GenerationStats[];
  height?: number;
  showDiversity?: boolean;
}

export function FitnessChart({ data, height = 300, showDiversity = false }: FitnessChartProps) {
  if (!data.length) {
    return (
      <div className="card flex items-center justify-center" style={{ height }}>
        <span className="text-gray-500">No generation data yet</span>
      </div>
    );
  }

  return (
    <div className="card">
      <h3 className="text-sm font-medium text-gray-300 mb-3">Fitness Over Generations</h3>
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <defs>
            <linearGradient id="fitGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="avgGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="generation"
            stroke="#6b7280"
            tick={{ fill: '#9ca3af', fontSize: 11 }}
          />
          <YAxis
            stroke="#6b7280"
            tick={{ fill: '#9ca3af', fontSize: 11 }}
            width={60}
          />
          {showDiversity && (
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="#f59e0b"
              tick={{ fill: '#f59e0b', fontSize: 11 }}
              width={50}
              domain={[0, 1]}
              tickFormatter={(v: number) => v.toFixed(1)}
            />
          )}
          <Tooltip
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '8px',
              color: '#e5e7eb',
              fontSize: 12,
            }}
            formatter={(value: number) => value?.toFixed(4)}
          />
          <Legend
            wrapperStyle={{ color: '#9ca3af', fontSize: 11 }}
          />
          <Area
            type="monotone"
            dataKey="best_fitness"
            name="Best"
            stroke="#3b82f6"
            fill="url(#fitGrad)"
            strokeWidth={2}
            dot={false}
          />
          <Area
            type="monotone"
            dataKey="avg_fitness"
            name="Average"
            stroke="#8b5cf6"
            fill="url(#avgGrad)"
            strokeWidth={1.5}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="worst_fitness"
            name="Worst"
            stroke="#ef4444"
            strokeWidth={1}
            dot={false}
            strokeDasharray="4 4"
          />
          {showDiversity && (
            <Line
              type="monotone"
              dataKey="genetic_diversity"
              name="Diversity"
              stroke="#f59e0b"
              strokeWidth={1}
              dot={false}
              yAxisId="right"
            />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

/* Compact sparkline version for list cards */
export function FitnessSparkline({ data }: { data: GenerationStats[] }) {
  if (data.length < 2) return null;
  return (
    <ResponsiveContainer width={120} height={32}>
      <LineChart data={data}>
        <Line
          type="monotone"
          dataKey="best_fitness"
          stroke="#3b82f6"
          strokeWidth={1.5}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
