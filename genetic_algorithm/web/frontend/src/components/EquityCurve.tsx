/**
 * Equity curve chart — shows cumulative profit over time for a backtest.
 *
 * Uses Recharts AreaChart with gradient fill.
 * Data is computed from sequential trade results.
 */

import { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import type { BacktestTrade } from '../types';

interface EquityCurveProps {
  trades: BacktestTrade[];
  height?: number;
  /** Starting balance (default 1000) */
  startingBalance?: number;
}

interface EquityPoint {
  time: string;        // formatted date
  timestamp: number;   // for sorting
  equity: number;      // cumulative balance
  drawdown: number;    // drawdown from peak (%)
  tradeNum: number;    // trade sequence number
}

export function EquityCurve({
  trades,
  height = 300,
  startingBalance = 1000,
}: EquityCurveProps) {
  const data = useMemo(() => {
    if (!trades || trades.length === 0) return [];

    // Sort trades by close_date
    const sorted = [...trades].sort(
      (a, b) => new Date(a.close_date).getTime() - new Date(b.close_date).getTime()
    );

    let balance = startingBalance;
    let peak = startingBalance;
    const points: EquityPoint[] = [
      {
        time: formatDate(sorted[0].open_date),
        timestamp: new Date(sorted[0].open_date).getTime(),
        equity: startingBalance,
        drawdown: 0,
        tradeNum: 0,
      },
    ];

    sorted.forEach((trade, i) => {
      balance += trade.profit_abs;
      peak = Math.max(peak, balance);
      const drawdown = peak > 0 ? ((peak - balance) / peak) * 100 : 0;

      points.push({
        time: formatDate(trade.close_date),
        timestamp: new Date(trade.close_date).getTime(),
        equity: Number(balance.toFixed(2)),
        drawdown: Number(drawdown.toFixed(2)),
        tradeNum: i + 1,
      });
    });

    return points;
  }, [trades, startingBalance]);

  if (data.length === 0) {
    return (
      <div className="text-center text-gray-500 text-xs py-8">
        No trade data available for equity curve
      </div>
    );
  }

  const minEquity = Math.min(...data.map((d) => d.equity));
  const maxEquity = Math.max(...data.map((d) => d.equity));
  const finalEquity = data[data.length - 1].equity;
  const totalReturn = ((finalEquity - startingBalance) / startingBalance) * 100;
  const maxDrawdown = Math.max(...data.map((d) => d.drawdown));

  return (
    <div className="space-y-2">
      {/* Summary badges */}
      <div className="flex gap-3 text-xs">
        <span className="text-gray-500">
          Final:{' '}
          <span className={finalEquity >= startingBalance ? 'text-profit' : 'text-loss'}>
            {finalEquity.toFixed(2)} ({totalReturn >= 0 ? '+' : ''}{totalReturn.toFixed(1)}%)
          </span>
        </span>
        <span className="text-gray-500">
          Max DD:{' '}
          <span className="text-loss">{maxDrawdown.toFixed(1)}%</span>
        </span>
        <span className="text-gray-500">
          Trades: <span className="text-gray-300">{trades.length}</span>
        </span>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
          <defs>
            <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 10, fill: '#6b7280' }}
            tickLine={{ stroke: '#374151' }}
            axisLine={{ stroke: '#374151' }}
            minTickGap={40}
          />
          <YAxis
            domain={[Math.floor(minEquity * 0.98), Math.ceil(maxEquity * 1.02)]}
            tick={{ fontSize: 10, fill: '#6b7280' }}
            tickLine={{ stroke: '#374151' }}
            axisLine={{ stroke: '#374151' }}
            width={60}
          />
          <Tooltip
            contentStyle={{
              background: '#1e2030',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 8,
              fontSize: 11,
            }}
            labelStyle={{ color: '#9ca3af' }}
            formatter={(value: number, name: string) => {
              if (name === 'equity') return [`$${value.toFixed(2)}`, 'Equity'];
              if (name === 'drawdown') return [`${value.toFixed(1)}%`, 'Drawdown'];
              return [value, name];
            }}
          />
          <ReferenceLine
            y={startingBalance}
            stroke="rgba(255,255,255,0.15)"
            strokeDasharray="4 4"
          />
          <Area
            type="monotone"
            dataKey="equity"
            stroke="#3b82f6"
            strokeWidth={1.5}
            fill="url(#equityGradient)"
            dot={false}
            activeDot={{ r: 3, fill: '#3b82f6' }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function formatDate(dateStr: string): string {
  try {
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch {
    return dateStr;
  }
}
