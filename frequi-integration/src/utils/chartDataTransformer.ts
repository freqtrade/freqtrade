/**
 * Data transformation utilities for converting Freqtrade API responses
 * to TradingView Lightweight Charts format
 */

import { CandlestickData, HistogramData, LineData, Time } from 'lightweight-charts';

export interface FreqtradeCandle {
  date: number; // timestamp in milliseconds
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
  [key: string]: any; // For additional indicators
}

export interface FreqtradePairHistory {
  strategy: string;
  pair: string;
  timeframe: string;
  timeframe_ms: number;
  columns: string[];
  data: any[][];
  annotations?: any[];
  length: number;
  buy_signals?: number;
  sell_signals?: number;
  enter_long_signals?: number;
  exit_long_signals?: number;
  enter_short_signals?: number;
  exit_short_signals?: number;
  last_analyzed?: string;
  last_analyzed_ts?: number;
  data_start_ts?: number;
  data_start?: string;
  data_stop?: string;
  data_stop_ts?: number;
}

/**
 * Convert Freqtrade API response to Lightweight Charts candlestick format
 *
 * @param pairHistory - The PairHistory object from Freqtrade API
 * @returns Array of candlestick data points
 */
export function transformToCandlestickData(
  pairHistory: FreqtradePairHistory
): CandlestickData[] {
  const { columns, data } = pairHistory;

  // Find column indices
  const dateIdx = columns.indexOf('date');
  const openIdx = columns.indexOf('open');
  const highIdx = columns.indexOf('high');
  const lowIdx = columns.indexOf('low');
  const closeIdx = columns.indexOf('close');

  if (dateIdx === -1 || openIdx === -1 || highIdx === -1 ||
      lowIdx === -1 || closeIdx === -1) {
    throw new Error('Missing required OHLC columns in data');
  }

  return data.map((candle) => ({
    time: Math.floor(candle[dateIdx] / 1000) as Time, // Convert ms to seconds
    open: candle[openIdx],
    high: candle[highIdx],
    low: candle[lowIdx],
    close: candle[closeIdx],
  }));
}

/**
 * Extract volume data from Freqtrade response
 *
 * @param pairHistory - The PairHistory object from Freqtrade API
 * @returns Array of volume histogram data or null if volume column doesn't exist
 */
export function transformToVolumeData(
  pairHistory: FreqtradePairHistory
): HistogramData[] | null {
  const { columns, data } = pairHistory;

  const dateIdx = columns.indexOf('date');
  const volumeIdx = columns.indexOf('volume');
  const openIdx = columns.indexOf('open');
  const closeIdx = columns.indexOf('close');

  if (dateIdx === -1 || volumeIdx === -1) {
    return null;
  }

  return data.map((candle) => {
    const isUp = closeIdx !== -1 && openIdx !== -1
      ? candle[closeIdx] >= candle[openIdx]
      : true;

    return {
      time: Math.floor(candle[dateIdx] / 1000) as Time,
      value: candle[volumeIdx],
      color: isUp ? '#26a69a' : '#ef5350',
    };
  });
}

/**
 * Extract a specific indicator data from Freqtrade response
 *
 * @param pairHistory - The PairHistory object from Freqtrade API
 * @param indicatorName - The name of the indicator column (e.g., 'sma_50', 'ema_200')
 * @returns Array of line data or null if indicator doesn't exist
 */
export function extractIndicatorData(
  pairHistory: FreqtradePairHistory,
  indicatorName: string
): LineData[] | null {
  const { columns, data } = pairHistory;

  const dateIdx = columns.indexOf('date');
  const indicatorIdx = columns.indexOf(indicatorName);

  if (dateIdx === -1 || indicatorIdx === -1) {
    return null;
  }

  // Filter out null/NaN values which are common in indicators during warm-up period
  return data
    .filter((candle) => {
      const value = candle[indicatorIdx];
      return value !== null && value !== undefined && !isNaN(value);
    })
    .map((candle) => ({
      time: Math.floor(candle[dateIdx] / 1000) as Time,
      value: candle[indicatorIdx],
    }));
}

/**
 * Extract all indicators from the response automatically
 * This will skip OHLCV columns and common signal columns
 *
 * @param pairHistory - The PairHistory object from Freqtrade API
 * @returns Map of indicator name to line data
 */
export function extractAllIndicators(
  pairHistory: FreqtradePairHistory
): Map<string, LineData[]> {
  const indicators = new Map<string, LineData[]>();
  const { columns } = pairHistory;

  // Skip OHLCV columns and common non-indicator columns
  const skipColumns = [
    'date', 'open', 'high', 'low', 'close', 'volume',
    'buy', 'sell', 'enter_long', 'exit_long', 'enter_short', 'exit_short',
    'buy_tag', 'enter_tag', 'exit_tag', 'exit_reason',
  ];

  columns.forEach((columnName) => {
    if (!skipColumns.includes(columnName)) {
      const data = extractIndicatorData(pairHistory, columnName);
      if (data && data.length > 0) {
        indicators.set(columnName, data);
      }
    }
  });

  return indicators;
}

/**
 * Extract entry/exit signals as markers data
 *
 * @param pairHistory - The PairHistory object from Freqtrade API
 * @returns Object containing long and short entry/exit signals
 */
export function extractSignals(pairHistory: FreqtradePairHistory) {
  const { columns, data } = pairHistory;

  const dateIdx = columns.indexOf('date');
  const closeIdx = columns.indexOf('close');

  // Try to find signal columns (support both old and new naming)
  const enterLongIdx = columns.indexOf('enter_long') !== -1
    ? columns.indexOf('enter_long')
    : columns.indexOf('buy');
  const exitLongIdx = columns.indexOf('exit_long') !== -1
    ? columns.indexOf('exit_long')
    : columns.indexOf('sell');
  const enterShortIdx = columns.indexOf('enter_short');
  const exitShortIdx = columns.indexOf('exit_short');

  const signals = {
    enterLong: [] as Array<{ time: Time; price: number }>,
    exitLong: [] as Array<{ time: Time; price: number }>,
    enterShort: [] as Array<{ time: Time; price: number }>,
    exitShort: [] as Array<{ time: Time; price: number }>,
  };

  if (dateIdx === -1 || closeIdx === -1) {
    return signals;
  }

  data.forEach((candle) => {
    const time = Math.floor(candle[dateIdx] / 1000) as Time;
    const price = candle[closeIdx];

    if (enterLongIdx !== -1 && candle[enterLongIdx] === 1) {
      signals.enterLong.push({ time, price });
    }

    if (exitLongIdx !== -1 && candle[exitLongIdx] === 1) {
      signals.exitLong.push({ time, price });
    }

    if (enterShortIdx !== -1 && candle[enterShortIdx] === 1) {
      signals.enterShort.push({ time, price });
    }

    if (exitShortIdx !== -1 && candle[exitShortIdx] === 1) {
      signals.exitShort.push({ time, price });
    }
  });

  return signals;
}

/**
 * Convert signals to Lightweight Charts markers format
 *
 * @param signals - Signals object from extractSignals()
 * @returns Array of marker objects
 */
export function signalsToMarkers(signals: ReturnType<typeof extractSignals>) {
  const markers: any[] = [];

  // Enter long (buy) signals
  signals.enterLong.forEach(({ time, price }) => {
    markers.push({
      time,
      position: 'belowBar',
      color: '#26a69a',
      shape: 'arrowUp',
      text: 'Buy',
    });
  });

  // Exit long (sell) signals
  signals.exitLong.forEach(({ time, price }) => {
    markers.push({
      time,
      position: 'aboveBar',
      color: '#ef5350',
      shape: 'arrowDown',
      text: 'Sell',
    });
  });

  // Enter short signals
  signals.enterShort.forEach(({ time, price }) => {
    markers.push({
      time,
      position: 'aboveBar',
      color: '#ef5350',
      shape: 'arrowDown',
      text: 'Short',
    });
  });

  // Exit short signals
  signals.exitShort.forEach(({ time, price }) => {
    markers.push({
      time,
      position: 'belowBar',
      color: '#26a69a',
      shape: 'arrowUp',
      text: 'Cover',
    });
  });

  return markers;
}

/**
 * Detect common indicator types and assign appropriate colors
 *
 * @param indicatorName - Name of the indicator
 * @returns Color hex code
 */
export function getIndicatorColor(indicatorName: string): string {
  const name = indicatorName.toLowerCase();

  // Moving averages - shades of blue
  if (name.includes('sma') || name.includes('ema')) {
    if (name.includes('50')) return '#2196F3';
    if (name.includes('100')) return '#1976D2';
    if (name.includes('200')) return '#FF9800';
    return '#42A5F5';
  }

  // Bollinger Bands - purple
  if (name.includes('bb_')) {
    if (name.includes('upper')) return '#9C27B0';
    if (name.includes('lower')) return '#9C27B0';
    if (name.includes('mid')) return '#BA68C8';
    return '#9C27B0';
  }

  // RSI - purple
  if (name.includes('rsi')) return '#9C27B0';

  // MACD - various colors
  if (name.includes('macd')) {
    if (name.includes('signal')) return '#FF9800';
    if (name.includes('hist')) return '#4CAF50';
    return '#2196F3';
  }

  // Stochastic - green
  if (name.includes('stoch') || name.includes('slowk') || name.includes('slowd')) {
    return '#4CAF50';
  }

  // ATR - orange
  if (name.includes('atr')) return '#FF9800';

  // ADX - red
  if (name.includes('adx')) return '#F44336';

  // Default color
  return '#607D8B';
}

/**
 * Complete data transformation from Freqtrade API to chart-ready format
 *
 * @param pairHistory - The PairHistory object from Freqtrade API
 * @returns Object containing all chart data ready to use
 */
export function transformPairHistory(pairHistory: FreqtradePairHistory) {
  const candles = transformToCandlestickData(pairHistory);
  const volume = transformToVolumeData(pairHistory);
  const signals = extractSignals(pairHistory);
  const markers = signalsToMarkers(signals);

  // Extract all indicators
  const indicatorMap = extractAllIndicators(pairHistory);
  const indicators = Array.from(indicatorMap.entries()).map(([name, data]) => ({
    name,
    type: 'line' as const,
    data,
    color: getIndicatorColor(name),
  }));

  return {
    candles,
    volume,
    indicators,
    markers,
    signals,
    annotations: pairHistory.annotations || [],
    metadata: {
      pair: pairHistory.pair,
      timeframe: pairHistory.timeframe,
      strategy: pairHistory.strategy,
      dataStart: pairHistory.data_start,
      dataStop: pairHistory.data_stop,
      lastAnalyzed: pairHistory.last_analyzed,
    },
  };
}
