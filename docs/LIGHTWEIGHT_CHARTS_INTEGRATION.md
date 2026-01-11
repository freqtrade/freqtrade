# TradingView Lightweight Charts Integration Guide

## Overview

This document provides a comprehensive guide for integrating TradingView's Lightweight Charts library into Freqtrade's FreqUI frontend to replace or enhance the current Plotly-based charting system.

## Table of Contents

1. [Why Lightweight Charts?](#why-lightweight-charts)
2. [Architecture Overview](#architecture-overview)
3. [Installation](#installation)
4. [Vue Component Implementation](#vue-component-implementation)
5. [Data Transformation](#data-transformation)
6. [Annotation Support](#annotation-support)
7. [Technical Indicators](#technical-indicators)
8. [Configuration](#configuration)

## Why Lightweight Charts?

TradingView's Lightweight Charts offers several advantages:

- **Performance**: Optimized for financial data visualization with HTML5 Canvas
- **Open Source**: MIT license, free to use
- **Lightweight**: Minimal bundle size (~50KB gzipped)
- **Feature Rich**: Supports candlesticks, lines, areas, histograms, and more
- **Mobile Friendly**: Touch-optimized and responsive
- **Customizable**: Extensive styling and configuration options
- **Active Development**: Well-maintained by TradingView team

## Architecture Overview

### Current Architecture
```
Freqtrade Backend (Python)
    ↓ (REST API)
/pair_candles endpoint
    ↓ (JSON: PairHistory schema)
FreqUI (Vue.js) - Uses Plotly
```

### New Architecture with Lightweight Charts
```
Freqtrade Backend (Python)
    ↓ (REST API)
/pair_candles endpoint
    ↓ (JSON: PairHistory schema)
FreqUI (Vue.js)
    ↓ (Transform data)
Lightweight Charts Library
```

## Installation

In your FreqUI project directory:

```bash
npm install lightweight-charts
# or
yarn add lightweight-charts
```

## Vue Component Implementation

### Basic Chart Component

Create a new file: `src/components/charts/LightweightChart.vue`

```vue
<template>
  <div ref="chartContainer" class="chart-container"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, shallowRef } from 'vue';
import { createChart, IChartApi, ISeriesApi, CandlestickData } from 'lightweight-charts';

interface Props {
  data: CandlestickData[];
  width?: number;
  height?: number;
  autoSize?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  width: 800,
  height: 400,
  autoSize: true,
});

const chartContainer = ref<HTMLDivElement | null>(null);
// Use shallowRef for chart instance (important for performance)
const chart = shallowRef<IChartApi | null>(null);
const candlestickSeries = shallowRef<ISeriesApi<'Candlestick'> | null>(null);

const initChart = () => {
  if (!chartContainer.value) return;

  // Create chart
  chart.value = createChart(chartContainer.value, {
    width: props.autoSize ? chartContainer.value.clientWidth : props.width,
    height: props.height,
    layout: {
      background: { color: '#1e222d' },
      textColor: '#d1d4dc',
    },
    grid: {
      vertLines: { color: '#2b2f3a' },
      horzLines: { color: '#2b2f3a' },
    },
    crosshair: {
      mode: 1, // Normal crosshair mode
    },
    rightPriceScale: {
      borderColor: '#2b2f3a',
    },
    timeScale: {
      borderColor: '#2b2f3a',
      timeVisible: true,
      secondsVisible: false,
    },
  });

  // Add candlestick series
  candlestickSeries.value = chart.value.addCandlestickSeries({
    upColor: '#26a69a',
    downColor: '#ef5350',
    borderVisible: false,
    wickUpColor: '#26a69a',
    wickDownColor: '#ef5350',
  });

  // Set initial data
  if (props.data.length > 0) {
    candlestickSeries.value.setData(props.data);
  }

  // Auto-resize handler
  if (props.autoSize) {
    const resizeObserver = new ResizeObserver((entries) => {
      if (chart.value && entries.length > 0) {
        const { width } = entries[0].contentRect;
        chart.value.applyOptions({ width });
      }
    });
    resizeObserver.observe(chartContainer.value);
  }
};

// Watch for data changes
watch(() => props.data, (newData) => {
  if (candlestickSeries.value && newData.length > 0) {
    candlestickSeries.value.setData(newData);
  }
}, { deep: true });

onMounted(() => {
  initChart();
});

onUnmounted(() => {
  if (chart.value) {
    chart.value.remove();
  }
});
</script>

<style scoped>
.chart-container {
  width: 100%;
  height: 100%;
  position: relative;
}
</style>
```

### Advanced Chart Component with Indicators

Create: `src/components/charts/TradingChart.vue`

```vue
<template>
  <div class="trading-chart-wrapper">
    <div ref="chartContainer" class="chart-container"></div>
    <div ref="volumeContainer" class="volume-container"></div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, shallowRef } from 'vue';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  HistogramData,
  LineData,
  Time,
} from 'lightweight-charts';

interface IndicatorData {
  name: string;
  type: 'line' | 'area' | 'histogram';
  data: LineData[];
  color?: string;
}

interface Props {
  candleData: CandlestickData[];
  volumeData?: HistogramData[];
  indicators?: IndicatorData[];
  height?: number;
}

const props = withDefaults(defineProps<Props>(), {
  height: 500,
});

const chartContainer = ref<HTMLDivElement | null>(null);
const volumeContainer = ref<HTMLDivElement | null>(null);
const chart = shallowRef<IChartApi | null>(null);
const candlestickSeries = shallowRef<ISeriesApi<'Candlestick'> | null>(null);
const volumeSeries = shallowRef<ISeriesApi<'Histogram'> | null>(null);
const indicatorSeries = shallowRef<Map<string, ISeriesApi<any>>>(new Map());

const chartOptions = {
  layout: {
    background: { color: '#1e222d' },
    textColor: '#d1d4dc',
  },
  grid: {
    vertLines: { color: '#2b2f3a' },
    horzLines: { color: '#2b2f3a' },
  },
  crosshair: {
    mode: 1,
  },
  rightPriceScale: {
    borderColor: '#2b2f3a',
  },
  timeScale: {
    borderColor: '#2b2f3a',
    timeVisible: true,
    secondsVisible: false,
  },
};

const initChart = () => {
  if (!chartContainer.value) return;

  const width = chartContainer.value.clientWidth;

  // Main chart
  chart.value = createChart(chartContainer.value, {
    ...chartOptions,
    width,
    height: props.height * 0.7, // 70% for main chart
  });

  // Candlestick series
  candlestickSeries.value = chart.value.addCandlestickSeries({
    upColor: '#26a69a',
    downColor: '#ef5350',
    borderVisible: false,
    wickUpColor: '#26a69a',
    wickDownColor: '#ef5350',
  });

  candlestickSeries.value.setData(props.candleData);

  // Add indicators
  if (props.indicators) {
    props.indicators.forEach((indicator) => {
      addIndicator(indicator);
    });
  }

  // Volume chart (if provided)
  if (props.volumeData && volumeContainer.value) {
    const volumeChart = createChart(volumeContainer.value, {
      ...chartOptions,
      width,
      height: props.height * 0.3, // 30% for volume
    });

    volumeSeries.value = volumeChart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: {
        type: 'volume',
      },
    });

    volumeSeries.value.setData(props.volumeData);

    // Sync time scales
    chart.value.timeScale().subscribeVisibleTimeRangeChange((timeRange) => {
      if (timeRange) {
        volumeChart.timeScale().setVisibleRange(timeRange);
      }
    });
  }
};

const addIndicator = (indicator: IndicatorData) => {
  if (!chart.value) return;

  let series: ISeriesApi<any>;

  switch (indicator.type) {
    case 'line':
      series = chart.value.addLineSeries({
        color: indicator.color || '#2196F3',
        lineWidth: 2,
        title: indicator.name,
      });
      break;
    case 'area':
      series = chart.value.addAreaSeries({
        topColor: indicator.color || 'rgba(33, 150, 243, 0.4)',
        bottomColor: 'rgba(33, 150, 243, 0.0)',
        lineColor: indicator.color || '#2196F3',
        lineWidth: 2,
        title: indicator.name,
      });
      break;
    case 'histogram':
      series = chart.value.addHistogramSeries({
        color: indicator.color || '#26a69a',
        title: indicator.name,
      });
      break;
    default:
      return;
  }

  series.setData(indicator.data);
  indicatorSeries.value.set(indicator.name, series);
};

watch(() => props.candleData, (newData) => {
  if (candlestickSeries.value && newData.length > 0) {
    candlestickSeries.value.setData(newData);
  }
});

onMounted(() => {
  initChart();
});

onUnmounted(() => {
  if (chart.value) {
    chart.value.remove();
  }
});

defineExpose({
  chart,
  candlestickSeries,
  addIndicator,
});
</script>

<style scoped>
.trading-chart-wrapper {
  width: 100%;
  display: flex;
  flex-direction: column;
}

.chart-container,
.volume-container {
  width: 100%;
  position: relative;
}
</style>
```

## Data Transformation

### Transforming Freqtrade API Response

Create: `src/utils/chartDataTransformer.ts`

```typescript
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
}

/**
 * Convert Freqtrade API response to Lightweight Charts candlestick format
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
 */
export function transformToVolumeData(
  pairHistory: FreqtradePairHistory
): HistogramData[] | null {
  const { columns, data } = pairHistory;

  const dateIdx = columns.indexOf('date');
  const volumeIdx = columns.indexOf('volume');

  if (dateIdx === -1 || volumeIdx === -1) {
    return null;
  }

  return data.map((candle) => ({
    time: Math.floor(candle[dateIdx] / 1000) as Time,
    value: candle[volumeIdx],
    color: candle[columns.indexOf('close')] >= candle[columns.indexOf('open')]
      ? '#26a69a'
      : '#ef5350',
  }));
}

/**
 * Extract indicator data (MA, EMA, etc.) from Freqtrade response
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

  return data
    .filter((candle) => candle[indicatorIdx] !== null && !isNaN(candle[indicatorIdx]))
    .map((candle) => ({
      time: Math.floor(candle[dateIdx] / 1000) as Time,
      value: candle[indicatorIdx],
    }));
}

/**
 * Extract all indicators from the response
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
    'buy_tag', 'enter_tag', 'exit_tag',
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
```

## Annotation Support

Freqtrade supports three types of annotations that can be mapped to Lightweight Charts:

### Annotation Types Mapping

```typescript
// src/utils/chartAnnotations.ts

import { Time, ISeriesApi } from 'lightweight-charts';

export interface FreqtradeAreaAnnotation {
  type: 'area';
  start: string; // ISO datetime
  end: string;
  y_start: number;
  y_end: number;
  color?: string;
  label?: string;
  z_level?: number;
}

export interface FreqtradeLineAnnotation {
  type: 'line';
  start: string;
  end: string;
  y_start: number;
  y_end: number;
  color?: string;
  label?: string;
  width?: number;
  line_style?: 'solid' | 'dashed' | 'dotted';
  z_level?: number;
}

export interface FreqtradePointAnnotation {
  type: 'point';
  x: string; // ISO datetime
  y: number;
  color?: string;
  label?: string;
  size?: number;
  shape?: 'circle' | 'rect' | 'roundRect' | 'triangle' | 'pin' | 'arrow' | 'none';
  z_level?: number;
}

export type FreqtradeAnnotation =
  | FreqtradeAreaAnnotation
  | FreqtradeLineAnnotation
  | FreqtradePointAnnotation;

/**
 * Convert ISO datetime string to Unix timestamp
 */
function isoToTime(isoString: string): Time {
  return Math.floor(new Date(isoString).getTime() / 1000) as Time;
}

/**
 * Add area annotation using price lines and shaded areas
 */
export function addAreaAnnotation(
  series: ISeriesApi<any>,
  annotation: FreqtradeAreaAnnotation
) {
  // Lightweight Charts doesn't have built-in area annotations
  // We can use price lines or create a separate area series
  const color = annotation.color || 'rgba(255, 193, 7, 0.2)';

  // Add horizontal lines to mark the area boundaries
  series.createPriceLine({
    price: annotation.y_start,
    color: color,
    lineWidth: 1,
    lineStyle: 2, // Dashed
    axisLabelVisible: true,
    title: annotation.label || '',
  });

  series.createPriceLine({
    price: annotation.y_end,
    color: color,
    lineWidth: 1,
    lineStyle: 2,
    axisLabelVisible: true,
  });
}

/**
 * Add line annotation
 */
export function addLineAnnotation(
  series: ISeriesApi<any>,
  annotation: FreqtradeLineAnnotation
) {
  const color = annotation.color || '#ffc107';
  const lineStyle = annotation.line_style === 'dashed' ? 2 :
                    annotation.line_style === 'dotted' ? 3 : 0;

  // For horizontal lines
  if (annotation.y_start === annotation.y_end) {
    series.createPriceLine({
      price: annotation.y_start,
      color: color,
      lineWidth: annotation.width || 2,
      lineStyle: lineStyle,
      axisLabelVisible: true,
      title: annotation.label || '',
    });
  }
}

/**
 * Add point annotation using markers
 */
export function addPointAnnotation(
  series: ISeriesApi<any>,
  annotation: FreqtradePointAnnotation,
  allData: any[]
) {
  const time = isoToTime(annotation.x);
  const color = annotation.color || '#2196F3';

  // Map Freqtrade shapes to Lightweight Charts marker shapes
  const shapeMap: Record<string, any> = {
    'circle': 'circle',
    'rect': 'square',
    'triangle': 'arrowUp',
    'pin': 'arrowDown',
    'arrow': 'arrowUp',
  };

  const position = annotation.y > (allData.find(d => d.time === time)?.close || 0)
    ? 'aboveBar'
    : 'belowBar';

  series.setMarkers([
    ...series.markers?.() || [],
    {
      time: time,
      position: position,
      color: color,
      shape: shapeMap[annotation.shape || 'circle'] || 'circle',
      text: annotation.label || '',
      size: annotation.size || 1,
    },
  ]);
}

/**
 * Add all annotations from Freqtrade response
 */
export function addAnnotations(
  series: ISeriesApi<any>,
  annotations: FreqtradeAnnotation[],
  candleData: any[]
) {
  annotations.forEach((annotation) => {
    switch (annotation.type) {
      case 'area':
        addAreaAnnotation(series, annotation);
        break;
      case 'line':
        addLineAnnotation(series, annotation);
        break;
      case 'point':
        addPointAnnotation(series, annotation, candleData);
        break;
    }
  });
}
```

## Technical Indicators

### Adding Moving Averages

```typescript
// Example: Adding MA indicators
import { LineData } from 'lightweight-charts';

interface MAIndicator {
  name: string;
  period: number;
  color: string;
  data: LineData[];
}

// In your component:
const addMovingAverages = (pairHistory: FreqtradePairHistory) => {
  const ma50 = extractIndicatorData(pairHistory, 'sma_50');
  const ma200 = extractIndicatorData(pairHistory, 'sma_200');

  if (ma50) {
    addIndicator({
      name: 'MA 50',
      type: 'line',
      data: ma50,
      color: '#2196F3',
    });
  }

  if (ma200) {
    addIndicator({
      name: 'MA 200',
      type: 'line',
      data: ma200,
      color: '#FF9800',
    });
  }
};
```

## Configuration

### Chart Theme Configuration

```typescript
// src/config/chartThemes.ts

export const darkTheme = {
  layout: {
    background: { color: '#1e222d' },
    textColor: '#d1d4dc',
  },
  grid: {
    vertLines: { color: '#2b2f3a' },
    horzLines: { color: '#2b2f3a' },
  },
  crosshair: {
    mode: 1,
  },
  rightPriceScale: {
    borderColor: '#2b2f3a',
  },
  timeScale: {
    borderColor: '#2b2f3a',
    timeVisible: true,
    secondsVisible: false,
  },
};

export const lightTheme = {
  layout: {
    background: { color: '#FFFFFF' },
    textColor: '#191919',
  },
  grid: {
    vertLines: { color: '#e1e1e1' },
    horzLines: { color: '#e1e1e1' },
  },
  crosshair: {
    mode: 1,
  },
  rightPriceScale: {
    borderColor: '#e1e1e1',
  },
  timeScale: {
    borderColor: '#e1e1e1',
    timeVisible: true,
    secondsVisible: false,
  },
};
```

## Complete Integration Example

```vue
<!-- src/views/ChartView.vue -->
<template>
  <div class="chart-view">
    <div class="chart-controls">
      <select v-model="selectedPair">
        <option v-for="pair in pairs" :key="pair" :value="pair">
          {{ pair }}
        </option>
      </select>
      <select v-model="selectedTimeframe">
        <option value="1m">1m</option>
        <option value="5m">5m</option>
        <option value="15m">15m</option>
        <option value="1h">1h</option>
        <option value="4h">4h</option>
        <option value="1d">1d</option>
      </select>
      <button @click="loadChartData">Load</button>
    </div>

    <TradingChart
      v-if="chartData"
      :candle-data="chartData.candles"
      :volume-data="chartData.volume"
      :indicators="chartData.indicators"
      :height="600"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import TradingChart from '@/components/charts/TradingChart.vue';
import {
  transformToCandlestickData,
  transformToVolumeData,
  extractAllIndicators,
} from '@/utils/chartDataTransformer';
import { addAnnotations } from '@/utils/chartAnnotations';

const selectedPair = ref('BTC/USDT');
const selectedTimeframe = ref('5m');
const pairs = ref<string[]>([]);
const chartData = ref<any>(null);

const loadChartData = async () => {
  try {
    const response = await fetch(
      `/api/v1/pair_candles?pair=${selectedPair.value}&timeframe=${selectedTimeframe.value}`
    );
    const pairHistory = await response.json();

    // Transform data
    const candles = transformToCandlestickData(pairHistory);
    const volume = transformToVolumeData(pairHistory);
    const indicatorMap = extractAllIndicators(pairHistory);

    // Convert indicator map to array
    const indicators = Array.from(indicatorMap.entries()).map(([name, data]) => ({
      name,
      type: 'line' as const,
      data,
      color: getIndicatorColor(name),
    }));

    chartData.value = {
      candles,
      volume,
      indicators,
      annotations: pairHistory.annotations || [],
    };
  } catch (error) {
    console.error('Failed to load chart data:', error);
  }
};

const getIndicatorColor = (name: string): string => {
  const colors: Record<string, string> = {
    sma_50: '#2196F3',
    sma_200: '#FF9800',
    ema_12: '#4CAF50',
    ema_26: '#F44336',
    rsi: '#9C27B0',
  };
  return colors[name] || '#607D8B';
};

onMounted(async () => {
  // Load available pairs
  const response = await fetch('/api/v1/whitelist');
  const data = await response.json();
  pairs.value = data.whitelist;

  // Load initial chart
  await loadChartData();
});
</script>

<style scoped>
.chart-view {
  padding: 20px;
}

.chart-controls {
  margin-bottom: 20px;
  display: flex;
  gap: 10px;
}

.chart-controls select,
.chart-controls button {
  padding: 8px 12px;
  border-radius: 4px;
  border: 1px solid #2b2f3a;
  background: #1e222d;
  color: #d1d4dc;
}

.chart-controls button {
  cursor: pointer;
}

.chart-controls button:hover {
  background: #2b2f3a;
}
</style>
```

## Next Steps

1. **Install the package** in FreqUI:
   ```bash
   npm install lightweight-charts
   ```

2. **Create the components** in the FreqUI repository using the examples above

3. **Test with Freqtrade API** to ensure data transformation works correctly

4. **Add customization options** like themes, indicator toggles, drawing tools

5. **Optimize performance** for real-time updates via WebSocket

## Performance Considerations

- Use `shallowRef` for chart instances in Vue to avoid reactivity overhead
- Implement virtualization for large datasets
- Use WebSocket for real-time updates instead of polling
- Debounce resize events
- Consider using Web Workers for data transformation

## Resources

- [TradingView Lightweight Charts Documentation](https://tradingview.github.io/lightweight-charts/)
- [Lightweight Charts GitHub](https://github.com/tradingview/lightweight-charts)
- [Vue.js Wrapper Tutorial](https://tradingview.github.io/lightweight-charts/tutorials/vuejs/wrapper)
- [Freqtrade API Documentation](https://www.freqtrade.io/en/stable/rest-api/)

## Troubleshooting

### Chart not rendering
- Ensure container has explicit height/width
- Check that data is in correct format (timestamps in seconds, not milliseconds)

### Performance issues
- Use `shallowRef` instead of `ref` for chart instances
- Limit the number of data points loaded initially
- Implement pagination or lazy loading

### Time zone issues
- Lightweight Charts uses UTC by default
- Convert timestamps correctly from Freqtrade's milliseconds to seconds
