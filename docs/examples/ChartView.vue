<template>
  <div class="chart-view">
    <div class="chart-header">
      <h2>{{ selectedPair }} - {{ selectedTimeframe }}</h2>
      <div class="chart-controls">
        <select v-model="selectedPair" class="control-select" @change="loadChartData">
          <option v-for="pair in availablePairs" :key="pair" :value="pair">
            {{ pair }}
          </option>
        </select>

        <select v-model="selectedTimeframe" class="control-select" @change="loadChartData">
          <option value="1m">1 minute</option>
          <option value="5m">5 minutes</option>
          <option value="15m">15 minutes</option>
          <option value="30m">30 minutes</option>
          <option value="1h">1 hour</option>
          <option value="4h">4 hours</option>
          <option value="1d">1 day</option>
        </select>

        <button @click="loadChartData" class="control-button" :disabled="loading">
          {{ loading ? 'Loading...' : 'Refresh' }}
        </button>

        <button @click="toggleTheme" class="control-button">
          {{ theme === 'dark' ? '☀️ Light' : '🌙 Dark' }}
        </button>
      </div>
    </div>

    <div class="chart-info" v-if="chartMetadata">
      <span>Strategy: {{ chartMetadata.strategy }}</span>
      <span>Last analyzed: {{ formatDateTime(chartMetadata.lastAnalyzed) }}</span>
      <span>Data range: {{ formatDateTime(chartMetadata.dataStart) }} - {{ formatDateTime(chartMetadata.dataStop) }}</span>
    </div>

    <div class="indicator-controls" v-if="availableIndicators.length > 0">
      <div class="indicator-toggles">
        <label v-for="indicator in availableIndicators" :key="indicator.name" class="indicator-toggle">
          <input
            type="checkbox"
            :checked="indicator.visible"
            @change="toggleIndicator(indicator.name)"
          />
          <span :style="{ color: indicator.color }">{{ indicator.name }}</span>
        </label>
      </div>
    </div>

    <div class="error-message" v-if="error">
      {{ error }}
    </div>

    <div class="chart-container">
      <TradingChart
        v-if="chartData && !loading"
        ref="chartRef"
        :candle-data="chartData.candles"
        :volume-data="chartData.volume"
        :indicators="visibleIndicators"
        :height="600"
        :show-volume="showVolume"
        :theme="theme"
        @crosshair-move="handleCrosshairMove"
        @click="handleChartClick"
      />

      <div v-if="loading" class="loading-overlay">
        <div class="spinner"></div>
        <p>Loading chart data...</p>
      </div>
    </div>

    <div class="chart-footer" v-if="crosshairData">
      <div class="price-info">
        <span>Time: {{ crosshairData.time }}</span>
        <span>Open: {{ crosshairData.open?.toFixed(2) }}</span>
        <span>High: {{ crosshairData.high?.toFixed(2) }}</span>
        <span>Low: {{ crosshairData.low?.toFixed(2) }}</span>
        <span>Close: {{ crosshairData.close?.toFixed(2) }}</span>
        <span v-if="crosshairData.volume">Volume: {{ formatVolume(crosshairData.volume) }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import TradingChart from './TradingChart.vue';
import type { IndicatorConfig } from './TradingChart.vue';
import {
  transformPairHistory,
  type FreqtradePairHistory,
} from './chartDataTransformer';
import { applyAnnotations } from './chartAnnotations';

// State
const selectedPair = ref('BTC/USDT');
const selectedTimeframe = ref('5m');
const availablePairs = ref<string[]>([]);
const chartData = ref<any>(null);
const chartMetadata = ref<any>(null);
const availableIndicators = ref<IndicatorConfig[]>([]);
const loading = ref(false);
const error = ref<string | null>(null);
const chartRef = ref<InstanceType<typeof TradingChart> | null>(null);
const crosshairData = ref<any>(null);
const theme = ref<'dark' | 'light'>('dark');
const showVolume = ref(true);

// Computed
const visibleIndicators = computed(() => {
  return availableIndicators.value.filter(ind => ind.visible !== false);
});

// Methods
const loadChartData = async () => {
  loading.value = true;
  error.value = null;

  try {
    // Fetch chart data from Freqtrade API
    const response = await fetch(
      `/api/v1/pair_candles?pair=${encodeURIComponent(selectedPair.value)}&timeframe=${selectedTimeframe.value}&limit=500`
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch chart data: ${response.statusText}`);
    }

    const pairHistory: FreqtradePairHistory = await response.json();

    // Transform data for Lightweight Charts
    const transformed = transformPairHistory(pairHistory);

    chartData.value = transformed;
    chartMetadata.value = transformed.metadata;

    // Update available indicators
    availableIndicators.value = transformed.indicators.map(ind => ({
      ...ind,
      visible: true,
    }));

    // Apply annotations if any
    if (transformed.annotations && transformed.annotations.length > 0 && chartRef.value) {
      setTimeout(() => {
        if (chartRef.value?.candlestickSeries) {
          applyAnnotations(
            chartRef.value.candlestickSeries,
            transformed.annotations,
            transformed.markers
          );
        }
      }, 100);
    } else if (transformed.markers && transformed.markers.length > 0 && chartRef.value) {
      // Just apply signal markers if no annotations
      setTimeout(() => {
        if (chartRef.value?.candlestickSeries) {
          chartRef.value.candlestickSeries.setMarkers(transformed.markers);
        }
      }, 100);
    }
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Unknown error occurred';
    console.error('Error loading chart data:', err);
  } finally {
    loading.value = false;
  }
};

const loadAvailablePairs = async () => {
  try {
    const response = await fetch('/api/v1/whitelist');
    const data = await response.json();
    availablePairs.value = data.whitelist || [];

    if (availablePairs.value.length > 0 && !selectedPair.value) {
      selectedPair.value = availablePairs.value[0];
    }
  } catch (err) {
    console.error('Error loading available pairs:', err);
  }
};

const toggleIndicator = (indicatorName: string) => {
  const indicator = availableIndicators.value.find(ind => ind.name === indicatorName);
  if (indicator) {
    indicator.visible = !indicator.visible;
  }
};

const toggleTheme = () => {
  theme.value = theme.value === 'dark' ? 'light' : 'dark';
};

const handleCrosshairMove = (param: any) => {
  if (!param || !param.time) {
    crosshairData.value = null;
    return;
  }

  const data = param.seriesData.get(chartRef.value?.candlestickSeries);
  if (data) {
    crosshairData.value = {
      time: new Date(param.time * 1000).toLocaleString(),
      open: data.open,
      high: data.high,
      low: data.low,
      close: data.close,
      volume: param.seriesData.get(chartRef.value?.volumeSeries)?.value,
    };
  }
};

const handleChartClick = (param: any) => {
  console.log('Chart clicked:', param);
};

const formatDateTime = (dateString?: string) => {
  if (!dateString) return 'N/A';
  return new Date(dateString).toLocaleString();
};

const formatVolume = (volume: number) => {
  if (volume >= 1e9) return `${(volume / 1e9).toFixed(2)}B`;
  if (volume >= 1e6) return `${(volume / 1e6).toFixed(2)}M`;
  if (volume >= 1e3) return `${(volume / 1e3).toFixed(2)}K`;
  return volume.toFixed(2);
};

// Lifecycle
onMounted(async () => {
  await loadAvailablePairs();
  await loadChartData();
});
</script>

<style scoped>
.chart-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 20px;
  background: #1a1e27;
  color: #d1d4dc;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.chart-header h2 {
  margin: 0;
  font-size: 24px;
  font-weight: 600;
}

.chart-controls {
  display: flex;
  gap: 12px;
  align-items: center;
}

.control-select,
.control-button {
  padding: 8px 16px;
  border-radius: 6px;
  border: 1px solid #2b2f3a;
  background: #1e222d;
  color: #d1d4dc;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}

.control-select:hover,
.control-button:hover:not(:disabled) {
  background: #2b2f3a;
  border-color: #3a3f4a;
}

.control-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.chart-info {
  display: flex;
  gap: 20px;
  margin-bottom: 12px;
  padding: 12px;
  background: #252930;
  border-radius: 6px;
  font-size: 13px;
  color: #9ca3af;
}

.indicator-controls {
  margin-bottom: 16px;
}

.indicator-toggles {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  padding: 12px;
  background: #252930;
  border-radius: 6px;
}

.indicator-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  font-size: 13px;
}

.indicator-toggle input[type="checkbox"] {
  cursor: pointer;
}

.error-message {
  padding: 12px;
  margin-bottom: 16px;
  background: #ef5350;
  color: white;
  border-radius: 6px;
  font-size: 14px;
}

.chart-container {
  position: relative;
  flex: 1;
  min-height: 600px;
  background: #1e222d;
  border-radius: 8px;
  overflow: hidden;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: rgba(30, 34, 45, 0.9);
  z-index: 10;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #2b2f3a;
  border-top-color: #2196F3;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.chart-footer {
  margin-top: 16px;
  padding: 12px;
  background: #252930;
  border-radius: 6px;
}

.price-info {
  display: flex;
  gap: 20px;
  font-size: 13px;
  font-family: 'Monaco', 'Menlo', monospace;
}

.price-info span {
  color: #9ca3af;
}

/* Light theme support */
:global(.theme-light) .chart-view {
  background: #f5f5f5;
  color: #191919;
}

:global(.theme-light) .control-select,
:global(.theme-light) .control-button {
  background: #ffffff;
  border-color: #e1e1e1;
  color: #191919;
}

:global(.theme-light) .chart-info,
:global(.theme-light) .indicator-toggles,
:global(.theme-light) .chart-footer {
  background: #ffffff;
  border: 1px solid #e1e1e1;
}
</style>
