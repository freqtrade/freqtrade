<template>
  <div ref="chartContainer" class="h-full w-full">
    <TradingChart
      v-if="candleData.length > 0"
      :candle-data="candleData"
      :volume-data="volumeData"
      :indicators="indicatorData"
      :height="containerHeight"
      :theme="theme === 'dark' ? 'dark' : 'light'"
      @crosshair-move="handleCrosshairMove"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue';
import type { PairHistory, PlotConfig, Trade, ChartSliderPosition } from '@/types';
import TradingChart from './TradingChart.vue';
import type { IndicatorConfig } from './TradingChart.vue';
import {
  transformToCandlestickData,
  transformToVolumeData,
  extractIndicatorData,
  getIndicatorColor,
  extractSignals,
  signalsToMarkers,
} from '@/utils/chartDataTransformer';
import { applyAnnotations } from '@/utils/chartAnnotations';

interface Props {
  dataset: PairHistory;
  trades?: Trade[];
  plotConfig: PlotConfig;
  heikinAshi?: boolean;
  showMarkArea?: boolean;
  useUTC?: boolean;
  theme?: string;
  sliderPosition?: ChartSliderPosition;
  colorUp?: string;
  colorDown?: string;
  startCandleCount?: number;
  labelSide?: 'left' | 'right';
}

const props = withDefaults(defineProps<Props>(), {
  trades: () => [],
  heikinAshi: false,
  showMarkArea: true,
  useUTC: false,
  theme: 'dark',
  sliderPosition: undefined,
  colorUp: '#26a69a',
  colorDown: '#ef5350',
  startCandleCount: 150,
  labelSide: 'right',
});

const chartContainer = ref<HTMLDivElement | null>(null);
const containerHeight = ref(600);
const chart = ref<any>(null);

// Transform data for Lightweight Charts
const candleData = computed(() => {
  if (!props.dataset || !props.dataset.data || props.dataset.data.length === 0) {
    return [];
  }

  try {
    return transformToCandlestickData(props.dataset);
  } catch (error) {
    console.error('Failed to transform candle data:', error);
    return [];
  }
});

const volumeData = computed(() => {
  if (!props.dataset || !props.dataset.data || props.dataset.data.length === 0) {
    return null;
  }

  try {
    return transformToVolumeData(props.dataset);
  } catch (error) {
    console.warn('Volume data not available:', error);
    return null;
  }
});

const indicatorData = computed((): IndicatorConfig[] => {
  if (!props.dataset || !props.plotConfig) {
    return [];
  }

  const indicators: IndicatorConfig[] = [];

  // Extract indicators from plot config
  if (props.plotConfig.main_plot) {
    Object.keys(props.plotConfig.main_plot).forEach((indicatorName) => {
      const data = extractIndicatorData(props.dataset, indicatorName);
      if (data && data.length > 0) {
        const config = props.plotConfig.main_plot[indicatorName];
        indicators.push({
          name: indicatorName,
          type: 'line',
          data: data,
          color: config?.color || getIndicatorColor(indicatorName),
          lineWidth: 2,
          visible: true,
        });
      }
    });
  }

  // Extract subplots (like RSI, MACD) - these could be on separate charts
  // For now, we'll add them to the main chart
  if (props.plotConfig.subplots) {
    Object.keys(props.plotConfig.subplots).forEach((subplotName) => {
      const subplot = props.plotConfig.subplots[subplotName];
      Object.keys(subplot).forEach((indicatorName) => {
        const data = extractIndicatorData(props.dataset, indicatorName);
        if (data && data.length > 0) {
          const config = subplot[indicatorName];
          indicators.push({
            name: indicatorName,
            type: config?.type === 'bar' ? 'histogram' : 'line',
            data: data,
            color: config?.color || getIndicatorColor(indicatorName),
            lineWidth: 2,
            visible: true,
          });
        }
      });
    });
  }

  return indicators;
});

// Handle crosshair events
function handleCrosshairMove(param: any) {
  // You can emit this to parent if needed
  // emit('crosshair-move', param);
}

// Update height based on container size
onMounted(() => {
  if (chartContainer.value) {
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        containerHeight.value = entry.contentRect.height || 600;
      }
    });

    resizeObserver.observe(chartContainer.value);

    // Initial height
    containerHeight.value = chartContainer.value.clientHeight || 600;
  }
});

onUnmounted(() => {
  // Cleanup if needed
});
</script>

<style scoped>
/* Ensure the container takes full height */
div {
  min-height: 400px;
}
</style>
