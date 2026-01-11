<template>
  <div class="trading-chart-wrapper">
    <div ref="mainChartContainer" class="main-chart-container"></div>
    <div v-if="showVolume && volumeData" ref="volumeChartContainer" class="volume-chart-container"></div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, shallowRef, computed } from 'vue';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  HistogramData,
  LineData,
  Time,
  ColorType,
} from 'lightweight-charts';

export interface IndicatorConfig {
  name: string;
  type: 'line' | 'area' | 'histogram';
  data: LineData[];
  color?: string;
  lineWidth?: number;
  visible?: boolean;
}

export interface ChartTheme {
  backgroundColor: string;
  textColor: string;
  gridColor: string;
  upColor: string;
  downColor: string;
}

interface Props {
  candleData: CandlestickData[];
  volumeData?: HistogramData[];
  indicators?: IndicatorConfig[];
  height?: number;
  showVolume?: boolean;
  theme?: 'dark' | 'light' | ChartTheme;
}

const props = withDefaults(defineProps<Props>(), {
  height: 500,
  showVolume: true,
  theme: 'dark',
});

const emit = defineEmits<{
  (e: 'crosshairMove', data: any): void;
  (e: 'click', data: any): void;
}>();

const mainChartContainer = ref<HTMLDivElement | null>(null);
const volumeChartContainer = ref<HTMLDivElement | null>(null);
const mainChart = shallowRef<IChartApi | null>(null);
const volumeChart = shallowRef<IChartApi | null>(null);
const candlestickSeries = shallowRef<ISeriesApi<'Candlestick'> | null>(null);
const volumeSeries = shallowRef<ISeriesApi<'Histogram'> | null>(null);
const indicatorSeriesMap = shallowRef<Map<string, ISeriesApi<any>>>(new Map());

// Compute theme colors
const themeColors = computed(() => {
  if (typeof props.theme === 'string') {
    return props.theme === 'dark' ? {
      backgroundColor: '#1e222d',
      textColor: '#d1d4dc',
      gridColor: '#2b2f3a',
      upColor: '#26a69a',
      downColor: '#ef5350',
    } : {
      backgroundColor: '#FFFFFF',
      textColor: '#191919',
      gridColor: '#e1e1e1',
      upColor: '#26a69a',
      downColor: '#ef5350',
    };
  }
  return props.theme;
});

const chartOptions = computed(() => ({
  layout: {
    background: { type: ColorType.Solid, color: themeColors.value.backgroundColor },
    textColor: themeColors.value.textColor,
  },
  grid: {
    vertLines: { color: themeColors.value.gridColor },
    horzLines: { color: themeColors.value.gridColor },
  },
  crosshair: {
    mode: 1,
  },
  rightPriceScale: {
    borderColor: themeColors.value.gridColor,
  },
  timeScale: {
    borderColor: themeColors.value.gridColor,
    timeVisible: true,
    secondsVisible: false,
  },
}));

const initChart = () => {
  if (!mainChartContainer.value) return;

  const width = mainChartContainer.value.clientWidth;
  const mainChartHeight = props.showVolume && props.volumeData
    ? props.height * 0.7
    : props.height;

  // Create main chart
  mainChart.value = createChart(mainChartContainer.value, {
    ...chartOptions.value,
    width,
    height: mainChartHeight,
  });

  // Add candlestick series
  candlestickSeries.value = mainChart.value.addCandlestickSeries({
    upColor: themeColors.value.upColor,
    downColor: themeColors.value.downColor,
    borderVisible: false,
    wickUpColor: themeColors.value.upColor,
    wickDownColor: themeColors.value.downColor,
  });

  if (props.candleData.length > 0) {
    candlestickSeries.value.setData(props.candleData);
  }

  // Add indicators
  if (props.indicators) {
    props.indicators.forEach((indicator) => {
      if (indicator.visible !== false) {
        addIndicator(indicator);
      }
    });
  }

  // Setup crosshair event
  mainChart.value.subscribeCrosshairMove((param) => {
    emit('crosshairMove', param);
  });

  // Setup click event
  mainChart.value.subscribeClick((param) => {
    emit('click', param);
  });

  // Create volume chart if needed
  if (props.showVolume && props.volumeData && volumeChartContainer.value) {
    const volumeHeight = props.height * 0.3;

    volumeChart.value = createChart(volumeChartContainer.value, {
      ...chartOptions.value,
      width,
      height: volumeHeight,
    });

    volumeSeries.value = volumeChart.value.addHistogramSeries({
      color: themeColors.value.upColor,
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'volume',
    });

    volumeSeries.value.setData(props.volumeData);

    // Synchronize time scales
    mainChart.value.timeScale().subscribeVisibleTimeRangeChange((timeRange) => {
      if (timeRange && volumeChart.value) {
        volumeChart.value.timeScale().setVisibleRange(timeRange);
      }
    });

    volumeChart.value.timeScale().subscribeVisibleTimeRangeChange((timeRange) => {
      if (timeRange && mainChart.value) {
        mainChart.value.timeScale().setVisibleRange(timeRange);
      }
    });
  }

  // Auto-resize handler
  const resizeObserver = new ResizeObserver((entries) => {
    if (entries.length === 0) return;

    const { width: newWidth } = entries[0].contentRect;

    if (mainChart.value) {
      mainChart.value.applyOptions({ width: newWidth });
    }

    if (volumeChart.value) {
      volumeChart.value.applyOptions({ width: newWidth });
    }
  });

  resizeObserver.observe(mainChartContainer.value);
};

const addIndicator = (indicator: IndicatorConfig) => {
  if (!mainChart.value || !indicator.data || indicator.data.length === 0) return;

  let series: ISeriesApi<any>;

  switch (indicator.type) {
    case 'line':
      series = mainChart.value.addLineSeries({
        color: indicator.color || '#2196F3',
        lineWidth: indicator.lineWidth || 2,
        title: indicator.name,
      });
      break;

    case 'area':
      series = mainChart.value.addAreaSeries({
        topColor: indicator.color
          ? `${indicator.color}66` // Add transparency
          : 'rgba(33, 150, 243, 0.4)',
        bottomColor: 'rgba(33, 150, 243, 0.0)',
        lineColor: indicator.color || '#2196F3',
        lineWidth: indicator.lineWidth || 2,
        title: indicator.name,
      });
      break;

    case 'histogram':
      series = mainChart.value.addHistogramSeries({
        color: indicator.color || '#26a69a',
        title: indicator.name,
      });
      break;

    default:
      return;
  }

  series.setData(indicator.data);
  indicatorSeriesMap.value.set(indicator.name, series);
};

const removeIndicator = (indicatorName: string) => {
  const series = indicatorSeriesMap.value.get(indicatorName);
  if (series && mainChart.value) {
    mainChart.value.removeSeries(series);
    indicatorSeriesMap.value.delete(indicatorName);
  }
};

const updateIndicator = (indicator: IndicatorConfig) => {
  removeIndicator(indicator.name);
  if (indicator.visible !== false) {
    addIndicator(indicator);
  }
};

// Watch for data changes
watch(() => props.candleData, (newData) => {
  if (candlestickSeries.value && newData.length > 0) {
    candlestickSeries.value.setData(newData);
  }
}, { deep: true });

watch(() => props.volumeData, (newData) => {
  if (volumeSeries.value && newData && newData.length > 0) {
    volumeSeries.value.setData(newData);
  }
}, { deep: true });

watch(() => props.indicators, (newIndicators, oldIndicators) => {
  if (!newIndicators) return;

  // Remove old indicators that are no longer present
  const newNames = new Set(newIndicators.map(ind => ind.name));
  if (oldIndicators) {
    oldIndicators.forEach((ind) => {
      if (!newNames.has(ind.name)) {
        removeIndicator(ind.name);
      }
    });
  }

  // Add or update indicators
  newIndicators.forEach((indicator) => {
    if (indicatorSeriesMap.value.has(indicator.name)) {
      updateIndicator(indicator);
    } else if (indicator.visible !== false) {
      addIndicator(indicator);
    }
  });
}, { deep: true });

// Watch for theme changes
watch(() => props.theme, () => {
  if (mainChart.value) {
    mainChart.value.applyOptions(chartOptions.value);
  }
  if (volumeChart.value) {
    volumeChart.value.applyOptions(chartOptions.value);
  }
}, { deep: true });

onMounted(() => {
  initChart();
});

onUnmounted(() => {
  if (mainChart.value) {
    mainChart.value.remove();
  }
  if (volumeChart.value) {
    volumeChart.value.remove();
  }
});

// Expose methods and refs for parent components
defineExpose({
  mainChart,
  volumeChart,
  candlestickSeries,
  volumeSeries,
  addIndicator,
  removeIndicator,
  updateIndicator,
});
</script>

<style scoped>
.trading-chart-wrapper {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.main-chart-container,
.volume-chart-container {
  width: 100%;
  position: relative;
}
</style>
