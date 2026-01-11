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

// Expose chart and series for parent components
defineExpose({
  chart,
  candlestickSeries,
});
</script>

<style scoped>
.chart-container {
  width: 100%;
  height: 100%;
  position: relative;
}
</style>
