# Lightweight Charts Integration Examples

This directory contains ready-to-use example files for integrating TradingView's Lightweight Charts library into Freqtrade's FreqUI frontend.

## Files Overview

### Vue Components

1. **LightweightChart.vue** - Basic chart component
   - Simple candlestick chart with minimal configuration
   - Auto-resizing support
   - Perfect for simple use cases

2. **TradingChart.vue** - Advanced chart component
   - Candlestick chart with volume
   - Support for multiple indicators
   - Theme support (dark/light)
   - Event handling (crosshair, click)
   - Synchronized time scales

3. **ChartView.vue** - Complete example view
   - Full integration with Freqtrade API
   - Pair and timeframe selection
   - Indicator toggles
   - Real-time data loading
   - Error handling
   - Crosshair data display

### Utilities

4. **chartDataTransformer.ts** - Data transformation utilities
   - Transform Freqtrade API responses to Lightweight Charts format
   - Extract indicators, volume, and signals
   - Automatic color assignment for common indicators
   - Signal to marker conversion

5. **chartAnnotations.ts** - Annotation support
   - Map Freqtrade annotations to chart elements
   - Support for area, line, and point annotations
   - Trade marker helpers
   - Price line utilities

## Installation

### 1. Install Dependencies

In your FreqUI project:

```bash
npm install lightweight-charts
# or
yarn add lightweight-charts
```

### 2. Copy Files to Your Project

Copy the example files to your FreqUI project:

```bash
# Vue components
cp LightweightChart.vue /path/to/frequi/src/components/charts/
cp TradingChart.vue /path/to/frequi/src/components/charts/
cp ChartView.vue /path/to/frequi/src/views/

# Utilities
cp chartDataTransformer.ts /path/to/frequi/src/utils/
cp chartAnnotations.ts /path/to/frequi/src/utils/
```

### 3. Update Your Routes (if using ChartView)

In your router configuration:

```typescript
import ChartView from '@/views/ChartView.vue';

const routes = [
  // ... other routes
  {
    path: '/chart',
    name: 'chart',
    component: ChartView,
  },
];
```

## Usage Examples

### Basic Usage

```vue
<template>
  <LightweightChart :data="candleData" :height="400" />
</template>

<script setup>
import { ref } from 'vue';
import LightweightChart from '@/components/charts/LightweightChart.vue';
import { transformToCandlestickData } from '@/utils/chartDataTransformer';

const candleData = ref([]);

// Fetch and transform data
async function loadData() {
  const response = await fetch('/api/v1/pair_candles?pair=BTC/USDT&timeframe=5m');
  const pairHistory = await response.json();
  candleData.value = transformToCandlestickData(pairHistory);
}

loadData();
</script>
```

### Advanced Usage with Indicators

```vue
<template>
  <TradingChart
    :candle-data="candles"
    :volume-data="volume"
    :indicators="indicators"
    :height="600"
    theme="dark"
    @crosshair-move="handleCrosshairMove"
  />
</template>

<script setup>
import { ref } from 'vue';
import TradingChart from '@/components/charts/TradingChart.vue';
import { transformPairHistory } from '@/utils/chartDataTransformer';

const candles = ref([]);
const volume = ref([]);
const indicators = ref([]);

async function loadData() {
  const response = await fetch('/api/v1/pair_candles?pair=BTC/USDT&timeframe=5m');
  const pairHistory = await response.json();

  const transformed = transformPairHistory(pairHistory);
  candles.value = transformed.candles;
  volume.value = transformed.volume;
  indicators.value = transformed.indicators;
}

function handleCrosshairMove(param) {
  console.log('Crosshair data:', param);
}

loadData();
</script>
```

### Using Annotations

```typescript
import { applyAnnotations } from '@/utils/chartAnnotations';

// After chart is initialized
const annotations = [
  {
    type: 'line',
    start: '2024-01-01T00:00:00Z',
    end: '2024-01-31T00:00:00Z',
    y_start: 50000,
    y_end: 50000,
    color: '#ffc107',
    label: 'Support Level',
    line_style: 'dashed',
  },
  {
    type: 'point',
    x: '2024-01-15T12:00:00Z',
    y: 51000,
    color: '#26a69a',
    label: 'Buy Signal',
    shape: 'circle',
  },
];

// Apply to your candlestick series
applyAnnotations(chartRef.value.candlestickSeries, annotations);
```

## API Integration

### Freqtrade Endpoints Used

1. **GET /api/v1/whitelist** - Get available trading pairs
2. **GET /api/v1/pair_candles** - Get candlestick data
   - Parameters: `pair`, `timeframe`, `limit`
3. **POST /api/v1/pair_candles** - Get filtered candlestick data
   - Body: `{ pair, timeframe, limit, columns }`

### Response Format

```typescript
interface PairHistory {
  strategy: string;
  pair: string;
  timeframe: string;
  timeframe_ms: number;
  columns: string[];
  data: any[][];
  annotations?: AnnotationType[];
  length: number;
}
```

## Customization

### Changing Theme Colors

Edit the theme configuration in `TradingChart.vue`:

```typescript
const darkTheme = {
  backgroundColor: '#1e222d',  // Your dark background
  textColor: '#d1d4dc',        // Your text color
  gridColor: '#2b2f3a',        // Your grid color
  upColor: '#26a69a',          // Bullish candle color
  downColor: '#ef5350',        // Bearish candle color
};
```

### Adding Custom Indicators

In `chartDataTransformer.ts`, customize indicator colors:

```typescript
export function getIndicatorColor(indicatorName: string): string {
  const name = indicatorName.toLowerCase();

  // Add your custom mappings
  if (name.includes('my_indicator')) return '#YOUR_COLOR';

  return '#607D8B'; // Default color
}
```

### Real-time Updates via WebSocket

```typescript
// Example WebSocket integration
const ws = new WebSocket('ws://localhost:8080/api/v1/ws');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  if (message.type === 'new_candle') {
    // Update the last candle or add new one
    const newCandle = {
      time: Math.floor(message.data.date / 1000),
      open: message.data.open,
      high: message.data.high,
      low: message.data.low,
      close: message.data.close,
    };

    chartRef.value.candlestickSeries.update(newCandle);
  }
};
```

## Performance Tips

1. **Use shallowRef** for chart instances (already implemented in components)
2. **Limit data points** - Load 500-1000 candles initially, implement pagination for more
3. **Debounce resize events** - Already handled by ResizeObserver
4. **Lazy load indicators** - Only load indicators that are visible
5. **Use Web Workers** - For heavy data transformations (not included in examples)

## Troubleshooting

### Chart not rendering

- Check that container has explicit height
- Verify data format (timestamps must be in seconds, not milliseconds)
- Check browser console for errors

### Performance issues

- Reduce number of indicators displayed
- Limit initial data points
- Use `shallowRef` instead of `ref` for chart instances

### Indicators not showing

- Verify column names match exactly (case-sensitive)
- Check that indicator data doesn't contain null/NaN values
- Ensure data is sorted by time

### Time zone issues

- Lightweight Charts uses UTC by default
- Freqtrade API returns millisecond timestamps
- Transformation functions handle conversion (ms → seconds)

## Further Resources

- [Lightweight Charts Documentation](https://tradingview.github.io/lightweight-charts/)
- [Lightweight Charts GitHub](https://github.com/tradingview/lightweight-charts)
- [Freqtrade API Documentation](https://www.freqtrade.io/en/stable/rest-api/)
- [Vue.js Documentation](https://vuejs.org/)

## License

These examples are provided as part of the Freqtrade project. TradingView Lightweight Charts is licensed under Apache License 2.0.

## Contributing

If you improve these examples or find issues, please contribute back to the Freqtrade project!
