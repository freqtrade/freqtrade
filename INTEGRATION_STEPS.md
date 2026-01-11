# Next Steps: Integrating Lightweight Charts into FreqUI

Since FreqUI is a **separate repository**, here's your step-by-step guide to complete the integration.

## Overview

FreqUI is maintained at: https://github.com/freqtrade/frequi

You have:
- ✅ Pulled the Lightweight Charts integration documentation
- ✅ Installed `lightweight-charts` npm package (where?)
- ⏳ Need to integrate into FreqUI

## Step 1: Clone FreqUI Repository

```bash
# Navigate to your workspace
cd /usr/src

# Clone the FreqUI repository
git clone https://github.com/freqtrade/frequi.git

# Enter the directory
cd frequi
```

## Step 2: Install Dependencies

```bash
# Install all FreqUI dependencies
npm install

# Install Lightweight Charts (if not already done)
npm install lightweight-charts

# Verify installation
npm list lightweight-charts
```

## Step 3: Copy Example Files to FreqUI

```bash
# From the freqtrade repository, copy the example files to FreqUI

# Create directories if they don't exist
mkdir -p src/components/charts
mkdir -p src/utils

# Copy Vue components
cp /usr/src/freqtrade/docs/examples/LightweightChart.vue src/components/charts/
cp /usr/src/freqtrade/docs/examples/TradingChart.vue src/components/charts/

# Copy utility files
cp /usr/src/freqtrade/docs/examples/chartDataTransformer.ts src/utils/
cp /usr/src/freqtrade/docs/examples/chartAnnotations.ts src/utils/

# Copy the complete view example
cp /usr/src/freqtrade/docs/examples/ChartView.vue src/views/
```

## Step 4: Update FreqUI Router (Optional)

If you want to add the new chart view to the navigation:

Edit `src/router/index.ts`:

```typescript
import ChartView from '@/views/ChartView.vue';

// Add to your routes array
{
  path: '/chart',
  name: 'Chart',
  component: ChartView,
  meta: {
    requiresAuth: true,
  },
}
```

## Step 5: Integrate into Existing Views

### Option A: Replace Existing Chart Component

Find where the current chart is used in FreqUI (likely in a trading or pair view) and replace it:

```vue
<!-- OLD -->
<PlotlyChart :data="chartData" />

<!-- NEW -->
<TradingChart
  :candle-data="candles"
  :volume-data="volume"
  :indicators="indicators"
  :height="600"
  theme="dark"
/>
```

### Option B: Add as New Tab/View

Add the chart as a new tab in an existing view:

```vue
<template>
  <b-tabs>
    <b-tab title="Old Chart">
      <PlotlyChart :data="chartData" />
    </b-tab>
    <b-tab title="New Chart">
      <TradingChart
        :candle-data="candles"
        :volume-data="volume"
        :indicators="indicators"
      />
    </b-tab>
  </b-tabs>
</template>
```

## Step 6: Transform Data for Lightweight Charts

In your component that fetches chart data, add the transformation:

```vue
<script setup lang="ts">
import { ref } from 'vue';
import TradingChart from '@/components/charts/TradingChart.vue';
import { transformPairHistory } from '@/utils/chartDataTransformer';
import { applyAnnotations } from '@/utils/chartAnnotations';

const candles = ref([]);
const volume = ref([]);
const indicators = ref([]);

async function loadChartData(pair: string, timeframe: string) {
  try {
    // Fetch from Freqtrade API
    const response = await fetch(
      `/api/v1/pair_candles?pair=${pair}&timeframe=${timeframe}&limit=500`
    );
    const pairHistory = await response.json();

    // Transform data
    const transformed = transformPairHistory(pairHistory);
    candles.value = transformed.candles;
    volume.value = transformed.volume;
    indicators.value = transformed.indicators;

    // Apply annotations if available
    if (transformed.annotations.length > 0) {
      // This will be applied after the chart is mounted
      // See the ChartView.vue example for implementation
    }
  } catch (error) {
    console.error('Failed to load chart data:', error);
  }
}
</script>
```

## Step 7: Run FreqUI Development Server

```bash
# In the frequi directory
npm run dev

# Or
npm run serve
```

The dev server will typically run on http://localhost:3000 or http://localhost:8080

## Step 8: Test the Integration

1. Open FreqUI in your browser
2. Navigate to the chart view
3. Select a trading pair
4. Verify:
   - ✅ Candlesticks render correctly
   - ✅ Volume chart appears below
   - ✅ Indicators display
   - ✅ Chart is responsive
   - ✅ Crosshair works
   - ✅ Zoom and pan work

## Troubleshooting

### Issue: Import errors with TypeScript

If you see TypeScript errors, ensure your `tsconfig.json` includes:

```json
{
  "compilerOptions": {
    "types": ["node"],
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

### Issue: Module not found 'lightweight-charts'

```bash
# Reinstall
npm install lightweight-charts --save

# Clear cache
rm -rf node_modules package-lock.json
npm install
```

### Issue: Chart not rendering

1. Check browser console for errors
2. Verify the container has explicit height
3. Ensure data is in correct format (timestamps in seconds, not milliseconds)
4. Check that the chart container is visible in the DOM

### Issue: Data transformation errors

1. Log the raw API response: `console.log(pairHistory)`
2. Check column names match (case-sensitive)
3. Verify timestamps are numbers, not strings

## Alternative: Quick Test Setup

If you want to quickly test without integrating into FreqUI's existing structure:

```bash
# In the frequi directory, create a test page
cat > src/views/TestChart.vue << 'EOF'
<template>
  <div class="test-chart">
    <h1>Lightweight Charts Test</h1>
    <TradingChart
      :candle-data="sampleData"
      :height="500"
      theme="dark"
    />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import TradingChart from '@/components/charts/TradingChart.vue';

const sampleData = ref([
  { time: '2024-01-01', open: 100, high: 110, low: 95, close: 105 },
  { time: '2024-01-02', open: 105, high: 115, low: 100, close: 110 },
  // Add more sample data...
]);
</script>

<style scoped>
.test-chart {
  padding: 20px;
}
</style>
EOF

# Add route to router
# Then visit /test-chart in browser
```

## Production Build

When ready for production:

```bash
# Build FreqUI
npm run build

# The built files will be in the dist/ directory
# You can then deploy them to your Freqtrade server
```

## Additional Resources

- **Integration Guide**: `/usr/src/freqtrade/docs/LIGHTWEIGHT_CHARTS_INTEGRATION.md`
- **Example Files**: `/usr/src/freqtrade/docs/examples/`
- **FreqUI Repository**: https://github.com/freqtrade/frequi
- **Lightweight Charts Docs**: https://tradingview.github.io/lightweight-charts/

## Need Help?

If you encounter issues:

1. Check the Freqtrade documentation: https://www.freqtrade.io/
2. FreqUI issues: https://github.com/freqtrade/frequi/issues
3. Lightweight Charts issues: https://github.com/tradingview/lightweight-charts/issues

---

**Next Command to Run:**

```bash
cd /usr/src && git clone https://github.com/freqtrade/frequi.git && cd frequi && npm install
```
