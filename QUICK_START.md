# 🚀 Quick Start: Lightweight Charts Integration

## TL;DR - Fastest Way to Get Started

```bash
# Run the automated setup script
cd /usr/src/freqtrade
./setup_lightweight_charts.sh
```

That's it! The script will:
1. Clone FreqUI (if needed)
2. Install all dependencies
3. Copy all example files to the correct locations
4. Optionally start the dev server

---

## Manual Setup (If you prefer)

### 1. Clone FreqUI
```bash
cd /usr/src
git clone https://github.com/freqtrade/frequi.git
cd frequi
```

### 2. Install Dependencies
```bash
npm install
npm install lightweight-charts
```

### 3. Copy Files
```bash
# Components
cp /usr/src/freqtrade/docs/examples/*.vue src/components/charts/

# Utils
cp /usr/src/freqtrade/docs/examples/*.ts src/utils/
```

### 4. Start Development Server
```bash
npm run dev
```

---

## What You Got

### 📦 Components

**Basic Chart** (`LightweightChart.vue`)
```vue
<LightweightChart :data="candleData" :height="400" />
```

**Advanced Chart** (`TradingChart.vue`)
```vue
<TradingChart
  :candle-data="candles"
  :volume-data="volume"
  :indicators="indicators"
  :height="600"
  theme="dark"
/>
```

### 🛠️ Utilities

**Data Transformer** (`chartDataTransformer.ts`)
```typescript
import { transformPairHistory } from '@/utils/chartDataTransformer';

const response = await fetch('/api/v1/pair_candles?pair=BTC/USDT&timeframe=5m');
const pairHistory = await response.json();
const { candles, volume, indicators } = transformPairHistory(pairHistory);
```

**Annotations** (`chartAnnotations.ts`)
```typescript
import { applyAnnotations } from '@/utils/chartAnnotations';

applyAnnotations(candlestickSeries, annotations);
```

---

## Quick Integration Example

```vue
<template>
  <div class="trading-view">
    <TradingChart
      v-if="candles.length > 0"
      :candle-data="candles"
      :volume-data="volume"
      :indicators="indicators"
      :height="600"
      theme="dark"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import TradingChart from '@/components/charts/TradingChart.vue';
import { transformPairHistory } from '@/utils/chartDataTransformer';

const candles = ref([]);
const volume = ref(null);
const indicators = ref([]);

async function loadData() {
  const response = await fetch('/api/v1/pair_candles?pair=BTC/USDT&timeframe=5m');
  const data = await response.json();

  const transformed = transformPairHistory(data);
  candles.value = transformed.candles;
  volume.value = transformed.volume;
  indicators.value = transformed.indicators;
}

onMounted(() => loadData());
</script>
```

---

## File Structure After Setup

```
frequi/
├── src/
│   ├── components/
│   │   └── charts/
│   │       ├── LightweightChart.vue    ← Basic chart
│   │       └── TradingChart.vue        ← Advanced chart
│   ├── utils/
│   │   ├── chartDataTransformer.ts    ← Data conversion
│   │   └── chartAnnotations.ts        ← Annotation support
│   └── views/
│       └── ChartView.vue              ← Complete example (optional)
└── package.json                        ← Contains lightweight-charts
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Chart not rendering | Check container has explicit height |
| Import errors | Verify file paths match your structure |
| Data not loading | Check API endpoint is accessible |
| TypeScript errors | Ensure `@/*` path alias is configured |

---

## Resources

📖 **Full Documentation**
- Integration guide: `/usr/src/freqtrade/docs/LIGHTWEIGHT_CHARTS_INTEGRATION.md`
- Step-by-step guide: `/usr/src/freqtrade/INTEGRATION_STEPS.md`

🔗 **Links**
- [Lightweight Charts Docs](https://tradingview.github.io/lightweight-charts/)
- [FreqUI Repository](https://github.com/freqtrade/frequi)
- [Freqtrade Docs](https://www.freqtrade.io/)

---

## Key Features

✅ **Fast** - ~50KB gzipped
✅ **Mobile-friendly** - Touch optimized
✅ **Professional** - TradingView quality
✅ **Annotations** - Full support
✅ **Indicators** - Unlimited
✅ **Themes** - Dark & light
✅ **Real-time** - WebSocket ready

---

## Need Help?

Run the automated setup:
```bash
./setup_lightweight_charts.sh
```

Or check the detailed guides in the `docs/` directory.

Happy trading! 📈
