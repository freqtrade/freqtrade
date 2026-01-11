import axios from 'axios';
import { TimeSummaryCols, TimeSummaryOptions } from '@/types';
import type { ThemeName, UiVersion } from '@/types';
import { FtWsMessageTypes } from '@/types/wsMessageTypes';

export enum OpenTradeVizOptions {
  showPill = 'showPill',
  asTitle = 'asTitle',
  noOpenTrades = 'noOpenTrades',
}

const notificationDefaults = {
  [FtWsMessageTypes.entryFill]: true,
  [FtWsMessageTypes.exitFill]: true,
  [FtWsMessageTypes.entryCancel]: true,
  [FtWsMessageTypes.exitCancel]: true,
};

export const useSettingsStore = defineStore('uiSettings', {
  // other options...
  state: () => {
    return {
      openTradesInTitle: OpenTradeVizOptions.showPill as string,
      timezone: 'UTC',
      backgroundSync: true,
      currentTheme: 'dark' as ThemeName,
      _uiVersion: 'dev',
      useHeikinAshiCandles: false,
      useLightweightCharts: true,
      showMarkArea: true,
      useReducedPairCalls: true,
      notifications: notificationDefaults,
      profitDistributionBins: 20,
      confirmDialog: true,
      chartLabelSide: 'right' as 'left' | 'right',
      chartDefaultCandleCount: 250,
      timeProfitPeriod: TimeSummaryOptions.daily,
      timeProfitPreference: TimeSummaryCols.abs_profit,
      multiPaneButtonsShowText: false,
      multiPairSelection: false,
      backtestAdditionalMetrics: ['profit_factor', 'expectancy'] as string[],
    };
  },
  getters: {
    isDarkTheme(state) {
      return ['dark', 'bootstrap_dark'].includes(state.currentTheme);
    },
    chartTheme(): 'dark' | 'light' {
      return this.isDarkTheme ? 'dark' : 'light';
    },
    uiVersion(state) {
      return `${state._uiVersion}-${__COMMIT_HASH__}`;
    },
  },
  actions: {
    async loadUIVersion() {
      if (import.meta.env.PROD) {
        try {
          const result = await axios.get<UiVersion>('/ui_version', {
            withCredentials: true,
          });
          const { version } = result.data;
          this._uiVersion = version ?? 'dev';
        } catch (error) {
          //
        }
      }
    },
  },
  persist: {
    key: 'ftUISettings',
  },
});

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useSettingsStore, import.meta.hot));
}
