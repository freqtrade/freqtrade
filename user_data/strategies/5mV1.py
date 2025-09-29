# --- TanakaAlpha5mV1.py ---
import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import (
    BooleanParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
)


try:
    # Optional external feature store (orderbook-based features)
    from freqtrade_ext.feature_store import load_orderbook_features
except Exception:  # pragma: no cover - allow strategy import without extras during tests

    def load_orderbook_features(*args, **kwargs):  # type: ignore
        return pd.DataFrame()


class TanakaAlpha5mV1(IStrategy):
    """
    目的: 勝率を取りにいくプルバック順張り(ロング) + 反発戻り売り(ショート)
    市況: BTC/USDT・ETH/USDT の 5m。1h のトレンドでフィルタ。
    """

    timeframe = "5m"
    informative_timeframes = {"1h": "1h"}

    can_short = True  # 先物前提 (Bybit 等)
    process_only_new_candles = True
    startup_candle_count = 240  # 指標ウォームアップ

    # まずは緩め。Hyperopt で詰める前提
    minimal_roi = {"0": 0.04, "30": 0.02, "120": 0.0}
    stoploss = -0.10
    trailing_stop = False
    use_custom_stoploss = True

    # Hyperopt 対象パラメータ (勝率を重視)
    buy_rsi_max = IntParameter(20, 55, default=38, space="buy", optimize=False)
    buy_adx_min = IntParameter(15, 35, default=20, space="buy", optimize=False)
    buy_mfi_max = IntParameter(20, 55, default=40, space="buy", optimize=False)

    sell_rsi_min = IntParameter(50, 85, default=64, space="sell", optimize=False)
    sell_adx_min = IntParameter(15, 35, default=20, space="sell", optimize=False)

    bbp_max = DecimalParameter(
        0.05, 0.45, default=0.22, decimals=2, space="buy", optimize=False
    )  # BB% 低い=下側
    bbp_min_short = DecimalParameter(
        0.55, 0.95, default=0.78, decimals=2, space="sell", optimize=False
    )  # BB% 高い=上側

    atr_mult = DecimalParameter(1.2, 3.5, default=2.2, decimals=1, space="stoploss")
    tsl_profit = DecimalParameter(0.03, 0.12, default=0.06, decimals=2, space="stoploss")

    # Orderbook-based probability model (heuristic logistic, hyperopt対象)
    ob_w_imb = DecimalParameter(-2.0, 2.0, default=1.0, decimals=2, space="buy")
    ob_w_ofi = DecimalParameter(-2.0, 2.0, default=0.8, decimals=2, space="buy")
    ob_w_depth = DecimalParameter(-2.0, 2.0, default=0.6, decimals=2, space="buy")
    ob_w_slope = DecimalParameter(-2.0, 2.0, default=0.3, decimals=2, space="buy")
    ob_w_spread = DecimalParameter(-2.0, 0.0, default=-0.5, decimals=2, space="buy")
    ob_w_micro = DecimalParameter(-2.0, 2.0, default=0.5, decimals=2, space="buy")
    ob_bias = DecimalParameter(-1.0, 1.0, default=0.0, decimals=2, space="buy")
    ob_zwin = IntParameter(144, 720, default=288, space="buy")
    ob_long_thresh = DecimalParameter(0.50, 0.70, default=0.58, decimals=2, space="buy")
    ob_short_thresh = DecimalParameter(0.50, 0.70, default=0.58, decimals=2, space="sell")
    # 追加トグル/閾値
    ob_use_dir = BooleanParameter(default=False, space="buy")
    ob_use_prem = BooleanParameter(default=False, space="buy")
    ob_spread_q = DecimalParameter(0.80, 0.99, default=0.95, decimals=2, space="buy")

    plot_config = {
        "main_plot": {
            "ema20": {"color": "blue"},
            "ema50": {"color": "orange"},
            "bb_middleband": {"color": "white"},
            "bb_lowerband": {"color": "grey"},
            "bb_upperband": {"color": "grey"},
        },
        "subplots": {
            "rsi": {"rsi": {"color": "purple"}},
            "adx": {"adx": {"color": "green"}},
            "mfi": {"mfi": {"color": "red"}},
            "ob_prob": {"ob_prob_long": {"color": "yellow"}},
        },
    }

    # Protections are now strategy-level (config-level is deprecated as of 2025.x)
    protections = [
        {"method": "CooldownPeriod", "stop_duration_candles": 10},
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 288,
            "trade_limit": 20,
            "stop_duration_candles": 144,
            "max_allowed_drawdown": 0.12,
        },
    ]

    def informative_pairs(self):
        return [(pair, "1h") for pair in self.dp.current_whitelist()]

    @staticmethod
    def _bollinger(dataframe: DataFrame, period: int = 20, nbdev: float = 2.0):
        # Use talib.abstract with full OHLCV dataframe to get named columns
        bb = ta.BBANDS(dataframe, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev, matype=0)
        dataframe["bb_upperband"] = bb["upperband"]
        dataframe["bb_middleband"] = bb["middleband"]
        dataframe["bb_lowerband"] = bb["lowerband"]
        dataframe["bb_width"] = (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe[
            "bb_middleband"
        ]
        dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lowerband"]) / (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # メインTF(5m)
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["vol_sma"] = ta.SMA(dataframe["volume"], timeperiod=20)
        self._bollinger(dataframe)

        # 情報TF(1h)
        informative = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1h")
        informative["ema200"] = ta.EMA(informative, timeperiod=200)
        informative["adx"] = ta.ADX(informative, timeperiod=14)
        informative["atr"] = ta.ATR(informative, timeperiod=14)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, "1h", ffill=True)

        # 板情報の特徴量をJOIN (存在すれば)。シグナルには未使用 (安全な拡張)。
        try:
            pair = metadata.get("pair", "BTC/USDT:USDT")
            exid = getattr(getattr(self, "exchange", None), "id", "bybit")
            # Define timerange based on available candles to avoid lookahead.
            if not dataframe.empty:
                tr = (
                    pd.to_datetime(dataframe.index.min()).tz_convert("UTC"),
                    pd.to_datetime(dataframe.index.max()).tz_convert("UTC"),
                )
            else:
                tr = None
            feats = load_orderbook_features(
                exid, pair, self.timeframe, timerange=tr, embargo_secs=1, depth=200
            )
            if isinstance(feats, pd.DataFrame) and not feats.empty:
                dataframe = dataframe.join(feats, how="left")
                # 安定化のためのNA処理
                dataframe = (
                    dataframe.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0)
                )
                # 主要な列をFreqAI向けにプレフィックス付きで複製 (衝突回避)
                feat_cols = [
                    "spread_bps",
                    "microprice",
                    "ob_imbalance",
                    "ob_depth_delta",
                    "ofi_top",
                    "book_slope",
                ]
                for c in feat_cols:
                    if c in dataframe.columns:
                        dataframe[f"feat__{c}"] = dataframe[c]
                # スプレッドが過大なときは取引回避 (5m: 1日=288本の分位)
                if "spread_bps" in dataframe.columns:
                    q = float(self.ob_spread_q.value) if hasattr(self, "ob_spread_q") else 0.95
                    q_spread = dataframe["spread_bps"].rolling(288, min_periods=60).quantile(q)
                    dataframe["no_trade"] = (
                        (dataframe["spread_bps"] > q_spread).astype(int).fillna(0)
                    )
                else:
                    dataframe["no_trade"] = 0

                # 簡易ロジスティックで勝率を算出 (特徴をzスコア正規化)
                def zscore(s: pd.Series, win: int) -> pd.Series:
                    m = s.rolling(win, min_periods=max(20, win // 10)).mean()
                    v = s.rolling(win, min_periods=max(20, win // 10)).std()
                    z = (s - m) / v.replace(0, np.nan)
                    return z.clip(-5, 5).fillna(0)

                zwin = int(self.ob_zwin.value) if hasattr(self, "ob_zwin") else 288

                z_imb = zscore(
                    dataframe.get("ob_imbalance", pd.Series(0, index=dataframe.index)), zwin
                )
                z_ofi = zscore(dataframe.get("ofi_top", pd.Series(0, index=dataframe.index)), zwin)
                z_depth = zscore(
                    dataframe.get("ob_depth_delta", pd.Series(0, index=dataframe.index)), zwin
                )
                z_slope = zscore(
                    dataframe.get("book_slope", pd.Series(0, index=dataframe.index)), zwin
                )
                z_spread = zscore(
                    dataframe.get("spread_bps", pd.Series(0, index=dataframe.index)), zwin
                )
                # micropriceプレミアム (close比bps)
                if "microprice" in dataframe.columns:
                    prem_bps = (dataframe["microprice"] / dataframe["close"] - 1.0) * 1e4
                else:
                    prem_bps = pd.Series(0, index=dataframe.index)
                z_micro = zscore(prem_bps, zwin)
                dataframe["prem_bps"] = prem_bps

                s = (
                    float(self.ob_bias.value)
                    + float(self.ob_w_imb.value) * z_imb
                    + float(self.ob_w_ofi.value) * z_ofi
                    + float(self.ob_w_depth.value) * z_depth
                    + float(self.ob_w_slope.value) * z_slope
                    + float(self.ob_w_spread.value) * z_spread
                    + float(self.ob_w_micro.value) * z_micro
                )
                # 数値安定化
                s = s.clip(-20, 20)
                p_long = 1.0 / (1.0 + np.exp(-s))
                p_short = 1.0 / (1.0 + np.exp(s))  # = 1 - p_long

                dataframe["ob_prob_long"] = p_long.astype(float)
                dataframe["ob_prob_short"] = p_short.astype(float)
        except Exception:
            # フィーチャーが無い/読み込み失敗時は無視して継続
            dataframe["no_trade"] = 0
            dataframe["ob_prob_long"] = 1.0
            dataframe["ob_prob_short"] = 1.0

        # トレンドフラグ (1hの大局 + 5mの位置)
        dataframe["uptrend"] = (dataframe["close_1h"] > dataframe["ema200_1h"]) & (
            dataframe["adx_1h"] > 15
        )
        dataframe["downtrend"] = (dataframe["close_1h"] < dataframe["ema200_1h"]) & (
            dataframe["adx_1h"] > 15
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long: 上昇トレンド中の下振れ拾い (勝率重視)
        long_cond = (
            (dataframe["uptrend"])
            & (dataframe["close"] > dataframe["ema200"])
            & (dataframe["bb_percent"] < float(self.bbp_max.value))
            & (dataframe["rsi"] <= int(self.buy_rsi_max.value))
            & (dataframe["mfi"] <= int(self.buy_mfi_max.value))
            & (dataframe["adx"] >= int(self.buy_adx_min.value))
            & (dataframe["volume"] > dataframe["vol_sma"])
            & (dataframe.get("no_trade", 0) == 0)
            & (dataframe.get("ob_prob_long", 1.0) >= float(self.ob_long_thresh.value))
        )
        if hasattr(self, "ob_use_dir") and bool(self.ob_use_dir.value):
            long_cond &= (dataframe.get("ob_imbalance", 0) > 0) & (dataframe.get("ofi_top", 0) > 0)
        if hasattr(self, "ob_use_prem") and bool(self.ob_use_prem.value):
            long_cond &= dataframe.get("prem_bps", 0) > 0

        # Short: 下降トレンド中の上振れ叩き
        short_cond = (
            (dataframe["downtrend"])
            & (dataframe["close"] < dataframe["ema200"])
            & (dataframe["bb_percent"] > float(self.bbp_min_short.value))
            & (dataframe["rsi"] >= int(self.sell_rsi_min.value))
            & (dataframe["adx"] >= int(self.sell_adx_min.value))
            & (dataframe["volume"] > dataframe["vol_sma"])
            & (dataframe.get("no_trade", 0) == 0)
            & (dataframe.get("ob_prob_short", 1.0) >= float(self.ob_short_thresh.value))
        )
        if hasattr(self, "ob_use_dir") and bool(self.ob_use_dir.value):
            short_cond &= (dataframe.get("ob_imbalance", 0) < 0) & (dataframe.get("ofi_top", 0) < 0)
        if hasattr(self, "ob_use_prem") and bool(self.ob_use_prem.value):
            short_cond &= dataframe.get("prem_bps", 0) < 0

        dataframe.loc[long_cond, ["enter_long", "enter_tag"]] = (1, "L_bbp_pullback")
        dataframe.loc[short_cond, ["enter_short", "enter_tag"]] = (1, "S_bbp_pullup")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 長く引っ張らず勝率重視でこまめに離脱
        exit_long = (dataframe["rsi"] >= int(self.sell_rsi_min.value)) | (
            dataframe["close"] >= dataframe["bb_middleband"]
        )
        exit_short = (dataframe["rsi"] <= int(self.buy_rsi_max.value)) | (
            dataframe["close"] <= dataframe["bb_middleband"]
        )
        dataframe.loc[exit_long, "exit_long"] = 1
        dataframe.loc[exit_short, "exit_short"] = 1
        return dataframe

    # ATR ベースの段階的トレーリング (勝率維持しつつDD抑制)
    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        # 直近足の ATR を利用
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return self.stoploss

        atr = dataframe["atr"].iloc[-1]
        base_sl = abs(float(self.stoploss))
        atr_sl = float(self.atr_mult.value) * (atr / current_rate)  # ATR を率に変換
        sl_now = max(base_sl, atr_sl)

        # 利益が乗ったらタイトに
        if current_profit is not None and current_profit > float(self.tsl_profit.value):
            sl_now = min(sl_now, 0.01)  # +1% まで追い込み

        return -float(sl_now)
