import logging
from datetime import datetime

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IStrategy, timeframe_to_minutes


try:
    from freqtrade_ext.feature_store import load_orderbook_features
except Exception:

    def load_orderbook_features(*args, **kwargs):  # type: ignore
        return pd.DataFrame()


logger = logging.getLogger(__name__)


class FreqAICustomStrategy(IStrategy):
    """
    カスタム FreqAI 戦略
    指定された10個の特徴量を使用してモデルを訓練
    """

    # 戦略のメタデータ
    INTERFACE_VERSION = 3
    minimal_roi = {"0": 10}
    stoploss = -0.99
    process_only_new_candles = True
    stoploss_on_exchange = False
    startup_candle_count: int = 300
    can_short = True
    # Enable position adjustment hooks (we'll return None when disabled via config)
    position_adjustment_enable: bool = True

    # トレーリングストップ設定
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # 売買の閾値パラメーター
    # Use realistic thresholds for 1m regression target (&-target ~ future return)
    buy_threshold = DecimalParameter(
        0.0001, 0.0050, default=0.0022, decimals=4, optimize=True, space="buy"
    )
    sell_threshold = DecimalParameter(
        0.0001, 0.0030, default=0.0003, decimals=4, optimize=True, space="sell"
    )

    # Debug: allow bypassing do_predict gate to diagnose zero-trade issues
    # Can be overridden via config: freqai.debug_skip_do_predict_gate: true
    debug_skip_do_predict_gate: bool = False

    # --- extension wiring (lazy-initialized) ---
    _ext_vol_sizer = None
    _ext_exit_policy = None
    _calib_store: dict | None = None

    def _ext_cfg(self) -> dict:
        return self.config.get("ext_risk", {}) if hasattr(self, "config") else {}

    def _get_timeframe_minutes(self) -> int:
        try:
            return int(timeframe_to_minutes(self.timeframe))
        except Exception:
            return 1

    def informative_pairs(self):
        """
        追加の通貨ペア情報 (必要に応じて)
        """
        whitelist_pairs = self.dp.current_whitelist()
        corr_pairs = self.config["freqai"]["feature_parameters"]["include_corr_pairlist"]
        informative_pairs = []
        for tf in self.config["freqai"]["feature_parameters"]["include_timeframes"]:
            for pair in whitelist_pairs:
                informative_pairs.append((pair, tf))
            for pair in corr_pairs:
                if pair in whitelist_pairs:
                    continue
                informative_pairs.append((pair, tf))
        return informative_pairs

    # Removed deprecated populate_any_indicators (migrated to feature_engineering_* methods)

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        FreqAI用の特徴量作成 (自動拡張対象)
        10個の主要特徴量+補助特徴量を作成。列名は % プレフィックス必須。
        """
        # 1. EMA5-EMA10 差分
        ema5 = ta.EMA(dataframe, timeperiod=5)
        ema10 = ta.EMA(dataframe, timeperiod=10)
        dataframe["%-ema5_ema10_diff"] = (ema5 - ema10) / dataframe["close"]

        # 2. RSI(14)
        dataframe["%-rsi_14"] = ta.RSI(dataframe, timeperiod=14)

        # 3. MACD ヒストグラム
        macd_line, macd_signal, macd_histogram = ta.MACD(
            dataframe, fastperiod=12, slowperiod=26, signalperiod=9
        )
        dataframe["%-macd_histogram"] = macd_histogram

        # 4. ボリンジャーバンド幅 (20) - tolerate different TA-Lib return types
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        bb_upper = bb_middle = bb_lower = None
        # Handle different TA-Lib return types explicitly
        if isinstance(bb, list | tuple) and len(bb) >= 3:
            bb_upper, bb_middle, bb_lower = bb[0], bb[1], bb[2]
        elif hasattr(bb, "columns"):
            cols = [str(c).lower() for c in bb.columns]

            def _col(name, fallback_idx):
                name_l = name.lower()
                if name_l in cols:
                    return bb.iloc[:, cols.index(name_l)]
                return bb.iloc[:, fallback_idx]

            bb_upper = _col("upperband", 0)
            bb_middle = _col("middleband", 1)
            bb_lower = _col("lowerband", 2)
        elif isinstance(bb, dict):
            bb_upper = bb.get("upperband")
            bb_middle = bb.get("middleband")
            bb_lower = bb.get("lowerband")
        if bb_upper is not None and bb_lower is not None and bb_middle is not None:
            dataframe["%-bb_width"] = (bb_upper - bb_lower) / bb_middle

        # 5. ATR(14)
        dataframe["%-atr_14"] = ta.ATR(dataframe, timeperiod=14) / dataframe["close"]

        # 6. OBV と変化率
        dataframe["%-obv"] = ta.OBV(dataframe)
        dataframe["%-obv_pct"] = dataframe["%-obv"].pct_change(periods=5)

        # 7. モメンタム (5)
        dataframe["%-momentum_5"] = ta.MOM(dataframe, timeperiod=5) / dataframe["close"]

        # 8. ストキャスティクス %K(14)
        slowk, slowd = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe["%-stoch_k"] = slowk

        # 9. Williams %R(14)
        dataframe["%-williams_r"] = ta.WILLR(dataframe, timeperiod=14)

        # 10. 出来高変化率
        dataframe["%-volume_pct"] = dataframe["volume"].pct_change()

        # 補助特徴量
        dataframe["%-price_pct"] = dataframe["close"].pct_change()
        ema20 = ta.EMA(dataframe, timeperiod=20)
        dataframe["%-price_vs_ema20"] = (dataframe["close"] - ema20) / dataframe["close"]
        dataframe["%-high_low_ratio"] = (dataframe["close"] - dataframe["low"]) / (
            (dataframe["high"] - dataframe["low"]).replace(0, np.nan)
        )
        dataframe["%-volatility"] = dataframe["close"].rolling(20).std() / dataframe["close"]

        return self._sanitize_features(dataframe)

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        自動拡張しない一般特徴量 (例: 曜日/時刻など)
        """
        # 時間特徴量 (インデックスは変更しない)
        if "date" in dataframe.columns:
            _times = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        else:
            # フォールバック: インデックスから
            _times = pd.to_datetime(dataframe.index, utc=True, errors="coerce")
        dataframe["%-hour"] = _times.dt.hour
        dataframe["%-day_of_week"] = _times.dt.dayofweek

        # Base EMAs used in entry/exit conditions
        try:
            dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
            dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        except Exception as e:
            logger.debug("EMA calculation failed: %s", e)

        # Join external orderbook features (1m aggregated)
        # right-open window, embargo & shift handled upstream
        dataframe = self._join_ob_features(dataframe, _times, metadata)

        # Market regime filters: btc turbulence + own volatility high-quantile
        dataframe = self._add_market_regime(dataframe)

        return self._sanitize_features(dataframe)

    def _join_ob_features(
        self, dataframe: DataFrame, _times: pd.Series, metadata: dict
    ) -> DataFrame:
        try:
            pair = (
                metadata.get("pair", "BTC/USDT:USDT")
                if isinstance(metadata, dict)
                else "BTC/USDT:USDT"
            )
            start, end = _times.min(), _times.max()
            if pd.notna(start) and pd.notna(end):
                feats = load_orderbook_features(
                    exchange=getattr(getattr(self, "exchange", None), "id", "bybit"),
                    pair=pair,
                    timeframe=self.timeframe,
                    timerange=(pd.Timestamp(start), pd.Timestamp(end)),
                    embargo_secs=1,
                    depth=200,
                )
                if not feats.empty:
                    out = dataframe.copy()
                    out["__tmp_idx__"] = _times  # tz-aware
                    out = out.set_index("__tmp_idx__").join(feats, how="left")
                    out = out.reset_index().rename(columns={"index": "date", "__tmp_idx__": "date"})
                    dataframe = out
                    dataframe = (
                        dataframe.replace([float("inf"), float("-inf")], pd.NA)
                        .fillna(method="ffill")
                        .fillna(0)
                    )
                    for c in [
                        "spread_bps",
                        "microprice",
                        "ob_imbalance",
                        "ob_depth_delta",
                        "ofi_top",
                        "book_slope",
                    ]:
                        if c in dataframe.columns:
                            dataframe[f"feat__{c}"] = dataframe[c]
        except Exception as e:
            logger.debug("Orderbook feature join failed: %s", e)
        return dataframe

    def _add_market_regime(self, dataframe: DataFrame) -> DataFrame:
        try:
            base_tf = getattr(self, "timeframe", "5m")
            btc_pair = "BTC/USDT:USDT"
            btc_df, _ = self.dp.get_analyzed_dataframe(btc_pair, base_tf)
            btc_ret = btc_df["close"].pct_change()
            btc_vol = btc_ret.rolling(96, min_periods=48).std()
            if isinstance(dataframe.index, pd.DatetimeIndex) and isinstance(
                btc_vol.index, pd.DatetimeIndex
            ):
                reidx = btc_vol.reindex(
                    dataframe.index,
                    method="nearest",
                    tolerance=pd.Timedelta(minutes=self._get_timeframe_minutes()),
                )
            else:
                reidx = pd.Series(0, index=dataframe.index)
            q_hi = reidx.rolling(
                1440 // max(1, self._get_timeframe_minutes()), min_periods=60
            ).quantile(0.95)
            dataframe["market_bad"] = (reidx >= q_hi).astype(int).fillna(0)
        except Exception as e:
            dataframe["market_bad"] = 0
            logger.debug("Market regime filter failed: %s", e)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        FreqAI の学習/推論を起動し、必要カラム (do_predict, &-target 等) を付与。
        先にこのクラスで作る基本指標は feature_engineering_* で生成されます。
        """
        # Ensure base EMAs exist (redundant safety)
        try:
            if "ema_20" not in dataframe.columns:
                dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
            if "ema_50" not in dataframe.columns:
                dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        except Exception as e:
            logger.debug("Base EMA ensure failed: %s", e)

        # Kick off FreqAI pipeline (targets + predictions + do_predict flags)
        dataframe = self.freqai.start(dataframe, metadata, self)
        try:
            # Lightweight debug to confirm acceptance/passing counts
            dp_series = dataframe.get("do_predict", pd.Series([0] * len(dataframe)))
            dp = int((dp_series == 1).sum())
            tgt = dataframe.get("&-target")
            bt = float(self.buy_threshold.value)
            if tgt is not None:
                hi_all = int((tgt > bt).sum())
                lo_all = int((tgt < -bt).sum())
                hi = int(((tgt > bt) & (dp_series == 1)).sum())
                lo = int(((tgt < -bt) & (dp_series == 1)).sum())
                logging.getLogger(__name__).info(
                    (
                        "[FreqAICustomStrategy] do_predict==1: %s | &-target>(%g): %s (dp1:%s) "
                        "| &-target<-(%g): %s (dp1:%s)"
                    ),
                    dp,
                    bt,
                    hi_all,
                    hi,
                    bt,
                    lo_all,
                    lo,
                )
        except Exception as e:
            logger.debug("FreqAI debug logging failed: %s", e)
        return dataframe

    # --- helpers ---
    def _sanitize_features(self, df: DataFrame) -> DataFrame:
        """Ensure engineered features are numeric for FreqAI pipeline.
        Coerce %-prefixed and feat__ columns to numeric, replace inf/NaN.
        """
        out = df.copy()
        cols = [c for c in out.columns if str(c).startswith("%-") or str(c).startswith("feat__")]
        if cols:
            out[cols] = out[cols].apply(pd.to_numeric, errors="coerce")
            out[cols] = out[cols].replace([float("inf"), float("-inf")], pd.NA)
            out[cols] = out[cols].fillna(0)
        return out

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        ラベル (目的変数) を作成。
        - 回帰: 将来のリターンを &-target
        - 参考: 三重バリア分類ラベル &-tb_label (将来のhit順序で±1/0)
        """
        lp = self.config["freqai"]["feature_parameters"].get("label_period_candles", 24)
        # Forward return (realized) - used as regression target or calibration target
        fwd_ret = (dataframe["close"].shift(-lp) / dataframe["close"]) - 1.0
        pmode_cfg = (
            self.config.get("freqai", {}).get("prediction_mode", "regression")
            if hasattr(self, "config")
            else "regression"
        )
        if str(pmode_cfg).lower() == "classification":
            # Classification mode: don't create a regression &-target label to avoid mixed task
            dataframe["ret_fwd"] = fwd_ret
        else:
            dataframe["&-target"] = fwd_ret

        # Optional: Triple-barrier label (non-invasive, enables classifier use)
        try:
            tb_cfg = (
                self.config.get("ext_risk", {}).get("tb", {}) if hasattr(self, "config") else {}
            )
            tp = float(tb_cfg.get("tp", 0.006))  # 0.6%
            sl = float(tb_cfg.get("sl", 0.006))  # 0.6%
            hz = int(tb_cfg.get("horizon_candles", lp))
            tb_num = self._triple_barrier_labels(dataframe, tp=tp, sl=sl, horizon=hz)
            dataframe["&-tb_label"] = tb_num
            # 予測モードに応じて分類ラベルは条件付きで生成
            pmode_cfg = (
                self.config.get("freqai", {}).get("prediction_mode", "regression")
                if hasattr(self, "config")
                else "regression"
            )
            if str(pmode_cfg).lower() == "classification":
                tb_str = tb_num.map({-1: "down", 0: "down", 1: "up"}).astype(object)
                dataframe["&s-tb_label"] = tb_str
                try:
                    if hasattr(self, "freqai"):
                        self.freqai.class_names = ["down", "up"]
                except Exception as e:
                    logger.debug("Setting class_names failed: %s", e)
        except Exception as e:
            # Fail silently without impacting regression
            logger.debug("Triple-barrier label generation failed: %s", e)
        return dataframe

    def _triple_barrier_labels(
        self, df: DataFrame, *, tp: float, sl: float, horizon: int
    ) -> pd.Series:
        """Compute simple triple-barrier labels (1: tp first, -1: sl first, 0: none by horizon).
        Uses a rolling forward window scan; horizon should be modest (<= ~96) for speed.
        """
        closes = df["close"].astype(float).values
        n = len(closes)
        out = np.zeros(n, dtype=np.int8)
        h = max(1, int(horizon))
        for i in range(n):
            if i + 1 >= n:
                break
            base = closes[i]
            upto = min(n, i + 1 + h)
            future = closes[i + 1 : upto]
            if future.size == 0 or base <= 0:
                continue
            rets = future / base - 1.0
            hit_up_idx = np.where(rets >= tp)[0]
            hit_dn_idx = np.where(rets <= -abs(sl))[0]
            up_first = hit_up_idx[0] if hit_up_idx.size else None
            dn_first = hit_dn_idx[0] if hit_dn_idx.size else None
            if up_first is None and dn_first is None:
                out[i] = 0
            elif up_first is None:
                out[i] = -1
            elif dn_first is None:
                out[i] = 1
            else:
                out[i] = 1 if up_first < dn_first else -1
        return pd.Series(out, index=df.index)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  # noqa: C901
        """
        エントリー条件を定義 (FreqAIの予測+品質フラグ)
        """
        conditions_long = []
        conditions_short = []

        # 予測の品質チェック (0=良, 1=除外, 2=モデル期限切れ等)
        # デバッグ用に do_predict ゲートを無効化できるトグルを用意
        dp_override = (
            self.config.get("freqai", {}).get("debug_skip_do_predict_gate", False)
            if hasattr(self, "config")
            else False
        )
        use_skip = bool(self.debug_skip_do_predict_gate or dp_override)

        dp_series = dataframe.get("do_predict")
        if use_skip or (dp_series is None):
            dp_mask = pd.Series([1] * len(dataframe), index=dataframe.index)
        else:
            dp_mask = dp_series == 1

        # 汎用: 予測列の解決 (回帰/分類の両方に対応)
        try:
            pmode = self.config.get("freqai", {}).get("prediction_mode", "regression")
        except Exception:
            pmode = "regression"

        if pmode == "classification":
            clf_cfg = (
                self.config.get("freqai", {}).get("classifier", {})
                if hasattr(self, "config")
                else {}
            )
            # 推奨: configで proba列名を指定
            long_key = clf_cfg.get("long_key")
            short_key = clf_cfg.get("short_key")
            thr_long = float(clf_cfg.get("long_threshold", 0.5))
            thr_short = float(clf_cfg.get("short_threshold", 0.5))

            # Optional: Calibrate thresholds on-the-fly for this pair
            try:
                calib_cfg = (
                    self.config.get("ext_risk", {}).get("calibration", {})
                    if hasattr(self, "config")
                    else {}
                )
                if calib_cfg.get("enabled", False) and (long_key or short_key):
                    if self._calib_store is None:
                        self._calib_store = {}
                    pair = metadata.get("pair") if isinstance(metadata, dict) else None
                    key = pair or "__default__"
                    if key not in self._calib_store:
                        from freqtrade_ext.calibration.thresholds import best_proba_thresholds

                        df_for_cal = dataframe
                        # Use realized forward return column for calibration in classification mode
                        target_col = "ret_fwd" if "ret_fwd" in df_for_cal.columns else "&-target"
                        res = best_proba_thresholds(
                            df_for_cal,
                            long_col=(long_key or "up"),
                            short_col=(short_key or "down"),
                            target_col=target_col,
                            dp_col="do_predict",
                        )
                        if res is not None:
                            self._calib_store[key] = (res.long_threshold, res.short_threshold)
                    if key in self._calib_store:
                        thr_long, thr_short = self._calib_store[key]
            except Exception as e:
                logger.debug("Calibration thresholds failed: %s", e)
            # 自動推測 (なければ 1 / 0 カラム、または up/down)
            if long_key is None:
                for cand in ["1", 1, "up", "long", "LONG"]:
                    if str(cand) in dataframe.columns:
                        long_key = str(cand)
                        break
            if short_key is None:
                for cand in ["0", 0, "down", "short", "SHORT"]:
                    if str(cand) in dataframe.columns:
                        short_key = str(cand)
                        break
            # ベース条件 (proba または ラベル)
            base_long = None
            base_short = None
            if long_key and long_key in dataframe.columns:
                base_long = dataframe[long_key] >= thr_long
            if short_key and short_key in dataframe.columns:
                base_short = dataframe[short_key] >= thr_short
            if base_long is None or base_short is None:
                if "&s-tb_label" in dataframe.columns:
                    base_long = (
                        (dataframe["&s-tb_label"] == "up") if base_long is None else base_long
                    )
                    base_short = (
                        (dataframe["&s-tb_label"] == "down") if base_short is None else base_short
                    )
            if base_long is None or base_short is None:
                rk = (
                    self.config.get("freqai", {}).get("prediction_key", "&-target")
                    if hasattr(self, "config")
                    else "&-target"
                )
                if rk in dataframe.columns:
                    base_long = (
                        (dataframe[rk] > self.buy_threshold.value)
                        if base_long is None
                        else base_long
                    )
                    base_short = (
                        (dataframe[rk] < -self.buy_threshold.value)
                        if base_short is None
                        else base_short
                    )

            # トレンド/スプレッドフィルタ適用 (回帰と同様)
            try:
                trend_long = (dataframe["ema_20"] > dataframe["ema_50"]).fillna(False)
                trend_short = (dataframe["ema_20"] < dataframe["ema_50"]).fillna(False)
            except Exception as e:
                logger.debug("Trend filter failed: %s", e)
                trend_long = trend_short = pd.Series([True] * len(dataframe), index=dataframe.index)
            spread_ok = None
            try:
                if "feat__spread_bps" in dataframe.columns:
                    tf_min = max(1, self._get_timeframe_minutes())
                    win = max(60 // tf_min, int(1440 // tf_min))
                    q = (
                        dataframe["feat__spread_bps"]
                        .rolling(win, min_periods=max(10, win // 6))
                        .quantile(0.95)
                    )
                    spread_ok = (dataframe["feat__spread_bps"] <= q).fillna(True)
            except Exception:
                spread_ok = None

            if base_long is not None:
                lm = base_long & dp_mask & trend_long & (dataframe.get("market_bad", 0) == 0)
                if spread_ok is not None:
                    lm &= spread_ok
                conditions_long.append(lm)
            if base_short is not None:
                sm = base_short & dp_mask & trend_short & (dataframe.get("market_bad", 0) == 0)
                if spread_ok is not None:
                    sm &= spread_ok
                conditions_short.append(sm)
        else:
            # 回帰 (デフォルト): 予測列キーの解決
            pred_key = (
                self.config.get("freqai", {}).get("prediction_key", "&-target")
                if hasattr(self, "config")
                else "&-target"
            )
            key = (
                pred_key
                if pred_key in dataframe.columns
                else ("&-target" if "&-target" in dataframe.columns else None)
            )
            if key is not None:
                # Optional calibration of absolute threshold
                try:
                    calib_cfg = (
                        self.config.get("ext_risk", {}).get("calibration", {})
                        if hasattr(self, "config")
                        else {}
                    )
                    thr_l = self.buy_threshold.value
                    thr_s = self.buy_threshold.value
                    if calib_cfg.get("enabled", False):
                        from freqtrade_ext.calibration.thresholds import best_abs_threshold

                        res = best_abs_threshold(dataframe, pred_col=key, dp_col="do_predict")
                        if res is not None:
                            thr_l, thr_s, _, _ = res
                except Exception as e:
                    logger.debug("Absolute calibration failed: %s", e)
                    thr_l = thr_s = self.buy_threshold.value

                base_long = (dataframe[key] > float(thr_l)) & dp_mask
                base_short = (dataframe[key] < -float(thr_s)) & dp_mask

                # トレンドフィルタ (EMA20 / EMA50)
                try:
                    trend_long = (dataframe["ema_20"] > dataframe["ema_50"]).fillna(False)
                    trend_short = (dataframe["ema_20"] < dataframe["ema_50"]).fillna(False)
                except Exception as e:
                    logger.debug("Trend filter failed (regression): %s", e)
                    trend_long = trend_short = pd.Series(
                        [True] * len(dataframe), index=dataframe.index
                    )

                # スプレッドフィルタ: 高スプレッド状態は回避
                spread_ok = None
                try:
                    if "feat__spread_bps" in dataframe.columns:
                        tf_min = max(1, self._get_timeframe_minutes())
                        win = max(60 // tf_min, int(1440 // tf_min))  # 1日相当、最低60分
                        q = (
                            dataframe["feat__spread_bps"]
                            .rolling(win, min_periods=max(10, win // 6))
                            .quantile(0.95)
                        )
                        spread_ok = (dataframe["feat__spread_bps"] <= q).fillna(True)
                except Exception as e:
                    logger.debug("Spread filter failed: %s", e)
                    spread_ok = None

            # Apply regime filter conservatively in classification mode only.
            # For regression, keep filters lighter.
            long_mask = base_long & trend_long
            short_mask = base_short & trend_short
            if spread_ok is not None:
                long_mask &= spread_ok
                short_mask &= spread_ok

            conditions_long.append(long_mask)
            conditions_short.append(short_mask)

        if conditions_long:
            long_mask = reduce(lambda x, y: x & y, conditions_long)
            dataframe.loc[long_mask, "enter_long"] = 1
        if conditions_short:
            short_mask = reduce(lambda x, y: x & y, conditions_short)
            dataframe.loc[short_mask, "enter_short"] = 1

        # 追加のデバッグ出力 (どの予測列/しきい値を使ったか)
        try:
            bt = float(self.buy_threshold.value)
            n = len(dataframe)
            n_dp1 = int(dp_mask.sum()) if "dp_mask" in locals() else 0
            # 集計は回帰をデフォルトに簡易表示。分類の場合は条件合致件数のみ記録。
            n_long_raw = 0
            n_short_raw = 0
            if pmode != "classification":
                col = (
                    self.config.get("freqai", {}).get("prediction_key", "&-target")
                    if hasattr(self, "config")
                    else "&-target"
                )
                series = dataframe.get(col, dataframe.get("&-target", 0))
                try:
                    n_long_raw = int((series > bt).sum())
                    n_short_raw = int((series < -bt).sum())
                except Exception as e:
                    logger.debug("Count raw signals failed: %s", e)
                    n_long_raw = n_short_raw = 0
            n_long = int((dataframe.get("enter_long", 0) == 1).sum())
            n_short = int((dataframe.get("enter_short", 0) == 1).sum())
            logging.getLogger(__name__).info(
                (
                    "[FreqAICustomStrategy] rows:%s dp1:%s mode:%s | thr>(%g):%s thr<-(%g):%s "
                    "| enter_long:%s enter_short:%s | skip_dp:%s"
                ),
                n,
                n_dp1,
                pmode,
                bt,
                n_long_raw,
                bt,
                n_short_raw,
                n_long,
                n_short,
                use_skip,
            )
        except Exception as e:
            logger.debug("Final debug logging failed: %s", e)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        エグジット条件を定義
        """
        conditions_long_exit = []
        conditions_short_exit = []

        if "do_predict" in dataframe.columns:
            pmode = (
                str(self.config.get("freqai", {}).get("prediction_mode", "regression")).lower()
                if hasattr(self, "config")
                else "regression"
            )
            if pmode == "classification":
                clf = (
                    self.config.get("freqai", {}).get("classifier", {})
                    if hasattr(self, "config")
                    else {}
                )
                thr_l = float(clf.get("long_threshold", 0.55))
                thr_s = float(clf.get("short_threshold", 0.55))
                # If probabilities exist, use reversal as exit signal
                p_up = dataframe.get("1")  # long proba
                p_dn = dataframe.get("0")  # short proba
                if p_dn is not None:
                    conditions_long_exit.append((p_dn >= thr_s) & (dataframe["do_predict"] == 1))
                if p_up is not None:
                    conditions_short_exit.append((p_up >= thr_l) & (dataframe["do_predict"] == 1))
            else:
                if "&-target" in dataframe.columns:
                    conditions_long_exit.append(
                        (dataframe["&-target"] < self.sell_threshold.value)
                        & (dataframe["do_predict"] == 1)
                    )
                    conditions_short_exit.append(
                        (dataframe["&-target"] > -self.sell_threshold.value)
                        & (dataframe["do_predict"] == 1)
                    )

        if conditions_long_exit:
            dataframe.loc[reduce(lambda x, y: x & y, conditions_long_exit), "exit_long"] = 1
        if conditions_short_exit:
            dataframe.loc[reduce(lambda x, y: x & y, conditions_short_exit), "exit_short"] = 1
        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """レバレッジを動的に調整 (VolTarget に委譲、未設定なら控えめにフォールバック)"""
        try:
            from freqtrade_ext.risk.vol_sizer import VolatilityTargetSizer

            if self._ext_vol_sizer is None:
                self._ext_vol_sizer = VolatilityTargetSizer(self._ext_cfg().get("vol_target", {}))
        except Exception as e:
            self._ext_vol_sizer = None
            logger.debug("VolSizer init failed: %s", e)

        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        except Exception as e:
            df = None
            logger.debug("DP get_analyzed_dataframe failed: %s", e)

        if self._ext_vol_sizer is not None:
            return float(
                self._ext_vol_sizer.suggest_leverage(
                    proposed_leverage=proposed_leverage,
                    max_leverage=max_leverage,
                    current_rate=current_rate,
                    max_stake=self.wallets.get_available_stake_amount() if self.wallets else 0.0,
                    min_stake=None,
                    ohlcv=df,
                )
            )

        # Fallback: modest cap
        return float(min(max(proposed_leverage or 1.0, 1.0), max_leverage))

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """VolTarget (R単位) でステーク (証拠金) を調整。未設定時は提案値を返す。"""
        try:
            from freqtrade_ext.risk.vol_sizer import VolatilityTargetSizer

            if self._ext_vol_sizer is None:
                self._ext_vol_sizer = VolatilityTargetSizer(self._ext_cfg().get("vol_target", {}))
        except Exception:
            self._ext_vol_sizer = None

        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        except Exception:
            df = None

        if self._ext_vol_sizer is None:
            return proposed_stake

        # Edge score: how much the prediction exceeds threshold (>=0)
        edge_score = None
        try:
            pred_key = (
                self.config.get("freqai", {}).get("prediction_key", "&-target")
                if hasattr(self, "config")
                else "&-target"
            )
            thr = float(self.buy_threshold.value)
            if df is not None and pred_key in df.columns and len(df) > 0:
                # Use the latest row up to current_time
                series = df[pred_key]
                if isinstance(series.index, pd.DatetimeIndex):
                    series = series[series.index <= pd.Timestamp(current_time, tz="UTC")]
                pv = float(series.iloc[-1]) if len(series) else float("nan")
                if np.isfinite(pv):
                    edge_score = max(0.0, abs(pv) - thr)
        except Exception:
            edge_score = None

        return float(
            self._ext_vol_sizer.suggest_stake(
                current_rate=current_rate,
                proposed_stake=proposed_stake,
                min_stake=min_stake,
                max_stake=max_stake,
                leverage=leverage,
                side=side,
                ohlcv=df,
                edge_score=edge_score,
            )
        )

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        """タイムストップ/トレイル等による強制エグジット判定 (ExitPolicy)。"""
        try:
            from freqtrade_ext.risk.exit_policy import ExitPolicy

            if self._ext_exit_policy is None:
                self._ext_exit_policy = ExitPolicy(
                    self._ext_cfg().get("exit_policy", {}),
                    timeframe_minutes=self._get_timeframe_minutes(),
                )
        except Exception as e:
            self._ext_exit_policy = None
            logger.debug("ExitPolicy init failed: %s", e)

        if self._ext_exit_policy is None or not getattr(trade, "open_date_utc", None):
            return None

        reason = self._ext_exit_policy.evaluate_custom_exit(
            trade_id=getattr(trade, "id", 0) or 0,
            trade_open_time=trade.open_date_utc,
            current_time=current_time,
            current_profit=current_profit,
        )
        return reason

    def adjust_trade_position(
        self,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None:
        """部分利確ステップ (ExitPolicy): 戻り値は±ステーク金額。Noneで不動作。"""
        try:
            from freqtrade_ext.risk.exit_policy import ExitPolicy

            if self._ext_exit_policy is None:
                self._ext_exit_policy = ExitPolicy(
                    self._ext_cfg().get("exit_policy", {}),
                    timeframe_minutes=self._get_timeframe_minutes(),
                )
        except Exception as e:
            self._ext_exit_policy = None
            logger.debug("ExitPolicy init failed (adjust): %s", e)

        if self._ext_exit_policy is None or not getattr(trade, "open_date_utc", None):
            return None

        # trade.stake_amount は (先物では) 証拠金ベース。部分利確はその割合で減額要求する。
        stake_now = getattr(trade, "stake_amount", None)
        if stake_now is None:
            return None

        return self._ext_exit_policy.evaluate_adjustment(
            trade_id=getattr(trade, "id", 0) or 0,
            trade_open_time=trade.open_date_utc,
            current_time=current_time,
            current_profit=current_profit,
            current_stake_amount=float(stake_now),
        )


# ヘルパー関数
def reduce(function, iterable, initializer=None):
    """
    functools.reduce の代替実装
    """
    it = iter(iterable)
    if initializer is None:
        value = next(it)
    else:
        value = initializer
    for element in it:
        value = function(value, element)
    return value
