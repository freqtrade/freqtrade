# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.linalg import LinAlgError

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


class LinearAlgebraStrategy(IStrategy):
    """
    線形代数を活用した高度なトレーディングストラテジー

    主な機能:
    1. 主成分分析(PCA)による次元削減と特徴抽出
    2. 相関行列分析による市場構造の理解
    3. 線形回帰による価格予測
    4. 固有値・固有ベクトル分析
    5. 特異値分解(SVD)によるノイズ除去
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04,
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.05

    # Trailing stoploss
    trailing_stop = False

    # Optimal timeframe for the strategy
    timeframe = "5m"

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    pca_components = IntParameter(low=2, high=5, default=3, space="buy", optimize=True, load=True)
    correlation_threshold = DecimalParameter(
        low=0.5, high=0.9, default=0.7, space="buy", optimize=True, load=True
    )
    regression_window = IntParameter(
        low=10, high=50, default=20, space="buy", optimize=True, load=True
    )
    eigenvalue_threshold = DecimalParameter(
        low=0.1, high=0.5, default=0.3, space="sell", optimize=True, load=True
    )
    signal_strength = DecimalParameter(
        low=0.6, high=0.9, default=0.75, space="buy", optimize=True, load=True
    )

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        線形代数を用いたインジケーターの計算
        """

        # 基本的なテクニカル指標
        dataframe["rsi"] = ta.RSI(dataframe)
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=12)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=26)
        dataframe["macd"] = ta.MACD(dataframe)["macd"]
        dataframe["bb_upper"] = ta.BBANDS(dataframe)["upperband"]
        dataframe["bb_lower"] = ta.BBANDS(dataframe)["lowerband"]
        dataframe["bb_middle"] = ta.BBANDS(dataframe)["middleband"]
        dataframe["adx"] = ta.ADX(dataframe)
        dataframe["volume_sma"] = ta.SMA(dataframe["volume"], timeperiod=20)

        # 価格変化率
        dataframe["returns"] = dataframe["close"].pct_change()
        dataframe["log_returns"] = np.log(dataframe["close"] / dataframe["close"].shift(1))

        # 価格の正規化
        dataframe["price_normalized"] = self.normalize_price(dataframe["close"])

        # 線形代数ベースのインジケーター
        dataframe = self.add_pca_indicators(dataframe)
        dataframe = self.add_correlation_indicators(dataframe)
        dataframe = self.add_regression_indicators(dataframe)
        dataframe = self.add_eigenvalue_indicators(dataframe)
        dataframe = self.add_svd_indicators(dataframe)

        return dataframe

    def normalize_price(self, price_series: pd.Series, window: int = 20) -> pd.Series:
        """価格データの正規化"""
        rolling_mean = price_series.rolling(window=window).mean()
        rolling_std = price_series.rolling(window=window).std()
        return (price_series - rolling_mean) / rolling_std

    def add_pca_indicators(self, dataframe: DataFrame) -> DataFrame:
        """主成分分析(PCA)によるインジケーター"""
        window = 50

        # 特徴量マトリックスの作成
        features = ["close", "high", "low", "volume", "rsi", "macd"]

        # PCAによる次元削減
        pca_data: list[list[float]] = []
        explained_variance: list[float] = []

        for i in range(window, len(dataframe)):
            try:
                # データの準備
                data_window = dataframe[features].iloc[i - window : i].values

                # 欠損値のチェック
                if np.isnan(data_window).any() or np.isinf(data_window).any():
                    pca_data.append([np.nan] * self.pca_components.value)
                    explained_variance.append(np.nan)
                    continue

                # 標準化
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_window)

                # PCA
                pca = PCA(n_components=self.pca_components.value)
                pca_result = pca.fit_transform(data_scaled)

                # 最新の主成分スコア
                pca_data.append(pca_result[-1].astype(float).tolist())
                explained_variance.append(pca.explained_variance_ratio_[0])

            except (LinAlgError, ValueError):
                pca_data.append([np.nan] * self.pca_components.value)
                explained_variance.append(np.nan)

        # データフレームに追加
        pca_arr = np.array(pca_data, dtype=float)
        for i in range(self.pca_components.value):
            dataframe.loc[window:, f"pca_component_{i + 1}"] = pca_arr[:, i]

        dataframe.loc[window:, "pca_explained_variance"] = explained_variance

        return dataframe

    def add_correlation_indicators(self, dataframe: DataFrame) -> DataFrame:
        """相関行列分析によるインジケーター"""
        window = 30

        price_volume_corr = []
        ohlc_corr_matrix_det = []

        for i in range(window, len(dataframe)):
            try:
                # 価格とボリュームの相関
                price_data = dataframe["close"].iloc[i - window : i]
                volume_data = dataframe["volume"].iloc[i - window : i]

                if len(price_data.dropna()) > 10 and len(volume_data.dropna()) > 10:
                    corr_pv = price_data.corr(volume_data)
                    price_volume_corr.append(corr_pv)
                else:
                    price_volume_corr.append(np.nan)

                # OHLC相関行列の行列式
                ohlc_data = dataframe[["open", "high", "low", "close"]].iloc[i - window : i]
                if not ohlc_data.isnull().any().any():
                    corr_matrix = ohlc_data.corr()
                    det = np.linalg.det(corr_matrix)
                    ohlc_corr_matrix_det.append(det)
                else:
                    ohlc_corr_matrix_det.append(np.nan)

            except (LinAlgError, ValueError):
                price_volume_corr.append(np.nan)
                ohlc_corr_matrix_det.append(np.nan)

        dataframe.loc[window:, "price_volume_corr"] = price_volume_corr
        dataframe.loc[window:, "ohlc_corr_det"] = ohlc_corr_matrix_det

        return dataframe

    def add_regression_indicators(self, dataframe: DataFrame) -> DataFrame:
        """線形回帰による価格予測インジケーター"""
        window = self.regression_window.value

        regression_slope = []
        regression_r2 = []
        regression_prediction = []

        for i in range(window, len(dataframe)):
            try:
                # 独立変数 (時間インデックス)
                X = np.arange(window).reshape(-1, 1)
                # 従属変数 (価格)
                y = dataframe["close"].iloc[i - window : i].values

                if not np.isnan(y).any():
                    # 線形回帰
                    reg = LinearRegression()
                    reg.fit(X, y)

                    # 傾き (トレンドの強さ)
                    slope = reg.coef_[0]

                    # 決定係数 (トレンドの信頼性)
                    r2 = reg.score(X, y)

                    # 次の価格予測
                    next_pred = reg.predict([[window]])[0]

                    regression_slope.append(slope)
                    regression_r2.append(r2)
                    regression_prediction.append(next_pred)
                else:
                    regression_slope.append(np.nan)
                    regression_r2.append(np.nan)
                    regression_prediction.append(np.nan)

            except (LinAlgError, ValueError):
                regression_slope.append(np.nan)
                regression_r2.append(np.nan)
                regression_prediction.append(np.nan)

        dataframe.loc[window:, "regression_slope"] = regression_slope
        dataframe.loc[window:, "regression_r2"] = regression_r2
        dataframe.loc[window:, "regression_prediction"] = regression_prediction

        # 予測価格と実際価格の差
        dataframe["prediction_error"] = dataframe["close"] - dataframe["regression_prediction"]

        return dataframe

    def add_eigenvalue_indicators(self, dataframe: DataFrame) -> DataFrame:
        """固有値・固有ベクトル分析によるインジケーター"""
        window = 25

        max_eigenvalue: list[float] = []
        eigenvalue_ratio: list[float] = []
        market_regime: list[float] = []

        for i in range(window, len(dataframe)):
            try:
                # 価格データの共分散行列
                price_data = dataframe[["open", "high", "low", "close"]].iloc[i - window : i]

                if not price_data.isnull().any().any():
                    # 共分散行列
                    cov_matrix = price_data.cov()

                    # 固有値の計算
                    eigenvalues = np.linalg.eigvals(cov_matrix)
                    eigenvalues = np.real(eigenvalues[eigenvalues.imag == 0])  # 実数部のみ
                    eigenvalues = np.sort(eigenvalues)[::-1]  # 降順ソート

                    if len(eigenvalues) > 0:
                        max_eigen = eigenvalues[0]
                        eigen_ratio = (
                            eigenvalues[0] / np.sum(eigenvalues) if len(eigenvalues) > 1 else 1.0
                        )

                        # マーケットレジーム判定 (主固有値の比率による)
                        regime = float(1 if eigen_ratio > self.eigenvalue_threshold.value else 0)

                        max_eigenvalue.append(max_eigen)
                        eigenvalue_ratio.append(eigen_ratio)
                        market_regime.append(regime)
                    else:
                        max_eigenvalue.append(np.nan)
                        eigenvalue_ratio.append(np.nan)
                        market_regime.append(np.nan)
                else:
                    max_eigenvalue.append(np.nan)
                    eigenvalue_ratio.append(np.nan)
                    market_regime.append(np.nan)

            except (LinAlgError, ValueError):
                max_eigenvalue.append(np.nan)
                eigenvalue_ratio.append(np.nan)
                market_regime.append(np.nan)

        dataframe.loc[window:, "max_eigenvalue"] = max_eigenvalue
        dataframe.loc[window:, "eigenvalue_ratio"] = eigenvalue_ratio
        dataframe.loc[window:, "market_regime"] = market_regime

        return dataframe

    def add_svd_indicators(self, dataframe: DataFrame) -> DataFrame:
        """特異値分解(SVD)によるノイズ除去インジケーター"""
        window = 20

        svd_trend = []
        svd_noise_ratio = []

        for i in range(window, len(dataframe)):
            try:
                # 価格データマトリックス
                price_matrix = (
                    dataframe[["open", "high", "low", "close"]].iloc[i - window : i].values.T
                )

                if not np.isnan(price_matrix).any():
                    # SVD分解
                    U, s, Vt = np.linalg.svd(price_matrix, full_matrices=False)

                    # 主要なトレンド成分 (最大特異値の成分)
                    trend_component = s[0] * np.outer(U[:, 0], Vt[0, :])
                    trend_signal = trend_component[3, -1]  # close価格のトレンド

                    # ノイズ比率 (小さな特異値の比率)
                    noise_ratio = np.sum(s[2:]) / np.sum(s) if len(s) > 2 else 0

                    svd_trend.append(trend_signal)
                    svd_noise_ratio.append(noise_ratio)
                else:
                    svd_trend.append(np.nan)
                    svd_noise_ratio.append(np.nan)

            except (LinAlgError, ValueError):
                svd_trend.append(np.nan)
                svd_noise_ratio.append(np.nan)

        dataframe.loc[window:, "svd_trend"] = svd_trend
        dataframe.loc[window:, "svd_noise_ratio"] = svd_noise_ratio

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        線形代数指標に基づくエントリーシグナル
        """

        # ロングエントリー条件
        long_conditions = [
            # PCA条件: 第1主成分が上昇トレンド
            (dataframe["pca_component_1"] > dataframe["pca_component_1"].shift(1)),
            # 回帰条件: 正の傾きと高い決定係数
            (dataframe["regression_slope"] > 0),
            (dataframe["regression_r2"] > self.signal_strength.value),
            # 相関条件: 価格とボリュームの正相関
            (dataframe["price_volume_corr"] > self.correlation_threshold.value),
            # 固有値条件: 明確なマーケットレジーム
            (dataframe["market_regime"] == 1),
            # SVD条件: 低ノイズ環境
            (dataframe["svd_noise_ratio"] < 0.3),
            # 予測条件: 予測価格が現在価格より高い
            (dataframe["regression_prediction"] > dataframe["close"]),
            # ボリューム条件
            (dataframe["volume"] > dataframe["volume_sma"]),
        ]

        # ショートエントリー条件
        short_conditions = [
            # PCA条件: 第1主成分が下降トレンド
            (dataframe["pca_component_1"] < dataframe["pca_component_1"].shift(1)),
            # 回帰条件: 負の傾きと高い決定係数
            (dataframe["regression_slope"] < 0),
            (dataframe["regression_r2"] > self.signal_strength.value),
            # 相関条件: 価格とボリュームの負相関
            (dataframe["price_volume_corr"] < -self.correlation_threshold.value),
            # 固有値条件: 明確なマーケットレジーム
            (dataframe["market_regime"] == 1),
            # SVD条件: 低ノイズ環境
            (dataframe["svd_noise_ratio"] < 0.3),
            # 予測条件: 予測価格が現在価格より低い
            (dataframe["regression_prediction"] < dataframe["close"]),
            # ボリューム条件
            (dataframe["volume"] > dataframe["volume_sma"]),
        ]

        # ロングエントリー
        dataframe.loc[reduce(lambda x, y: x & y, long_conditions), "enter_long"] = 1

        # ショートエントリー
        dataframe.loc[reduce(lambda x, y: x & y, short_conditions), "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        線形代数指標に基づくエグジットシグナル
        """

        # ロングエグジット条件
        long_exit_conditions = [
            # PCA条件: 第1主成分の反転
            (dataframe["pca_component_1"] < dataframe["pca_component_1"].shift(1)),
            # 回帰条件: 傾きの反転または決定係数の低下
            (dataframe["regression_slope"] < 0) | (dataframe["regression_r2"] < 0.5),
            # 相関条件: 相関の弱化
            (abs(dataframe["price_volume_corr"]) < 0.3),
            # 予測エラーの拡大
            (abs(dataframe["prediction_error"]) > dataframe["close"] * 0.02),
        ]

        # ショートエグジット条件
        short_exit_conditions = [
            # PCA条件: 第1主成分の反転
            (dataframe["pca_component_1"] > dataframe["pca_component_1"].shift(1)),
            # 回帰条件: 傾きの反転または決定係数の低下
            (dataframe["regression_slope"] > 0) | (dataframe["regression_r2"] < 0.5),
            # 相関条件: 相関の弱化
            (abs(dataframe["price_volume_corr"]) < 0.3),
            # 予測エラーの拡大
            (abs(dataframe["prediction_error"]) > dataframe["close"] * 0.02),
        ]

        # ロングエグジット
        dataframe.loc[reduce(lambda x, y: x | y, long_exit_conditions), "exit_long"] = 1

        # ショートエグジット
        dataframe.loc[reduce(lambda x, y: x | y, short_exit_conditions), "exit_short"] = 1

        return dataframe


def reduce(function, iterable, initializer=None):
    """reduce関数の実装"""
    it = iter(iterable)
    if initializer is None:
        value = next(it)
    else:
        value = initializer
    for element in it:
        value = function(value, element)
    return value
