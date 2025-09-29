# Freqtrade フォーク改造仕様（板情報→特徴量→FreqAI 学習まで一気通貫）— **Full v1**

**目的**  
- 板情報（Orderbook, L2/L1）を WebSocket で収集・**要約（秒足）**し、**Parquet（列指向）**で**Feature Store**化  
- バックテスト / Hyperopt / FreqAI 学習 / ライブで**同一パイプライン**から読み出し、**情報リーク無し**で特徴量として利用  
- 将来の本家マージ影響を最小化するため、**外部拡張（最小侵襲）**で実装

**非目標**  
- 本家 `freqtrade` コアへの大規模改造  
- 生の L2 全ティックのフル保存（容量・I/O 爆増を回避）

---

## Definition of Done（成果物）

- [ ] `tools/ob_collector.py`：  
  - ccxtpro で板を収集 → **1秒粒度で代表値（last）要約**  
  - **1分バッチ**で Parquet 書込（**ZSTD圧縮**、**年/月/日パーティション**、RowGroup≈60–600）  
  - **再接続（指数バックオフ）**・ハートビート監視・NTP 同期前提  
  - **スキーマ固定**（`schema_version` 付与）

- [ ] `freqtrade_ext/feature_store.py`：  
  - Parquet（秒足）を **右開窓 [t-60s, t)** + **embargo（末尾 n 秒除外）**で 1分へダウンサンプル  
  - 最終的に **`shift(1)`**（リーク防止）  
  - **相対化・派生特徴量**（`spread_bps`, `microprice`, `imbalance (L1/L5/L10/L50)`, `book_slope`, 簡易 `OFI_top` など）  
  - `timerange` に応じた**必要最小限読み込み**（パーティションプルーニング）

- [ ] 例示戦略 `user_data/strategies/MyOBStrategy.py`：  
  - OHLCV に板特徴量を**左結合**  
  - FreqAI に渡す列を **`feat__*` 接頭辞**で明示（バージョン差を吸収）  
  - **レジーム/ノートレード**（スプレッド分位・RV・約定率など）スイッチ  
  - 予測確率の**較正（Platt/Isotonic）** → **ポジションサイズ連動**（Vol 目標 or Fractional-Kelly）

- [ ] FreqAI 用サンプル設定／フック：
  - **PurgedKFold + Embargo** の CV  
  - **分類 + EV しきい値最適化**（手数料/スリッページ込み）  
  - ラベルは **トリプルバリア / EV分類**（実装名の差は戦略側で吸収）

- [ ] （任意）`freqtrade_ext/execution_model.py`：  
  - **手数料・資金調達（funding）・スリッページ**内生化  
  - 成行/指値 **fill ロジック**（BestBid/BestAsk 到達）

- [ ] **DQ（データ品質）レポート**自動生成（欠損・遅延・外れ値）`*_dq.json` を同階層へ出力

- [ ] ドキュメント（本書）に沿って `backtesting` / `freqai-train` / ライブが**コマンド一発**で動く

---

## ディレクトリ構成（フォーク）

```
freqtrade-fork/
 ├─ freqtrade/                        # 本家（最小改造方針）
 ├─ freqtrade_ext/                    # ← 追加：拡張コードを隔離
 │   ├─ __init__.py
 │   ├─ feature_store.py              # Parquet読込/整形/リーク対策/派生特徴
 │   └─ execution_model.py            # (任意) 約定モデル（コスト/impact内生化）
 ├─ tools/
 │   └─ ob_collector.py               # 板収集→秒足要約→1分バッチでParquet保存
 ├─ user_data/
 │   └─ strategies/
 │       └─ MyOBStrategy.py           # 特徴量結合＆FreqAI連携（分類+サイズ連動）
 ├─ user_data/featurestore/           # Parquet出力ルート（年/月/日パーティション）
 │   └─ bybit/BTCUSDT/1s/...
 ├─ requirements-ext.txt              # 追加依存（ccxtpro, pyarrow, etc.）
 └─ docs/ORDERBOOK_FREQAI.md          # ← 本ドキュメント
```

---

## 依存関係

`requirements-ext.txt`
```
ccxtpro>=1.0.0
pyarrow>=15.0.0
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0   # Platt/Isotonic など
```

> 追加パッケージは本家とは別管理（仮想環境/extra-requirements）し、衝突回避。

---

## 実装

### 1) 板収集コレクタ（WebSocket → 秒足要約 → **1分バッチParquet**）

`tools/ob_collector.py`
```python
# Python 3.10+
import asyncio, time, math, random
from collections import deque
from datetime import datetime, timezone
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import ccxtpro

EXCHANGE    = "bybit"                         # 変更可
PAIR        = "BTC/USDT:USDT"                 # 先物例
DEPTH       = 200                             # 取引所仕様に合わせる
ROOT_DIR    = "/user_data/featurestore/bybit/BTCUSDT/1s"
SCHEMA_VER  = 1
FLUSH_EVERY = 60                              # 1分でバッチ書込
HEARTBEAT_TOUT_SEC = 15

def utc_now_floor_s(ts=None):
    return pd.Timestamp.utcnow().floor("S").to_pydatetime().replace(tzinfo=timezone.utc)

def _minute_key(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M")

async def run():
    ex = getattr(ccxtpro, EXCHANGE)({"enableRateLimit": True})
    buf = deque()
    last_sec = None
    last_tick_ts = time.time()
    current_minute = None

    async def _write_minute_batch(df_minute: pd.DataFrame):
        if df_minute.empty:
            return
        df_minute = df_minute.copy()
        # preserve index 'ts'
        df_minute["year"]  = df_minute.index.year
        df_minute["month"] = df_minute.index.month
        df_minute["day"]   = df_minute.index.day
        table = pa.Table.from_pandas(df_minute, preserve_index=True)
        ds.write_dataset(
            data=table,
            base_dir=ROOT_DIR,
            format="parquet",
            partitioning=["year","month","day"],
            file_options=pq.ParquetFileWriteOptions(compression="zstd"),
            min_rows_per_group=60,
            max_rows_per_group=600,
            existing_data_behavior="create"
        )

    while True:
        try:
            ob = await asyncio.wait_for(ex.watch_order_book(PAIR, limit=DEPTH), timeout=HEARTBEAT_TOUT_SEC)
            last_tick_ts = time.time()
            ts = utc_now_floor_s()
            bids, asks = ob["bids"][:DEPTH], ob["asks"][:DEPTH]
            if not bids or not asks:
                continue

            best_bid = float(bids[0][0]); best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            mid = (best_ask + best_bid) / 2.0
            bid_vol = float(sum(q for _, q in bids)); ask_vol = float(sum(q for _, q in asks))
            imb = bid_vol / max(bid_vol + ask_vol, 1e-9)
            depth_delta = bid_vol - ask_vol

            buf.append({
                "ts": ts,                      # 秒に丸めたUTC
                "schema_version": SCHEMA_VER,
                "exchange": EXCHANGE,
                "pair": PAIR,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "mid": mid,
                f"imb_{DEPTH}": imb,
                f"depth_delta_{DEPTH}": depth_delta,
                # 追加のtop-of-book量（OFI用）
                "top_bid_qty": float(bids[0][1]),
                "top_ask_qty": float(asks[0][1]),
            })

            # 秒ごとに代表値化（同一秒の最後のみ使用）
            if last_sec is None:
                last_sec = ts
                current_minute = _minute_key(ts)

            if ts > last_sec:
                # 直近秒のレコードで代表値（last）
                sec_df = pd.DataFrame([x for x in buf if x["ts"] == last_sec])
                if not sec_df.empty:
                    out = sec_df.groupby("ts").last().sort_index()

                    # バッファの分単位蓄積（メモリ上に minute_df を保持）
                    minute_key = _minute_key(last_sec)
                    # minute rollover -> 書き出し
                    if minute_key != current_minute and 'minute_df' in locals():
                        await _write_minute_batch(minute_df)
                        del minute_df
                        current_minute = minute_key

                    if 'minute_df' not in locals():
                        minute_df = out
                    else:
                        minute_df = pd.concat([minute_df, out], axis=0)

                # バッファ掃除
                buf = deque([x for x in buf if x["ts"] > last_sec])
                last_sec = ts

        except asyncio.TimeoutError:
            # ハートビート切れ → 再接続
            await asyncio.sleep(1.0)
            try:
                await ex.close()
            except Exception:
                pass
            # バックオフ
            await asyncio.sleep(3.0 + random.random() * 2.0)
            ex = getattr(ccxtpro, EXCHANGE)({"enableRateLimit": True})
        except Exception as e:
            # ログだけ出して継続
            print(f"[collector] error: {e}")
            await asyncio.sleep(1.0)

if __name__ == "__main__":
    asyncio.run(run())
```

**設計ポイント**  
- **小ファイル回避**：**1分バッチ**で `pyarrow.dataset.write_dataset`、**ZSTD 圧縮**、**年/月/日パーティション**  
- **スキーマ固定**＆`schema_version` 付与で後方互換の分岐が可能  
- **再接続・ハートビート**：WS 切断耐性  
- 時刻はすべて **UTC**（NTP 必須）  

---

### 2) Feature Store ローダ（右開窓 + embargo → 1分 / 派生特徴 / shift(1)）

`freqtrade_ext/feature_store.py`
```python
from __future__ import annotations
import math, pathlib
from typing import Optional, Tuple, List
import pandas as pd
import pyarrow.dataset as ds

def _safe_shift(df: pd.DataFrame, cols: list[str], n: int = 1) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].shift(n)
    return df

def _scanner_for_timerange(root: str, start: pd.Timestamp, end: pd.Timestamp):
    # 年/月/日パーティションに合わせた簡易フィルタ
    part_filter = (
        (ds.field("year") >= start.year) & (ds.field("year") <= end.year)
    )
    return ds.dataset(root, format="parquet"), part_filter

def _minute_last_right_open(df_sec: pd.DataFrame, embargo_secs: int = 1) -> pd.DataFrame:
    """
    秒足から、各分の右開窓 [t-60, t) の最後の観測を採用。
    末尾 embargo_secs は除外してから last を取る。
    """
    s = df_sec.copy()
    # embargo: 分の最後 n 秒を捨てる（例: 1秒）
    s = s[s.index.second <= (59 - embargo_secs)]
    out = s.groupby(s.index.floor("T")).last()
    out.index.name = "date"
    return out

def _derive_features(df: pd.DataFrame, depth: int) -> pd.DataFrame:
    df = df.copy()
    # 相対スプレッド・bps
    df["rel_spread"]  = df["spread"] / df["mid"].clip(lower=1e-9)
    df["spread_bps"]  = 1e4 * df["rel_spread"]
    # 既存列の正規化名
    if f"imb_{depth}" in df.columns:
        df["ob_imbalance"] = df[f"imb_{depth}"]
    if f"depth_delta_{depth}" in df.columns:
        df["ob_depth_delta"] = df[f"depth_delta_{depth}"]
    # Microprice（簡易）
    df["microprice"] = (
        (df["best_ask"] * df.get("top_bid_qty", 0).fillna(0))
        + (df["best_bid"] * df.get("top_ask_qty", 0).fillna(0))
    ) / (df.get("top_bid_qty", 0).fillna(0) + df.get("top_ask_qty", 0).fillna(0)).replace(0, pd.NA)

    # OFI (top-of-book, 簡易)
    for col in ["best_bid", "best_ask", "top_bid_qty", "top_ask_qty"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["d_best_bid"]  = df["best_bid"].diff()
    df["d_best_ask"]  = df["best_ask"].diff()
    df["d_top_bid_q"] = df["top_bid_qty"].diff()
    df["d_top_ask_q"] = df["top_ask_qty"].diff()
    # An Empirical OFI-esque proxy
    df["ofi_top"] = (df["d_top_bid_q"].clip(lower=0) - df["d_top_ask_q"].clip(lower=0)).fillna(0)

    # book slope（価格 vs 累積量の傾きの代理：ここでは depth_delta の変化で近似）
    if "ob_depth_delta" in df.columns:
        df["book_slope"] = df["ob_depth_delta"].diff()

    # 欠損/inf整理
    df = df.replace([float("inf"), float("-inf")], pd.NA)

    return df

def load_orderbook_features(
    exchange: str,
    pair: str,
    timeframe: str,
    timerange: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    embargo_secs: int = 1,
    depth: int = 200
) -> pd.DataFrame:
    """
    Parquet(秒足) → 右開 [t-60,t) + embargo → timeframe(例 '1m')へ整形 → 最終的に shift(1)
    返り値 index: DatetimeIndex(name='date')
    """
    base = pair.replace("/", "").replace(":", "")
    root = f"/user_data/featurestore/{exchange}/{base}/1s"

    if timerange is not None:
        start, end = timerange
    else:
        # 広めに読む（必要に応じて呼び出し側で渡す）
        end = pd.Timestamp.utcnow().tz_localize("UTC")
        start = end - pd.Timedelta(days=7)

    dataset, part_filter = _scanner_for_timerange(root, start, end)
    try:
        tbl = dataset.to_table(filter=part_filter)
    except Exception:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

    if tbl.num_rows == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

    df = tbl.to_pandas(types_mapper=None)
    if "ts" not in df.columns:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

    df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    df = df.set_index(pd.to_datetime(df["ts"], utc=True)).drop(columns=["ts"])

    # 1分へ（右開窓 + embargo）
    agg = _minute_last_right_open(df, embargo_secs=embargo_secs)

    # 派生特徴量
    agg = _derive_features(agg, depth=depth)

    # ---- リーク防止 ----
    agg = _safe_shift(agg, list(agg.columns), n=1)

    return agg
```

**ポイント**  
- **右開窓 [t-60,t)** + **embargo（末尾 n 秒）** → その後 **`shift(1)`** でリーク防止を二重に担保  
- **相対化/派生特徴量**（`spread_bps`, `microprice`, `ofi_top`, `book_slope` 等）を **Loader 側で一元生成**  
- `timerange` で**必要日だけ**読み出す（パーティションプルーニングによるI/O削減）

---

### 3) 戦略（OHLCV 左結合 → FreqAI へ明示的に列を渡す／レジーム/サイズ連動）

`user_data/strategies/MyOBStrategy.py`
```python
from freqtrade.strategy.interface import IStrategy
import pandas as pd
from typing import Dict, Any, Tuple
from freqtrade_ext.feature_store import load_orderbook_features

class MyOBStrategy(IStrategy):
    timeframe = "1m"
    can_short = True
    use_exit_signal = True

    FEAT_PREFIX = "feat__"
    FEAT_COLS = [
        "spread_bps", "microprice", "ob_imbalance", "ob_depth_delta",
        "ofi_top", "book_slope"
    ]

    def _timerange(self, df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
        return (pd.to_datetime(df.index.min()).tz_convert("UTC"),
                pd.to_datetime(df.index.max()).tz_convert("UTC"))

    def populate_indicators(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        pair = metadata["pair"]
        exid = self.exchange.id

        # 外部特徴量を結合（BT/HO/学習/ライブで同一コード）
        if self.dp and self.dp.runmode.value in ("backtest","hyperopt","freqai","dry_run","live"):
            feats = load_orderbook_features(exid, pair, self.timeframe,
                                            timerange=self._timerange(df),
                                            embargo_secs=1, depth=200)
            if not feats.empty:
                df = df.join(feats, how="left")

        # 簡易欠損処理（学習時のスケーリング/前処理は FreqAI パイプライン側でfit/transform）
        df = df.replace([float("inf"), float("-inf")], pd.NA).fillna(method="ffill").fillna(0)

        # FreqAIへ渡す列に接頭辞
        for c in self.FEAT_COLS:
            if c in df.columns:
                df[f"{self.FEAT_PREFIX}{c}"] = df[c]

        # --- レジームフィルタ（例） ---
        # スプレッドが広すぎ・ボラ過大などの環境は取引停止
        q_spread = df["spread_bps"].rolling(1440, min_periods=60).quantile(0.95)
        df["no_trade"] = (df["spread_bps"] > q_spread).astype(int).fillna(0)

        return df

    def populate_buy_trend(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        df.loc[:, "buy"] = 0
        cond = (df["close"] > df["close"].rolling(20).mean()) & (df["no_trade"] == 0)
        df.loc[cond, "buy"] = 1
        return df

    def populate_sell_trend(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        df.loc[:, "sell"] = 0
        cond = (df["close"] < df["close"].rolling(20).mean()) & (df["no_trade"] == 0)
        df.loc[cond, "sell"] = 1
        return df

    # ---- FreqAI 連携（バージョン差吸収） ----
    @property
    def freqai_info(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "feature_parameters": {
                "feature_columns": [f"{self.FEAT_PREFIX}{c}" for c in self.FEAT_COLS]
            },
            # ラベルは EV/トリプルバリア相当（実装キー名は環境で調整）
            "label_parameters": {
                "label_type": "classification",
                "label_function": "triple_barrier_ev",  # 拡張側で提供する想定
                "future_period": 5,          # 例：t+5
                "barrier_up_bps": 15,        # +0.15%
                "barrier_dn_bps": 10,        # -0.10%
                "holding_limit": 30,         # 最大30本で時間切れ
                "cost_bps": 2.5              # 手数料・スリッページ概算を含める
            },
            "cv_parameters": {
                "cv_type": "purged_kfold",
                "n_splits": 5,
                "embargo_bars": 1            # 分足→1–5を推奨
            },
            "calibration": {
                "enabled": True,
                "method": "isotonic"         # or "platt"
            },
            "threshold_optimization": {
                "objective": "expected_value", # 手数料/スリッページ込みEV最大
                "metric_cost_bps": 2.5
            },
            "sizing": {
                "mode": "vol_target",        # or "fractional_kelly"
                "target_vol_bps": 50,        # 1トレード当たりの目標ボラ
                "max_leverage": 3.0,
                "daily_loss_limit_bps": 200
            }
        }
```

---

### 4) （任意）実行モデル（スプレッド/深さ・手数料/資金調達込み）

`freqtrade_ext/execution_model.py`
```python
def fill_price(
    mid: float, spread: float, side: str, qty: float,
    top_depth: float, rel_fee_bp: float, funding_bp: float = 0.0
) -> tuple[float, float]:
    """
    単純だが実運用寄りの近似：
      base  = mid ± 0.5*spread
      impact= (qty / top_depth)^γ * (spread/mid) * k * mid
      fee   = px * rel_fee_bp * 1e-4
      funding = px * funding_bp * 1e-4
    戻り値: (exec_price, total_cost)
    """
    rel_spread = spread / max(mid, 1e-9)
    k, gamma = 1.0, 0.7
    impact = (qty / max(top_depth, 1e-9))**gamma * rel_spread * k * mid
    base = mid + (0.5 if side.lower()=="buy" else -0.5) * spread
    px = base + (impact if side.lower()=="buy" else -impact)
    fee = px * rel_fee_bp * 1e-4
    funding = px * funding_bp * 1e-4
    return px, (fee + funding)

def limit_fill(best_bid: float, best_ask: float, price: float, side: str) -> bool:
    """
    指値到達による約定可否の簡易判定。
    buy: price >= best_ask, sell: price <= best_bid
    """
    if side.lower() == "buy":
        return price >= best_ask
    else:
        return price <= best_bid
```

> Freqtrade の既存手数料設定と二重計上しないよう**単一責務トグル**を用意（片側のみ有効）。

---

### 5) DQ（データ品質）レポート（自動生成）

- 欠損率、時間逆転、スプレッド/深さの外れ値（分位で閾値）を分単位で計測  
- 該当日の Parquet 階層に `YYYY-MM-DD_dq.json` を保存  
- 学習/BT 前に**DQを読み、閾値超過は no-trade または期間スキップ**

（実装は省略可：`tools/dq_check.py` のユーティリティとして分離推奨）

---

## コマンドと動作確認

1. **板収集の起動（別ターミナル）**
```bash
python tools/ob_collector.py
# 数分～数時間流し、/user_data/featurestore/.../1s/year=YYYY/... に parquet が生成されること
```

2. **バックテスト（板特徴量を読み込む）**
```bash
freqtrade backtesting --strategy MyOBStrategy \
  --timerange=20250101-20250107 \
  --timeframe 1m
```

3. **FreqAI 学習（列の受け渡し/ラベル/校正/CV）**
```bash
freqtrade freqai-train --strategy MyOBStrategy \
  --timerange=20250101-20250121 \
  --timeframe 1m
# ログに feature_columns=feat__* / CV=PurgedKFold / calibration が出力されること
```

> **同一パイプライン保証**：Live/BT/Train 全てで `feature_store.load_orderbook_features()` を通す。

---

## テスト（必須）

- `pytest`：  
  - `feature_store.py`：  
    - `embargo_secs` > 0 で**同分最後の n 秒が集計から除外**される  
    - その後の **`shift(1)` が必ず効く**（`close` の同時刻値を参照しない）  
    - 相対化列（`spread_bps` 等）・派生列の**非NaN率**  
  - `MyOBStrategy`：`feat__*` 列の存在、no-trade ガードが True で発注ゼロ  
- 速度：  
  - 1日の BT/Train で Parquet 読み出しが**パーティションプルーニング**され、I/O が抑制される

---

## パフォーマンス要件

- Parquet 書込：**1分バッチ**、ZSTD、RowGroup≈60–600  
- 読込：`timerange` に合わせて**日付パーティションのみ**スキャン  
- 前処理：`ffill → 0` はあくまで推論パイプラインでの欠損緩和。**学習パイプラインでは fit/transform** を分離。

---

## よくある落とし穴と回避策

- **現在バー値リーク**：右開窓 + embargo + **`shift(1)`** の三段構え  
- **小ファイル地獄**：**1分バッチ**、パーティション & 圧縮、RowGroup調整  
- **スキーマ揺れ**：`schema_version` で分岐、正規化名（`ob_imbalance` 等）を Loader で統一  
- **過学習**：**PurgedKFold + Embargo**、確率**較正**、**レジームOFF**  
- **コスト未反映**：**実行モデル**で手数料/資金調達/impact を内生化し**評価指標も同一基準**

---

## 次の拡張（任意）

- Feature Store **バージョニング**（`_v1`, `_v2`）  
- 不均衡の複数スケール & zscore  
- **PurgedKFold** の汎用ユーティリティ化  
- 実行モデルの**マーケットインパクト**関数を銘柄依存に最適化（学習）

---

## ライセンスと運用

- 本家は **GPLv3**。フォーク配布時はライセンス継承に留意。  
- 社内利用オンリーなら通常は公開義務なし。外部提供（SaaS 等）は法務確認。  
- 取引所API利用規約（WSポリシー・レート制限・データ再配布可否）も合わせて確認。

---

## 作業順序（エージェント用）

1. `freqtrade_ext/feature_store.py` を実装（右開+embargo+shift、派生特徴、timerange読込）→ Pytest  
2. `tools/ob_collector.py` を実装（1分バッチ/ZSTD/パーティション、再接続）→ Parquet 生成確認  
3. `MyOBStrategy.py` を配置 → `backtesting` 実行 → `feat__*` 列・no-trade が効くことを確認  
4. `freqai-train` → PurgedKFold/校正/EV最適化のログ確認  
5. （任意）`execution_model.py` を組込み → コスト内生化の on/off 比較  
6. DQ チェックを学習前フックに追加 → 異常日のスキップ/警告を確認

---

### 付録：設定のヒント

- **交換所・ペア**は `.env` に外出しし、Collector/Strategy の両方で参照  
- 手数料/資金調達は取引所設定と重複しないよう**片側**で適用  
- 重要メトリクス（シャープ、Calmar、回転率、約定率、滑り、DD）は**コスト込み**で統一

---
