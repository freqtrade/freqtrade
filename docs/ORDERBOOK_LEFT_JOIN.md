目的: 別LLMへ渡す用の「板情報(オーダーブック特徴量)を左結合して使う」仕様の要約

- 参照実装(左結合):
  - user_data/strategies/FreqAICustomStrategy.py:209
  - user_data/strategies/MyOBStrategy.py:49
- 特徴量の読み出し元:
  - freqtrade_ext/feature_store.py:15

概要
- 特徴量生成: `load_orderbook_features(exchange, pair, timeframe, timerange, embargo_secs, depth)` が1秒パーケットから集計し、日時インデックス(name='date')のDataFrameを返す。
  - 1分集計は右開区間[t-60s, t)で実施。
  - エンバーゴ(末尾n秒のドロップ)適用。
  - 最後に `shift(1)` で将来情報リークを防止。
  - 代表的な列: `spread_bps`, `microprice`, `ob_imbalance`, `ob_depth_delta`, `ofi_top`, `book_slope` など。
- 左結合: ローカルのOHLCV DataFrame側(左)に対し、同じ時刻キーで `how='left'` で結合。左側のローソク足行は保持される。
  - FreqAICustomStrategyでは、時刻列を一旦 `__tmp_idx__` としてindex化し、`join`で特徴量を左結合 -> `reset_index()` で`date`に戻す。
  - MyOBStrategyでは、DataFrame同士をindexで直接 `df.join(feats, how='left')` する。
- NA処理: 結合後に `inf/-inf` をNAに置換し、`ffill` と `fillna(0)` で最小限の穴埋めを実施しランタイム安定性を確保。

結合イメージ(擬似コード)

from pandas import DataFrame
import pandas as pd

# 左側: ローカルのOHLCV (date列 or DatetimeIndex)
ohlcv: DataFrame  # columns: [date, open, high, low, close, volume, ...]

# 右側: OB特徴量(DatetimeIndex name='date')
feats: DataFrame  # columns: [spread_bps, microprice, ob_imbalance, ...], index=DatetimeIndex('date')

# 左結合 (index結合)。左側全行を保持。
out = ohlcv.copy()
if 'date' in out.columns:
    out = out.set_index(pd.to_datetime(out['date'], utc=True))
out = out.join(feats, how='left')
out = out.reset_index().rename(columns={'index': 'date'})

# 簡易NA処理
out = out.replace([float('inf'), float('-inf')], pd.NA).fillna(method='ffill').fillna(0)

# 任意: OB列へ接頭辞
for c in ['spread_bps', 'microprice', 'ob_imbalance', 'ob_depth_delta', 'ofi_top', 'book_slope']:
    if c in out.columns:
        out[f'feat__{c}'] = out[c]

コード参照(クリック可能)
- user_data/strategies/FreqAICustomStrategy.py:209
- user_data/strategies/MyOBStrategy.py:49
- freqtrade_ext/feature_store.py:15

注意
- `load_orderbook_features` が返すDataFrameは、将来情報リーク防止のために `shift(1)` 済み。
- 結合キーはUTCのDatetimeIndex('date')。左結合のため、ローソク足側に存在しない時刻に特徴量だけがあっても行は増えない。
