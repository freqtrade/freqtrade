# 拡張ガイド: リスク管理・サイジング・FreqAI 連携

このドキュメントは、Freqtrade 本体へ手を入れずに追加した拡張（risk/sizing/exit/FreqAI 連携）と、戦略側の配線をまとめたものです。次の担当者がすぐに状況把握し、継続作業できることを目的にしています。

含まれる内容: 追加/変更点、設定方法、回帰/分類エントリーの構造、NaN 回避のデータ要件、BT/再学習のワークフロー。

## 概要

Freqtrade コア無改造で実現したもの:

- ボラティリティ目標サイジング（ATR）＋シグナル強度（エッジ）による倍率調整。
- タイムストップ＋部分利確＋ドローダウントレイルを ExitPolicy として再利用可に。
- 「方向 × 期待値」ゲート（現状は回帰で運用、分類 A/B も準備済み）。
- しきい値自動較正（回帰は |pred|、分類は確率）。
- レジームフィルタ（スプレッド高分位、EMA トレンド、BTC 乱高下）と安全なフォールバック。

すべて設定で有効化。コア改変は不要です。

## 追加/変更ファイル

追加（非侵襲モジュール）:

- `freqtrade_ext/risk/vol_sizer.py`: ATR サイジング＋エッジ倍率。
- `freqtrade_ext/risk/exit_policy.py`: ExitPolicy（タイムストップ、部分利確、トレイル/ハードDD）。
- `freqtrade_ext/calibration/thresholds.py`: しきい値較正ユーティリティ
  - `best_abs_threshold()`（回帰 |pred|）
  - `best_proba_thresholds()`（分類 up/down 確率）

変更（戦略配線/設定）:

- `user_data/strategies/FreqAICustomStrategy.py`
  - 目的変数/TB ラベル: `set_freqai_targets()`
  - エントリー（回帰/分類）: `populate_entry_trend()`
  - サイジング: `custom_stake_amount()` が VolatilityTargetSizer + edge を使用
  - Exit: `custom_exit()` / `adjust_trade_position()` が ExitPolicy を使用
- `user_data/config_freqai.json`
  - `ext_risk` セクション（vol_target, exit_policy, tb, calibration）
  - 5m 安定用の FreqAI 設定（下記「データ要件」参照）

参照しやすい導線:

- `user_data/strategies/FreqAICustomStrategy.py:1`
- `freqtrade_ext/risk/vol_sizer.py:1`
- `freqtrade_ext/risk/exit_policy.py:1`
- `freqtrade_ext/calibration/thresholds.py:1`
- `user_data/config_freqai.json:1`

## 設定（クイックリファレンス）

`user_data/config_freqai.json` の `ext_risk` を中心に切替できます。

- `ext_risk.vol_target`
  - `enabled`: true/false
  - `mode`: "unit_atr"
  - `risk_pct_per_trade`: 例 0.001（=0.1%）
  - `atr_period`: 14, `atr_k`: 2.0（想定ストップ距離）
  - `max_leverage`: 3（先物）
  - エッジ倍率（任意）: `edge_enabled`, `edge_scale`, `edge_min`, `edge_max`

- `ext_risk.exit_policy`
  - `enabled`: true/false
  - `time_stop_candles`: 例 36（5m×36=3h）
  - `partial_tps`: `{ profit_pct, reduce_pct, min_hold_candles }` の配列
  - `trail_from_profit_pct`, `trail_step_pct`, `hard_stop_dd_pct`

- `ext_risk.tb`（トリプルバリア）
  - `tp`, `sl`, `horizon_candles`（分類時のみ `&s-tb_label` を生成）

- `ext_risk.calibration`
  - `enabled`: true/false（動的しきい値較正の有効化）

FreqAI 設定の注意:

- `include_timeframes`: 広域検証時は `["5m"]` を推奨（欠損連鎖を回避）。
- `include_corr_pairlist`: `[]`（相関ペア欠損での NaN 連鎖回避）。
- `train_period_days`: 5m で 20 日（≈ 4332 本）を確保。
- `prediction_mode`: 回帰（現状）/分類（A/B 準備済み）。

## サイジング: VolatilityTargetSizer

場所: `freqtrade_ext/risk/vol_sizer.py`

基本（先物/現物共通）:

- 停止距離 ≈ `atr_k * ATR(period)`
- リスク予算 ≈ `max_stake * risk_pct_per_trade`
- ノーション ≈ `budget * price / stop_distance`
- 証拠金ステーク ≈ `notional / leverage`

エッジ倍率（任意）:

- `edge_score = max(0, |prediction|-threshold)`（分類なら確率超過量）
- 倍率 ∈ [`edge_min`, `edge_max`]、傾きは `edge_score / edge_scale`

戦略配線: `custom_stake_amount()` が edge を集約し `suggest_stake()` を呼び出します。

## Exit: ExitPolicy（タイムストップ/部分利確/トレイル）

場所: `freqtrade_ext/risk/exit_policy.py`

- `custom_exit()`: `time_stop`, `trail_take`, `hard_dd_stop` を返却。
- `adjust_trade_position()`: 部分利確時にマイナスのステーク（削減量）を返す。
- 状態は `trade.id` 単位でメモリ管理（消化済み段、最大含み益）。

有効化: 戦略で `position_adjustment_enable: true`（既に設定済み）。

## エントリー（回帰 / 分類）

共通ゲート:

- 品質: `do_predict == 1`（デバッグ時はスキップ可能）。
- トレンド: EMA20 vs EMA50（long: EMA20>EMA50 / short: EMA20<EMA50）。
- スプレッド: `feat__spread_bps` がローリング 95% 超は回避。
- BTC 乱高下（分類のみ）: `market_bad==1` で停止。

回帰（現運用）:

- 方向: `&-target` の符号。
- 期待値: `&-target` の絶対値しきい値。
- 較正: `best_abs_threshold()` がペア別に最適しきい値（long/short）を選定。
- 実装: `populate_entry_trend()` の回帰ブランチ。

分類（A/B 準備済み）:

- 方向: `&s-tb_label`（'up'/'down'）または確率列（`long_key`/`short_key`）。
- 期待値: 較正後の確率超過量（`best_proba_thresholds()`）。
- 実装: `populate_entry_trend()` の分類ブランチ、`set_freqai_targets()` で TB ラベル生成。分類有効時は `self.freqai.class_names` を設定。

## 目的変数/ラベル

`set_freqai_targets()`:

- 将来リターンを常に算出: `fwd_ret = close.shift(-lp)/close - 1`。
- 回帰モード: `&-target = fwd_ret` を作成。
- 分類モード: 校正用に `ret_fwd`（通常列）を保持し、TB ラベルを作成：
  - 数値 TB: `&-tb_label ∈ { -1, 0, 1 }`
  - 文字列 TB（二値, `0->down`): `&s-tb_label ∈ { 'down', 'up' }`

## データ要件（NaN 落ち回避）

- 5m で `train_period_days` とインジケータ初期窓を満たすだけの“過去データ”が必要。スライス先頭で不足すると DataKitchen が全落ちさせ、BT 失敗になります。
- 広域検証中は `["5m"]` のみ・相関なしを推奨。
- BT 開始前に十分な 5m データを用意してください。

コマンド例:

- 取得: `freqtrade download-data -c user_data/config_freqai.json -t 5m --timerange 20250615-20250824 -p BTC/USDT:USDT ETH/USDT:USDT`
- 範囲確認: `freqtrade list-data -c user_data/config_freqai.json --trading-mode futures --show-timerange -p BTC/USDT:USDT ETH/USDT:USDT -v`

## 典型ワークフロー

BT（回帰ベース）:

- 安全レンジ（既存ローカルでOK）: `--timerange 20250730-20250824`
- データ拡張後の広域: `--timerange 20250701-20250824`

分類 A/B（方向∧確率超過量）:

- 切替: `freqaimodel: LightGBMClassifier`, `prediction_mode: classification`
- 確率列（'up'/'down'）を `classifier.long_key/short_key` に合わせる。
- `ext_risk.calibration.enabled: true` で自動較正を有効化。

ライブ（先物）:

- `live_retrain_hours` を 6–12h、`expiration_hours` を 1–2h で新鮮さを維持。
- `risk_pct_per_trade`, `atr_k`, `edge_*` を DD と資金利用状況で微調整。

## トラブルシュート

- 「all training data dropped due to NaNs」:
  - データ開始が遅い → 5m を十分前から取得。
  - `include_timeframes: ["5m"]`、`include_corr_pairlist: []` を確認。
  - テスト時は `train_period_days` を一時的に 14→10 へ短縮も可。

- 取引が発生しない:
  - ログ `[FreqAICustomStrategy] rows:... enter_long:... enter_short:...` を確認。
  - `do_predict==1` のカバレッジ、トレンド/スプレッドゲートの厳しさを確認。
  - 分類時は `classifier.long_key/short_key` が実列名と一致しているか。

- サイジングが大/小すぎる:
  - `risk_pct_per_trade`, `atr_k`, `max_leverage` を調整。
  - `edge_scale`, `edge_min`, `edge_max` を調整。

## 本リポジトリ差分（変更履歴）

- 追加: `freqtrade_ext/risk/vol_sizer.py`（ATR サイジング＋エッジ倍率）
- 追加: `freqtrade_ext/risk/exit_policy.py`（タイムストップ/部分利確/トレイル/ハードDD）
- 追加: `freqtrade_ext/calibration/thresholds.py`（回帰/分類のしきい値較正）
- 戦略拡張: `user_data/strategies/FreqAICustomStrategy.py`
  - 目的変数/TB（`set_freqai_targets`）
  - 回帰/分類エントリー（`populate_entry_trend`）
  - サイジング（`custom_stake_amount`）、Exit（`custom_exit`）、部分利確（`adjust_trade_position`）
- 設定拡張: `user_data/config_freqai.json`（`ext_risk` 各種と 5m 安定初期値）

---

次の作業（分類 A/B 有効化・期間拡大・ExitPolicy 調整など）は「典型ワークフロー」「トラブルシュート」に従って実行してください。較正ユーティリティを併用することで、固定値チューニングの手間を減らせます。
