# 仮想通貨自動取引Bot工場 AIエージェント向け指示書

## 0. この指示書の目的

このドキュメントは、仮想通貨自動取引Botを **自律的に作成・評価・ペーパー運用・小額本番運用・監視・改善** する「Bot工場」を構築するための、AIエージェント向け実装指示書である。

ここでいう「Bot工場」とは、単一の売買Botではなく、以下の一連の流れを継続的に回すシステムを指す。

```text
市場データ収集
→ 特徴量生成
→ 戦略案生成
→ Botコード生成
→ 静的検査
→ バックテスト
→ Walk-forward検証
→ ペーパー取引
→ 小額本番
→ 本番監視
→ 停止・改善・再学習
```

## 1. 最重要方針

### 1.1 LLMに直接売買判断をさせない

LLM/AIエージェントは以下を行ってよい。

- 戦略アイデアの生成
- Botコードの生成
- バックテスト実行
- 結果分析
- 改善案の提案
- レポート作成
- 設定ファイルの作成
- ペーパー環境へのデプロイ提案

ただし、LLM/AIエージェントに以下を直接行わせてはならない。

- 人間承認なしの本番Bot昇格
- 人間承認なしの実資金投入
- APIキー権限の変更
- リスク制限の緩和
- 損失中Botのナンピン判断
- 予測だけに基づく成行注文
- 失敗したBotの自動再投入

実注文は、必ず `Risk Governor` と `Execution Gateway` を通すこと。

---

## 2. システム全体像

```text
[Market Data Sources]
  ├─ Exchange REST API
  ├─ Exchange WebSocket
  ├─ OHLCV
  ├─ Trades
  ├─ Orderbook
  ├─ Funding Rate
  ├─ Open Interest
  └─ Account / Position Data

        ↓

[Data Layer]
  ├─ Raw Data Store: Parquet
  ├─ Query Store: DuckDB / ClickHouse / QuestDB
  ├─ Metadata DB: PostgreSQL
  └─ Experiment Store: MLflow

        ↓

[Bot Factory]
  ├─ Strategy Generator Agent
  ├─ Code Generator Agent
  ├─ Test Agent
  ├─ Backtest Agent
  ├─ ML/FreqAI Agent
  ├─ Reviewer Agent
  └─ Risk Agent

        ↓

[Evaluation Pipeline]
  ├─ Static Check
  ├─ Unit Test
  ├─ Lookahead Bias Check
  ├─ Fast Backtest
  ├─ Detailed Backtest
  ├─ Walk-forward Test
  ├─ Paper Trading
  └─ Canary Live Trading

        ↓

[Operation Layer]
  ├─ Freqtrade
  ├─ FreqAI
  ├─ Hummingbot
  ├─ Risk Governor
  ├─ Execution Gateway
  ├─ Monitoring
  └─ Alerting
```

---

## 3. 推奨技術スタック

### 3.1 MVP構成

まずは以下で構築する。

```text
Python
Freqtrade
FreqAI
CCXT
DuckDB
Parquet
PostgreSQL
MLflow
Prefect
Docker Compose
Grafana
Slack / Chatwork通知
```

### 3.2 本格運用構成

```text
Data:
  - Parquet
  - DuckDB
  - ClickHouse or QuestDB
  - PostgreSQL
  - MinIO or S3-compatible storage

Workflow:
  - Prefect or Dagster

Backtest:
  - vectorbt
  - Freqtrade backtesting
  - Hummingbot backtesting where applicable

ML:
  - FreqAI
  - LightGBM
  - XGBoost
  - CatBoost
  - scikit-learn
  - MLflow

Execution:
  - Freqtrade
  - Hummingbot
  - Custom Risk Governor
  - Custom Execution Gateway

Monitoring:
  - Prometheus
  - Grafana
  - Loki
  - Alertmanager
  - Slack / Chatwork
```

---

## 4. リポジトリ構成

以下の構成を基本とする。

```text
crypto-bot-factory/
  README.md
  docker-compose.yml
  .env.example
  pyproject.toml

  docs/
    architecture.md
    risk_policy.md
    promotion_rules.md
    operation_manual.md
    incident_response.md

  apps/
    dashboard/
    research_api/
    risk_governor/
    execution_gateway/
    reporter/

  workers/
    data_ingestor/
    feature_builder/
    strategy_generator/
    backtest_runner/
    freqai_trainer/
    paper_trader/
    live_deployer/
    monitor/

  bots/
    freqtrade/
      strategies/
      configs/
      user_data/
    hummingbot/
      controllers/
      configs/

  data/
    raw/
    normalized/
    features/
    backtests/
    paper_trading/
    live/

  registry/
    strategies/
    models/
    experiments/
    deployments/

  tests/
    unit/
    integration/
    backtest_validation/
    risk/
    e2e/

  scripts/
    setup.sh
    download_data.py
    run_backtest.py
    run_walk_forward.py
    promote_bot.py
    stop_bot.py
    generate_report.py
```

---

## 5. エージェントの役割

## 5.1 Research Agent

### 目的

新しい売買戦略の仮説を作る。

### 入力

- 市場データの概要
- 既存Botの成績
- 不採用になった戦略一覧
- 使用可能な特徴量
- 直近の市場レジーム
- リスクポリシー

### 出力

`registry/strategies/proposals/YYYYMMDD_strategy_name.md`

### 出力フォーマット

```markdown
# Strategy Proposal: <strategy_name>

## Summary

## Target Market
- Exchange:
- Symbol:
- Timeframe:
- Long/Short:
- Spot/Futures:

## Hypothesis

## Entry Logic

## Exit Logic

## Risk Logic

## Required Data

## Expected Weakness

## Overfitting Risk

## Backtest Plan

## Rejection Conditions
```

### 禁止事項

- 「儲かりそう」だけの曖昧な戦略を作らない
- 未来データを使う前提の戦略を作らない
- 流動性の低い銘柄だけに依存しない
- 手数料・スリッページを無視しない

---

## 5.2 Code Generator Agent

### 目的

Research Agentの戦略案を、実行可能なBotコードへ変換する。

### 対象

- Freqtrade Strategy
- FreqAI Strategy
- Hummingbot Controller
- Feature Builder
- Test Code

### 出力先

```text
bots/freqtrade/strategies/
bots/hummingbot/controllers/
workers/feature_builder/
tests/
```

### 実装ルール

- すべての戦略コードにdocstringを付ける
- エントリー条件とイグジット条件を関数分割する
- 設定値はハードコードせず、configまたはclass parameterに逃がす
- future data / shift(-1) / 現在足の終値確定前利用を禁止する
- 注文サイズは戦略側で直接決定しない
- 実注文の可否はRisk Governorに委ねる

### Freqtrade Strategyの最低要件

- `populate_indicators`
- `populate_entry_trend`
- `populate_exit_trend`
- `minimal_roi`
- `stoploss`
- `timeframe`
- `startup_candle_count`

---

## 5.3 Test Agent

### 目的

生成されたBotコードが安全に評価可能か検査する。

### 必須テスト

```text
1. importできること
2. FreqtradeのStrategyとして読み込めること
3. 必須メソッドが存在すること
4. NaNが異常に発生しないこと
5. future data参照がないこと
6. shift(-1)が使われていないこと
7. iloc[-1]の危険な利用がないこと
8. 注文サイズを直接操作していないこと
9. 例外時に発注しないこと
10. 設定ファイルがschema validationを通ること
```

### 静的検査例

以下を検出した場合は原則として失格にする。

```text
shift(-1)
future
lookahead
iloc[-1] in indicator generation
hardcoded API key
hardcoded secret
requests.post to exchange order API
manual order placement outside execution gateway
```

---

## 5.4 Backtest Agent

### 目的

Bot候補をバックテストし、採用可能性を評価する。

### 実行順序

```text
1. データ存在確認
2. データ品質チェック
3. 高速バックテスト
4. 詳細バックテスト
5. 手数料・スリッページ反映
6. 複数期間テスト
7. 複数銘柄テスト
8. レポート出力
```

### 評価対象期間

最低でも以下を含める。

```text
bull_market
bear_market
range_market
high_volatility
low_volatility
crash_period
recent_period
```

### 出力

```text
data/backtests/<strategy_name>/<run_id>/
  result.json
  trades.csv
  equity_curve.csv
  metrics.json
  report.md
```

### 主要指標

```yaml
metrics:
  total_return:
  cagr:
  max_drawdown:
  sharpe:
  sortino:
  calmar:
  profit_factor:
  win_rate:
  average_win:
  average_loss:
  trade_count:
  expectancy:
  turnover:
  fee_paid:
  slippage_sensitivity:
  longest_drawdown_days:
```

---

## 5.5 Walk-forward Agent

### 目的

過学習したBotを排除する。

### 検証方針

以下のように、学習期間と検証期間をずらして複数回評価する。

```text
Train: 2021-01-01 ~ 2021-12-31
Test : 2022-01-01 ~ 2022-03-31

Train: 2021-04-01 ~ 2022-03-31
Test : 2022-04-01 ~ 2022-06-30

Train: 2021-07-01 ~ 2022-06-30
Test : 2022-07-01 ~ 2022-09-30
```

### 合格基準

```yaml
walk_forward_rules:
  min_pass_rate: 0.7
  max_single_period_dependency: 0.4
  max_drawdown_in_any_period: 0.2
  min_profitable_periods_ratio: 0.6
```

---

## 5.6 FreqAI Agent

### 目的

FreqAIを利用して、予測モデルを含むBot候補を作成・評価する。

### 予測対象の例

```text
5分後リターン
15分後リターン
1時間後リターン
上昇確率
下落確率
ボラティリティ
市場レジーム
```

### 特徴量候補

```text
OHLCV
volume change
volatility
RSI
MACD
Bollinger Bands
ATR
VWAP deviation
funding rate
open interest
spread
orderbook imbalance
trade imbalance
liquidation data
```

### 禁止事項

- 未来のリターンを特徴量に混ぜない
- ラベル生成後に時系列をシャッフルしない
- 検証データを学習に使わない
- 特徴量重要度だけを根拠に本番投入しない
- 精度だけで評価しない

### 評価指標

分類モデルの場合。

```yaml
classification_metrics:
  accuracy:
  precision:
  recall:
  f1:
  auc:
  calibration_error:
  profit_when_signal_used:
```

回帰モデルの場合。

```yaml
regression_metrics:
  mae:
  rmse:
  directional_accuracy:
  rank_correlation:
  profit_when_signal_used:
```

---

## 5.7 Reviewer Agent

### 目的

Bot候補の採用・却下・再検証を判断する。

### 入力

- strategy proposal
- source code
- unit test result
- backtest result
- walk-forward result
- paper trading result
- risk report
- correlation report

### 出力

```text
registry/strategies/reviews/<strategy_name>/<run_id>_review.md
```

### レビュー観点

```text
1. 利益は一部期間に偏っていないか
2. trade countは十分か
3. 手数料負けしていないか
4. スリッページに弱すぎないか
5. 最大DDが許容範囲か
6. 既存Botと相関が高すぎないか
7. コードが単純すぎる最適化になっていないか
8. ペーパー取引でバックテストと乖離していないか
9. 本番投入時の損失上限が定義されているか
10. 停止条件が明確か
```

---

## 5.8 Risk Agent / Risk Governor

### 目的

全Bot横断で資金・ポジション・損失・異常状態を管理する。

### Risk Governorの原則

すべての注文はRisk Governorを通す。

```text
Bot
 ↓ order request
Risk Governor
 ↓ approved / rejected / resized
Execution Gateway
 ↓
Exchange
```

### グローバルリスク制限

初期値は以下を採用する。

```yaml
global_risk:
  max_total_exposure_pct: 30
  max_single_bot_exposure_pct: 5
  max_single_symbol_exposure_pct: 10
  max_daily_loss_pct: 3
  max_weekly_loss_pct: 7
  max_total_drawdown_pct: 15
  max_leverage: 1
  allow_short: false
```

### Bot単位制限

```yaml
bot_risk:
  max_bot_daily_loss_pct: 1
  max_bot_drawdown_pct: 5
  max_position_count: 3
  max_order_per_minute: 5
  max_consecutive_losses: 5
```

### 自動停止条件

以下に該当した場合、Botを停止する。

```text
日次損失上限に到達
週次損失上限に到達
最大DD上限に到達
想定外ポジションを検出
取引所APIエラー率が閾値超過
WebSocket遅延が閾値超過
注文拒否が連続発生
残高不一致
バックテストと実績の乖離が大きい
モデル予測分布が学習時から大きく変化
```

---

## 5.9 Operator Agent

### 目的

Botの起動・停止・再起動・ペーパー環境へのデプロイを行う。

### 許可される操作

- ペーパー取引Botの起動
- ペーパー取引Botの停止
- レポート生成
- ログ確認
- 異常時の本番Bot停止
- 本番Botの状態確認

### 人間承認が必要な操作

- 本番Botの新規起動
- 本番Botの資金増額
- レバレッジ利用
- ショート許可
- リスク制限の緩和
- APIキー権限の変更

---

## 5.10 Reporter Agent

### 目的

運用状況を人間が判断しやすい形に要約する。

### 日次レポート

```markdown
# Daily Bot Factory Report

## Summary
- Active Bots:
- Paper Bots:
- New Candidates:
- Stopped Bots:
- Total PnL:
- Daily Drawdown:

## Best Bot

## Worst Bot

## Risk Alerts

## Backtest vs Live Drift

## Promotion Candidates

## Demotion Candidates

## Required Human Actions
```

### 週次レポート

```markdown
# Weekly Bot Factory Report

## Portfolio Performance

## Strategy Performance by Type

## Market Regime Analysis

## Correlation Analysis

## Failed Experiments

## Lessons Learned

## Next Week Plan
```

---

## 6. Bot昇格ゲート

Botは以下のゲートを順番に通過しなければならない。

```text
G0: Strategy Proposal
G1: Code Generation
G2: Static Check
G3: Unit Test
G4: Fast Backtest
G5: Detailed Backtest
G6: Walk-forward Test
G7: Paper Trading
G8: Canary Live
G9: Production
```

---

## 6.1 G0: Strategy Proposal

合格条件。

```yaml
g0_rules:
  has_clear_hypothesis: true
  has_entry_logic: true
  has_exit_logic: true
  has_risk_logic: true
  defines_required_data: true
  defines_rejection_conditions: true
```

---

## 6.2 G1-G3: Code / Static / Unit Test

合格条件。

```yaml
g1_g3_rules:
  import_success: true
  no_syntax_error: true
  no_hardcoded_secret: true
  no_future_data_reference: true
  no_direct_exchange_order_call: true
  unit_test_passed: true
```

---

## 6.3 G4-G5: Backtest

初期合格条件。

```yaml
backtest_rules:
  min_trades: 200
  min_profit_factor: 1.25
  max_drawdown_pct: 15
  min_sortino: 1.2
  max_fee_to_profit_ratio: 0.35
  max_slippage_degradation_pct: 30
```

---

## 6.4 G6: Walk-forward

合格条件。

```yaml
walk_forward_rules:
  min_pass_rate: 0.7
  min_profitable_windows_ratio: 0.6
  max_drawdown_pct_any_window: 20
  no_single_window_profit_dependency: true
```

---

## 6.5 G7: Paper Trading

合格条件。

```yaml
paper_trading_rules:
  min_days: 21
  max_backtest_live_pnl_deviation_pct: 40
  max_execution_error_rate_pct: 2
  max_order_rejection_rate_pct: 2
  no_unexpected_position: true
  no_balance_mismatch: true
```

---

## 6.6 G8: Canary Live

合格条件。

```yaml
canary_live_rules:
  max_capital_allocation_pct: 1
  min_days: 14
  max_daily_loss_pct: 0.5
  no_critical_incident: true
  human_review_required: true
```

---

## 6.7 G9: Production

Production昇格には人間承認を必須とする。

```yaml
production_rules:
  human_approval_required: true
  max_initial_capital_allocation_pct: 3
  leverage_allowed: false
  short_allowed: false
  rollback_plan_required: true
  kill_switch_required: true
```

---

## 7. データ設計

## 7.1 保存対象

最低限、以下を保存する。

```text
ohlcv
trades
orderbook_l1
orderbook_l2
funding_rate
open_interest
liquidations
account_balance
orders
fills
positions
bot_decisions
bot_logs
risk_events
```

---

## 7.2 Rawデータ

```text
data/raw/
  exchange=<exchange>/
    symbol=<symbol>/
      data_type=<type>/
        date=YYYY-MM-DD/
          part-000.parquet
```

例。

```text
data/raw/exchange=binance/symbol=BTC_USDT/data_type=ohlcv_1m/date=2026-04-25/part-000.parquet
```

---

## 7.3 Featureデータ

```text
data/features/
  feature_set=<feature_set_name>/
    exchange=<exchange>/
      symbol=<symbol>/
        timeframe=<timeframe>/
          date=YYYY-MM-DD/
            part-000.parquet
```

---

## 7.4 データ品質チェック

以下を必ず確認する。

```text
欠損率
重複
タイムスタンプ逆転
異常価格
異常出来高
スプレッド異常
OHLCの整合性
取引所停止時間
データ取得遅延
```

異常検出時は、バックテストを中止する。

---

## 8. 戦略タイプ

Bot工場は、以下の戦略タイプを扱う。

## 8.1 Trend Following

```text
移動平均
ブレイクアウト
高値更新
ドンチャンチャネル
ボラティリティ拡大
```

## 8.2 Mean Reversion

```text
RSI
Bollinger Bands
VWAP乖離
短期急落リバウンド
過熱感逆張り
```

## 8.3 ML Prediction

```text
短期リターン予測
上昇確率分類
下落確率分類
ボラティリティ予測
レジーム分類
```

## 8.4 Market Making

```text
spread capture
inventory skew
volatility-adjusted quoting
orderbook imbalance
```

## 8.5 Relative Value / Arbitrage

```text
現物先物basis
funding rate arbitrage
取引所間価格差
ペアトレード
CEX/DEX差
```

---

## 9. 実装フェーズ

## Phase 1: Backtest Factory

### 目的

実取引なしで、戦略生成とバックテストを自動化する。

### 実装内容

```text
OHLCV取得
Parquet保存
Freqtrade strategy生成
静的検査
バックテスト
結果保存
ランキング表示
レポート生成
```

### 完了条件

```yaml
phase_1_done:
  can_download_ohlcv: true
  can_generate_strategy: true
  can_run_static_check: true
  can_run_backtest: true
  can_save_result_to_mlflow: true
  can_generate_report: true
```

---

## Phase 2: FreqAI Factory

### 目的

機械学習モデルを含むBot候補を生成・検証する。

### 実装内容

```text
特徴量生成
ラベル生成
FreqAI設定生成
モデル学習
Walk-forward検証
特徴量重要度出力
MLflow登録
```

### 完了条件

```yaml
phase_2_done:
  can_build_features: true
  can_train_freqai_model: true
  can_run_walk_forward: true
  can_register_model: true
```

---

## Phase 3: Paper Trading Factory

### 目的

バックテスト合格Botをペーパー環境に投入する。

### 実装内容

```text
dry-run config生成
paper bot起動
注文ログ保存
実績監視
バックテストとの差分分析
停止条件実装
```

### 完了条件

```yaml
phase_3_done:
  can_start_paper_bot: true
  can_monitor_paper_bot: true
  can_compare_backtest_and_paper: true
  can_stop_abnormal_bot: true
```

---

## Phase 4: Canary Live

### 目的

小額本番運用を安全に行う。

### 実装内容

```text
APIキー権限制限
Risk Governor実装
Execution Gateway実装
小額資金割当
Kill Switch実装
アラート実装
```

### 完了条件

```yaml
phase_4_done:
  risk_governor_required: true
  execution_gateway_required: true
  kill_switch_required: true
  human_approval_required: true
```

---

## Phase 5: Multi-Bot Portfolio

### 目的

複数Botをポートフォリオとして管理する。

### 実装内容

```text
Bot間相関計算
資金配分
重複ポジション制御
戦略タイプ分散
ポートフォリオDD管理
週次レビュー
```

---

## Phase 6: Hummingbot Integration

### 目的

マーケットメイク・裁定系Botを追加する。

### 実装内容

```text
Hummingbot controller生成
paper trading
inventory risk管理
spread監視
約定品質分析
```

---

## 10. AIエージェントへの共通指示

すべてのAIエージェントは、以下を守ること。

## 10.1 基本姿勢

```text
安全性を利益より優先する
再現性を重視する
すべての判断理由を記録する
すべての実験を追跡可能にする
曖昧な場合は本番投入しない
バックテスト結果を過信しない
```

---

## 10.2 コード生成ルール

```text
型ヒントを付ける
docstringを書く
例外処理を書く
テストを書ける構造にする
設定値を外出しする
ログを出す
secretをコードに書かない
本番/検証/ペーパーを明確に分ける
```

---

## 10.3 失格条件

以下を検出したら即失格とする。

```text
APIキーのハードコード
secretのハードコード
未来データ参照
直接発注API呼び出し
Risk Governorを迂回
本番資金の無断利用
レバレッジの無断利用
ショートの無断利用
ナンピン前提の損失隠し
バックテスト期間の恣意的選択
```

---

## 10.4 出力物の記録

すべての生成物に以下を記録する。

```yaml
metadata:
  created_at:
  created_by_agent:
  strategy_name:
  strategy_type:
  source_proposal:
  data_version:
  code_version:
  backtest_run_id:
  mlflow_run_id:
  status:
  rejection_reason:
```

---

## 11. 実装タスク一覧

## 11.1 初期セットアップ

```text
[ ] リポジトリ作成
[ ] Docker Compose作成
[ ] PostgreSQL起動
[ ] MLflow起動
[ ] Freqtrade環境作成
[ ] データ保存ディレクトリ作成
[ ] .env.example作成
[ ] README作成
```

---

## 11.2 データ収集

```text
[ ] CCXTでOHLCV取得
[ ] Parquet保存
[ ] データ品質チェック
[ ] DuckDBで読み込み確認
[ ] 複数銘柄対応
[ ] 複数timeframe対応
```

---

## 11.3 戦略生成

```text
[ ] strategy proposalテンプレート作成
[ ] Freqtrade strategyテンプレート作成
[ ] Research Agentプロンプト作成
[ ] Code Generator Agentプロンプト作成
[ ] 生成コード保存
```

---

## 11.4 テスト

```text
[ ] 静的検査スクリプト
[ ] import test
[ ] Freqtrade strategy load test
[ ] future data detection
[ ] hardcoded secret detection
[ ] direct order call detection
```

---

## 11.5 バックテスト

```text
[ ] Freqtrade backtest runner
[ ] 結果JSON保存
[ ] trades.csv保存
[ ] metrics計算
[ ] MLflow登録
[ ] Markdownレポート生成
```

---

## 11.6 Walk-forward

```text
[ ] rolling window分割
[ ] 各期間バックテスト
[ ] 結果集計
[ ] 合格判定
[ ] レポート生成
```

---

## 11.7 ペーパー取引

```text
[ ] dry-run config生成
[ ] paper bot起動
[ ] paper result保存
[ ] 実績監視
[ ] バックテストとの差分分析
```

---

## 11.8 Risk Governor

```text
[ ] 注文リクエストschema定義
[ ] exposure計算
[ ] symbol別上限制御
[ ] bot別上限制御
[ ] DD計算
[ ] daily loss計算
[ ] kill switch
[ ] alert通知
```

---

## 11.9 Dashboard

```text
[ ] bot一覧
[ ] backtestランキング
[ ] paper trading状況
[ ] live trading状況
[ ] DDグラフ
[ ] PnLグラフ
[ ] risk alert一覧
[ ] promotion candidate一覧
```

---

## 12. エージェント用プロンプト雛形

## 12.1 Research Agent Prompt

```markdown
あなたは仮想通貨自動取引Bot工場のResearch Agentです。

目的:
新しいBot候補の戦略仮説を作成してください。

制約:
- 未来データを使う戦略は禁止
- 手数料とスリッページを考慮すること
- 過学習しやすい条件を避けること
- リスク管理ロジックを必ず含めること
- 本番投入ではなく、まずバックテスト用の提案に限定すること

入力:
- 利用可能データ:
- 対象取引所:
- 対象銘柄:
- timeframe:
- 既存Botの弱点:
- 現在の市場レジーム:

出力:
以下の形式でMarkdownを出力してください。

# Strategy Proposal: <name>

## Summary

## Hypothesis

## Market Condition

## Entry Logic

## Exit Logic

## Risk Logic

## Required Data

## Parameters

## Expected Failure Cases

## Backtest Plan

## Rejection Conditions
```

---

## 12.2 Code Generator Agent Prompt

```markdown
あなたは仮想通貨自動取引Bot工場のCode Generator Agentです。

目的:
与えられたStrategy ProposalをFreqtrade Strategyとして実装してください。

制約:
- Pythonで実装すること
- Freqtrade Strategyとして読み込めること
- future data参照は禁止
- shift(-1)は禁止
- APIキーやsecretをコードに書かないこと
- 直接注文APIを呼ばないこと
- 注文サイズを戦略コード内で直接決めないこと
- 例外時に発注しないこと
- docstringと型ヒントを付けること

出力:
- strategy python file
- unit test file
- config example
- 実装上の注意点
```

---

## 12.3 Backtest Agent Prompt

```markdown
あなたは仮想通貨自動取引Bot工場のBacktest Agentです。

目的:
指定されたBot候補のバックテストを実行し、昇格可能性を評価してください。

実行内容:
1. データ品質チェック
2. Freqtrade strategy load確認
3. バックテスト実行
4. 指標計算
5. 手数料・スリッページ影響確認
6. 結果レポート作成
7. MLflow登録

失格条件:
- trade_count < 200
- max_drawdown > 15%
- profit_factor < 1.25
- sortino < 1.2
- slippage degradation > 30%

出力:
- metrics.json
- trades.csv
- report.md
- recommendation: pass / fail / retry
```

---

## 12.4 Reviewer Agent Prompt

```markdown
あなたは仮想通貨自動取引Bot工場のReviewer Agentです。

目的:
Bot候補をレビューし、次のゲートに進めるか判断してください。

入力:
- strategy proposal
- source code
- static check result
- unit test result
- backtest result
- walk-forward result
- paper trading result

判断:
- promote
- reject
- retry_with_modification
- need_human_review

レビュー観点:
- 過学習リスク
- 期間依存性
- 手数料耐性
- スリッページ耐性
- DD耐性
- 既存Botとの相関
- 本番運用時の停止条件
- リスク制限との整合性

出力:
Markdownでレビュー結果を出力してください。
```

---

## 12.5 Risk Agent Prompt

```markdown
あなたは仮想通貨自動取引Bot工場のRisk Agentです。

目的:
Botからの注文リクエストを評価し、許可・拒否・縮小の判断を行ってください。

絶対ルール:
- max_total_exposure_pctを超える注文は禁止
- max_single_bot_exposure_pctを超える注文は禁止
- max_single_symbol_exposure_pctを超える注文は禁止
- daily loss limit到達後の新規注文は禁止
- 予期しないポジションがある場合は新規注文禁止
- APIエラー率が高い場合は新規注文禁止
- balance mismatchがある場合は全Bot停止

出力:
{
  "decision": "approve | reject | resize | stop_bot | stop_all",
  "reason": "...",
  "approved_size": "...",
  "risk_flags": []
}
```

---

## 13. Cursor / Claude Code / Codex向け実装指示

AIコーディングエージェントに渡す場合は、以下を最初の指示として使う。

```markdown
あなたは「仮想通貨自動取引Bot工場」を実装するシニアソフトウェアエンジニア兼MLOpsエンジニアです。

目的:
Freqtrade / FreqAI / Hummingbot / MLflow / Prefect / PostgreSQL / Parquet を使って、
Botの作成・評価・ペーパー運用・小額本番運用・監視を行うシステムを段階的に構築してください。

最重要制約:
- いきなり本番取引を実装しない
- まずバックテスト工場を完成させる
- 実注文はRisk GovernorとExecution Gatewayを通す
- APIキーやsecretをコードに書かない
- LLMに直接発注権限を与えない
- 人間承認なしに本番Botを起動しない
- レバレッジとショートは初期実装では無効

最初に実装する範囲:
Phase 1: Backtest Factory

具体的に作るもの:
1. docker-compose.yml
2. PostgreSQL
3. MLflow
4. Freqtrade用ディレクトリ
5. OHLCVダウンロードスクリプト
6. Parquet保存
7. サンプルFreqtrade Strategy
8. バックテスト実行スクリプト
9. metrics保存
10. Markdownレポート生成

完了条件:
- `docker compose up` で基盤が起動する
- OHLCVを取得してParquetに保存できる
- サンプル戦略のバックテストが実行できる
- 結果がMLflowまたはPostgreSQLに保存される
- `data/backtests/.../report.md` が生成される

作業方針:
- 小さく実装する
- 各ステップで動作確認する
- テストを書く
- READMEを更新する
- 危険な本番取引処理はスタブにする
```

---

## 14. 最初に作るべきMVPタスク

AIエージェントは、最初に以下を実装する。

```text
Task 1:
  Create repository skeleton.

Task 2:
  Add docker-compose for PostgreSQL and MLflow.

Task 3:
  Add data downloader using CCXT.

Task 4:
  Save OHLCV to Parquet.

Task 5:
  Add sample Freqtrade strategy.

Task 6:
  Add backtest runner script.

Task 7:
  Parse backtest result.

Task 8:
  Save metrics to local JSON and MLflow.

Task 9:
  Generate Markdown report.

Task 10:
  Add static safety checker for strategy code.
```

---

## 15. MVPの成功条件

MVPは以下を満たしたら成功とする。

```yaml
mvp_success:
  exchange_data_download: true
  parquet_storage: true
  sample_strategy_backtest: true
  backtest_metrics_saved: true
  mlflow_tracking: true
  markdown_report_generated: true
  static_safety_check: true
  no_live_trading: true
```

---

## 16. 運用上の禁止事項

以下は禁止。

```text
他人資金の運用
投資助言としての提供
APIキーの共有
出金権限付きAPIキーの利用
レバレッジ前提の初期運用
損失中Botの自動増額
人間承認なしの本番投入
バックテスト結果のみでの本番投入
税務・法務確認なしの商用提供
```

---

## 17. セキュリティ要件

```text
APIキーは環境変数またはsecret managerで管理
出金権限は必ず無効化
IP制限可能なら有効化
本番キーと検証キーを分離
ログにsecretを出さない
注文ログを保存
残高差分を監視
Docker imageを固定
依存ライブラリを定期更新
```

---

## 18. インシデント対応

以下のイベントはインシデントとして扱う。

```text
想定外注文
想定外ポジション
残高不一致
APIキー漏洩疑い
異常な連続損失
Bot暴走
取引所API異常
WebSocket停止
価格データ異常
Risk Governor停止
```

### 初動

```text
1. 全Bot停止
2. 新規注文停止
3. 未約定注文キャンセル
4. 現在ポジション確認
5. APIキー無効化判断
6. ログ保全
7. 原因調査
8. 再発防止策作成
```

---

## 19. 判断基準

AIエージェントが迷った場合は、以下を優先する。

```text
1. 資金保護
2. 再現性
3. 透明性
4. 検証可能性
5. 利益
```

つまり、利益よりも安全性を優先する。

---

## 20. 最終ゴール

最終的な状態は以下。

```text
AIが新しい戦略案を作る
AIがBotコードを書く
AIがテストする
AIがバックテストする
AIがWalk-forward検証する
AIがペーパー運用に出す
AIが結果を評価する
AIが昇格候補を提案する
人間が本番投入を承認する
Risk Governorが実運用を制御する
異常時は自動停止する
週次で改善サイクルが回る
```

ただし、本番資金の利用とリスク制限の変更は、必ず人間承認を必要とする。
