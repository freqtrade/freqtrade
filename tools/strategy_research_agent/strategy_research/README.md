# Strategy Research Agent

这个目录是本地策略研究员 Agent 的工作台。它的目标是把策略研究变成可复现的闭环，而不是替代 Freqtrade 的交易 bot。

## 仓库化说明

可版本化源码位于：

```text
tools/strategy_research_agent/
```

运行区位于：

```text
user_data/strategy_research/
```

从仓库源码部署到运行区：

```bash
tools/strategy_research_agent/install_runtime.sh
```

`user_data` 继续存放本地数据、报告、dashboard、候选池、外部来源快照和私有配置，不应提交到 Git。

## 边界

允许：

- 登记内部策略和外部资料来源。
- 在隔离目录里生成研究策略或策略变体。
- 调用 Freqtrade 官方 `backtesting`、`recursive-analysis`、`lookahead-analysis`。
- 检查本地 K 线覆盖、缺口和数据路径。
- 生成 JSON、Markdown、候选/淘汰归档和 dashboard。

禁止：

- 不连接实盘 API。
- 不读取或打印私有 key。
- 不自动修改 dry-run/live 默认策略。
- 不自动启动 live trading。
- 不自动安装或运行外部 repo 代码。
- 不把网上策略直接当成可交易策略。

## 手动触发

推荐手动入口：

```bash
user_data/strategy_research/start_manual_research.sh --quick
```

这会先运行预检，再刷新报告和 dashboard，但不会重新跑大回测。

自主生成策略假设并跑短区间 smoke：

```bash
user_data/strategy_research/start_manual_research.sh --autonomous-smoke
```

它会调用 `autonomous_strategy_lab.py`，从本地可审计蓝图生成多家族策略候选、策略注册表、实验定义和假设台账，然后用 Freqtrade 官方回测系统跑短区间 smoke。

只做启动前体检：

```bash
user_data/strategy_research/start_manual_research.sh --preflight-only
```

真正跑完整研究循环：

```bash
user_data/strategy_research/start_manual_research.sh --full
```

如果还要重新尝试拉取 funding/mark 辅助数据：

```bash
user_data/strategy_research/start_manual_research.sh --full-with-aux
```

完整研究循环：

```bash
user_data/strategy_research/run_full_research_cycle.sh --skip-aux-fetch
```

每次完整研究循环都会先尝试增量补齐 BTC/ETH Binance USDT-M `1m` 主回测 K 线。如果需要重新尝试拉取 Binance 静态 funding/mark 辅助数据，可以去掉 `--skip-aux-fetch`。

完整研究矩阵：

```bash
./.venv/bin/python user_data/strategy_research/run_research_agent.py
```

只跑一个策略：

```bash
./.venv/bin/python user_data/strategy_research/run_research_agent.py \
  --strategy BtcEthFuturesRegime10xPullbackShortOnlyStrategy
```

只预览命令，不跑回测：

```bash
./.venv/bin/python user_data/strategy_research/run_research_agent.py --dry-run
```

短区间冒烟：

```bash
./.venv/bin/python user_data/strategy_research/run_research_agent.py \
  --strategy BtcEthFuturesEthSelfPullbackShortOnlyStrategy \
  --timerange 20260101-20260201
```

可选 recursive / lookahead：

```bash
./.venv/bin/python user_data/strategy_research/run_research_agent.py \
  --strategy BtcEthFuturesRegime10xPullbackShortOnlyStrategy \
  --run-recursive \
  --run-lookahead
```

旧版轻量回测脚本仍可用：

```bash
./.venv/bin/python user_data/strategy_research/run_strategy_research.py \
  --registry user_data/strategy_research/strategy_registry.json
```

## 外部来源登记

登记但不抓取正文：

```bash
./.venv/bin/python user_data/strategy_research/ingest_source.py \
  --url https://example.com/strategy-note \
  --title "Example strategy note" \
  --kind web_article
```

抓取一个最多 1MB 的隔离快照：

```bash
./.venv/bin/python user_data/strategy_research/ingest_source.py \
  --url https://example.com/strategy-note \
  --title "Example strategy note" \
  --kind web_article \
  --fetch
```

外部来源默认状态是 `quarantined_for_review`，只能阅读、摘要、转译到隔离策略，不能安装依赖、运行外部代码或进入实盘。

## 外部来源审查与转译

审查已登记来源并生成转译草案：

```bash
./.venv/bin/python user_data/strategy_research/review_sources.py
```

从通过审查的草案生成隔离策略：

```bash
./.venv/bin/python user_data/strategy_research/generate_source_strategies.py
```

回测来源转译策略：

```bash
./.venv/bin/python user_data/strategy_research/run_research_agent.py \
  --experiment user_data/strategy_research/experiments/source_translated_experiment.json
```

当前已验证过一个真实外部来源：

```text
https://raw.githubusercontent.com/freqtrade/freqtrade/stable/freqtrade/templates/sample_strategy.py
```

该来源只被抓取为隔离快照、审查并转译为本地研究策略；没有安装依赖、没有 import 外部代码、没有运行外部源码。

## 自主策略实验室

生成本地自主策略假设：

```bash
./.venv/bin/python user_data/strategy_research/autonomous_strategy_lab.py
```

输出：

```text
user_data/strategies/research_generated/autonomous_research_strategies.py
user_data/strategy_research/experiments/autonomous_strategy_registry.json
user_data/strategy_research/experiments/autonomous_strategy_experiment.json
user_data/strategy_research/experiments/autonomous_hypothesis_ledger.md
```

当前蓝图覆盖趋势回踩、震荡均值回归、波动压缩突破、失败反弹做空、微动量确认和防御型低杠杆基线。它的目标是让 Agent 主动提出可回测假设，而不是只复跑手工策略。

## 市场状态与成本矩阵

从本地 BTC futures 1m OHLCV 自动生成市场状态切片和成本场景：

```bash
./.venv/bin/python user_data/strategy_research/build_experiment_matrix.py
```

它会写入：

```text
user_data/strategy_research/market_regimes/btc_market_regime_slices.json
user_data/strategy_research/experiments/candidate_regime_matrix_base_cost.json
user_data/strategy_research/experiments/candidate_regime_matrix_stress_cost.json
```

运行候选策略的市场状态矩阵：

```bash
./.venv/bin/python user_data/strategy_research/run_research_agent.py \
  --experiment user_data/strategy_research/experiments/candidate_regime_matrix_base_cost.json

./.venv/bin/python user_data/strategy_research/run_research_agent.py \
  --experiment user_data/strategy_research/experiments/candidate_regime_matrix_stress_cost.json
```

合并 base/stress 矩阵为韧性摘要：

```bash
./.venv/bin/python user_data/strategy_research/summarize_matrix.py \
  --report user_data/strategy_research/reports/agent_research_20260623T031046Z.json \
  --report user_data/strategy_research/reports/agent_research_20260623T031130Z.json
```

当前矩阵窗口：

```text
bull:     20241022-20250120
bear:     20251222-20260322
range:    20240507-20240805
high_vol: 20260118-20260418
```

当前成本场景：

```text
base fee:   0.05%
stress fee: 0.10%
```

注意：stress fee 是手续费/滑点压力的粗代理。当前已经下载并审计 Binance 静态 funding rate 与 mark price 数据，并已转换为 Freqtrade 可识别的 `1h funding_rate` 与 `1h mark` 本地数据文件；覆盖期到 `2026-05-31`，`2026-06` 静态包暂不可用。`estimate_trade_costs.py` 仍保留为交易级 funding/滑点校正视角。

## 自动生成策略变体

从当前候选池生成隔离杠杆变体：

```bash
./.venv/bin/python user_data/strategy_research/generate_variants.py
```

它会写入：

```text
user_data/strategies/research_generated/generated_leverage_variants.py
user_data/strategy_research/experiments/generated_variant_registry.json
user_data/strategy_research/experiments/generated_leverage_variants_experiment.json
```

回测生成变体实验：

```bash
./.venv/bin/python user_data/strategy_research/run_research_agent.py \
  --experiment user_data/strategy_research/experiments/generated_leverage_variants_experiment.json
```

## 安全定时巡检

手动入口：

```bash
user_data/strategy_research/run_daily_research.sh
```

自动触发采用 macOS `launchd`，不是常驻进程。当前提供两条任务：

- 每天 08:30：运行 `run_full_research_cycle.sh --skip-aux-fetch`，刷新本地矩阵、摘要、报告和 dashboard。
- 每周日 09:15：运行 `run_full_research_cycle.sh`，额外尝试拉取 Binance 静态 funding/mark 辅助数据。

两条任务都会先尝试增量补最新 BTC/ETH USDT-M `1m` 主回测 K 线；差别只在 weekly 会额外刷新 funding/mark 辅助数据。

安装：

```bash
user_data/strategy_research/automation/install_launchd.sh
```

查看状态和最近日志：

```bash
user_data/strategy_research/automation/status_launchd.sh
```

卸载：

```bash
user_data/strategy_research/automation/uninstall_launchd.sh
```

任务定义：

```text
user_data/strategy_research/automation/com.wangsen.freqtrade.strategy-research.daily.plist
user_data/strategy_research/automation/com.wangsen.freqtrade.strategy-research.weekly-aux.plist
```

这些任务只用于研究巡检，不会启动 live trading，也不会改实盘配置。当前已通过安装脚本注册到用户级 `~/Library/LaunchAgents`。

## 主要路径

- 配置：`agent_config.json`
- 策略登记：`strategy_registry.json`
- 外部来源登记：`source_registry.json`
- 外部来源审查：`sources/reviews/`
- 外部来源转译草案：`sources/translation_drafts/`
- 市场状态切片：`market_regimes/`
- 1m K 线更新报告：`data_updates/`
- 矩阵韧性摘要：`matrix_summaries/`
- 策略评分和失败归因：`strategy_assessments/`
- 合约成本数据审计：`cost_audits/`
- 策略级成本校正：`cost_adjustments/`
- 实验定义：`experiments/`
- 生成策略隔离区：`../strategies/research_generated/`
- 定时任务示例：`automation/`
- 完整研究循环：`run_full_research_cycle.sh`
- 报告：`reports/`
- 候选池：`candidates/`
- 观察池：`watchlist/`
- 淘汰池：`rejected/`
- Dashboard：`dashboard/index.html`

## 当前验收

- `run_research_agent.py --dry-run` 已通过。
- 短区间真实回测已通过。
- 全样本 BTC/ETH futures 1m 矩阵已通过。
- 生成策略变体已通过 Freqtrade `list-strategies` 识别。
- 生成变体短区间真实回测已通过。
- 真实 GitHub 外部来源已完成隔离抓取、审查、转译、短区间 smoke 和全样本回测。
- `recursive-analysis` 和 `lookahead-analysis` 已在来源转译策略短区间 smoke 上通过命令级验证。
- 候选策略已完成 bull/bear/range/high-vol × base/stress fee 矩阵回测；两条候选在矩阵摘要中均为 `fragile`。
- 已下载 Binance BTC/ETH USDT 永续 funding rate 与 mark price 静态数据并生成成本审计。
- 已将 funding/mark 辅助数据转换为 Freqtrade 内建 futures 数据格式；`list-data --show-timerange` 已识别 BTC/ETH 的 `1h funding_rate` 与 `1h mark`，覆盖到 `2026-05-31`。
- 已用官方 `backtesting --export trades` 验证 Freqtrade 回测交易明细里出现非零 `funding_fees`。
- 已用 Freqtrade `--export trades` 交易明细估算候选策略的 funding 与 4 bps 往返滑点影响；两条候选的校正后收益约为 `+1.69%` 与 `+1.76%`。
- 主回测 BTC/ETH USDT-M `1m` K 线已接入增量补数；最新状态见 `data_updates/latest_ohlcv_1m_update.md`。
- 已新增策略评分卡与失败归因报告；最新状态见 `strategy_assessments/latest_strategy_assessment.md`，并已接入 dashboard。
- 完整研究循环入口 `run_full_research_cycle.sh --help` 已通过。
- launchd 自动触发任务 plist 已通过 `plutil -lint`，安装/卸载/状态脚本已通过 shell 语法检查；当前已安装每日和每周两条用户级定时任务。
- 定时巡检入口 `run_daily_research.sh --help` 已通过。
- 最新来源转译全样本报告：`reports/agent_research_20260623T030602Z.md`
- 最新矩阵韧性摘要：`matrix_summaries/latest_matrix_summary.md`
- 最新合约成本数据审计：`cost_audits/latest_futures_cost_audit.md`
- 最新策略级成本校正：`cost_adjustments/latest_trade_cost_estimate.md`
- 最新 dashboard：`dashboard/index.html`
