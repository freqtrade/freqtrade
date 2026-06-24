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

根据上一轮自主策略失败结果生成 V2 并跑短区间 smoke：

```bash
user_data/strategy_research/start_manual_research.sh --iterate-smoke
```

它会调用 `strategy_iteration_engine.py`，读取最新自主 smoke 报告，把“交易太少 / PF 低 / 负收益”等失败原因转成下一版策略假设，并写入迭代台账。

对当前迭代策略跑固定窗口 walk-forward：

```bash
user_data/strategy_research/start_manual_research.sh --walk-forward
```

它会生成 2024H1、2024H2、2025H1、2025H2、2026H1 五个固定窗口实验，用同一组策略、同一手续费、同一 1m 数据跑回测，再输出稳定性汇总。

只做启动前体检：

```bash
user_data/strategy_research/start_manual_research.sh --preflight-only
```

真正跑完整研究循环：

```bash
user_data/strategy_research/start_manual_research.sh --full
```

一键强研究员 smoke 循环：

```bash
user_data/strategy_research/start_manual_research.sh --strong-researcher-smoke
```

它会串联 source scout、strategy lineage、research memory、memory-guided hypothesis planning、memory-guided strategy generation、Freqtrade `list-strategies`、短区间 smoke backtesting、评分/失败归因、成熟研究员决策计划和 dashboard/report 刷新。这个模式仍然是 research-only，不读取私钥、不改 dry-run/live 配置、不启动交易。

只刷新成熟研究员诊断和下一步实验计划：

```bash
user_data/strategy_research/start_manual_research.sh --mature-researcher
```

成熟研究员决策器会读取最新回测、交易行为、失败归因、评分卡和晋级闸门，自动判断：

- 高频但亏损：停止加杠杆，转向反向信号、低杠杆 edge 网格、手续费压力、延迟入场和短持仓退出实验。
- 多空都亏：拆 long-only / short-only，按方向跑 regime matrix，不允许用混合收益掩盖双侧失败。
- 入场时机差：用 MFE/MAE 证据触发延迟入场、价格先朝有利方向移动、入场前后对照实验。
- 成本敏感：自动要求 base fee / stress fee / slippage 检查，不允许 base-fee-only 晋级。
- 样本不足：每次只放宽一个条件，并保持最多 3 个确认条件，避免复杂过拟合。
- 候选策略：必须补齐 recursive/lookahead、regime matrix、walk-forward 和 stress-cost 才能进入人工 dry-run 评审。

输出：

```text
user_data/strategy_research/mature_researcher/latest_researcher_decision.json
user_data/strategy_research/mature_researcher/latest_researcher_decision.md
user_data/strategy_research/mature_researcher/latest_response_queue.json
user_data/strategy_research/mature_researcher/latest_response_queue.md
```

固定研究迭代闭环：

```bash
user_data/strategy_research/start_manual_research.sh --research-iteration
```

这条命令把你总结的路线固化为默认动作：

```text
生成/刷新研究记忆
-> 生成记忆驱动策略
-> 跑一轮 Freqtrade 回测
-> 分析交易行为和失败归因
-> 生成成熟研究员决策与响应队列
-> 复盘 Agent 研究员本身的问题
-> 写入下一轮 Agent 升级队列
```

复盘输出：

```text
user_data/strategy_research/agent_iterations/latest_iteration_review.json
user_data/strategy_research/agent_iterations/latest_iteration_review.md
user_data/strategy_research/agent_iterations/improvement_queue.json
user_data/strategy_research/agent_iterations/improvement_queue.md
```

只把成熟研究员决策转成可执行队列：

```bash
user_data/strategy_research/start_manual_research.sh --mature-researcher-queue
```

执行最高优先级的安全队列项：

```bash
user_data/strategy_research/start_manual_research.sh --execute-mature-researcher
```

队列执行器会记录每个 `strategy + experiment + command` 的执行历史，并默认在 6 小时内跳过重复项，避免长时间研究循环反复打同一个靶子。需要调试时可直接调用底层脚本：

```bash
./.venv/bin/python user_data/strategy_research/mature_researcher_queue.py --cooldown-hours 1
./.venv/bin/python user_data/strategy_research/mature_researcher_queue.py --execute-next --cooldown-hours 1
```

执行器一次只跑一个队列项，会写入：

```text
user_data/strategy_research/mature_researcher/latest_response_execution.json
user_data/strategy_research/mature_researcher/latest_response_execution.md
user_data/strategy_research/mature_researcher/response_execution_history.jsonl
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

生成外部来源发现与审查队列：

```bash
user_data/strategy_research/start_manual_research.sh --source-scout
```

直接调用脚本：

```bash
./.venv/bin/python user_data/strategy_research/scout_external_sources.py
```

登记并抓取一个新 URL 的有界快照：

```bash
./.venv/bin/python user_data/strategy_research/scout_external_sources.py \
  --url https://example.com/strategy-note \
  --title "Example strategy note" \
  --kind web_article \
  --fetch
```

输出：

```text
user_data/strategy_research/source_discovery/latest_source_discovery.json
user_data/strategy_research/source_discovery/latest_source_discovery.md
```

scout 只登记、抓取最多 1MB 的本地快照、生成审查队列和下一步命令；不会安装依赖、不会 import 或运行外部代码、不会启动交易。

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

## 失败驱动迭代

从最新自主 smoke 结果生成第二代策略：

```bash
./.venv/bin/python user_data/strategy_research/strategy_iteration_engine.py
```

输出：

```text
user_data/strategies/research_generated/iterative_research_strategies.py
user_data/strategy_research/experiments/iterative_strategy_registry.json
user_data/strategy_research/experiments/iterative_strategy_experiment.json
user_data/strategy_research/experiments/iterative_hypothesis_ledger.md
```

迭代规则：

- 交易太少：放宽入场容忍度，但保留市场状态过滤。
- PF 低或负收益：降低杠杆、加强确认、加快退出。
- 高频亏损：不加杠杆，先收紧触发条件。

## Walk-Forward 稳健性验证

构建固定窗口实验：

```bash
./.venv/bin/python user_data/strategy_research/walk_forward_validator.py build --source iterative --limit 6
```

运行实验：

```bash
./.venv/bin/python user_data/strategy_research/run_research_agent.py \
  --experiment user_data/strategy_research/experiments/walk_forward_validation_experiment.json
```

汇总实验：

```bash
./.venv/bin/python user_data/strategy_research/walk_forward_validator.py summarize \
  --report user_data/strategy_research/reports/<walk-forward-report>.json
```

输出：

```text
user_data/strategy_research/experiments/walk_forward_validation_experiment.json
user_data/strategy_research/walk_forward_summaries/latest_walk_forward_summary.json
user_data/strategy_research/walk_forward_summaries/latest_walk_forward_summary.md
```

候选至少需要跨多个固定时间窗表现稳定。单个窗口好看不能进入 dry-run 候选。

## 晋级闸门

对当前候选池和观察池做保守晋级检查：

```bash
./.venv/bin/python user_data/strategy_research/promotion_gate.py
```

或通过手动入口运行并刷新 dashboard：

```bash
user_data/strategy_research/start_manual_research.sh --promotion-gate
```

输出：

```text
user_data/strategy_research/promotion_reports/latest_promotion_report.json
user_data/strategy_research/promotion_reports/latest_promotion_report.md
user_data/strategy_research/promotion_candidates/
user_data/strategy_research/promotion_blocks/
```

通过闸门只代表“可进入人工 dry-run 复核”，不会自动改配置、不会读取私钥、不会启动实盘。

## 研究议程

基于晋级闸门的阻断原因，自动生成下一轮策略研究议程：

```bash
./.venv/bin/python user_data/strategy_research/research_agenda.py
```

或通过手动入口运行并刷新 dashboard：

```bash
user_data/strategy_research/start_manual_research.sh --agenda
```

输出：

```text
user_data/strategy_research/research_agendas/latest_research_agenda.json
user_data/strategy_research/research_agendas/latest_research_agenda.md
```

议程会记录优先级、阻断原因、研究假设、下一步命令、成功闸门和风险备注。它只生成研究任务，不启动交易。

## 议程执行回执

选择下一项安全研究任务并写入 dry-run 回执：

```bash
user_data/strategy_research/start_manual_research.sh --next-agenda
```

显式执行下一项非长任务：

```bash
user_data/strategy_research/start_manual_research.sh --execute-next-agenda
```

直接调用脚本：

```bash
./.venv/bin/python user_data/strategy_research/agenda_executor.py
```

输出：

```text
user_data/strategy_research/agenda_runs/latest_agenda_run.json
user_data/strategy_research/agenda_runs/latest_agenda_run.md
```

执行器只允许 allowlist 中的研究命令。walk-forward/full-cycle 等长任务需要额外传 `--allow-long`，默认不会执行。

## 交易行为分析

分析 Freqtrade 导出的交易明细，拆解策略为什么赚钱或亏钱：

```bash
user_data/strategy_research/start_manual_research.sh --trade-behavior
```

直接调用脚本：

```bash
./.venv/bin/python user_data/strategy_research/analyze_trade_behavior.py
```

默认读取 `cost_adjustments/latest_trade_cost_estimate.json` 中记录的 backtest zip。输出：

```text
user_data/strategy_research/trade_behavior/latest_trade_behavior.json
user_data/strategy_research/trade_behavior/latest_trade_behavior.md
```

当前分析项包括：胜率、平均盈亏、payoff、profit factor、多空比例、多空盈亏、持仓时长、止损退出、最大连续亏损、MFE/MAE、pair/exit/tag 拆分和诊断备注。

## 行为驱动实验计划

基于交易行为分析，自动规划下一批实验：

```bash
user_data/strategy_research/start_manual_research.sh --behavior-experiments
```

直接调用脚本：

```bash
./.venv/bin/python user_data/strategy_research/plan_behavior_experiments.py
```

输出：

```text
user_data/strategy_research/behavior_experiments/latest_behavior_experiment_plan.json
user_data/strategy_research/behavior_experiments/latest_behavior_experiment_plan.md
```

计划会把止损亏损、连续亏损、short-only 偏置、弱 pair、MFE/MAE 等问题转成实验假设、change set、预期效果、成功闸门和风险备注。

## 行为实验策略变体

基于行为驱动实验计划，生成可被 Freqtrade 发现和回测的策略变体：

```bash
user_data/strategy_research/start_manual_research.sh --behavior-variants
```

直接调用脚本：

```bash
./.venv/bin/python user_data/strategy_research/generate_behavior_experiment_strategies.py
```

输出：

```text
user_data/strategies/research_generated/behavior_experiment_strategies.py
user_data/strategy_research/experiments/behavior_experiment_strategy_registry.json
user_data/strategy_research/experiments/behavior_experiment_strategy_experiment.json
user_data/strategy_research/experiments/behavior_experiment_hypothesis_ledger.md
```

这些变体继承现有候选策略，只覆盖受控的研究参数或过滤条件，例如更强入场确认、stoploss/ROI sweep、坏微观状态过滤、short-only regime 拆分、弱 pair 禁用。它们仍然是 research-only，不能直接晋级 dry-run 或实盘。

## 失败归因

合并评分卡、晋级阻断、交易行为和行为实验计划，生成策略级失败归因：

```bash
user_data/strategy_research/start_manual_research.sh --failure-attribution
```

直接调用脚本：

```bash
./.venv/bin/python user_data/strategy_research/attribute_strategy_failures.py
```

输出：

```text
user_data/strategy_research/failure_attribution/latest_failure_attribution.json
user_data/strategy_research/failure_attribution/latest_failure_attribution.md
```

归因会给出每条策略的 top failure mode、严重度、证据、推荐动作和关联实验。它用于决定下一轮研究重点，而不是替代回测或晋级闸门。

## 策略库与版本族谱

合并本地策略登记、生成变体、来源转译、自主策略、迭代策略、行为实验策略、候选池、观察池、淘汰池、评分卡、晋级闸门、交易行为和失败归因，生成策略库族谱：

```bash
user_data/strategy_research/start_manual_research.sh --strategy-lineage
```

直接调用脚本：

```bash
./.venv/bin/python user_data/strategy_research/build_strategy_lineage.py
```

输出：

```text
user_data/strategy_research/strategy_library/latest_strategy_lineage.json
user_data/strategy_research/strategy_library/latest_strategy_lineage.md
```

族谱会记录每条策略的 generation、parent、root、children、pool status、推荐研究状态、核心回测指标、评分、晋级阻断、交易行为摘要和 top failure mode。它用于让 Agent 知道“这个策略从哪里来、为什么被保留或淘汰、下一步应该继续改哪一支”，不会授予任何 dry-run 或实盘权限。

## 研究记忆

从最新族谱、失败归因、评分卡和研究议程生成下一轮策略设计记忆：

```bash
user_data/strategy_research/start_manual_research.sh --research-memory
```

直接调用脚本：

```bash
./.venv/bin/python user_data/strategy_research/build_research_memory.py
```

输出：

```text
user_data/strategy_research/research_memory/latest_research_memory.json
user_data/strategy_research/research_memory/latest_research_memory.md
```

研究记忆包含 active roots、avoid patterns、next focus、knowledge gaps 和 durable rules。它的用途是给下一轮自动策略发明提供上下文：哪些 root 还值得继续，哪些失败模式不要重复，哪些证据缺口必须先补齐。它不会替代回测、不会改配置、不会启动交易。

## 记忆驱动假设规划

从研究记忆生成下一批策略研发假设：

```bash
user_data/strategy_research/start_manual_research.sh --memory-guided-hypotheses
```

直接调用脚本：

```bash
./.venv/bin/python user_data/strategy_research/plan_memory_guided_hypotheses.py
```

输出：

```text
user_data/strategy_research/experiments/memory_guided_hypothesis_plan.json
user_data/strategy_research/experiments/memory_guided_hypothesis_ledger.md
user_data/strategy_research/experiments/memory_guided_strategy_experiment.json
```

该规划器把 research memory 里的 active roots、avoid patterns、next focus 和 knowledge gaps 转成明确的 hypothesis id、目标策略、阻断原因、拟议入场/退出/风控变化、成功闸门和下一步命令。它是策略代码生成前的研究设计层，不直接创建 live 代码，也不直接回测。

## 记忆驱动策略变体

从可行动的记忆驱动假设生成隔离策略子类：

```bash
user_data/strategy_research/start_manual_research.sh --memory-guided-strategies
```

直接调用脚本：

```bash
./.venv/bin/python user_data/strategy_research/generate_memory_guided_strategies.py
```

输出：

```text
user_data/strategies/research_generated/memory_guided_research_strategies.py
user_data/strategy_research/experiments/memory_guided_strategy_registry.json
user_data/strategy_research/experiments/memory_guided_strategy_experiment.json
user_data/strategy_research/experiments/memory_guided_strategy_ledger.md
```

生成器会跳过 `bias_checks_missing` 这类验证型 blocker，只为 cost、matrix、walk-forward、exit quality 等可通过逻辑变体测试的问题生成策略。所有变体仍是 research-only，必须先通过 Freqtrade 识别、回测、评分卡、矩阵、walk-forward、成本和偏差检查。

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
- 外部来源发现：`source_discovery/`
- 外部来源审查：`sources/reviews/`
- 外部来源转译草案：`sources/translation_drafts/`
- 市场状态切片：`market_regimes/`
- 1m K 线更新报告：`data_updates/`
- 矩阵韧性摘要：`matrix_summaries/`
- 策略评分和失败归因：`strategy_assessments/`
- 晋级闸门：`promotion_reports/`、`promotion_candidates/`、`promotion_blocks/`
- 研究议程：`research_agendas/`
- 议程执行回执：`agenda_runs/`
- 交易行为分析：`trade_behavior/`
- 行为驱动实验计划：`behavior_experiments/`
- 行为实验策略变体：`experiments/behavior_experiment_*`、`../strategies/research_generated/behavior_experiment_strategies.py`
- 失败归因：`failure_attribution/`
- 策略库与版本族谱：`strategy_library/`
- 研究记忆：`research_memory/`
- 记忆驱动假设：`experiments/memory_guided_*`
- 记忆驱动策略变体：`experiments/memory_guided_strategy_*`、`../strategies/research_generated/memory_guided_research_strategies.py`
- 合约成本数据审计：`cost_audits/`
- 策略级成本校正：`cost_adjustments/`
- 实验定义：`experiments/`
- 生成策略隔离区：`../strategies/research_generated/`
- 定时任务示例：`automation/`
- 完整研究循环：`run_full_research_cycle.sh`
- 强研究员 smoke 循环：`run_strong_researcher_smoke.sh`
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
- 已新增外部来源发现队列；最新状态见 `source_discovery/latest_source_discovery.md`，并已接入 dashboard 与预检。
- 已新增强研究员 smoke 循环；入口为 `start_manual_research.sh --strong-researcher-smoke`。
- `recursive-analysis` 和 `lookahead-analysis` 已在来源转译策略短区间 smoke 上通过命令级验证。
- 候选策略已完成 bull/bear/range/high-vol × base/stress fee 矩阵回测；两条候选在矩阵摘要中均为 `fragile`。
- 已下载 Binance BTC/ETH USDT 永续 funding rate 与 mark price 静态数据并生成成本审计。
- 已将 funding/mark 辅助数据转换为 Freqtrade 内建 futures 数据格式；`list-data --show-timerange` 已识别 BTC/ETH 的 `1h funding_rate` 与 `1h mark`，覆盖到 `2026-05-31`。
- 已用官方 `backtesting --export trades` 验证 Freqtrade 回测交易明细里出现非零 `funding_fees`。
- 已用 Freqtrade `--export trades` 交易明细估算候选策略的 funding 与 4 bps 往返滑点影响；两条候选的校正后收益约为 `+1.69%` 与 `+1.76%`。
- 主回测 BTC/ETH USDT-M `1m` K 线已接入增量补数；最新状态见 `data_updates/latest_ohlcv_1m_update.md`。
- 已新增策略评分卡与失败归因报告；最新状态见 `strategy_assessments/latest_strategy_assessment.md`，并已接入 dashboard。
- 已新增晋级闸门；最新状态见 `promotion_reports/latest_promotion_report.md`，并已接入 dashboard 与预检。
- 已新增研究议程；最新状态见 `research_agendas/latest_research_agenda.md`，并已接入 dashboard 与预检。
- 已新增议程执行回执；最新状态见 `agenda_runs/latest_agenda_run.md`，并已接入 dashboard 与预检。
- 已新增交易行为分析；最新状态见 `trade_behavior/latest_trade_behavior.md`，并已接入 dashboard 与预检。
- 已新增行为驱动实验计划；最新状态见 `behavior_experiments/latest_behavior_experiment_plan.md`，并已接入 dashboard 与预检。
- 已新增行为实验策略变体生成；最新状态见 `experiments/behavior_experiment_hypothesis_ledger.md`。
- 已新增失败归因；最新状态见 `failure_attribution/latest_failure_attribution.md`，并已接入 dashboard 与预检。
- 已新增策略库与版本族谱；最新状态见 `strategy_library/latest_strategy_lineage.md`，并已接入 dashboard 与预检。
- 已新增研究记忆；最新状态见 `research_memory/latest_research_memory.md`，并已接入 dashboard 与预检。
- 已新增记忆驱动假设规划；最新状态见 `experiments/memory_guided_hypothesis_ledger.md`，并已接入 dashboard 与预检。
- 已新增记忆驱动策略变体生成；最新状态见 `experiments/memory_guided_strategy_ledger.md`。
- 完整研究循环入口 `run_full_research_cycle.sh --help` 已通过。
- launchd 自动触发任务 plist 已通过 `plutil -lint`，安装/卸载/状态脚本已通过 shell 语法检查；当前已安装每日和每周两条用户级定时任务。
- 定时巡检入口 `run_daily_research.sh --help` 已通过。
- 最新来源转译全样本报告：`reports/agent_research_20260623T030602Z.md`
- 最新矩阵韧性摘要：`matrix_summaries/latest_matrix_summary.md`
- 最新合约成本数据审计：`cost_audits/latest_futures_cost_audit.md`
- 最新策略级成本校正：`cost_adjustments/latest_trade_cost_estimate.md`
- 最新 dashboard：`dashboard/index.html`
