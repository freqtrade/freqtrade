# Freqtrade 个人实盘前检查清单

这份清单用于准备个人 Freqtrade 自动交易环境，目标是在实盘前尽量降低技术、配置、风控和操作风险。它不是投资建议。一个能运行的 bot、一次漂亮的回测、或者几天 dry-run，都不能证明策略适合实盘。

## 1. 环境准备

- 当前状态：已切到 `stable` 分支，已用 Python 3.11 创建 `.venv`，已安装 Freqtrade `2026.5.1`，已安装 FreqUI，`user_data` 目录已存在，命令行验证通过。
- 优先从 `stable` 分支安装 Freqtrade。
- 创建独立的 `user_data` 目录。
- 使用独立的 Python 虚拟环境，或者使用 Docker。
- 确认 `freqtrade --version` 和 `freqtrade --help` 可以正常运行。
- 配好 `.gitignore`，确保 API key、私有配置、交易数据库不会被提交到 Git。

## 2. 交易所与 API 安全

- 当前状态：已创建本地忽略文件 `user_data/config_dryrun.json`；配置为 Binance 现货 dry-run；交易所 `key`/`secret` 留空；WebUI/REST API 只监听 `127.0.0.1:8080`；密码、`jwt_secret_key`、`ws_token` 已随机生成；配置已通过 Freqtrade 加载校验。
- 一开始只使用 dry-run 或测试网。
- 不要把 API key 贴到聊天里，也不要提交到仓库。
- 实盘 API key 只给交易权限。
- 不要给 bot 使用的 API key 开提现权限。
- 如果交易所支持，开启 IP 白名单。
- Freqtrade WebUI 和 REST API 只监听 `127.0.0.1`。
- 不要把 WebUI 或 REST API 直接暴露到公网。
- 使用强密码、随机 `jwt_secret_key`、随机 `ws_token`。

## 3. 策略准备

- 当前状态：已从个人 `crypto` 仓库迁移全部已识别策略定义：17 个现货基础策略、12 个现货变体策略、7 个合约方向策略，共 36 个源策略定义；同时保留了 2 个先前手工封装的主策略包装，并新增 1 个现货熊市过滤策略、2 个 1h 50x 合约趋势/震荡过滤研究策略、1 个 1m 50x 合约研究策略、4 个杠杆/周期矩阵研究策略、9 个入场确认/方向过滤研究策略和 5 个 BTC 领先 ETH 研究策略。因此当前 Freqtrade 可识别 60 个策略类。
- 当前状态：现货策略文件包括 `user_data/strategies/btc_eth_core_signal_strategy.py`、`user_data/strategies/spot_strategy_library.py`、`user_data/strategies/btc_eth_risk_controlled_strategies.py`；`user_data/config_dryrun.json` 已指向熊市过滤现货策略 `BtcEthSpotBearMarketGuardStrategy`；已把 BTC/ETH 1h 数据转换为 Freqtrade 本地现货数据；现货策略库 29 个策略类已完成编译、`list-strategies` 识别、本地 BTC/ETH 信号 smoke；核心现货策略、风险受控现货策略和熊市过滤现货策略已完成官方 `backtesting` 和 `recursive-analysis`。
- 当前状态：合约策略文件包括 `user_data/strategies/btc_eth_futures_dual_momentum_strategy.py`、`user_data/strategies/futures_directional_strategy_library.py`、`user_data/strategies/btc_eth_risk_controlled_strategies.py`；`user_data/config_futures_dryrun.json` 已配置 isolated futures dry-run、`can_short` 策略、1x leverage callback，并指向风险适配合约策略 `BtcEthFuturesRiskAdaptedStrategy`；合约方向策略库 7 个策略类已完成编译、`list-strategies` 识别、本地 BTC/ETH long/short 信号 smoke；核心合约策略和风险适配合约策略已完成官方 `backtesting` 和 `recursive-analysis`。
- 当前状态：已新增本地策略研究员 Agent 工作台，位置为 `user_data/strategy_research/`；它不会连接实盘 key，也不会自动改 Freqtrade 默认策略。当前能力包括：读取 `strategy_registry.json` 和实验定义、检查 BTC/ETH futures 1m 数据完整性、批量调用 Freqtrade 官方 `backtesting`、可选调用 `recursive-analysis`/`lookahead-analysis`、解析收益/回撤/PF/交易次数/market change、写入候选池/观察池/淘汰池、生成 JSON/Markdown 报告和 `dashboard/index.html`。外部资料通过 `ingest_source.py` 进入隔离来源登记，通过 `review_sources.py` 生成来源审查与转译草案，通过 `generate_source_strategies.py` 生成本地隔离策略；默认不安装、不 import、不运行外部代码。当前已用 Freqtrade 官方 GitHub sample strategy raw 文件完成真实外部来源的隔离抓取、审查、转译、短区间 smoke 和全样本回测，官方 sample 转译策略全样本 `-28.01%` 被拒绝，本地 RSI/EMA pullback seed 转译策略全样本 `+0.27%` 进入观察池。候选池策略可通过 `generate_variants.py` 自动生成 3x/5x/10x 隔离杠杆变体，并由 `generated_leverage_variants_experiment.json` 进入回测；`build_experiment_matrix.py` 可从本地 BTC futures 1m OHLCV 自动生成 bull/bear/range/high-vol 市场状态切片，并产生 base fee `0.05%` 与 stress fee `0.10%` 两套候选策略矩阵；`summarize_matrix.py` 已把 base/stress 矩阵合并为 `matrix_summaries/latest_matrix_summary.md`，当前两条候选在 8 次 regime/cost 组合里均被评为 `fragile`，主要问题是 90 天切片交易数太少且 stress fee 下部分窗口转负。已新增 `analyze_strategy_research.py` 生成策略评分卡、失败归因和下一步建议，最新报告位于 `strategy_assessments/latest_strategy_assessment.md` 并已接入 dashboard。`run_full_research_cycle.sh` 可先增量补齐 BTC/ETH USDT-M `1m` 主回测 K 线，再串起辅助数据、成本审计、矩阵回测、摘要、评分归因和 dashboard 刷新；已新增并安装 macOS `launchd` 自动触发任务：每日 08:30 跑本地完整研究循环但跳过 aux 下载，周日 09:15 跑带 aux 下载的完整研究循环；两条任务都会先尝试补最新主回测 1m K 线；安装、状态、卸载脚本分别是 `automation/install_launchd.sh`、`automation/status_launchd.sh`、`automation/uninstall_launchd.sh`。
- 当前状态：Agent 源码、实验模板、自动化脚本和生成策略源码已仓库化到 `tools/strategy_research_agent/`；`user_data/strategy_research/` 继续作为本地运行区，保留数据、报告、dashboard、候选池和本地状态。macOS/Linux 重新部署运行区可执行 `tools/strategy_research_agent/install_runtime.sh`，Windows 可执行 `tools/strategy_research_agent/install_runtime.ps1` 并按 `tools/strategy_research_agent/README_WINDOWS.md` 安装 Task Scheduler 定时任务；这些脚本会保护候选池、淘汰池、观察池、外部来源快照、报告、dashboard、数据和本地配置，不会把这些运行时产物纳入 Git。
- 当前状态：已从 Binance 静态 archive 下载并审计 BTC/ETH USDT 永续 funding rate 与 mark price 数据；静态数据当前覆盖到 `2026-05-31`，`2026-06` 月份静态包暂不可用。已新增 `convert_aux_to_freqtrade_futures.py`，把辅助数据转换为 Freqtrade 可识别的 `1h funding_rate` 与 `1h mark` 本地数据；`list-data --show-timerange` 已能看到 BTC/ETH 的 funding/mark 文件，官方 `backtesting --export trades` 已验证交易明细出现非零 `funding_fees`。已新增 `estimate_trade_costs.py`，可基于 Freqtrade `--export trades` 的交易明细估算 funding 与往返滑点影响；在 4 bps 往返滑点假设下，当前两条合约候选的校正后收益约为 `+1.69%` 和 `+1.76%`，说明策略毛利润很薄，不能作为直接实盘依据。
- 当前限制：当前网络无法稳定访问 Binance/Kraken/OKX 等 public market metadata，因此官方命令通过显式启用 `PYTHONPATH=user_data/offline_exchange` 的离线 market stub 跑通；当前 BTC/ETH 1h 数据覆盖 `2024-01-01` 到 `2026-06-21`，每个交易对约 21672 根 K 线；BTC/ETH USDT 永续 1m 数据覆盖 `2024-01-01 00:00:00 UTC` 到 `2026-06-21 23:59:00 UTC`，每个交易对 1300320 根 K 线且缺口为 0；funding/mark 已转换为 Freqtrade 内建 futures 数据格式但只覆盖到 `2026-05-31`，因此覆盖 2026 年 6 月的官方合约回测仍不能完全替代严肃实盘前仿真。
- 先用一个极简策略跑通完整流程。
- 不要一开始就做复杂 AI、高频、多市场混合逻辑。
- 明确交易市场：现货还是合约。
- 明确交易对，例如 `BTC/USDT`、`ETH/USDT`。
- 明确时间周期，例如 `15m` 或 `1h`。
- 明确入场规则、出场规则、止损规则、最大同时持仓数。
- 避免未来函数和数据泄漏。
- 在相信结果前，运行 `lookahead-analysis` 和 `recursive-analysis`。

## 4. 历史数据与回测

- 当前状态：已从 Binance 静态 archive `data.binance.vision` 下载并转换 BTC/ETH 现货与 USDT 永续 1h K 线；4 个 Freqtrade feather 文件均覆盖 `2024-01-01 00:00:00 UTC` 到 `2026-06-21 23:00:00 UTC`，每个文件 21672 根 K 线，小时级缺口为 0。
- 当前状态：已从 Binance 静态 archive `data.binance.vision` 下载并转换 BTC/ETH USDT 永续 1m K 线；文件位于 `user_data/data/binance/futures/BTC_USDT_USDT-1m-futures.feather` 和 `user_data/data/binance/futures/ETH_USDT_USDT-1m-futures.feather`；每个文件覆盖 `2024-01-01 00:00:00 UTC` 到 `2026-06-21 23:59:00 UTC`，共 1300320 根 K 线，分钟级缺口为 0。
- 当前状态：核心现货策略 `BtcEthCoreSignalStrategy` 已完成长样本回测，显式手续费 `0.1%`、初始资金 `1000 USDT`、最大同时持仓 2；全样本 `2024-01-06` 到 `2026-06-21` 结果为 2008 笔交易、总收益 `-19.89%`、最大回撤 `20.31%`、胜率 `49.2%`、Profit factor `0.60`，同期 Freqtrade market change 为 `+9.55%`；样本切分结果为 2024 年 `-4.71%`、2025-2026 年 `-15.18%`。
- 当前状态：核心合约策略 `BtcEthFuturesDualMomentumStrategy` 已完成长样本回测，显式手续费 `0.05%`、初始资金 `1000 USDT`、最大同时持仓 1、1x leverage；全样本 `2024-01-09` 到 `2026-06-21` 结果为 3520 笔交易、总收益 `-8.23%`、最大回撤 `8.80%`、胜率 `60.0%`、Profit factor `0.83`，同期 Freqtrade market change 为 `+5.21%`；样本切分结果为 2024 年 `-1.88%`、2025-2026 年 `-6.31%`。
- 当前状态：风险受控现货策略 `BtcEthSpotRiskHoldStrategy` 已复用个人 `crypto` 仓库里的现货风控：4% 止损、8% 止盈、高 ATR/24h 大跌/RSI 过热过滤；全样本 `2024-01-09` 到 `2026-06-21` 结果为 220 笔交易、总收益 `-1.80%`、最大回撤 `7.25%`、胜率 `33.2%`、Profit factor `0.94`；样本切分结果为 2024 年 `+3.58%`、2025-2026 年 `-5.51%`；`recursive-analysis` 未发现指标 lookahead/recursive bias。
- 当前状态：熊市过滤现货策略 `BtcEthSpotBearMarketGuardStrategy` 在上述现货风控基础上新增市场状态过滤，只在价格位于 200h 均线之上且中期趋势或 30 日反弹满足条件时开仓；全样本 `2024-01-09` 到 `2026-06-21` 结果为 133 笔交易、总收益 `+3.22%`、最大回撤 `2.65%`、胜率 `38.3%`、Profit factor `1.19`；2025-2026 下跌窗口从上一版 `-5.51%` 改善到 `-0.06%`，最大回撤从 `6.74%` 降到 `2.53%`；`recursive-analysis` 未发现 lookahead，720h 指标需要至少 800 根 startup candles。
- 当前状态：风险适配合约策略 `BtcEthFuturesRiskAdaptedStrategy` 已复用个人 `crypto` 仓库里的合约入场过滤、4 小时保本退出思想，并把原 50x 场景下的 0.6% 价格止损适配为当前 1x dry-run 的 2% 止损；全样本 `2024-01-09` 到 `2026-06-21` 结果为 2920 笔交易、总收益 `-7.10%`、最大回撤 `7.89%`、胜率 `60.4%`、Profit factor `0.82`；样本切分结果为 2024 年 `-2.31%`、2025-2026 年 `-4.79%`；`recursive-analysis` 未发现指标 lookahead/recursive bias。
- 当前状态：50x 合约趋势/震荡过滤研究策略 `BtcEthFuturesRegime50xStrategy` 和 `BtcEthFuturesRegime50xStrictStrategy` 已完成初步回测；基础 50x 版全样本总收益 `-95.39%`、最大回撤 `95.49%`，极严格版全样本总收益 `-56.55%`、最大回撤 `59.50%`，2025-2026 极严格版仍为 `-34.67%`。因此当前 1h BTC/ETH 动量信号不适合作为 50x 实盘候选，这两个策略只保留为研究样本，不作为 dry-run 默认策略。
- 当前状态：1m 版 50x 合约研究策略 `BtcEthFuturesRegime50xOneMinuteStrategy` 已完成官方 `backtesting`；该策略把 1h 策略里的 24h/72h/7d 语义改为 1440/4320/10080 根 1m K 线，而不是直接把 168 根分钟线误当 7 天。全样本 `2024-01-02 16:00` 到 `2026-06-21 23:59` 结果为 981 笔交易、总收益 `-95.09%`、最大回撤 `95.09%`、胜率 `30.2%`、Profit factor `0.67`，同期 market change 为 `+5.97%`；2025-2026 子样本结果为 658 笔交易、总收益 `-55.96%`、最大回撤 `59.06%`、胜率 `31.2%`、Profit factor `0.70`，同期 market change 为 `-40.63%`。
- 当前状态：杠杆/周期矩阵已完成官方 `backtesting`。慢信号 `24h/72h/7d` 降到 10x 后全样本 209 笔、总收益 `+2.76%`、最大回撤 `2.79%`、胜率 `47.4%`、Profit factor `1.15`，但低于同期 market change `+5.97%`；同信号 20x 为 348 笔、总收益 `+0.60%`、最大回撤 `6.62%`、胜率 `42.5%`、Profit factor `1.01`。快信号 `1h/4h` 的 30x 为 1321 笔、总收益 `-95.16%`、最大回撤 `95.21%`、胜率 `35.0%`、Profit factor `0.65`；同信号 50x 为 632 笔、总收益 `-95.09%`、最大回撤 `95.15%`、胜率 `26.6%`、Profit factor `0.50`。
- 当前状态：入场确认实验已完成官方 `backtesting`。在 `24h/72h/7d + 10x` 基线之上，回踩恢复确认为 173 笔、总收益 `+1.60%`、最大回撤 `3.60%`、PF `1.10`；局部突破确认为 111 笔、总收益 `-1.88%`、最大回撤 `3.62%`、PF `0.86`；短动量堆叠确认为 169 笔、总收益 `+1.74%`、最大回撤 `2.57%`、PF `1.11`；压缩扩张确认为 19 笔、总收益 `-1.15%`、最大回撤 `1.21%`、PF `0.29`；严格组合确认为 130 笔、总收益 `-0.20%`、最大回撤 `2.16%`、PF `0.98`。这些确认没有打败原始 10x 基线的 `+2.76%`。
- 当前状态：方向过滤补充实验显示空头侧更干净。原始 10x 只做多为 95 笔、总收益 `+0.87%`、最大回撤 `2.17%`、PF `1.11`；原始 10x 只做空为 114 笔、总收益 `+1.88%`、最大回撤 `1.50%`、PF `1.20`；回踩确认只做空为 92 笔、总收益 `+2.56%`、最大回撤 `1.16%`、PF `1.35`；动量确认只做空为 91 笔、总收益 `+1.46%`、最大回撤 `1.81%`、PF `1.18`。当前最佳研究候选是 `BtcEthFuturesRegime10xPullbackShortOnlyStrategy`，但仍低于同期 market change `+5.97%`。
- 当前状态：BTC 领先 ETH 实验已完成官方 `backtesting`。该组只交易 ETH，BTC 只作为空头市场过滤器，ETH 仍必须满足自身回踩恢复下跌的进场确认。ETH 自身回踩做空基线为 72 笔、总收益 `+2.47%`、最大回撤 `1.44%`、PF `1.42`；BTC 同步过滤为 16 笔、总收益 `-0.30%`、最大回撤 `0.80%`、PF `0.85`；BTC 领先/延迟 15m 为 13 笔、总收益 `-0.69%`、最大回撤 `0.89%`、PF `0.56`；60m 为 13 笔、总收益 `-1.65%`、最大回撤 `1.65%`、PF `0.22`；240m 为 5 笔、总收益 `-0.74%`、最大回撤 `0.74%`、PF `0.00`。因此当前这种“BTC 慢周期空头信号硬过滤 ETH 入场”的做法会错过 ETH 自身有效机会，未体现可用领先优势。
- 当前结论：第四步的数据、策略改进、回测和 recursive-analysis 已完成；现货熊市过滤版已把 2025-2026 下跌期亏损基本压平，并让全样本转正，但全样本收益仍低于同期 market change `+5.26%`，因此可以进入更长 dry-run 观察，仍不建议直接实盘；合约虽然回撤被压低但仍是负期望，50x 版本在 1h 和 1m 两种口径下都风险不可接受；慢信号降到 10x 后风险显著降低但收益仍跑输基准，20x 已接近无优势；快信号 30x/50x 明确不可用；入场确认本身没有解决收益不足，方向过滤里的“10x 慢信号 + 回踩确认 + 只做空”是目前最干净的合约研究候选，但还不能作为实盘策略；BTC 领先 ETH 的硬过滤实验暂时失败，后续若继续研究，应改为相关性/相对强弱/短窗口冲击响应，而不是用 BTC 慢趋势直接卡 ETH 入场。当前合约回测缺少真实 mark/funding 数据，只能作为方向性评估。
- 为选定交易对和周期下载足够长的历史数据。
- 数据要覆盖上涨、下跌、震荡三类行情。
- 回测必须考虑手续费。
- 不只看总收益，还要看最大回撤、胜率、盈亏比、交易次数、连续亏损。
- 不要只在一个有利时间段上优化。
- 做 walk-forward 或样本外验证。
- 不要直接相信网上现成策略的漂亮回测结果。

## 5. Dry-Run 与前向测试

- 设置 `dry_run` 为 `true`。
- 连续跑几天到几周 dry-run。
- 对比 dry-run 和 backtest 的差异。
- 检查信号延迟、成交假设、limit order 不成交、滑点假设等问题。
- 重启 bot，确认状态恢复正常。
- 确认日志、数据库记录、WebUI 状态你都能看懂。

## 6. 资金与风控

- 第一阶段只使用可以承受亏损的小资金。
- 优先从现货开始，不建议一开始做合约。
- 设置单笔最大投入金额。
- 设置最大同时持仓数量。
- 设置硬止损。
- 定义每日或每周最大亏损停机规则。
- 拉黑流动性差、异常波动、暴涨暴跌的交易对。
- 不要一开始使用高杠杆。
- 不要一开始使用全仓合约。
- 不要让 bot 控制整个交易所账户余额。

## 7. 运维与监控

- 把日志写入文件。
- 配置 Telegram、WebUI 或其他监控方式。
- 明确知道如何暂停、停止、强制退出交易。
- 理解断网、断电、交易所故障、程序崩溃时会发生什么。
- 实盘前确认开仓、止损、退出、取消订单流程都可控。
- 定期备份配置文件和交易数据库。
- 每次策略和配置变更都要留下记录。

## 8. 实盘前最终检查

- dry-run 已经稳定运行。
- 策略没有已知的 lookahead 或 recursive-analysis 问题。
- API key 和私有配置没有进入 Git。
- WebUI 和 REST API 没有暴露到公网。
- 实盘使用全新的数据库，不混用 dry-run 数据库。
- API key 没有提现权限。
- 最大亏损在资金和心理上都可以接受。
- 开始实盘前已经定义好停机条件。

## 9. 第一阶段实盘

- 使用尽可能小的实盘资金。
- 只运行一个策略。
- 只交易少量高流动性交易对。
- 优先做现货。
- 每天复盘交易记录。
- 不要因为前几天盈利就立刻加资金。
- 如果出现异常订单、连续报错、亏损超过预设限制，立即停机。

## 推荐执行顺序

```text
安装 Freqtrade
-> 用 sample strategy 跑 dry-run
-> 改写一个简单的个人策略
-> 回测
-> 运行 lookahead-analysis 和 recursive-analysis
-> dry-run 1-2 周
-> 小资金现货实盘
-> 复盘稳定后再考虑加资金
```

下一步应该先安装 Freqtrade，并完成第一次 dry-run。不要在 dry-run 和安全检查完成前连接实盘 API key。
