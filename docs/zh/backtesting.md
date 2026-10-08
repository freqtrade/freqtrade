<!-- 本文件为中文翻译版，由 AI 根据 docs/backtesting.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件（如 exchanges.md），这些文件将在逐步翻译过程中补齐。 -->
<!-- 图片与 includes 引用使用 ../ 指向英文原文档资源，以保证显示正常。 -->

# 回测

本页介绍如何通过回测来验证你的策略表现。

回测需要历史数据可用。
要了解如何获取你感兴趣的交易对和交易所的数据，请前往文档的 [数据下载](data-download.md) 部分。

回测也可以在 [Web 服务器模式](freq-ui.md#backtesting) 下使用，这允许你通过 Web 界面运行回测。

## 回测命令参考

--8<-- "commands/backtesting.md"

## 使用回测测试你的策略

现在你有了良好的入场和出场策略以及一些历史数据，你想要用真实数据来测试它。这就是我们所说的[回测](https://en.wikipedia.org/wiki/Backtesting)。

回测将使用你配置文件中的加密货币（交易对），并默认从 `user_data/data/<exchange>` 加载历史 K 线（OHLCV）数据。
如果没有可用的交易所 / 交易对 / 时间框架组合的数据，回测会要求你先使用 `freqtrade download-data` 下载数据。
有关下载的详细信息，请参阅文档中的 [数据下载](data-download.md) 部分。

回测的结果将确认你的机器人盈利的概率是否大于亏损的概率。

所有利润计算都包含手续费，freqtrade 将使用交易所的默认手续费进行计算。

!!! Warning "在回测中使用动态交易对列表"
    使用动态交易对列表是可行的（并非所有处理器都允许在回测模式下使用），但它依赖于当前的市场状况——这并不能反映交易对列表的历史状态。
    此外，当使用 StaticPairlist 以外的交易对列表时，无法保证回测结果的可重现性。
    请阅读 [交易对列表文档](plugins.md#pairlists) 了解更多信息。

    为了获得可重现的结果，最好通过 [`test-pairlist`](utils.md#test-pairlist) 命令生成交易对列表，并将其用作静态交易对列表。

!!! Note
    默认情况下，Freqtrade 会将回测结果导出到 `user_data/backtest_results`。
    导出的交易数据可用于[进一步分析](#further-backtest-result-analysis)，也可供脚本目录中的[绘图子命令](plotting.md#plot-price-and-indicators)（`freqtrade plot-dataframe`）使用。


### 起始余额

回测需要一个起始余额，可以通过 `--dry-run-wallet <balance>` 或 `--starting-balance <balance>` 命令行参数提供，也可以通过 `dry_run_wallet` 配置项设置。
该金额必须高于 `stake_amount`，否则机器人将无法模拟任何交易。

### 动态投入金额

回测支持[动态投入金额](configuration.md#dynamic-stake-amount)，通过将 `stake_amount` 配置为 `"unlimited"`，这将把起始余额分成 `max_open_trades` 份。
早期交易的利润将导致后续更高的投入金额，从而在回测期间实现利润复利。

### 回测命令示例

使用 5 分钟 K 线（OHLCV）数据（默认）

```bash
freqtrade backtesting --strategy AwesomeStrategy
```

其中 `--strategy AwesomeStrategy` / `-s AwesomeStrategy` 指的是策略的类名，该策略位于 `user_data/strategies` 目录中的 Python 文件内。

---

使用 1 分钟 K 线（OHLCV）数据

```bash
freqtrade backtesting --strategy AwesomeStrategy --timeframe 1m
```

---

提供自定义起始余额 1000（以投入币种计）

```bash
freqtrade backtesting --strategy AwesomeStrategy --dry-run-wallet 1000
```

---

使用不同的磁盘历史 K 线（OHLCV）数据源

假设你从 Binance 交易所下载了历史数据并保存在 `user_data/data/binance-20180101` 目录中。
你可以按如下方式使用这些数据进行回测：

```bash
freqtrade backtesting --strategy AwesomeStrategy --datadir user_data/data/binance-20180101 
```

---

比较多个策略

```bash
freqtrade backtesting --strategy-list SampleStrategy1 AwesomeStrategy --timeframe 5m
```

其中 `SampleStrategy1` 和 `AwesomeStrategy` 指的是策略的类名。

---

阻止将交易导出到文件

```bash
freqtrade backtesting --strategy backtesting --export none --config config.json 
```

仅在你确定不需要进一步绘图或分析结果时使用此选项。

---

将交易导出到指定自定义目录的文件

```bash
freqtrade backtesting --strategy backtesting --export trades --backtest-directory=user_data/custom-backtest-results
```

---

另请阅读关于[策略启动周期](strategy-customization.md#strategy-startup-period)的内容。

---

提供自定义手续费值

有时你的账户享有特定的手续费减免（基于一定账户规模或月交易量的手续费降低），这些对 ccxt 不可见。
为了在回测中考虑这一点，你可以使用 `--fee` 命令行选项将此值提供给回测。
此手续费必须是一个比率，并将被应用两次（一次在交易入场时，一次在交易出场时）。

例如，如果每笔订单的手续费为 0.1%（即以比率表示为 0.001），则你可以按如下方式运行回测：

```bash
freqtrade backtesting --fee 0.001
```

!!! Note
    仅在你想要尝试不同的手续费值时才提供此选项（或相应的配置参数）。默认情况下，回测从交易所的交易对/市场信息中获取默认手续费。

---

使用时间范围运行较小测试集的回测

使用 `--timerange` 参数来更改你想要使用的测试集范围。

例如，使用 `--timerange=20190501-` 选项运行回测将使用你输入数据中从 2019 年 5 月 1 日开始的所有可用数据。

```bash
freqtrade backtesting --timerange=20190501-
```

你也可以指定特定的日期范围。

完整的时间范围规范：

- 使用截至 2018/01/31 的数据：`--timerange=-20180131`
- 使用 2018/01/31 以来的数据：`--timerange=20180131-`
- 使用 2018/01/31 到 2018/03/01 的数据：`--timerange=20180131-20180301`
- 使用 POSIX / 纪元时间戳 1527595200 到 1527618600 之间的数据：`--timerange=1527595200-1527618600`

## 理解回测结果

回测中最重要的是理解结果。

回测结果将如下所示：

```
                                               BACKTESTING REPORT                                                
┏━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          Pair ┃ Trades ┃ Avg Profit % ┃  Tot Profit ┃ Tot Profit % ┃    Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ LTC/USDT:USDT │     16 │         1.01 │      56.882 │         5.69 │        16:16:00 │   16     0     0   100 │
│ ETC/USDT:USDT │     12 │         0.73 │      31.513 │         3.15 │         9:55:00 │   11     0     1  91.7 │
│ ETH/USDT:USDT │      8 │         0.69 │      18.659 │         1.87 │ 1 day, 13:55:00 │    7     0     1  87.5 │
│ XLM/USDT:USDT │     10 │          0.3 │      10.694 │         1.07 │        12:08:00 │    9     0     1  90.0 │
│ BTC/USDT:USDT │      8 │         0.22 │       7.502 │         0.75 │ 3 days, 1:24:00 │    6     0     2  75.0 │
│ XRP/USDT:USDT │      9 │        -0.13 │      -6.837 │        -0.68 │        21:18:00 │    8     0     1  88.9 │
│ DOT/USDT:USDT │      6 │        -0.39 │      -9.169 │        -0.92 │         5:35:00 │    4     0     2  66.7 │
│ ADA/USDT:USDT │      8 │        -1.75 │     -52.089 │        -5.21 │        11:38:00 │    6     0     2  75.0 │
│         TOTAL │     77 │         0.23 │      57.157 │         5.72 │        22:12:00 │   67     0    10  87.0 │
└───────────────┴────────┴──────────────┴─────────────┴──────────────┴─────────────────┴────────────────────────┘
                                             LEFT OPEN TRADES REPORT                                              
┏━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          Pair ┃ Trades ┃ Avg Profit % ┃  Tot Profit ┃ Tot Profit % ┃     Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ BTC/USDT:USDT │      1 │        -4.14 │      -9.930 │        -0.99 │ 17 days, 8:00:00 │    0     0     1     0 │
│ ETC/USDT:USDT │      1 │        -4.24 │     -15.365 │        -1.54 │         10:40:00 │    0     0     1     0 │
│ DOT/USDT:USDT │      1 │        -5.29 │     -19.166 │        -1.92 │         11:30:00 │    0     0     1     0 │
│         TOTAL │      3 │        -4.56 │     -44.461 │        -4.45 │  6 days, 2:03:00 │    0     0     3     0 │
└───────────────┴────────┴──────────────┴─────────────┴──────────────┴──────────────────┴────────────────────────┘
                                              ENTER TAG STATS                                              
┏━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Enter Tag ┃ Entries ┃ Avg Profit % ┃  Tot Profit ┃ Tot Profit % ┃ Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│     OTHER │      77 │         0.23 │      57.157 │         5.72 │     22:12:00 │   67     0    10  87.0 │
│     TOTAL │      77 │         0.23 │      57.157 │         5.72 │     22:12:00 │   67     0    10  87.0 │
└───────────┴─────────┴──────────────┴─────────────┴──────────────┴──────────────┴────────────────────────┘
                                              EXIT REASON STATS                                               
┏━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Exit Reason ┃ Exits ┃ Avg Profit % ┃  Tot Profit ┃ Tot Profit % ┃    Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│         roi │    67 │         1.06 │     245.117 │        24.51 │        15:49:00 │   67     0     0   100 │
│ exit_signal │     4 │        -2.23 │     -31.226 │        -3.12 │  1 day, 8:38:00 │    0     0     4     0 │
│  force_exit │     3 │        -4.56 │     -44.461 │        -4.45 │ 6 days, 2:03:00 │    0     0     3     0 │
│   stop_loss │     3 │       -10.14 │    -112.273 │       -11.23 │  1 day, 3:05:00 │    0     0     3     0 │
│       TOTAL │    77 │         0.23 │      57.157 │         5.72 │        22:12:00 │   67     0    10  87.0 │
└─────────────┴───────┴──────────────┴─────────────┴──────────────┴─────────────────┴────────────────────────┘
                                                      MIXED TAG STATS                                                      
┏━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Enter Tag ┃ Exit Reason ┃ Trades ┃ Avg Profit % ┃  Tot Profit ┃ Tot Profit % ┃    Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│           │         roi │     67 │         1.06 │     245.117 │        24.51 │        15:49:00 │   67     0     0   100 │
│           │ exit_signal │      4 │        -2.23 │     -31.226 │        -3.12 │  1 day, 8:38:00 │    0     0     4     0 │
│           │  force_exit │      3 │        -4.56 │     -44.461 │        -4.45 │ 6 days, 2:03:00 │    0     0     3     0 │
│           │   stop_loss │      3 │       -10.14 │    -112.273 │       -11.23 │  1 day, 3:05:00 │    0     0     3     0 │
│     TOTAL │             │     77 │         0.23 │      57.157 │         5.72 │        22:12:00 │   67     0    10  87.0 │
└───────────┴─────────────┴────────┴──────────────┴─────────────┴──────────────┴─────────────────┴────────────────────────┘
                                   SUMMARY METRICS                                    
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                                 ┃ Value                                     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Backtesting from                       │ 2025-07-01 00:00:00                       │
│ Backtesting to                         │ 2025-08-01 00:00:00                       │
│ Trading Mode                           │ Isolated Futures                          │
│ Max open trades                        │ 3                                         │
│                                        │                                           │
│ Total/Daily Avg Trades                 │ 77 / 2.48                                 │
│ Starting balance                       │ 1000 USDT                                 │
│ Final balance                          │ 1057.157 USDT                             │
│ Absolute profit                        │ 57.157 USDT                               │
│ Total profit %                         │ 5.72%                                     │
│ CAGR %                                 │ 92.41%                                    │
│ Sharpe (closed trades)                 │ 3.89                                      │
│ Sortino (closed trades)                │ 2.57                                      │
│ Calmar (closed trades)                 │ 43.03                                     │
│ SQN                                    │ 0.71                                      │
│ Mean profit p-value                    │ 0.4768                                    │
│ Profit factor                          │ 1.30                                      │
│ Expectancy (Ratio)                     │ 0.74 (0.04)                               │
│ Avg. daily profit                      │ 1.844 USDT                                │
│ Avg. stake amount                      │ 345.478 USDT                              │
│ Market change                          │ 30.51%                                    │
│ Total trade volume                     │ 53390.788 USDT                            │
│                                        │                                           │
│ Long / Short trades                    │ 67 / 10                                   │
│ Long / Short profit %                  │ 9.19% / -3.48%                            │
│ Long / Short profit USDT               │ 91.940 / -34.783                          │
│                                        │                                           │
│ Best Pair                              │ LTC/USDT:USDT 5.69%                       │
│ Worst Pair                             │ ADA/USDT:USDT -5.21%                      │
│ Best trade                             │ XRP/USDT:USDT 2.00%                       │
│ Worst trade                            │ ADA/USDT:USDT -10.17%                     │
│ Best day                               │ 27.031 USDT                               │
│ Worst day                              │ -47.826 USDT                              │
│ Days win/draw/lose                     │ 20 / 6 / 5                                │
│ Min/Max/Avg. Duration Winners          │ 0d 00:35 / 5d 18:15 / 0d 15:49            │
│ Min/Max/Avg. Duration Losers           │ 0d 10:40 / 17d 08:00 / 2d 17:00           │
│ Max Consecutive Wins / Loss            │ 36 / 3                                    │
│ Rejected Entry signals                 │ 258                                       │
│ Entry/Exit Timeouts                    │ 0 / 0                                     │
│                                        │                                           │
│ Min/Max balance (closed trades)        │ 1003.205 USDT / 1151.425 USDT             │
│ Max % of account underwater            │ 8.19%                                     │
│ Absolute drawdown                      │ 94.268 USDT (8.19%)                       │
│ Drawdown duration                      │ 9 days 08:50:00                           │
│ Profit at drawdown start               │ 151.425 USDT                              │
│ Profit at drawdown end                 │ 57.157 USDT                               │
│ Drawdown start                         │ 2025-07-22 15:10:00                       │
│ Drawdown end                           │ 2025-08-01 00:00:00                       │
│                                        │                                           │
│ Wallet based Metrics                   │                                           │
│ Min/Max balance (wallet balance)       │ 1000 USDT / 1151.425 USDT                 │
│ Min/Max balance dates (wallet balance) │ 2025-07-01 00:05:00 / 2025-07-22 15:15:00 │
│ Max % of account underwater (balance)  │ 5.01%                                     │
│ Absolute drawdown (wallet balance)     │ 54.76 USDT (4.76%)                        │
│ Drawdown duration                      │ 7 days 20:35:00                           │
│ Profit at drawdown start               │ 151.425 USDT                              │
│ Profit at drawdown end                 │ 96.664 USDT                               │
│ Drawdown start                         │ 2025-07-22 15:15:00                       │
│ Drawdown end                           │ 2025-07-30 11:50:00                       │
│ Sharpe (daily wallet balance)          │ 4.42                                      │
│ Sortino (daily wallet balance)         │ 4.35                                      │
│ Calmar (daily wallet balance)          │ 136.07                                    │
└────────────────────────────────────────┴───────────────────────────────────────────┘

Backtested 2025-07-01 00:00:00 -> 2025-08-01 00:00:00 | Max open trades : 3
                                                        STRATEGY SUMMARY                                                        
┏━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃       Strategy ┃ Trades ┃ Avg Profit % ┃  Tot Profit ┃ Tot Profit % ┃ Avg Duration ┃  Win  Draw  Loss  Win% ┃       Drawdown ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ SampleStrategy │     77 │         0.23 │      57.157 │         5.72 │     22:12:00 │   67     0    10  87.0 │ 94.268   8.19% │
└────────────────┴────────┴──────────────┴─────────────┴──────────────┴──────────────┴────────────────────────┴────────────────┘

```

### 回测报告表格

第一个表格包含机器人进行的所有交易，包括"未平仓交易"。

最后一行将显示你策略的总体表现：

```
│         TOTAL │     77 │         0.22 │          54.774 │         5.48 │        22:12:00 │   67     0    10  87.0 │
```

机器人进行了 `77` 笔交易，平均持续时间为 `22:12:00`，表现为 `5.48%`（利润），这意味着它从 1000 USDT 的初始资本开始，总共赚取了 `54.774 USDT`。

`Avg Profit %` 列显示所有交易的平均利润。
`Tot Profit %` 列则显示相对于起始余额的总利润百分比。

在上述结果中，我们有 1000 USDT 的起始余额和 54.774 USDT 的绝对利润——因此 `Tot Profit %` 为 `(54.774 / 1000) * 100 ~= 5.48%`。

你的策略表现受入场策略、出场策略以及你设置的 `minimal_roi` 和 `stop_loss` 影响。

例如，如果你的 `minimal_roi` 仅为 `"0":  0.01`，你不能期望机器人获得超过 1% 的利润（因为它每次在交易达到 1% 时就会退出）。

```json
"minimal_roi": {
    "0":  0.01
},
```

另一方面，如果你设置了过高的 `minimal_roi`，如 `"0":  0.55`（55%），机器人几乎没有机会达到这个利润。
因此，请记住，你的表现是策略的所有不同元素、你的配置以及你设置的加密货币交易对的综合结果。

### 未平仓交易表格

第二个表格包含机器人在回测期结束时不得不 `force_exit` 的所有交易，以向你展示完整画面。
这是模拟真实行为所必需的，因为回测期必须在某个时间点结束，而实际上你可以让机器人永远运行。
这些交易也包含在第一个表格中，但为了清晰起见，在此表中单独显示。

### 入场标签统计表格

第三个表格按入场标签（如 `enter_long`、`enter_short`）对交易进行分类，显示每个标签的入场次数、平均利润百分比、以投入币种计的总利润、总利润百分比、平均持续时间，以及赢、平、亏的次数。

### 出场原因统计表格

第四个表格包含出场原因的回顾（如 `exit_signal`、`roi`、`stop_loss`、`force_exit`）。此表格可以告诉你哪个方面需要额外改进（例如，如果许多 `exit_signal` 交易是亏损的，你应该改进出场信号或考虑禁用它）。

### 混合标签统计表格

第五个表格结合了入场标签和出场原因，提供了不同入场标签在特定出场原因下的表现详细视图。这有助于识别哪些入场和出场策略的组合最有效。

### 汇总指标

回测报告的最后一个元素是汇总指标表格。
它包含关于你的策略在回测数据上表现的关键指标。

```
                                   SUMMARY METRICS                                    
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                                 ┃ Value                                     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Backtesting from                       │ 2025-07-01 00:00:00                       │
│ Backtesting to                         │ 2025-08-01 00:00:00                       │
│ Trading Mode                           │ Isolated Futures                          │
│ Max open trades                        │ 3                                         │
│                                        │                                           │
│ Total/Daily Avg Trades                 │ 77 / 2.48                                 │
│ Starting balance                       │ 1000 USDT                                 │
│ Final balance                          │ 1057.157 USDT                             │
│ Absolute profit                        │ 57.157 USDT                               │
│ Total profit %                         │ 5.72%                                     │
│ CAGR %                                 │ 92.41%                                    │
│ Sharpe (closed trades)                 │ 3.89                                      │
│ Sortino (closed trades)                │ 2.57                                      │
│ Calmar (closed trades)                 │ 43.03                                     │
│ SQN                                    │ 0.71                                      │
│ Mean profit p-value                    │ 0.4768                                    │
│ Profit factor                          │ 1.30                                      │
│ Expectancy (Ratio)                     │ 0.74 (0.04)                               │
│ Avg. daily profit                      │ 1.844 USDT                                │
│ Avg. stake amount                      │ 345.478 USDT                              │
│ Market change                          │ 30.51%                                    │
│ Total trade volume                     │ 53390.788 USDT                            │
│                                        │                                           │
│ Long / Short trades                    │ 67 / 10                                   │
│ Long / Short profit %                  │ 9.19% / -3.48%                            │
│ Long / Short profit USDT               │ 91.940 / -34.783                          │
│                                        │                                           │
│ Best Pair                              │ LTC/USDT:USDT 5.69%                       │
│ Worst Pair                             │ ADA/USDT:USDT -5.21%                      │
│ Best trade                             │ XRP/USDT:USDT 2.00%                       │
│ Worst trade                            │ ADA/USDT:USDT -10.17%                     │
│ Best day                               │ 27.031 USDT                               │
│ Worst day                              │ -47.826 USDT                              │
│ Days win/draw/lose                     │ 20 / 6 / 5                                │
│ Min/Max/Avg. Duration Winners          │ 0d 00:35 / 5d 18:15 / 0d 15:49            │
│ Min/Max/Avg. Duration Losers           │ 0d 10:40 / 17d 08:00 / 2d 17:00           │
│ Max Consecutive Wins / Loss            │ 36 / 3                                    │
│ Rejected Entry signals                 │ 258                                       │
│ Entry/Exit Timeouts                    │ 0 / 0                                     │
│                                        │                                           │
│ Min/Max balance (closed trades)        │ 1003.205 USDT / 1151.425 USDT             │
│ Max % of account underwater            │ 8.19%                                     │
│ Absolute drawdown                      │ 94.268 USDT (8.19%)                       │
│ Drawdown duration                      │ 9 days 08:50:00                           │
│ Profit at drawdown start               │ 151.425 USDT                              │
│ Profit at drawdown end                 │ 57.157 USDT                               │
│ Drawdown start                         │ 2025-07-22 15:10:00                       │
│ Drawdown end                           │ 2025-08-01 00:00:00                       │
│                                        │                                           │
│ Wallet based Metrics                   │                                           │
│ Min/Max balance (wallet balance)       │ 1000 USDT / 1151.425 USDT                 │
│ Min/Max balance dates (wallet balance) │ 2025-07-01 00:05:00 / 2025-07-22 15:15:00 │
│ Max % of account underwater (balance)  │ 5.01%                                     │
│ Absolute drawdown (wallet balance)     │ 54.76 USDT (4.76%)                        │
│ Drawdown duration                      │ 7 days 20:35:00                           │
│ Profit at drawdown start               │ 151.425 USDT                              │
│ Profit at drawdown end                 │ 96.664 USDT                               │
│ Drawdown start                         │ 2025-07-22 15:15:00                       │
│ Drawdown end                           │ 2025-07-30 11:50:00                       │
│ Sharpe (daily wallet balance)          │ 4.42                                      │
│ Sortino (daily wallet balance)         │ 4.35                                      │
│ Calmar (daily wallet balance)          │ 136.07                                    │
└────────────────────────────────────────┴───────────────────────────────────────────┘
```

- `Backtesting from` / `Backtesting to`：回测范围（通常通过 `--timerange` 选项定义）。
- `Trading Mode`：现货或期货交易。
- `Max open trades`：`max_open_trades`（或 `--max-open-trades`）的设置——或交易对列表中的交易对数量（取较小值）。
- `Total/Daily Avg Trades`：与回测输出表格中的总交易数相同 / 总交易数除以回测天数（这将为你提供策略预期交易数量的信息）。
- `Starting balance`：起始余额——由 dry-run-wallet（配置或命令行）给出。
- `Final balance`：最终余额——起始余额 + 绝对利润。
- `Absolute profit`：以投入币种计的利润。
- `Total profit %`：总利润。与第一个表格中 `TOTAL` 行的 `Tot Profit %` 一致。计算公式为 `(End capital − Starting capital) / Starting capital`。
- `CAGR %`：复合年增长率。
- `Sharpe (closed trades)`：年化夏普比率，仅包含已平仓交易（忽略有盈亏的未平仓交易）。
- `Sortino (closed trades)`：年化索提诺比率，仅包含已平仓交易（忽略有盈亏的未平仓交易）。
- `Calmar (closed trades)`：年化卡尔马比率，仅包含已平仓交易（忽略有盈亏的未平仓交易）。
- `SQN`：系统质量数（SQN）——由 Van Tharp 提出。
- `Mean profit p-value`：针对"平均每笔交易收益为零"这一零假设的单样本 Student t 检验的双侧 p 值——简而言之，"平均利润是否可以与噪声区分开？"。较小的值（通常标准是低于 `0.05`）意味着观察到的优势不太可能归因于偶然。其底层 t 统计量与 `SQN` 相同。有关实际应用中的解读方式，请参阅下面的注释。
- `Profit factor`：所有盈利交易的利润总和除以所有亏损交易的损失总和。
- `Expectancy (Ratio)`：期望比率，即每笔交易的平均盈利或亏损。负的期望比率意味着你的策略不盈利。
- `Avg. daily profit`：日均利润，计算公式为 `(Total Profit / Backtest Days)`。
- `Avg. stake amount`：平均投入金额，可以是 `stake_amount` 或使用动态投入金额时的平均值。
- `Market change`：回测期间市场的变化。计算方式为所有交易对从第一根 K 线到最后一根 K 线使用"收盘价"列的变化平均值。
- `Total trade volume`：为达到上述利润而在交易所产生的交易量。
- `Long / Short trades`：多头/空头交易数量的拆分（仅在有空头交易时显示）。
- `Long / Short profit %`：多头和空头交易的利润百分比（仅在有空头交易时显示）。
- `Long / Short profit USDT`：多头和空头交易以投入币种计的利润（仅在有空头交易时显示）。
- `Best Pair` / `Worst Pair`：表现最好和最差的交易对（基于总利润百分比），及其对应的 `Tot Profit %`。
- `Best trade` / `Worst trade`：最大的单笔盈利交易和最大的单笔亏损交易。
- `Best day` / `Worst day`：基于日利润的最好和最差的一天。
- `Days win/draw/lose`：盈利/亏损天数（平局通常是无已平仓交易的天数）。
- `Min/Max/Avg. Duration Winners`：盈利交易的最小、最大和平均持续时间。
- `Min/Max/Avg. Duration Losers`：亏损交易的最小、最大和平均持续时间。
- `Max Consecutive Wins / Loss`：最大连续盈利/亏损次数。
- `Rejected Entry signals`：由于达到 `max_open_trades` 而无法执行的入场信号。
- `Entry/Exit Timeouts`：未成交的入场/出场订单（仅在使用自定义定价时适用）。
- `Min/Max balance (closed trades)`：回测期间基于已平仓交易的最低和最高钱包余额。
- `Max % of account underwater`：自模拟开始以来账户从最高点下降的最大百分比。计算公式为 `(Max Balance - Current Balance) / (Max Balance)` 的最大值。
- `Absolute drawdown`：经历的最大绝对回撤，包括相对于账户的百分比，计算公式为 `(Absolute Drawdown) / (DrawdownHigh + startingBalance)`。
- `Absolute drawdown (wallet balance)`：基于未实现余额经历的最大绝对回撤，包括相对于账户的百分比，计算公式为 `(Absolute Drawdown) / (DrawdownHigh + startingBalance)`。
- `Drawdown duration`：最大回撤期的持续时间。
- `Profit at drawdown start` / `Profit at drawdown end`：最大回撤期开始和结束时的利润。
- `Drawdown start` / `Drawdown end`：最大回撤的开始和结束日期时间（也可通过 `plot-dataframe` 子命令进行可视化）。
- `Min/Max balance (wallet balance)`：回测期间的最低和最高钱包余额——包括锁定在未平仓交易中的资金。
- `Min/Max balance dates (wallet balance)`：最低和最高未实现余额发生的日期。
- `Sharpe (wallet balance)`：包含未实现利润的年化夏普比率计算。
- `Sortino (wallet balance)`：包含未实现利润的年化索提诺比率计算。
- `Calmar (wallet balance)`：包含未实现利润的年化卡尔马比率计算。

??? Note "如何解读平均利润 p 值"
    将 p 值视为一个问题的答案：*如果该策略真的没有优势，纯粹的偶然仍然会给你一个至少偏离零这么远的平均每笔交易结果的概率有多大？* 因此，`0.4768` 的值意味着大约有 48% 的概率仅凭随机性就能出现这么大的波动——换句话说，平均利润与运气无法区分。p 值越低，结果是偶然的可能性就越小，一个常见的经验法则是将低于 `0.05`（5% 的概率）的值视为"统计显著"。

    有两个因素使这个检验保持诚实。该检验假设交易是独立且同分布的，但真实策略很少满足这一条件（交易在时间上重叠和聚集），因此该数字是一个*乐观*的下界——真实的不确定性通常更大。而且由于回测和超参数优化会评估许多策略，有些策略仅凭偶然就能获得较低的 p 值，所以一个小的值只能告诉你一个结果难以用噪声来解释；它本身并不能证明存在真正的优势。

!!! Tip "基于钱包的指标"
    "基于钱包的指标"部分下的指标是基于未实现余额计算的，其中包括锁定在未平仓交易中的资金。这提供了策略表现的更全面视图，因为它同时考虑了已实现和未实现的盈亏。

### 每日 / 每周 / 每月 / 每年细分

你可以使用 `--breakdown <>` 开关获取每日、每周、每月或每年的结果概览。

要显示每月和每年的细分，可以使用以下命令：

``` bash
freqtrade backtesting --strategy MyAwesomeStrategy --breakdown month year
```

``` output
                                 MONTH BREAKDOWN
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃      Month ┃ Trades ┃ Tot Profit USDT ┃ Profit Factor ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 31/01/2020 │     12 │          44.451 │          7.28 │   10     0     2  83.3 │
│ 29/02/2020 │     30 │           45.41 │          2.36 │   17     0    13  56.7 │
│ 31/03/2020 │     35 │         142.024 │          2.42 │   14     0    21  40.0 │
│ 30/04/2020 │     67 │         -23.692 │          0.81 │   24     0    43  35.8 │
...
...
│ 30/04/2025 │    203 │          -63.43 │          0.81 │   73     0   130  36.0 │
│ 31/05/2025 │    142 │         104.675 │          1.28 │   59     0    83  41.5 │
│ 30/06/2025 │    177 │          -1.014 │           1.0 │   85     0    92  48.0 │
│ 31/07/2025 │    155 │         232.762 │           1.6 │   63     0    92  40.6 │
└────────────┴────────┴─────────────────┴───────────────┴────────────────────────┘
                                  YEAR BREAKDOWN
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃       Year ┃ Trades ┃ Tot Profit USDT ┃ Profit Factor ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 31/12/2020 │    896 │         868.889 │          1.46 │  351     0   545  39.2 │
│ 31/12/2021 │   1778 │        4487.163 │          1.93 │  745     0  1033  41.9 │
│ 31/12/2022 │   1736 │          938.27 │          1.27 │  698     0  1038  40.2 │
│ 31/12/2023 │   1712 │        1677.126 │          1.68 │  670     0  1042  39.1 │
│ 31/12/2024 │   1609 │        3198.424 │          2.22 │  773     0   836  48.0 │
│ 31/12/2025 │   1042 │         716.174 │          1.33 │  420     0   622  40.3 │
└────────────┴────────┴─────────────────┴───────────────┴────────────────────────┘
```

输出将显示包含所选期间已实现绝对利润（以投入币种计）的表格，以及额外的统计数据，如交易数量、利润因子，以及在该期间实现（已平仓）的赢、平、亏分布。

### 回测结果缓存

为了节省时间，默认情况下回测会重用最近一天内的缓存结果，当回测的策略和配置与之前的回测匹配时。要强制进行新的回测（尽管存在相同运行的现有结果），请指定 `--cache none` 参数。

!!! Warning
    对于开放式时间范围（`--timerange 20210101-`），缓存会自动禁用，因为 freqtrade 无法可靠地确保底层数据未更改。如果原始回测在末尾有缺失数据（通过下载更多数据修复），它也可能在不应该使用缓存结果时使用缓存结果。
    在这种情况下，请使用 `--cache none` 一次以强制进行全新的回测。

### 进一步的回测结果分析

为了进一步分析你的回测结果，freqtrade 默认会将交易导出到文件。
然后你可以加载交易以执行进一步分析，如[数据分析](strategy_analysis_example.md#load-backtest-results-to-pandas-dataframe)回测部分所示。

此外，你可以在 [Web 服务器模式](freq-ui.md#backtesting) 下使用 freqtrade，在 Web 界面中可视化回测结果。
此模式还允许你加载现有的回测结果，因此你可以无需再次运行回测即可分析它们。
对于此模式——`--notes "<notes>"` 可用于向回测结果添加注释，这些注释将在 Web 界面中显示。

### 回测输出文件

freqtrade 生成的输出文件是一个 zip 文件，包含以下文件：

- JSON 格式的回测报告
- Feather 格式的市场变化数据
- 策略文件的副本
- 策略参数的副本（如果使用了参数文件）
- 配置文件的清理副本

这将确保结果可重现——前提是相同的数据可用。

zip 文件中仅包含策略文件和配置文件，不包含最终的依赖项。

## 回测做出的假设

由于回测缺乏关于 K 线内部发生什么的详细信息，它需要做一些假设：

- 遵守交易所[交易限制](#trading-limits-in-backtesting)
- 入场以开盘价发生，除非指定了自定义价格逻辑
- 只要价格在 K 线的高/低范围内，所有订单都以请求价格成交（无滑点）
- 出场信号出场发生在下一根 K 线的开盘价
- 出场释放其交易槽位用于不同交易对的新交易
- 出场信号优先于止损，因为出场信号假设在 K 线开盘时触发
- ROI
  - 出场与最高价比较——但使用 ROI 值（例如 ROI = 2%，最高价 = 5%——所以出场将在 2%）
  - 出场永远不会"低于 K 线"，因此 2% 的 ROI 可能导致在 2.4% 出场（如果最低价在 2.4% 的利润处）
  - 在触发 K 线上生效的 ROI 条目（例如 1 小时 K 线的 `120: 0.02`，从 `60: 0.05`）将使用 K 线的开盘价作为出场价格
  - 由 `<N>=-1` ROI 条目引起的强制出场使用最低价作为出场值，除非 N 恰好在 K 线开盘时（例如 1 小时 K 线的 `120: -1`）
- 止损出场恰好在止损价格发生，即使最低价更低，但损失将比止损价格高 `2 * 手续费`
- 止损在一根 K 线内优先于 ROI 评估。因此你经常可以看到比在模拟/实盘交易模式下使用相同策略获得的结果更多的 `stoploss` 出场原因的交易
- 最低价发生在最高价之前用于止损，优先保护资金
- 移动止损
  - 仅当移动止损低于 K 线最低价时才会调整（否则它将被触发）
  - 在触发移动止损的交易入场 K 线上，假设使用"最小偏移"（`stop_positive_offset`）（而不是最高价）——止损从这一点计算。此规则不适用于自定义止损场景，因为没有关于止损逻辑的可用信息。
  - 最高价先发生——调整止损
  - 最低价使用调整后的止损（因此具有较大高-低价差的出场被正确回测）
  - ROI 优先于移动止损应用，确保在 ROI 和移动止损都适用时利润被"上限限制"在 ROI
- 出场原因不解释交易是正还是负，只解释是什么触发了出场（如果使用负的 ROI 值，这可能看起来很奇怪）
- 评估顺序（如果同一 K 线上发生多个信号）
  - 出场信号
  - 止损
  - ROI
  - 移动止损
- 仓位反转（仅限期货）发生在现有交易平仓的 K 线上触发了与平仓方向相反的入场信号时。

考虑到这些假设，回测试图尽可能接近真实交易。然而，回测**永远**无法替代在模拟模式下运行策略。
此外，请记住过去的结果不能保证未来的成功。

除了上述假设外，策略作者还应仔细阅读[常见错误](strategy-customization.md#common-mistakes-when-developing-strategies)部分，以避免在回测中使用在真实市场条件下不可用的数据。

### 回测中的交易限制

交易所有一定的交易限制，如最小（和最大）基础货币，或最小/最大投入（报价）货币。
这些限制通常在交易所文档中列为"交易规则"或类似名称，在不同交易对之间可能差异很大。

回测（以及实盘和模拟）确实遵守这些限制，并确保止损可以设置在此值以下——因此该值将略高于交易所指定的值。
然而，Freqtrade 没有关于历史限制的信息。

这可能导致交易限制因使用历史价格而被抬高，导致最小金额 > 50\$。

例如：

BTC 最小可交易数量为 0.001。
BTC 今天交易价格为 22.000\$（0.001 BTC 与此相关）——但回测期间包括高达 50.000\$ 的价格。
今天的最小值将是 `0.001 * 22_000`——即 22\$。
然而在某些历史环境中，该限制也可能是 50\$——基于 `0.001 * 50_000`。

#### 交易精度限制

大多数交易所对价格和数量都有精度限制，因此你不能购买 1.0020401 个某交易对，或以 1.24567123123 的价格购买。
相反，这些价格和数量将根据交易所定义进行四舍五入或截断到定义的交易精度。
上述值可能例如被四舍五入为数量 1.002，价格 1.24567。

这些精度值基于当前的交易所限制（如[上一节](#trading-limits-in-backtesting)所述），因为历史精度限制不可用。

## 提高回测精度

回测的一个主要限制是它无法知道价格在 K 线内如何移动（最高价在收盘价之前，还是反之亦然？）。
因此假设你使用 1 小时时间框架运行回测，该 K 线将有 4 个价格（开盘价、最高价、最低价、收盘价）。

虽然回测确实对此做了一些假设（如上所述）——但这永远不会完美，并且总是会有某种方式的偏差。
为了缓解这一点，freqtrade 可以使用更低（更快）的时间框架来模拟 K 线内运动。

要使用此功能，你可以在常规回测命令后附加 `--timeframe-detail 5m`。

``` bash
freqtrade backtesting --strategy AwesomeStrategy --timeframe 1h --timeframe-detail 5m
```

这将加载 1 小时数据（主时间框架）以及 5 分钟数据（详细时间框架）用于所选时间范围。
策略将使用 1 小时时间框架进行分析。
可能发生活动的 K 线（有活跃信号、交易对在交易中）将在 5 分钟时间框架下评估。
这将允许更准确地模拟 K 线内运动——并且可能导致不同的结果，特别是在较高时间框架上。

入场通常仍将发生在主 K 线的开盘价，但释放的交易槽位可能会更早释放（如果出场信号在 5 分钟 K 线上触发），然后可以用于不同交易对的新交易。

所有回调函数（`custom_exit()`、`custom_stoploss()` 等）将在交易开仓后为每根 5 分钟 K 线运行（因此在上面 1 小时时间框架和 5 分钟详细时间框架的示例中运行 12 次）。

`--timeframe-detail` 必须小于原始时间框架，否则回测将无法启动。

显然，这将需要更多内存（5 分钟数据比 1 小时数据更大），并且还会影响运行时间（取决于交易数量和交易持续时间）。
此外，数据必须已可用 / 已下载。

!!! Tip
    你可以将此功能用作策略开发的最后部分，以确保你的策略没有利用[回测假设](#assumptions-made-by-backtesting)中的某一个。在此模式下表现同样良好的策略在模拟/实盘模式下也有很大机会表现良好（尽管只有前向测试（模拟模式）才能真正确认一个策略）。

??? Sample "极端差异示例"
    在极端示例上使用 `--timeframe-detail`（以下所有交易对在 10:00 K 线有入场信号）可能导致以下回测交易序列（max_open_trades 为 1）：

    | Pair | Entry Time | Exit Time | Duration |
    |------|------------|-----------| -------- |
    | BTC/USDT | 2024-01-01 10:00:00 | 2021-01-01 10:05:00 | 5m |
    | ETH/USDT | 2024-01-01 10:05:00 | 2021-01-01 10:15:00 | 10m |
    | XRP/USDT | 2024-01-01 10:15:00 | 2021-01-01 10:30:00 | 15m |
    | SOL/USDT | 2024-01-01 10:15:00 | 2021-01-01 11:05:00 | 50m |
    | BTC/USDT | 2024-01-01 11:05:00 | 2021-01-01 12:00:00 | 55m |

    不使用 timeframe-detail 时，这将如下所示：

    | Pair | Entry Time | Exit Time | Duration |
    |------|------------|-----------| -------- |
    | BTC/USDT | 2024-01-01 10:00:00 | 2021-01-01 11:00:00 | 1h |
    | BTC/USDT | 2024-01-01 11:00:00 | 2021-01-01 12:00:00 | 1h |

    差异是显著的，因为没有详细数据，每根 K 线只评估前 `max_open_trades` 个信号，并且交易槽位仅在 K 线结束时释放，允许在下一根 K 线开仓新交易。


## 回测多个策略

要比较多个策略，可以向回测提供一个策略列表。

这限制为每次运行 1 个时间框架值。然而，数据只从磁盘加载一次，因此如果你有多个
想要比较的策略，这将提供不错的运行时提升。

所有列出的策略需要在同一目录中，除非还指定了 `--recursive-strategy-search`，此时策略目录中的子目录也会被考虑。

``` bash
freqtrade backtesting --timerange 20180401-20180410 --timeframe 5m --strategy-list Strategy001 Strategy002 --export trades
```

这将把结果保存到 `user_data/backtest_results/backtest-result-<datetime>.json`，包括 `Strategy001` 和 `Strategy002` 的结果。
将有一个额外的表格比较不同策略的赢/亏（与第一个表格中的"Total"行相同）。
所有策略的详细输出将依次可用，因此请确保向上滚动查看每个策略的详细信息。

```
================================================== STRATEGY SUMMARY ===================================================================
| Strategy    |  Trades |   Avg Profit % |   Tot Profit BTC |   Tot Profit % | Avg Duration   |  Wins |  Draws | Losses | Drawdown % |
|-------------+---------+----------------+------------------+----------------+----------------+-------+--------+--------+------------|
| Strategy1   |     429 |           0.36 |       0.00762792 |          76.20 | 4:12:00        |   186 |      0 |    243 |       45.2 |
| Strategy2   |    1487 |          -0.13 |      -0.00988917 |         -98.79 | 4:43:00        |   662 |      0 |    825 |     241.68 |
```

## 下一步

太好了，你的策略是盈利的。如果机器人能告诉你策略使用的最佳参数呢？
你的下一步是学习[如何使用 Hyperopt 找到最佳参数](hyperopt.md)