<!-- 本文件为中文翻译版，由 AI 根据 docs/bot-basics.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，图片与 includes 引用使用 ../ 指向英文原文档资源。 -->

# Freqtrade 基础

本页介绍 Freqtrade 的工作原理与运行机制的一些基本概念。

## Freqtrade 术语

* **策略（Strategy）**：你的交易策略，告诉机器人该做什么。
* **交易（Trade）**：未平仓头寸（open position）。
* **挂单（Open Order）**：当前已提交到交易所、尚未完全成交的订单。
* **交易对（Pair）**：可交易的交易对，格式通常为 基础币/报价币（例如现货 `XRP/USDT`，合约 `XRP/USDT:USDT`）。
* **时间周期（Timeframe）**：使用的蜡烛长度（例如 `"5m"`、`"1h"` 等）。
* **指标（Indicators）**：技术指标（SMA、EMA、RSI 等）。
* **限价单（Limit order）**：限价单，在设定的限价或更优价格成交。
* **市价单（Market order）**：保证成交，但会根据订单大小移动价格。
* **当前盈亏（Current Profit）**：该笔交易当前待结算（未实现）的盈亏。主要在机器人和 UI 中使用。
* **已实现盈亏（Realized Profit）**：已经实现的盈亏。仅与[部分平仓](strategy-callbacks.md#adjust-trade-position)结合时相关——相关计算逻辑也在该处说明。
* **总盈亏（Total Profit）**：已实现的与未实现的盈亏之和。相对数值（%）是针对该笔交易的总投入来计算的。

## 手续费处理

Freqtrade 的所有盈亏计算都包含手续费。对于回测 / Hyperopt / Dry-run（模拟运行）模式，使用交易所的默认手续费（交易所的最低档费率）。对于实盘操作，使用交易所实际收取的手续费（包含 BNB 返佣等）。

## 交易对命名

Freqtrade 遵循 [ccxt 命名规范](https://docs.ccxt.com/#/README?id=consistency-of-base-and-quote-currencies)。

在不正确的市场中使用了不正确的命名，通常会导致机器人无法识别该交易对，进而出现类似"该交易对不可用"的错误。

### 现货交易对命名

现货交易对的命名为 `基础币/报价币`（例如 `ETH/USDT`）。

### 合约交易对命名

合约交易对的命名为 `基础币/报价币:结算币`（例如 `ETH/USDT:USDT`）。

## 机器人执行逻辑

以 dry-run（模拟运行）或实盘（live）模式启动 freqtrade（使用 `freqtrade trade`）会启动机器人并开启机器人的迭代循环。
这同时也会运行 `bot_start()` 回调。

默认情况下，机器人循环每几秒运行一次（`internals.process_throttle_secs`），并执行以下操作：

* 从持久化（persistence）中读取未平仓交易。
* 计算当前可交易的交易对列表。
* 为交易对列表（包含全部[信息型交易对](strategy-customization.md#get-data-for-non-tradeable-pairs)）下载 OHLCV 数据。
  此步骤每根蜡烛只执行一次，以避免不必要的网络流量。
* 调用 `bot_loop_start()` 策略回调。
* 逐交易对分析策略。
  * 调用 `populate_indicators()`
  * 调用 `populate_entry_trend()`
  * 调用 `populate_exit_trend()`
* 从交易所更新交易的挂单状态。
  * 对成交的订单调用 `order_filled()` 策略回调。
  * 检查挂单超时情况。
    * 对未成交的入场订单调用 `check_entry_timeout()` 策略回调。
    * 对未成交的离场订单调用 `check_exit_timeout()` 策略回调。
    * 对挂单调用 `adjust_order_price()` 策略回调。
      * 对未成交的入场订单调用 `adjust_entry_price()` 策略回调。*仅在未实现 `adjust_order_price()` 时才会调用*
      * 对未成交的离场订单调用 `adjust_exit_price()` 策略回调。*仅在未实现 `adjust_order_price()` 时才会调用*
* 核实现有头寸，并在需要时提交离场订单。
  * 考虑止损（stoploss）、ROI 和离场信号、`custom_exit()` 和 `custom_stoploss()`。
  * 基于 `exit_pricing` 配置项或使用 `custom_exit_price()` 回调来确定离场价格。
  * 在提交离场订单之前，会调用 `confirm_trade_exit()` 策略回调。
* 如果启用，调用 `adjust_trade_position()` 检查未平仓交易的头寸调整，并在需要时提交附加订单。
* 检查是否仍有交易槽位可用（是否已到达 `max_open_trades`）。
* 验证入场信号，尝试建立新头寸。
  * 基于 `entry_pricing` 配置项，或使用 `custom_entry_price()` 回调来确定入场价格。
  * 在保证金（Margin）和合约（Futures）模式下，调用 `leverage()` 策略回调来确定期望的杠杆。
  * 通过调用 `custom_stake_amount()` 回调来确定仓位大小。
  * 在提交入场订单之前，会调用 `confirm_trade_entry()` 策略回调。

该循环会不断重复，直到机器人停止。

## 回测 / Hyperopt 执行逻辑

[回测](backtesting.md) 或 [hyperopt](hyperopt.md) 只执行上述逻辑的一部分，因为大部分交易操作都是完全模拟的。

* 为配置的交易对列表加载历史数据。
* 调用一次 `bot_start()`。
* 计算指标（每个交易对调用一次 `populate_indicators()`）。
* 计算入场 / 离场信号（每个交易对调用一次 `populate_entry_trend()` 和 `populate_exit_trend()`）。
* 逐蜡烛循环，模拟入场和离场点。
  * 调用 `bot_loop_start()` 策略回调。
  * 检查订单超时，通过 `unfilledtimeout` 配置，或通过 `check_entry_timeout()` / `check_exit_timeout()` 策略回调。
  * 对挂单调用 `adjust_order_price()` 策略回调。
    * 对未成交的入场订单调用 `adjust_entry_price()` 策略回调。*仅在未实现 `adjust_order_price()` 时才会调用！*
    * 对未成交的离场订单调用 `adjust_exit_price()` 策略回调。*仅在未实现 `adjust_order_price()` 时才会调用！*
  * 检查交易入场信号（`enter_long` / `enter_short` 列）。
  * 确认交易入场 / 离场（若策略中已实现，则调用 `confirm_trade_entry()` 和 `confirm_trade_exit()`）。
  * 调用 `custom_entry_price()`（若策略中已实现）以确定入场价格（价格会被移动到开盘蜡烛范围内）。
  * 在保证金和合约模式下，调用 `leverage()` 策略回调以确定期望的杠杆。
  * 通过调用 `custom_stake_amount()` 回调来确定仓位大小。
  * 若已启用，则检查未平仓交易的头寸调整，并调用 `adjust_trade_position()` 以确认是否有附加订单请求。
  * 对成交的入场订单调用 `order_filled()` 策略回调。
  * 调用 `custom_stoploss()` 和 `custom_exit()` 以寻找自定义离场点。
  * 对于基于离场信号、自定义离场和部分平仓的离场：调用 `custom_exit_price()` 以确定离场价格（价格会被移动到收盘蜡烛范围内）。
  * 对成交的离场订单调用 `order_filled()` 策略回调。
* 生成回测报告输出

!!! Note
    回测和 Hyperopt 的计算都包含交易所的默认手续费。可以通过指定 `--fee` 参数将自定义手续费传递给回测 / hyperopt。

!!! Warning "回调调用频率"
    回测中每个回调最多每根蜡烛调用一次（`--timeframe-detail` 会将此行为修改为每个细分蜡烛调用一次）。
    大多数回调在实盘中每个迭代都会调用一次（通常约每 5 秒一次）——这可能导致回测与实盘的不一致。
