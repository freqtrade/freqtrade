# 止损（Stop Loss）

`stoploss` 配置参数是一个应触发卖出的亏损比率。例如，值 `-0.10` 将在某笔交易的利润跌破 -10% 时导致立即卖出。此参数是可选的。止损计算确实包含手续费，因此 -10% 的止损正好放置在入场点下方 10% 处。

大多数策略文件已经包含了最优的 `stoploss` 值。

!!! Info
    本文件中提到的所有止损属性都可以在策略或配置中设置。
    <ins>配置值将覆盖策略值。</ins>

## 交易所止损 / Freqtrade 止损

这些止损模式可以是 *on exchange*（交易所端）或 *off exchange*（机器人端）。

这些模式可以使用以下值进行配置：

``` python
    'emergency_exit': 'market',
    'stoploss_on_exchange': False
    'stoploss_on_exchange_interval': 60,
    'stoploss_on_exchange_limit_ratio': 0.99
```

交易所止损仅受以下交易所支持，并非所有交易所都同时支持 stop-limit 和 stop-market。如果只有一种模式可用，订单类型将被忽略。

??? info "支持的交易所和止损类型"

    --8<-- "includes/exchange-features.md"

!!! Note "过紧的止损"
    使用交易所止损时，请勿设置过低/过紧的止损值！
    如果设置得过低/过紧，你将面临订单无法成交的更大风险，止损将不起作用。

!!! Warning "过松的止损"
    使用交易所止损并设置非常宽的止损（例如 -1）可能会由于交易所限制而失败，无法在交易所上放置止损订单。
    在这种情况下，机器人将回退到使用 `emergency_exit` 订单类型来放置市价订单，因为放置止损订单失败了。
    Freqtrade 目前没有实现限制来避免这种情况，所以请确保你的止损值在交易所的合理限制范围内，或者禁用交易所止损。

### 交易所止损使用哪种订单类型？

交易所止损使用的订单类型由 `stoploss` 值和交易所能力决定。
如果你选择的交易所同时支持 stop-limit 和 stop-market 订单，那么 `stoploss` 值将决定交易所止损使用哪种订单类型。
如果你的交易所仅支持这两种订单类型中的一种，你必须相应地配置你的 `stoploss` 值，否则机器人将无法启动。

### 我应该使用哪种订单类型作为交易所止损？

如果我们将两种止损订单类型翻译成人类语言——它们会是这样的：

* **stoploss-market（止损市价）** -> “当止损触发时，无论什么价格，让我赶紧离场”。
* **stoploss-limit（止损限价）** -> “当止损触发时，在止损价格下方 x% 放置一个限价订单。我最坏情况下接受 'stoploss + 1%' 的亏损——但如果价格进一步跳空——我接受等待价格回落到我这里，这可能导致比 'stoploss + 1%' 大得多的亏损”。

因此，我们建议尽可能使用 stoploss-market 订单，因为止损的主要目的是在市场崩盘时将你带出仓位，而在这种情况下，你会希望以最佳可用价格立即退出仓位，而不是冒险让限价订单无法成交并可能招致更大的亏损。选择最终取决于你，但请意识到使用 stoploss-limit 订单的风险，尤其是在波动的市场中。

### stoploss_on_exchange 和 stoploss_on_exchange_limit_ratio

启用或禁用交易所止损。
如果止损是 *on exchange*（交易所端），这意味着一旦买入订单成交，就会在交易所立即放置一个止损限价订单。这将保护你免受市场的突然崩盘，因为订单执行纯粹发生在交易所内部，并且没有潜在的网络开销。

如果 `stoploss_on_exchange` 使用限价订单，交易所需要 2 个价格，stoploss_price 和 Limit price。
`stoploss` 定义了放置限价订单的止损价格——并且限价应该略微低于此价格。
如果交易所同时支持限价和市价止损订单，那么 `stoploss` 的值将用于确定止损类型。

计算示例：我们以 100$ 买入资产。
止损价格为 95$，那么限价将是 `95 * 0.99 = 94.05$` - 因此限价订单成交可能发生在 95$ 和 94.05$ 之间。

例如，假设止损在交易所，并且启用了追踪止损，且市场正在上涨，那么机器人会自动取消之前的止损订单，并放置一个新的、止损值高于之前止损订单的订单。

!!! Note
    如果 `stoploss_on_exchange` 已启用，并且止损在交易所上被手动取消，那么机器人将创建一个新的止损订单。

### stoploss_on_exchange_interval

在使用交易所止损的情况下，还有另一个参数叫做 `stoploss_on_exchange_interval`。这配置了机器人检查止损并在必要时更新它的间隔（以秒为单位）。
机器人不能每 5 秒（每次迭代）都执行这些操作，否则它会被交易所封禁。
因此，此参数将告诉机器人应该多久更新一次止损订单。默认值为 60（1 分钟）。
如果你不小心取消了止损订单，同样的逻辑将重新在交易所上应用一个止损订单。

### stoploss_price_type

!!! Warning "仅适用于合约（futures）"
    `stoploss_price_type` 仅适用于合约市场（在支持它的交易所上）。
    Freqtrade 将在启动时对此设置执行验证，如果为你的交易所选择了无效设置，将无法启动。
    支持的价格类型在每个交易所之间会有所不同。请向你的交易所查询它支持哪些价格类型。
    在现货（spot）市场中，此设置被忽略且不被验证，因为大多数交易所对现货市场的止损订单仅支持一种价格类型。

交易所上的合约止损可以根据不同的价格类型触发。这些价格在交易所术语中的命名通常有所不同，但通常是围绕 "last"（或 "contract price"）、"mark" 和 "index" 的某种表述。

此设置的可接受值为 `"last"`、`"mark"` 和 `"index"` - freqtrade 会自动将它们转换为相应的 API 类型，并相应地放置[交易所止损](#stoploss_on_exchange-and-stoploss_on_exchange_limit_ratio)订单。

### force_exit

`force_exit` 是一个可选值，默认与 `exit` 相同，用于从 Telegram 或 Rest API 发送 `/forceexit` 命令时。

### force_entry

`force_entry` 是一个可选值，默认与 `entry` 相同，用于从 Telegram 或 Rest API 发送 `/forceentry` 命令时。

### emergency_exit

`emergency_exit` 是一个可选值，默认是 `market`，用于创建交易所止损订单失败时。
以下是如果未在策略或配置文件中更改时使用的默认值。

来自策略文件的示例：

``` python
order_types = {
    "entry": "limit",
    "exit": "limit",
    "emergency_exit": "market",
    "stoploss": "market",
    "stoploss_on_exchange": True,
    "stoploss_on_exchange_interval": 60,
    "stoploss_on_exchange_limit_ratio": 0.99
}
```

## 止损类型

目前该机器人包含以下止损支持模式：

1. 静态止损（Static stop loss）。
2. 追踪止损（Trailing stop loss）。
3. 追踪止损，自定义的正数亏损（custom positive loss）。
4. 仅在交易达到某个偏移量后才追踪止损。
5. [自定义止损函数](strategy-callbacks.md#custom-stoploss)

### 静态止损

这非常简单，你定义一个止损为 x（作为价格的比率，即 x * 100% 的价格）。一旦亏损超过定义的亏损，这将尝试卖出资产。

止损示例：

``` python
    stoploss = -0.10
```

例如，简化的数学计算：

* 机器人以 100$ 的价格买入资产
* 止损定义为 -10%
* 一旦资产跌破 90$，止损就会触发

### 追踪止损

此功能的初始值是 `stoploss`，就像你定义静态止损一样。
要启用追踪止损：

``` python
    stoploss = -0.10
    trailing_stop = True
```

这将现在激活一个算法，该算法会在你的资产价格每次上涨时自动将止损上移。

例如，简化的数学计算：

* 机器人以 100$ 的价格买入资产
* 止损定义为 -10%
* 一旦资产跌破 90$，止损就会触发
* 假设资产现在上涨到 102$
* 止损现在将是 102$ 的 -10% = 91.8$
* 现在资产价值跌至 101\$，止损仍将是 91.8$，并将在 91.8$ 触发。

总而言之：止损将被调整为始终是所观察到的最高价格的 -10%。

### 追踪止损，不同的正数亏损

当你买入处于亏损状态时（买入 - 手续费），你也可以有一个默认的止损，但一旦你达到正的结果（或你定义的偏移量），系统将使用一个具有不同值的新止损。例如，你的默认止损是 -10%，但一旦你达到盈利（例如 0.1%），就会使用不同的追踪止损。

!!! Note
    如果你希望止损仅在你达到盈亏平衡或盈利时才更改（这是大多数用户想要的），请参阅下一节[启用偏移量](#trailing-stop-loss-only-once-the-trade-has-reached-a-certain-offset)。

这两个值都需要将 `trailing_stop` 设置为 true，并且 `trailing_stop_positive` 带有一个值。

``` python
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False  # 默认值 - 此示例不需要
```

例如，简化的数学计算：

* 机器人以 100$ 的价格买入资产
* 止损定义为 -10%
* 一旦资产跌破 90$，止损就会触发
* 假设资产现在上涨到 102$
* 止损现在将是 102$ 的 -2% = 99.96$（99.96$ 的止损将被锁定，并将随着资产价格的增量以 -2% 跟随）
* 现在资产价值跌至 101\$，止损仍将是 99.96$，并将在 99.96$ 触发

0.02 将转换为 -2% 的止损。
在此之前，`stoploss` 用于追踪止损。

!!! Tip "使用偏移量来更改你的止损"
    使用 `trailing_stop_positive_offset` 以确保你的新追踪止损将处于盈利状态，方法是将 `trailing_stop_positive_offset` 设置得高于 `trailing_stop_positive`。那么你的第一个新止损值将已经锁定了利润。

    简化数学计算示例：

    ``` python
        stoploss = -0.10
        trailing_stop = True
        trailing_stop_positive = 0.02
        trailing_stop_positive_offset = 0.03
    ```

    * 机器人以 100$ 的价格买入资产
    * 止损定义为 -10%，因此一旦资产跌破 90$，止损就会触发
    * 假设资产现在上涨到 102$
    * 止损现在将在 91.8$ - 最高观察汇率下方 10%
    * 假设资产现在上涨到 103.5$（高于配置的偏移量）
    * 止损现在将是 103.5$ 的 -2% = 101.43$
    * 现在资产价值跌至 102\$，止损仍将是 101.43$，并将在价格跌破 101.43$ 时触发

### 仅在交易达到某个偏移量后才追踪止损

你也可以保持静态止损，直到达到偏移量，然后在市场转向时追踪交易以获取利润。

如果 `trailing_only_offset_is_reached = True`，则仅在达到偏移量后才激活追踪止损。在此之前，止损保持在配置的 `stoploss` 并且不追踪。
将此值保持为 `trailing_only_offset_is_reached=False` 将允许追踪止损在资产价格上涨超过初始入场价格后立即开始追踪。

此选项可以与或不与 `trailing_stop_positive` 一起使用，但使用 `trailing_stop_positive_offset` 作为偏移量。

配置（偏移量是买入价 + 3%）：

``` python
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
```

例如，简化的数学计算：

* 机器人以 100$ 的价格买入资产
* 止损定义为 -10%
* 一旦资产跌破 90$，止损就会触发
* 止损将保持在 90$，除非资产上涨到配置的偏移量或以上
* 假设资产现在上涨到 103$（我们配置了偏移量的地方）
* 止损现在将是 103$ 的 -2% = 100.94$
* 现在资产价值跌至 101\$，止损仍将是 100.94$，并将在 100.94$ 触发

!!! Tip
    确保此值（`trailing_stop_positive_offset`）低于最小 ROI，否则最小 ROI 会先应用并卖出交易。

## 止损与杠杆

止损应该被视为“这笔交易的风险”——因此在 100$ 交易上 10% 的止损意味着你愿意在这笔交易上损失 10$（10%），如果价格向下移动 10%，这将触发。

使用杠杆时，应用相同的原则——止损定义交易的风险（你愿意损失的金额）。

因此，10x 交易上 10% 的止损将在 1% 的价格变动时触发。
如果你的 stake 金额（自有资本）是 100$ - 这笔交易在 10x（杠杆后）将是 1000$。
如果价格移动 1% - 你已经损失了自有资本的 10$ - 因此在这种情况下将触发止损。

请务必注意这一点，并避免使用过紧的止损（在 10x 杠杆下，10% 的风险可能太小，无法让交易“喘口气”）。

## 更改未平仓交易的止损

可以通过更改配置或策略中的值并使用 `/reload_config` 命令（或者，完全停止并重新启动机器人也可以）来更改未平仓交易的止损。

新的止损值将应用于未平仓交易（并将生成相应的日志消息）。

### 限制

如果启用了 `trailing_stop` 并且止损已经被调整，则无法更改止损值。
