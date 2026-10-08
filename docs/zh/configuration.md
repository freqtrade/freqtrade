<!-- 本文件为中文翻译版，由 AI 根据 docs/configuration.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# 配置机器人

Freqtrade 有许多可配置的功能和可能性。
默认情况下，这些设置都是通过配置文件（见下文）配置的。

## Freqtrade 配置文件

机器人在运行过程中使用一组配置参数，它们共同构成了机器人的配置。机器人通常从文件（Freqtrade 配置文件）中读取其配置。

默认情况下，机器人会从当前工作目录下的 `config.json` 文件加载配置。

你可以通过 `-c/--config` 命令行选项指定机器人所使用的不同配置文件。

如果你使用的是[快速开始](docker_quickstart.md#docker-quick-start)方法安装机器人，安装脚本应该已经为你创建了默认的配置文件（`config.json`）。

如果默认配置文件没有被创建，我们建议使用 `freqtrade new-config --config user_data/config.json` 来生成一个基础配置文件。

Freqtrade 配置文件采用 JSON 格式编写。

除了标准 JSON 语法外，你还可以在配置文件中使用单行 `// ...` 和多行 `/* ... */` 注释，以及列表末尾的逗号（trailing commas）。

如果你不熟悉 JSON 格式也不用担心——只需用你喜欢的编辑器打开配置文件，修改你需要的参数，保存改动，最后重启机器人；或者如果它之前已停止，就用你修改后的配置重新运行它。机器人会在启动时校验配置文件的语法，如果你在编辑时出了错，它会发出警告，并指出有问题的行。

### 环境变量

通过环境变量设置 Freqtrade 配置中的选项。这会覆盖配置或策略中相应的取值。

环境变量必须以 `FREQTRADE__` 为前缀才能被加载到 freqtrade 配置中。

`__` 用作层级分隔符，因此使用的格式应对应 `FREQTRADE__{section}__{key}`。
因此——定义为 `export FREQTRADE__STAKE_AMOUNT=200` 的环境变量，会产生 `{stake_amount: 200}`。

一个更复杂的例子可能是 `export FREQTRADE__EXCHANGE__KEY=<yourExchangeKey>`，用于将你的交易所 key 保密。这会将取值移动到配置的 `exchange.key` 段。
使用此方案，所有配置设置也都可以通过环境变量来设置。

请注意，环境变量会覆盖你配置中的相应设置，但命令行参数始终优先。

常见示例：

``` bash
FREQTRADE__TELEGRAM__CHAT_ID=<telegramchatid>
FREQTRADE__TELEGRAM__TOKEN=<telegramToken>
FREQTRADE__EXCHANGE__KEY=<yourExchangeKey>
FREQTRADE__EXCHANGE__SECRET=<yourExchangeSecret>
```

JSON 列表会以 json 形式解析——因此你可以使用以下命令来设置一个交易对列表：

``` bash
export FREQTRADE__EXCHANGE__PAIR_WHITELIST='["BTC/USDT", "ETH/USDT"]'
```

!!! Note
    检测到的环境变量会在启动时记录到日志中——所以如果你找不到某个取值为什么和你基于配置认为的不一致，请确认它不是从环境变量加载的。

!!! Tip "校验合并结果"
    你可以使用 [show-config 子命令](utils.md#show-config) 来查看最终的合并配置。

??? Warning "加载顺序"
    环境变量在初始配置之后加载。因此，你无法通过环境变量来提供配置文件的路径。请使用 `--config path/to/config.json`。
    这在一定程度上也适用于 `user_dir`——虽然用户目录可以通过环境变量设置，但配置**不会**从那个位置加载。

### 多个配置文件

机器人可以指定并使用多个配置文件，或者机器人可以从进程标准输入流中读取其配置参数。

你可以在 `add_config_files` 中指定附加配置文件。该参数中指定的文件会被加载，并与初始配置文件合并。这些文件是相对于初始配置文件来解析的。
这类似于使用多个 `--config` 参数，但使用起来更简单，因为你不必为所有命令都指定全部文件。

!!! Tip "校验合并结果"
    你可以使用 [show-config 子命令](utils.md#show-config) 来查看最终的合并配置。

!!! Tip "用多个配置文件保守秘密"
    你可以使用第二个包含你的密钥的配置文件。这样你就可以分享你的"主"配置文件，同时仍然保守你自己的 API key。
    第二个文件应当只指定你打算覆盖的内容。
    如果某个 key 出现在多个配置中，则"最后指定的配置"获胜（在上例中为 `config-private.json`）。

    对于一次性命令，你也可以使用下面的语法，指定多个 "--config" 参数。

    ``` bash
    freqtrade trade --config user_data/config1.json --config user_data/config-private.json <...>
    ```

    下面等价于上面的例子——但在配置中放入 2 个配置文件，便于复用。

    ``` json title="user_data/config.json"
    "add_config_files": [
        "config1.json",
        "config-private.json"
    ]
    ```

    ``` bash
    freqtrade trade --config user_data/config.json <...>
    ```

??? Note "配置冲突处理"
    如果相同的配置项同时出现在 `config.json` 和 `config-import.json` 中，则父配置获胜。
    在下面的例子中，合并后的 `max_open_trades` 会是 3——因为可复用的"import"配置中的这个 key 被覆盖了。

    ``` json title="user_data/config.json"
    {
        "max_open_trades": 3,
        "stake_currency": "USDT",
        "add_config_files": [
            "config-import.json"
        ]
    }
    ```

    ``` json title="user_data/config-import.json"
    {
        "max_open_trades": 10,
        "stake_amount": "unlimited",
    }
    ```

    合并后的配置结果为：

    ``` json title="Result"
    {
        "max_open_trades": 3,
        "stake_currency": "USDT",
        "stake_amount": "unlimited"
    }
    ```

    如果 `add_config_files` 段中有多个文件，则它们被视为同一层级，以最后出现的覆盖较早的配置（除非父配置已经定义了这样的 key）。

## 编辑器自动补全与校验

如果你使用的编辑器支持 JSON schema，你可以通过在配置文件顶部添加以下行，使用 Freqtrade 提供的 schema 来获得配置文件的自动补全和校验：

``` json
{
    "$schema": "https://schema.freqtrade.io/schema.json",
}
```

??? Note "开发版 schema"
    开发版 schema 地址为 `https://schema.freqtrade.io/schema_dev.json`——不过为了获得最佳体验，我们建议坚持使用稳定版。

## 配置参数

下表将列出所有可用的配置参数。

Freqtrade 也可以通过命令行（CLI）参数加载许多选项（详情请查看各命令的 `--help` 输出）。

### 配置选项优先级

所有选项的优先级如下：

* 命令行参数覆盖其他任何选项
* [环境变量](#environment-variables)
* 配置文件按序使用（最后一个文件获胜）并覆盖策略配置。
* 策略配置仅在不通过配置或命令行参数设置时才使用。这些选项在下表中标记为 [策略覆盖](#parameters-in-the-strategy)。

### 参数表

必填参数标记为 **Required**，意味着它们必须通过某种方式之一进行设置。

|  参数 | 说明 |
|------------|-------------|
| `max_open_trades` | **必填。** 机器人允许持有的未平仓交易数量。每个交易对只允许一笔未平仓交易，因此你的交易对列表长度是一个可能适用的额外限制。如果为 -1 则忽略该限制（即潜在地无限多笔未平仓交易，受交易对列表限制）。[详见下文](#configuring-amount-per-trade)。[策略覆盖](#parameters-in-the-strategy)。<br> **数据类型：** 正整数或 -1。 |
| `stake_currency` | **必填。** 用于交易的加密货币。<br> **数据类型：** 字符串 |
| `stake_amount` | **必填。** 机器人每笔交易将使用的加密货币数量。设为 `"unlimited"` 可让机器人使用全部可用余额。[详见下文](#configuring-amount-per-trade)。<br> **数据类型：** 正浮点数或 `"unlimited"`。 |
| `tradable_balance_ratio` | 机器人允许交易的账户总余额占比。[详见下文](#configuring-amount-per-trade)。<br>*默认为 `0.99`（99%）。*<br> **数据类型：** 介于 `0.1` 和 `1.0` 之间的正浮点数。 |
| `available_capital` | 机器人的可用起始资金。在同一交易所账户上运行多个机器人时很有用。[详见下文](#configuring-amount-per-trade)。<br> **数据类型：** 正浮点数。 |
| `amend_last_stake_amount` | 如有必要，使用减少后的末笔仓位数量。<br>*默认为 `false`。* <br> **数据类型：** 布尔值 |
| `last_stake_amount_min_ratio` | 定义必须保留并执行的最小仓位数量。仅适用于末笔仓位被调减时（即 `amend_last_stake_amount` 设为 `true` 时）。[详见下文](#configuring-amount-per-trade)。<br>*默认为 `0.5`。* <br> **数据类型：** 浮点数（作为比例） |
| `amount_reserve_percent` | 在最小交易对仓位数量中预留一定量。机器人在计算最小交易对仓位数量时会预留 `amount_reserve_percent` + 止损值，以避免可能的交易被拒。<br>*默认为 `0.05`（5%）。* <br> **数据类型：** 正浮点数（作为比例） |
| `timeframe` | 使用的时间周期（例如 `1m`、`5m`、`15m`、`30m`、`1h` ...）。通常在配置文件中缺失，而是在策略中指定。[策略覆盖](#parameters-in-the-strategy)。<br> **数据类型：** 字符串 |
| `fiat_display_currency` | 用于显示你盈亏的法币。<br> **数据类型：** 字符串 |
| `dry_run` | **必填。** 定义机器人必须处于 Dry Run（模拟运行）还是生产模式。<br>*默认为 `true`。* <br> **数据类型：** 布尔值 |
| `dry_run_wallet` | 定义在 Dry Run 模式下机器人所使用的模拟钱包的起始金额（以 stake 计价货币计）。[详见下文](#dry-run-wallet)<br>*默认为 `1000`。* <br> **数据类型：** 浮点数或字典 |
| `cancel_open_orders_on_exit` | 在发出 `/stop` RPC 命令、按下 `Ctrl+C` 或机器人意外终止时，取消挂单。设为 `true` 时，可让你在市场崩盘时使用 `/stop` 来取消未成交和部分成交的订单。它不影响未平仓头寸。<br>*默认为 `false`。* <br> **数据类型：** 布尔值 |
| `process_only_new_candles` | 仅在新蜡烛到达时启用指标计算。如果为 false，则每个循环都会填充指标，这意味着同一根蜡烛会被处理多次，造成系统负载，但如果你的策略依赖 tick 数据而不仅仅是蜡烛数据，这会很有用。[策略覆盖](#parameters-in-the-strategy)。<br>*默认为 `true`。*  <br> **数据类型：** 布尔值 |
| `minimal_roi` | **必填。** 设置机器人用于离场交易的阈值（作为比例）。[详见下文](#understand-minimal_roi)。[策略覆盖](#parameters-in-the-strategy)。<br> **数据类型：** 字典 |
| `stoploss` |  **必填。** 机器人使用的止损值（作为比例）。更多细节见[止损文档](stoploss.md)。[策略覆盖](#parameters-in-the-strategy)。  <br> **数据类型：** 浮点数（作为比例） |
| `trailing_stop` | 启用移动止损（trailing stoploss，基于配置或策略文件中的 `stoploss`）。更多细节见[止损文档](stoploss.md#trailing-stop-loss)。[策略覆盖](#parameters-in-the-strategy)。<br> **数据类型：** 布尔值 |
| `trailing_stop_positive` | 一旦达到盈利，改变止损。[止损文档](stoploss.md#trailing-stop-loss-different-positive-loss)。[策略覆盖](#parameters-in-the-strategy)。<br> **数据类型：** 浮点数 |
| `trailing_stop_positive_offset` | 应用 `trailing_stop_positive` 的偏移量。应为正数的百分比值。[止损文档](stoploss.md#trailing-stop-loss-only-once-the-trade-has-reached-a-certain-offset)。[策略覆盖](#parameters-in-the-strategy)。<br>*默认为 `0.0`（无偏移）。* <br> **数据类型：** 浮点数 |
| `trailing_only_offset_is_reached` | 仅当达到偏移量时才应用移动止损。[止损文档](stoploss.md)。[策略覆盖](#parameters-in-the-strategy)。<br>*默认为 `false`。*  <br> **数据类型：** 布尔值 |
| `fee` | 回测 / dry-run 期间使用的手续费。通常不应配置此项，从而让 freqtrade 回退到交易所默认手续费。以比例设置（例如 0.001 = 0.1%）。每次交易收取两次手续费，买入时一次，卖出时一次。<br> **数据类型：** 浮点数（作为比例） |
| `futures_funding_rate` | 当交易所无法提供历史资金费率时，用户指定的资金费率。这不会覆盖真实的历史费率。建议除非你在测试某个特定币种并了解资金费率将如何影响 freqtrade 的盈亏计算，否则将其设为 0。[更多说明](leverage.md#unavailable-funding-rates) <br>*默认为 `None`。*<br> **数据类型：** 浮点数 |
| `trading_mode` | 指定你是想常规交易、带杠杆交易，还是交易价格由相应加密货币价格衍生的合约。[杠杆文档](leverage.md)。<br>*默认为 `"spot"`（现货）。*  <br> **数据类型：** 字符串 |
| `margin_mode` | 使用杠杆交易时，这决定交易者拥有的抵押物是共享的还是隔离到每个交易对的。[杠杆文档](leverage.md)。<br> **数据类型：** 字符串 |
| `liquidation_buffer` | 一个比例，指定在强平价（liquidation price）和止损之间放置多大范围的安全垫，以防止头寸触及强平价。[杠杆文档](leverage.md)。<br>*默认为 `0.05`。*  <br> **数据类型：** 浮点数 |
| | **未成交超时（Unfilled timeout）** |
| `unfilledtimeout.entry` | **必填。** 机器人会等待未成交的入场订单完成多久（以分钟或秒计），超时后订单将被取消。[策略覆盖](#parameters-in-the-strategy)。<br> **数据类型：** 整数 |
| `unfilledtimeout.exit` | **必填。** 机器人会等待未成交的离场订单完成多久（以分钟或秒计），超时后订单将被取消，并在当前（新）价格处重新提交，只要有信号。[策略覆盖](#parameters-in-the-strategy)。<br> **数据类型：** 整数 |
| `unfilledtimeout.unit` | 未成交超时设置中使用的单位。注意：如果你将 `unfilledtimeout.unit` 设为 "seconds"，则 "internals.process_throttle_secs" 必须小于或等于超时值。[策略覆盖](#parameters-in-the-strategy)。<br> *默认为 `"minutes"`（分钟）。* <br> **数据类型：** 字符串 |
| `unfilledtimeout.exit_timeout_count` | 离场订单可超时的次数。一旦达到此超时次数，将触发紧急离场。设为 0 则禁用，并允许无限次取消订单。[策略覆盖](#parameters-in-the-strategy)。<br>*默认为 `0`。* <br> **数据类型：** 整数 |
| | **定价（Pricing）** |
| `entry_pricing.price_side` | 选择机器人查看以获取入场汇率的买卖价差（spread）一侧。[详见下文](#entry-price)。<br> *默认为 `"same"`（同侧）。* <br> **数据类型：** 字符串（可选 `ask`、`bid`、`same` 或 `other`） |
| `entry_pricing.price_last_balance` | **必填。** 对出价（bidding）价格进行插值。详见[下文](#entry-price-without-orderbook-enabled)。 |
| `entry_pricing.use_order_book` | 启用使用[订单簿入场](#entry-price-with-orderbook-enabled)中的汇率入场。<br> *默认为 `true`。*<br> **数据类型：** 布尔值 |
| `entry_pricing.order_book_top` | 机器人将使用订单簿"price_side"中的前 N 档汇率来入场。例如值为 2 时机器人会选取[订单簿入场](#entry-price-with-orderbook-enabled)中的第 2 档。<br>*默认为 `1`。*  <br> **数据类型：** 正整数 |
| `entry_pricing. check_depth_of_market.enabled` | 当订单簿中买盘与卖盘的差距达到设定值时，不入场。[查看市场深度](#check-depth-of-market)。<br>*默认为 `false`。* <br> **数据类型：** 布尔值 |
| `entry_pricing. check_depth_of_market.bids_to_ask_delta` | 订单簿中买盘与卖盘发现的数量差之比。小于 1 的值意味着卖盘规模更大，而大于 1 的值意味着买盘规模更大。[查看市场深度](#check-depth-of-market) <br> *默认为 `0`。*  <br> **数据类型：** 浮点数（作为比例） |
| `exit_pricing.price_side` | 选择机器人查看以获取离场汇率的价差一侧。[详见下文](#exit-price-side)。<br> *默认为 `"same"`（同侧）。* <br> **数据类型：** 字符串（可选 `ask`、`bid`、`same` 或 `other`） |
| `exit_pricing.price_last_balance` | 对离场价格进行插值。详见[下文](#exit-price-without-orderbook-enabled)。 |
| `exit_pricing.use_order_book` | 启用使用[订单簿离场](#exit-price-with-orderbook-enabled)对未平仓交易进行离场。<br> *默认为 `true`。*<br> **数据类型：** 布尔值 |
| `exit_pricing.order_book_top` | 机器人将使用订单簿"price_side"中的前 N 档汇率来离场。例如值为 2 时机器人会选取[订单簿离场](#exit-price-with-orderbook-enabled)中的第 2 档卖单价。<br>*默认为 `1`。* <br> **数据类型：** 正整数 |
| `custom_price_max_distance_ratio` | 配置当前价格与自定义入场或离场价格之间的最大距离比例。<br>*默认为 `0.02`（2%）。*<br> **数据类型：** 正浮点数 |
| | **订单 / 信号处理** |
| `use_exit_signal` | 在 `minimal_roi` 之外，还使用策略产生的离场信号。<br>设为 false 会禁用 `"exit_long"` 和 `"exit_short"` 列的使用。对其他离场方法（止损、ROI、回调）无影响。[策略覆盖](#parameters-in-the-strategy)。<br>*默认为 `true`。* <br> **数据类型：** 布尔值 |
| `exit_profit_only` | 在机器人达到 `exit_profit_offset` 之前，等待才做离场决策。[策略覆盖](#parameters-in-the-strategy)。<br>*默认为 `false`。* <br> **数据类型：** 布尔值 |
| `exit_profit_offset` | 离场信号仅在该值之上才生效。仅在与 `exit_profit_only=True` 组合时生效。[策略覆盖](#parameters-in-the-strategy)。<br>*默认为 `0.0`。* <br> **数据类型：** 浮点数（作为比例） |
| `ignore_roi_if_entry_signal` | 如果入场信号仍然活跃，则不离场。此设置优先于 `minimal_roi` 和 `use_exit_signal`。[策略覆盖](#parameters-in-the-strategy)。<br>*默认为 `false`。* <br> **数据类型：** 布尔值 |
| `ignore_buying_expired_candle_after` | 指定多少秒后买入信号不再被使用。<br> **数据类型：** 整数 |
| `order_types` | 根据动作（`"entry"`、`"exit"`、`"stoploss"`、`"stoploss_on_exchange"`）配置订单类型。[详见下文](#understand-order_types)。[策略覆盖](#parameters-in-the-strategy)。<br> **数据类型：** 字典 |
| `order_time_in_force` | 配置入场和离场订单的有效时间（time in force）。[详见下文](#understand-order_time_in_force)。[策略覆盖](#parameters-in-the-strategy)。<br> **数据类型：** 字典 |
| `position_adjustment_enable` | 启用策略进行头寸调整（额外的买入或卖出）。[详见此处](strategy-callbacks.md#adjust-trade-position)。<br> [策略覆盖](#parameters-in-the-strategy)。<br>*默认为 `false`。*<br> **数据类型：** 布尔值 |
| `max_entry_position_adjustment` | 除首笔入场订单外，每笔未平仓交易的最大附加订单数。设为 `-1` 表示无限附加订单。[详见此处](strategy-callbacks.md#adjust-trade-position)。<br> [策略覆盖](#parameters-in-the-strategy)。<br>*默认为 `-1`。*<br> **数据类型：** 正整数或 -1 |
| | **交易所（Exchange）** |
| `exchange.name` | **必填。** 要使用的交易所类的名称。<br> **数据类型：** 字符串 |
| `exchange.key` | 用于该交易所的 API key。仅在生产模式下需要。<br>**请保密，不要公开披露。** <br> **数据类型：** 字符串 |
| `exchange.secret` | 用于该交易所的 API secret。仅在生产模式下需要。<br>**请保密，不要公开披露。** <br> **数据类型：** 字符串 |
| `exchange.password` | 用于该交易所的 API 密码。仅在生产模式下且对使用密码进行 API 请求的交易所需要。<br>**请保密，不要公开披露。** <br> **数据类型：** 字符串 |
| `exchange.uid` | 用于该交易所的 API uid。仅在生产模式下且对使用 uid 进行 API 请求的交易所需要。<br>**请保密，不要公开披露。** <br> **数据类型：** 字符串 |
| `exchange.pair_whitelist` | 机器人用于交易、以及在回测期间检查潜在交易的交易对列表。支持正则交易对如 `.*/BTC`。不被 VolumePairList 使用。[更多说明](plugins.md#pairlists-and-pairlist-handlers)。<br> **数据类型：** 列表 |
| `exchange.pair_blacklist` | 机器人必须绝对避免用于交易和回测的交易对列表。[更多说明](plugins.md#pairlists-and-pairlist-handlers)。<br> **数据类型：** 列表 |
| `exchange.ccxt_config` | 传递给两个 ccxt 实例（同步和异步）的附加 CCXT 参数。这通常是放置附加 ccxt 配置的正确位置。参数因交易所而异，记录在 [ccxt 文档](https://docs.ccxt.com/#/README?id=overriding-exchange-properties-upon-instantiation) 中。请避免在此处添加交易所密钥（改用专门的字段），因为它们可能包含在日志中。<br> **数据类型：** 字典 |
| `exchange.ccxt_sync_config` | 传递给常规（同步）ccxt 实例的附加 CCXT 参数。参数因交易所而异，记录在 [ccxt 文档](https://docs.ccxt.com/#/README?id=overriding-exchange-properties-upon-instantiation)。<br> **数据类型：** 字典 |
| `exchange.ccxt_async_config` | 传递给异步 ccxt 实例的附加 CCXT 参数。参数因交易所而异，记录在 [ccxt 文档](https://docs.ccxt.com/#/README?id=overriding-exchange-properties-upon-instantiation)。<br> **数据类型：** 字典 |
| `exchange.enable_ws` | 启用对该交易所 Websocket 的使用。<br>[详见此处](#consuming-exchange-websockets)。<br>*默认为 `true`。* <br> **数据类型：** 布尔值 |
| `exchange.markets_refresh_interval` | 重新加载市场数据的间隔（分钟）。<br>*默认为 `60` 分钟。* <br> **数据类型：** 正整数 |
| `exchange.skip_open_order_update` | 如果交易所导致问题，跳过启动时的挂单更新。仅与实盘相关。<br>*默认为 `false`*<br> **数据类型：** 布尔值 |
| `exchange.unknown_fee_rate` | 计算交易手续费时使用的回退值。对于手续费以不可交易货币计价的交易所很有用。此处提供的值将与"fee cost"相乘。<br>*默认为 `None`*<br> **数据类型：** 浮点数 |
| `exchange.log_responses` | 记录相关的交易所响应。仅用于调试模式——请谨慎使用。<br>*默认为 `false`*<br> **数据类型：** 布尔值 |
| `exchange.only_from_ccxt` | 阻止从 data.binance.vision 下载数据。保持为 false 可大大加快下载速度，但如果该站点不可用可能会出现问题。<br>*默认为 `false`*<br> **数据类型：** 布尔值 |
| `experimental.block_bad_exchanges` | 阻止已知无法与 freqtrade 配合工作的交易所。除非你想测试该交易所现在是否可用，否则保持默认。<br>*默认为 `true`。* <br> **数据类型：** 布尔值 |
| | **插件（Plugins）** |
| `pairlists` | 定义一个或多个要使用的交易对列表。| [更多说明](plugins.md#pairlists-and-pairlist-handlers)。<br>*默认为 `StaticPairList`。*  <br> **数据类型：** 字典列表 |
| | **Telegram** |
| `telegram.enabled` | 启用 Telegram 的使用。<br> **数据类型：** 布尔值 |
| `telegram.token` | 你的 Telegram 机器人 token。仅当 `telegram.enabled` 为 `true` 时需要。<br>**请保密，不要公开披露。** <br> **数据类型：** 字符串 |
| `telegram.chat_id` | 你的个人 Telegram 账户 id。仅当 `telegram.enabled` 为 `true` 时需要。<br>**请保密，不要公开披露。** <br> **数据类型：** 字符串 |
| `telegram.balance_dust_level` | 零钱级别（以 stake 计价货币计）——余额低于此值的币种将不会由 `/balance` 显示。<br> **数据类型：** 浮点数 |
| `telegram.reload` | 允许 Telegram 消息上的"reload（重新加载）"按钮。<br>*默认为 `true`。*<br> **数据类型：** 布尔值 |
| `telegram.notification_settings.*` | 详细的通知设置。详情请参阅 [telegram 文档](telegram-usage.md)。<br> **数据类型：** 字典 |
| `telegram.allow_custom_messages` | 启用通过 dataprovider.send_msg() 函数从策略发送 Telegram 消息。<br> **数据类型：** 布尔值 |
| | **Webhook** |
| `webhook.enabled` | 启用 Webhook 通知的使用。<br> **数据类型：** 布尔值 |
| `webhook.url` | Webhook 的 URL。仅当 `webhook.enabled` 为 `true` 时需要。更多细节请参阅 [webhook 文档](webhook-config.md)。<br> **数据类型：** 字符串 |
| `webhook.entry` | 入场时发送的数据载荷。仅当 `webhook.enabled` 为 `true` 时需要。[webhook 文档](webhook-config.md)。<br> **数据类型：** 字符串 |
| `webhook.entry_cancel` | 入场订单取消时发送的数据载荷。仅当 `webhook.enabled` 为 `true` 时需要。[webhook 文档](webhook-config.md)。<br> **数据类型：** 字符串 |
| `webhook.entry_fill` | 入场订单成交时发送的数据载荷。仅当 `webhook.enabled` 为 `true` 时需要。[webhook 文档](webhook-config.md)。<br> **数据类型：** 字符串 |
| `webhook.exit` | 离场时发送的数据载荷。仅当 `webhook.enabled` 为 `true` 时需要。[webhook 文档](webhook-config.md)。<br> **数据类型：** 字符串 |
| `webhook.exit_cancel` | 离场订单取消时发送的数据载荷。仅当 `webhook.enabled` 为 `true` 时需要。[webhook 文档](webhook-config.md)。<br> **数据类型：** 字符串 |
| `webhook.exit_fill` | 离场订单成交时发送的数据载荷。仅当 `webhook.enabled` 为 `true` 时需要。[webhook 文档](webhook-config.md)。<br> **数据类型：** 字符串 |
| `webhook.status` | 状态调用时发送的数据载荷。仅当 `webhook.enabled` 为 `true` 时需要。[webhook 文档](webhook-config.md)。<br> **数据类型：** 字符串 |
| `webhook.allow_custom_messages` | 启用通过 dataprovider.send_msg() 函数从策略发送 Webhook 消息。<br> **数据类型：** 布尔值 |
| | **Rest API / FreqUI / 生产者-消费者（Producer-Consumer）** |
| `api_server.enabled` | 启用 API 服务器的使用。更多细节请参阅 [API 服务器文档](rest-api.md)。<br> **数据类型：** 布尔值 |
| `api_server.listen_ip_address` | 绑定的 IP 地址。更多细节请参阅 [API 服务器文档](rest-api.md)。<br> **数据类型：** IPv4 |
| `api_server.listen_port` | 绑定的端口。更多细节请参阅 [API 服务器文档](rest-api.md)。<br>**数据类型：** 1024 到 65535 之间的整数 |
| `api_server.verbosity` | 日志详细程度。`info` 会打印所有 RPC 调用，而 "error" 只显示错误。<br>**数据类型：** 枚举，可选 `info` 或 `error`。默认为 `info`。 |
| `api_server.username` | API 服务器的用户名。更多细节请参阅 [API 服务器文档](rest-api.md)。<br>**请保密，不要公开披露。**<br> **数据类型：** 字符串 |
| `api_server.password` | API 服务器的密码。更多细节请参阅 [API 服务器文档](rest-api.md)。<br>**请保密，不要公开披露。**<br> **数据类型：** 字符串 |
| `api_server.ws_token` | 消息 WebSocket 的 API token。更多细节请参阅 [API 服务器文档](rest-api.md)。<br>**请保密，不要公开披露。** <br> **数据类型：** 字符串 |
| `bot_name` | 机器人的名称。通过 API 传递给客户端——可用于区分 / 命名机器人。<br> *默认为 `freqtrade`*<br> **数据类型：** 字符串 |
| `external_message_consumer` | 启用[生产者/消费者模式](producer-consumer.md)。更多细节请参阅该文档。<br> **数据类型：** 字典 |
| | **其他（Other）** |
| `initial_state` | 定义应用的初始状态。如果设为 stopped，则必须通过 `/start` RPC 命令显式启动机器人。<br>*默认为 `stopped`。* <br> **数据类型：** 枚举，可选 `running`、`paused` 或 `stopped` |
| `force_entry_enable` | 启用 RPC 命令以强制交易入场。详见下文。<br> **数据类型：** 布尔值 |
| `disable_dataframe_checks` | 禁用对策略方法返回的 OHLCV dataframe 的正确性检查。仅当你有意修改 dataframe 且了解自己在做什么时使用。[策略覆盖](#parameters-in-the-strategy)。<br> *默认为 `False`*。 <br> **数据类型：** 布尔值 |
| `internals.process_throttle_secs` | 设置进程节流，即一次机器人迭代循环的最小循环时长。单位秒。<br>*默认为 `5` 秒。* <br> **数据类型：** 正整数 |
| `internals.heartbeat_interval` | 每 N 秒打印一次心跳消息。设为 0 可禁用心跳消息。<br>*默认为 `60` 秒。* <br> **数据类型：** 正整数或 0 |
| `internals.sd_notify` | 启用 sd_notify 协议的使用，以向 systemd 服务管理器告知机器人状态的变化，并发出保活（keep-alive）心跳。[详见此处](advanced-setup.md#configure-the-bot-running-as-a-systemd-service)。<br> **数据类型：** 布尔值 |
| `strategy` | **必填** 定义要使用的策略类。建议通过 `--strategy NAME` 设置。<br> **数据类型：** 类名 |
| `strategy_path` | 添加一个额外的策略查找路径（必须是一个目录）。<br> **数据类型：** 字符串 |
| `recursive_strategy_search` | 设为 `true` 以递归搜索 `user_data/strategies` 内的子目录来寻找策略。<br> **数据类型：** 布尔值 |
| `user_data_dir` | 包含用户数据的目录。<br> *默认为 `./user_data/`*。 <br> **数据类型：** 字符串 |
| `db_url` | 声明要使用的数据库 URL。注意：如果 `dry_run` 为 `true`，则默认 `sqlite:///tradesv3.dryrun.sqlite`；对于生产实例，默认 `sqlite:///tradesv3.sqlite`。<br> **数据类型：** 字符串，SQLAlchemy 连接字符串 |
| `logfile` | 指定日志文件名。对日志文件采用滚动策略，最多 10 个文件，每个文件 1MB 上限。<br> **数据类型：** 字符串 |
| `add_config_files` | 附加配置文件。这些文件会被加载并与当前配置文件合并。文件相对于初始文件解析。<br> *默认为 `[]`*。 <br> **数据类型：** 字符串列表 |
| `dataformat_ohlcv` | 用于存储历史蜡烛（OHLCV）数据的数据格式。<br> *默认为 `feather`*。 <br> **数据类型：** 字符串 |
| `dataformat_trades` | 用于存储历史交易数据的数据格式。<br> *默认为 `feather`*。 <br> **数据类型：** 字符串 |
| `reduce_df_footprint` | 将所有数值列重新转换为 float32/int32，以减少内存/磁盘占用（并减少回测/hyperopt 以及 FreqAI 中的训练/推理耗时）。<br> 默认：`False`。<br> **数据类型：** 布尔值。 |
| `log_config` | 包含 python logging 日志配置的字典。[更多信息](advanced-setup.md#advanced-logging) <br> 默认：`FtRichHandler` <br> **数据类型：** 字典。 |

### 策略中的参数

以下参数可以在配置文件或策略中设置。
配置文件中设置的值始终覆盖策略中设置的值。

* `minimal_roi`
* `timeframe`
* `stoploss`
* `max_open_trades`
* `trailing_stop`
* `trailing_stop_positive`
* `trailing_stop_positive_offset`
* `trailing_only_offset_is_reached`
* `use_custom_stoploss`
* `process_only_new_candles`
* `order_types`
* `order_time_in_force`
* `unfilledtimeout`
* `disable_dataframe_checks`
* `use_exit_signal`
* `exit_profit_only`
* `exit_profit_offset`
* `ignore_roi_if_entry_signal`
* `ignore_buying_expired_candle_after`
* `position_adjustment_enable`
* `max_entry_position_adjustment`

### 配置每笔交易的金额

有几种方法可以配置机器人用于入场交易的 stake 计价货币的数量。所有这些方法都遵循[可用余额配置](#tradable-balance)，如下所述。

#### 最小交易仓位

最小仓位数量取决于交易所和交易对，通常列在交易所的支持页面上。

假设 XRP/USD 的最小可交易数量为 20 XRP（由交易所给出），价格为 0.6 美元，则买入该交易对的最小仓位数量为 `20 * 0.6 ~= 12`。
该交易所对 USD 还有一个限制——所有订单必须 > 10 美元——但在此例中并不适用。

为了保证安全执行，freqtrade 不允许以 10.1 美元的仓位数量买入，相反，它会确保有足够的空间在该交易对下方放置一个止损（加上一个偏移量，由 `amount_reserve_percent` 定义，默认 5%）。

预留 5% 时，最小仓位数量约为 12.6 美元（`12 * (1 + 0.05)`）。如果我们在其上再考虑 10% 的止损——我们最终会得到约 14 美元的值（`12.6 / (1 - 0.1)`）。

为了在大止损值情况下限制这个计算，计算出的最小仓位限制永远不会超过真实限制的 50% 以上。

!!! Warning
    由于交易所的限制通常是稳定的、不经常更新，一些交易对会显示出相当高的最小限制，仅仅是因为自交易所上次调整限制以来价格已经大幅上涨。Freqtrade 会将仓位数量调整到这个值，除非它比计算/期望的仓位数量高出 30% 以上——在这种情况下交易会被拒绝。

#### Dry-run 钱包

以 dry-run 模式运行时，机器人将使用一个模拟钱包来执行交易。该钱包的起始余额由 `dry_run_wallet` 定义（默认为 1000）。
对于更复杂的场景，你也可以为 `dry_run_wallet` 分配一个字典，来定义每种货币的起始余额。

```json
"dry_run_wallet": {
    "BTC": 0.01,
    "ETH": 2,
    "USDT": 1000
}
```

命令行选项（`--dry-run-wallet`）可用于覆盖配置值，但仅适用于浮点值，不适用于字典。如果你想使用字典，请调整配置文件。

!!! Note
    非 stake 计价货币的余额不会用于交易，但会作为钱包余额的一部分显示。
    在跨保证金（Cross-margin）交易所上，钱包余额可能用于计算可用于交易的抵押物。

#### 可交易余额（Tradable balance）

默认情况下，机器人假设 `完整金额 - 1%` 可供其支配，当使用[动态仓位数量](#dynamic-stake-amount)时，它会将完整余额按 `max_open_trades` 分成每个交易一份。
Freqtrade 会在入场时为可能的手续费预留 1%，因此默认不会动用这部分。

你可以通过 `tradable_balance_ratio` 设置来配置"不被触碰"的金额。

例如，如果你的钱包在交易所有 10 ETH 可用，且 `tradable_balance_ratio=0.5`（即 50%），那么机器人最多会使用 5 ETH 进行交易，并将其视为可用余额。钱包的其余部分不受影响。

!!! Danger
    在同一账户上运行多个机器人时，**不应**使用此设置。请改用[分配给机器人的可用资金](#assign-available-capital)。

!!! Warning
    `tradable_balance_ratio` 设置适用于当前余额（可用余额 + 交易中占用的余额）。因此，假设起始余额为 1000，配置 `tradable_balance_ratio=0.99` 并不能保证交易所上始终有 10 个货币单位可用。例如，如果总余额减少到 500（无论是由于亏损还是提取了余额），可用金额可能会减少到 5 个单位。

#### 分配可用资金（Assign available Capital）

为了在同一交易所账户上运行多个机器人时充分利用复利收益，你会希望将每个机器人限制在一定的起始余额内。
这可以通过将 `available_capital` 设置为所需的起始余额来实现。

假设你的账户有 10000 USDT，你想在该交易所上运行 2 个不同的策略。
你会设置 `available_capital=5000`——为每个机器人授予 5000 USDT 的初始资金。
然后机器人会将此起始余额平均拆分为 `max_open_trades` 份。
盈利的交易将导致该机器人的仓位数量增加——而不会影响另一个机器人的仓位数量。

调整 `available_capital` 需要重新加载配置才能生效。调整 `available_capital` 会增加先前 `available_capital` 与新 `available_capital` 之间的差额。在交易未平仓时减少可用资金不会了结这些交易。差额会在交易结束时返还给钱包。其结果因调整和了结交易之间的价格变动而异。

!!! Warning "与 `tradable_balance_ratio` 不兼容"
    设置此选项将替换任何 `tradable_balance_ratio` 的配置。

#### 修正末笔仓位数量（Amend last stake amount）

假设我们的可交易余额为 1000 USDT，`stake_amount=400`，且 `max_open_trades=3`。
机器人会开 2 笔交易，并且无法填满最后一个交易槽位，因为所请求的 400 USDT 已不再可用（800 USDT 已绑定在其他交易中）。

为了克服这一点，可以将选项 `amend_last_stake_amount` 设为 `True`，这将使机器人能够将 stake_amount 减少为可用余额，以填满最后一份交易槽位。

在上面的例子中，这意味着：

* 交易1：400 USDT
* 交易2：400 USDT
* 交易3：200 USDT

!!! Note
    此选项仅适用于[静态仓位数量](#static-stake-amount)——因为[动态仓位数量](#dynamic-stake-amount)会平均分配余额。

!!! Note
    可以使用 `last_stake_amount_min_ratio` 配置最小末笔仓位数量——默认为 0.5（50%）。这意味着曾经使用的最小仓位数量为 `stake_amount * 0.5`。这避免了过低的仓位数量，这种数量接近该交易对的最小可交易数量，可能会被交易所拒绝。

#### 静态仓位数量（Static stake amount）

`stake_amount` 配置会静态地配置你的机器人每笔交易将使用的 stake 计价货币的数量。

最小配置值为 0.0001，不过，请检查你的交易所对所使用的 stake 计价货币的交易最小值，以避免出现问题。

此设置与 `max_open_trades` 配合使用。交易中投入的最大资金为 `stake_amount * max_open_trades`。
例如，假设配置为 `max_open_trades=3` 且 `stake_amount=0.05`，机器人最多会使用（0.05 BTC x 3）= 0.15 BTC。

!!! Note
    此设置遵循[可用余额配置](#tradable-balance)。

#### 动态仓位数量（Dynamic stake amount）

或者，你可以使用动态仓位数量，它将使用交易所上的可用余额，并按允许的交易数量（`max_open_trades`）平均分配。

要配置此选项，设置 `stake_amount="unlimited"`。我们还建议设置 `tradable_balance_ratio=0.99`（99%）——为可能的手续费保留最小余额。

在这种情况下，交易金额的计算方式为：

```python
currency_balance / (max_open_trades - current_open_trades)
```

要允许机器人交易你账户中所有可用的 `stake_currency`（减去 `tradable_balance_ratio`），设置：

```json
"stake_amount" : "unlimited",
"tradable_balance_ratio": 0.99,
```

!!! Tip "复利收益"
    此配置将允许根据机器人的表现增加/减少仓位（如果机器人亏损则降低仓位，如果机器人有盈利记录则提高仓位，因为可用余额更高），并将产生复利收益。

!!! Note "使用 Dry-Run 模式时"
    当使用 `"stake_amount" : "unlimited"` 与 Dry-Run、回测或 Hyperopt 组合时，余额将从 `dry_run_wallet` 定义的仓位开始模拟并随之演变。
    因此，将 `dry_run_wallet` 设置为一个合理的值是很重要的（例如 BTC 设为 0.05 或 0.01，USDT 设为 1000 或 100），否则它可能会一次性模拟 100 BTC（或更多）或 0.05 USDT（或更少）的交易——这可能与你的真实可用余额不符，或低于交易所对该 stake 计价货币订单金额的最小限制。

#### 带头寸调整的动态仓位数量

当你想将无限仓位与头寸调整一起使用时，你还必须实现 `custom_stake_amount`，根据策略返回一个值。
典型的值在建议仓位的 25% - 50% 范围内，但很大程度上取决于你的策略，以及你希望为头寸调整缓冲留出多少到钱包中。

例如，如果你的头寸调整假设它可以用相同的仓位数量做 2 次额外买入，那么你的缓冲应该是初始建议无限仓位的 66.6667%。

或者另一个例子，如果你的头寸调整假设它可以用 3 倍于原始仓位数量做一次额外买入，那么 `custom_stake_amount` 应返回建议仓位数量的 25%，并留出 75% 用于以后可能的头寸调整。

--8<-- "../includes/pricing.md"

## 更多配置细节

### 理解 minimal_roi

`minimal_roi` 配置参数是一个 JSON 对象，其中键是时长（分钟），值是作为比例的 ROI 最小值。
见下例：

```json
"minimal_roi": {
    "40": 0.0,    # 40 分钟后离场，如果利润不为负
    "30": 0.01,   # 30 分钟后离场，如果至少 1% 利润
    "20": 0.02,   # 20 分钟后离场，如果至少 2% 利润
    "0":  0.04    # 立即离场，如果至少 4% 利润
},
```

大多数策略文件已经包含了最优的 `minimal_roi` 值。
此参数可以在策略或配置文件中设置。如果你在配置文件中使用它，它将覆盖策略文件中的 `minimal_roi` 值。
如果在策略和配置中都未设置，则使用默认值 1000% `{"0": 10}`，并且最小 ROI 被禁用，除非你的交易产生 1000% 的利润。

!!! Note "强制在指定时间后离场的特殊情况"
    一种特殊情况是使用 `"<N>": -1` 作为 ROI。这会强制机器人 N 分钟后离场一笔交易，无论它是正还是负，因此代表一个限时强制离场。

### 理解 force_entry_enable

`force_entry_enable` 配置参数启用了通过 Telegram 和 REST API 使用强制入场（`/forcelong`、`/forceshort`）命令。
出于安全原因，默认禁用，如果启用，freqtrade 会在启动时显示警告消息。
例如，你可以向机器人发送 `/forceenter ETH/BTC`，这将导致 freqtrade 买入该交易对并持有，直到出现常规离场信号（ROI、止损、`/forceexit`）。

某些策略下这可能很危险，请谨慎使用。

使用详情请参阅 [telegram 文档](telegram-usage.md)。

### 忽略过期蜡烛

在处理较大的时间周期（例如 1h 或更大）并使用较低的 `max_open_trades` 值时，一旦有交易槽位可用，最后一根蜡烛可能立即被处理。在处理最后一根蜡烛时，这可能导致不希望在该蜡烛上使用买入信号的情况。例如，当你的策略中使用了一个交叉（cross-over）条件时，那个点可能已经过去太久，不适合在其上开始一笔交易。

在这些情况下，你可以通过将 `ignore_buying_expired_candle_after` 设为一个正数，来启用忽略超过指定时长的蜡烛的功能，该值表示买入信号过期所需的秒数。

例如，如果你的策略使用 1h 时间周期，并且你只想在新蜡烛到来后的前 5 分钟内买入，你可以向你的策略添加以下配置：

``` json
  {
    //...
    "ignore_buying_expired_candle_after": 300,
    // ...
  }
```

!!! Note
    此设置随着每根新蜡烛重置，因此它不会阻止在第 2 或第 3 根蜡烛上仍然活跃的"粘连信号"（sticking-signals）执行。最好对买入信号使用"触发器"选择器，使其仅在一根蜡烛内活跃。

### 理解 order_types

`order_types` 配置参数将动作（`entry`、`exit`、`stoploss`、`emergency_exit`、`force_exit`、`force_entry`）映射到订单类型（`market`、`limit` 等），并配置止损在交易所端（on exchange），以及定义交易所端止损的更新间隔（秒）。

这允许使用限价单入场、限价单离场，以及使用市价单创建止损。
它还允许设置交易所端止损（"stoploss on exchange"），这意味着一旦买入订单成交，止损订单将立即被放置。

配置文件中设置的 `order_types` 会整体覆盖策略中设置的值，因此你需要在一个地方配置整个 `order_types` 字典。

如果进行了配置，需要存在以下 4 个值（`entry`、`exit`、`stoploss` 和 `stoploss_on_exchange`），否则机器人将无法启动。

关于（`emergency_exit`、`force_exit`、`force_entry`、`stoploss_on_exchange`、`stoploss_on_exchange_interval`、`stoploss_on_exchange_limit_ratio`）的信息，请参阅止损文档 [交易所端止损](stoploss.md)。

策略中的语法：

```python
order_types = {
    "entry": "limit",
    "exit": "limit",
    "emergency_exit": "market",
    "force_entry": "market",
    "force_exit": "market",
    "stoploss": "market",
    "stoploss_on_exchange": False,
    "stoploss_on_exchange_interval": 60,
    "stoploss_on_exchange_limit_ratio": 0.99,
}
```

配置：

```json
"order_types": {
    "entry": "limit",
    "exit": "limit",
    "emergency_exit": "market",
    "force_entry": "market",
    "force_exit": "market",
    "stoploss": "market",
    "stoploss_on_exchange": false,
    "stoploss_on_exchange_interval": 60
}
```

!!! Note "市价单支持"
    并非所有交易所都支持"市价"订单。
    如果你的交易所不支持市价订单，会显示以下消息：`"Exchange <yourexchange> does not support market orders."` 并且机器人会拒绝启动。

!!! Warning "使用市价单"
    使用市价单时，请仔细阅读[市价单定价](#market-order-pricing)章节。

!!! Note "交易所端止损"
    `order_types.stoploss_on_exchange_interval` 不是必填项。如果你不确定自己在做什么，请不要更改其值。关于止损如何工作的更多信息，请参阅[止损文档](stoploss.md)。

    如果 `order_types.stoploss_on_exchange` 被启用，并且止损在交易所端被手动取消，那么机器人将创建一个新的止损订单。

!!! Warning "警告：order_types.stoploss_on_exchange 失败"
    如果交易所端止损创建由于某种原因失败，则会启动一个"紧急离场"（emergency exit）。默认情况下，这将使用市价单离场该交易。紧急离场的订单类型可以通过设置 `order_types` 字典中的 `emergency_exit` 值来更改——但不建议这样做。

### 理解 order_time_in_force

`order_time_in_force` 配置参数定义了订单在交易所端的执行策略。
常用的时间有效（time in force）类型有：

**GTC（Good Till Canceled，一直有效直到取消）：**

这在大多数情况下是默认的时间有效类型。它意味着订单将留在交易所，直到被用户取消。它可以完全成交或部分成交。如果部分成交，剩余部分将留在交易所直到被取消。

**FOK（Fill Or Kill，要么全部成交要么取消）：**

这意味着如果订单没有立即且完全成交，则会被交易所取消。

**IOC（Immediate Or Canceled，立即成交否则取消）：**

它与上面的 FOK 相同，只是它可以部分成交。剩余部分由交易所自动取消。

不一定推荐使用，因为这可能导致低于最小交易规模的部分成交。

**PO（Post only，仅挂单）：**

仅挂单（Post only）订单。订单要么作为 maker 订单放置，要么被取消。
这意味着订单必须在订单簿上以未成交状态停留至少一段时间。

请查看[交易所文档](exchanges.md)了解你的交易所支持的时间有效类型。

#### time_in_force 配置

`order_time_in_force` 参数包含一个字典，带有 entry 和 exit 的时间有效策略值。
这可以在配置文件或策略中设置。配置文件中设置的值按照常规的[优先级规则](#configuration-option-prevalence)覆盖策略中的值。

可能的值有：`GTC`（默认）、`FOK` 或 `IOC`。

``` python
"order_time_in_force": {
    "entry": "GTC",
    "exit": "GTC"
},
```

!!! Warning
    除非你知道自己在做什么，并且已经研究过对特定交易所使用不同值的影响，否则请不要更改默认值。

### 法币转换（Fiat conversion）

Freqtrade 使用 Coingecko API 将币种价值转换为相应的法币价值，用于 Telegram 报告。
FIAT 货币可以在配置文件中设置为 `fiat_display_currency`。

从配置中完全移除 `fiat_display_currency` 会跳过 coingecko 的初始化，并且不会显示任何法币转换。这对机器人的正常运行没有影响。

#### fiat_display_currency 可以使用哪些值？

`fiat_display_currency` 配置参数设置用于机器人 Telegram 报告中从币种到法币转换的基础货币。

有效值有：

```json
"AUD", "BRL", "CAD", "CHF", "CLP", "CNY", "CZK", "DKK", "EUR", "GBP", "HKD", "HUF", "IDR", "ILS", "INR", "JPY", "KRW", "MXN", "MYR", "NOK", "NZD", "PHP", "PKR", "PLN", "RUB", "SEK", "SGD", "THB", "TRY", "TWD", "ZAR", "USD"
```

除了法币外，还支持一系列加密货币。

有效值有：

```json
"BTC", "ETH", "XRP", "LTC", "BCH", "BNB"
```

#### Coingecko 速率限制问题

在某些 IP 段上，coingecko 的速率限制非常严格。
在这种情况下，你可能想将你的 coingecko API key 添加到配置中。

``` json
{
    "fiat_display_currency": "USD",
    "coingecko": {
        "api_key": "your-api",
        "is_demo": true
    }
}
```

Freqtrade 同时支持 Demo 和 Pro 版 coingecko API key。

Coingecko API key 不是机器人正确运行所必需的。
它仅用于 Telegram 报告中的币种到法币转换，这通常没有 API key 也能工作。

## 消费交易所 Websocket

Freqtrade 可以通过 ccxt.pro 消费 websocket。

Freqtrade 旨在确保数据始终可用。
如果 websocket 连接失败（或被禁用），机器人将回退到 REST API 调用。

如果你怀疑某些问题是由 websocket 引起的，可以通过设置 `exchange.enable_ws`（默认为 true）来禁用它们。

```jsonc
"exchange": {
    // ...
    "enable_ws": false,
    // ...
}
```

如果你需要使用代理，请参阅[在 Freqtrade 中使用代理](#using-a-proxy-with-freqtrade)章节了解更多。

!!! Info "逐步推出"
    我们正在缓慢地推出此功能，以确保你的机器人稳定。
    目前，使用仅限于 ohlcv 数据流。
    它也仅限于少数交易所，新交易所正在持续添加中。

## 使用 Dry-run 模式

我们建议先在 Dry-run 模式下启动机器人，看看你的机器人会如何表现，以及你的策略表现如何。在 Dry-run 模式下，机器人不会动用你的资金。它只运行实时模拟，而不会在交易所创建交易。

1. 编辑你的 `config.json` 配置文件。
2. 将 `dry-run` 切换为 `true`，并指定 `db_url` 用于持久化数据库。

```json
"dry_run": true,
"db_url": "sqlite:///tradesv3.dryrun.sqlite",
```

3. 移除你的交易所 API key 和 secret（将它们改为空值或伪造的凭据）：

```json
"exchange": {
    "name": "binance",
    "key": "key",
    "secret": "secret",
    ...
}
```

一旦你对 Dry-run 模式下机器人的表现感到满意，你就可以将其切换到生产模式。

!!! Note
    Dry-run 模式下会提供一个模拟钱包，并假定起始资金为 `dry_run_wallet`（默认为 1000）。

### Dry-run 的注意事项

* 可以提供也可以不提供 API key。在 dry-run 模式下只执行只读操作（即不会改变账户状态的操作）。
* 钱包（`/balance`）是基于 `dry_run_wallet` 模拟的。
* 订单是模拟的，不会提交到交易所。
* 市价单根据下单时刻的订单簿成交量成交，最大滑点为 5%。
* 限价单在价格达到设定水平时成交——或根据 `unfilledtimeout` 设置超时。
* 如果限价单跨越价格超过 1%，它将被转换为市价单，并根据常规市价单规则立即成交（见上面关于市价单的说明）。
* 与 `stoploss_on_exchange` 结合时，假定 stop_loss 价格已被成交。
* 挂单（不是交易，交易存储在数据库中）在机器人重启后保持打开状态，假定它们在离线期间未被成交。

## 切换到生产模式

在生产模式下，机器人会动用你的资金。请小心，因为错误的策略可能会让你损失所有资金。
当你在生产模式下运行时，要清楚自己在做什么。

切换到生产模式时，请确保使用不同的 / 全新的数据库，以避免 dry-run 交易干扰你的交易所资金，并最终污染你的统计。

### 设置你的交易所账户

你需要从交易所网站创建 API Key（通常你会得到 `key` 和 `secret`，某些交易所还需要一个额外的 `password`），并将其插入到配置的相应字段中，或在 `freqtrade new-config` 命令询问时提供。API Key 通常仅在进行实盘交易（用真实资金交易，即机器人运行在"生产模式"、在交易所执行真实订单）时需要，在 dry-run（交易模拟）模式下不需要。当你在 dry-run 模式下设置机器人时，可以将这些字段填为空值。

### 将你的机器人切换到生产模式

**编辑你的 `config.json` 文件。**

**将 dry-run 切换为 false，并且别忘了（如果设置了）调整你的数据库 URL：**

```json
"dry_run": false,
```

**插入你的交易所 API key（将其改为伪造的 API key）：**

```json
{
    "exchange": {
        "name": "binance",
        "key": "af8ddd35195e9dc500b9a6f799f6f5c93d89193b",
        "secret": "08a9dc6db3d7b53e1acebd9275677f4b0a04f1a5",
        //"password": "", // 可选，并非所有交易所都需要
        // ...
    }
    //...
}
```

你还应该确保阅读文档的[交易所](exchanges.md)章节，以了解特定于你交易所的潜在配置细节。

!!! Hint "保守你的秘密"
    为了保守你的秘密，我们建议使用第二个配置文件来存放你的 API key。
    只需将上面的代码片段放到一个新的配置文件（例如 `config-private.json`）中，并将你的设置保存在该文件中。
    然后你可以用 `freqtrade trade --config user_data/config.json --config user_data/config-private.json <...>` 启动机器人，以加载你的 key。

    **永远不要**与任何人分享你的私有配置文件或你的交易所 key！

## 在 Freqtrade 中使用代理

要在 freqtrade 中使用代理，请使用 `"HTTP_PROXY"` 和 `"HTTPS_PROXY"` 变量导出你的代理设置，并设置为适当的值。
这将把代理设置应用到所有地方（telegram、coingecko 等）**除了**交易所请求。

``` bash
export HTTP_PROXY="http://addr:port"
export HTTPS_PROXY="http://addr:port"
freqtrade
```

### 代理交易所请求

要为交易所连接使用代理——你必须将代理定义为 ccxt 配置的一部分。

``` json
{ 
  "exchange": {
    "ccxt_config": {
      "httpsProxy": "http://addr:port",
      "wsProxy": "http://addr:port",
    }
  }
}
```

有关可用代理类型的更多信息，请参阅 [ccxt 代理文档](https://docs.ccxt.com/#/README?id=proxy)。

## 下一步

现在你已经配置了 config.json，下一步是[启动你的机器人](bot-usage.md)。
