<!-- 本文件为中文翻译版，由 AI 根据 docs/telegram-usage.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件（如 exchanges.md），这些文件将在逐步翻译过程中补齐。 -->
<!-- 图片与 includes 引用使用 ../ 指向英文原文档资源，以保证显示正常。 -->

# Telegram 使用方法

## 设置你的 Telegram 机器人

下面我们介绍如何创建你的 Telegram 机器人，以及如何获取你的 Telegram 用户 ID。

### 1. 创建你的 Telegram 机器人

与 [Telegram BotFather](https://telegram.me/BotFather) 开始对话

发送消息 `/newbot`。

*BotFather 的回复：*

> Alright, a new bot. How are we going to call it? Please choose a name for your bot.

选择你的机器人的公开名称（例如 `Freqtrade bot`）

*BotFather 的回复：*

> Good. Now let's choose a username for your bot. It must end in `bot`. Like this, for example: TetrisBot or tetris_bot.

选择你的机器人的用户名 ID 并发送给 BotFather（例如 "`My_own_freqtrade_bot`"）

*BotFather 的回复：*

> Done! Congratulations on your new bot. You will find it at `t.me/yourbots_name_bot`. You can now add a description, about section and profile picture for your bot, see /help for a list of commands. By the way, when you've finished creating your cool bot, ping our Bot Support if you want a better username for it. Just make sure the bot is fully operational before you do this.

> Use this token to access the HTTP API: `22222222:APITOKEN`

> For a description of the Bot API, see this page: https://core.telegram.org/bots/api Father bot will return you the token (API key)

复制 API Token（上面示例中的 `22222222:APITOKEN`），并将其用于配置参数 `token`。

别忘了点击 `/START` 按钮来启动与机器人的对话

### 2. Telegram user_id

#### 获取你的用户 ID

与 [userinfobot](https://telegram.me/userinfobot) 对话

获取你的 "Id"，你将把它用于配置参数 `chat_id`。

#### 使用群组 ID

要获取群组 ID，你可以将机器人添加到群组中，启动 freqtrade，然后发送 `/tg_info` 命令。
这会将群组 ID 返回给你，无需使用其他随机机器人。
虽然 "chat_id" 仍然是必需的，但对于此命令，它不需要设置为这个特定的群组 ID。

如果需要，响应还将包含 "topic_id" — 两者都以可直接复制粘贴到配置中的格式呈现。

``` json
 {
    "enabled": true,
    "token": "********",
    "chat_id": "-1001332619709",
    "topic_id": "122"
}
```

对于 Freqtrade 配置，你可以使用包含 `-` 的完整值作为字符串：

```json
   "chat_id": "-1001332619709"
```

!!! Warning "使用 Telegram 群组"
    使用 Telegram 群组时，你将允许群组中的每个成员访问你的 freqtrade 机器人以及通过 Telegram 可用的所有命令。请确保你信任 Telegram 群组中的每个人，以避免不愉快的情况。

##### 群组话题 ID

要在群组中使用特定话题，你可以在配置中使用 `topic_id` 参数。这将允许你在群组中的特定话题中使用机器人。
如果不设置此项，当群聊启用了话题功能时，机器人将始终在群组的通用频道中回复。

```json
   "chat_id": "-1001332619709",
   "topic_id": "3"
```

与群组 ID 类似 — 你可以在话题/主题中使用 `/tg_info` 来获取正确的话题 ID。

#### 授权用户

对于群组，限制谁可以向机器人发送命令是很有用的。

如果 `"authorized_users": []` 存在且为空，则不允许任何用户控制机器人。
在下面的示例中，只有 ID 为 "1234567" 的用户被允许控制机器人 — 所有其他用户只能接收消息。

```json
   "chat_id": "-1001332619709",
   "topic_id": "3",
   "authorized_users": ["1234567"]
```

## 控制 Telegram 消息噪音

Freqtrade 提供了控制 Telegram 机器人消息详细程度的方法。
每个设置有以下可能的值：

* `on` - 消息将被发送，用户将收到通知。
* `silent` - 消息将被发送，通知将没有声音/振动。
* `off` - 完全跳过发送某种消息类型。

以下是展示不同设置的示例配置：

``` json
"telegram": {
    "enabled": true,
    "token": "your_telegram_token",
    "chat_id": "your_telegram_chat_id",
    "allow_custom_messages": true,
    "notification_settings": {
        "status": "silent",
        "warning": "on",
        "startup": "off",
        "entry": "silent",
        "entry_fill": "on",
        "entry_cancel": "silent",
        "exit": {
            "roi": "silent",
            "emergency_exit": "on",
            "force_exit": "on",
            "exit_signal": "silent",
            "trailing_stop_loss": "on",
            "stop_loss": "on",
            "stoploss_on_exchange": "on",
            "custom_exit": "silent",  // custom_exit without specifying an exit reason
            "partial_exit": "on",
            // "custom_exit_message": "silent",  // Disable individual custom exit reasons
            "*": "off"  // Disable all other exit reasons
        },
        // "exit": "off",  // Simplistic configuration to disable all exit messages
        "exit_cancel": "on",
        "exit_fill": "off",
        "protection_trigger": "off",
        "protection_trigger_global": "on",
        "strategy_msg": "off",
        "show_candle": "off"
    },
    "reload": true,
    "balance_dust_level": 0.01
},
```

* `entry` 通知在订单下达时发送，而 `entry_fill` 通知在交易所订单成交时发送。
* `exit` 通知在订单下达时发送，而 `exit_fill` 通知在交易所订单成交时发送。
    退出消息（`exit` 和 `exit_fill`）可以在各个退出原因级别进一步控制，使用特定的退出原因作为键。所有退出原因的默认值为 `on` — 但可以通过特殊的 `*` 键进行配置 — 该键将作为所有未明确定义的退出原因的通配符。
* `*_fill` 通知默认关闭，必须显式启用。
* `protection_trigger` 通知在保护触发时发送，而 `protection_trigger_global` 通知在全局保护触发时发送。
* `strategy_msg` - 接收来自策略的通知，通过策略中的 `self.dp.send_msg()` 发送 [更多详情](strategy-customization.md#send-notification)。
* `show_candle` - 在入场/退出消息中显示 K 线值。唯一可能的值为 `"ohlc"` 或 `"off"`。
* `balance_dust_level` 将定义 `/balance` 命令将什么视为 "dust" — 余额低于此值的币种将被显示。
* `allow_custom_messages` 完全禁用策略消息。
* `reload` 允许你在选定的消息上禁用重新加载按钮。

## 创建自定义键盘（命令快捷按钮）

Telegram 允许我们创建带有命令按钮的自定义键盘。
默认的自定义键盘如下所示。

```python
[
    ["/daily", "/profit", "/balance"], # row 1, 3 commands
    ["/status", "/status table", "/performance"], # row 2, 3 commands
    ["/count", "/start", "/stop", "/help"] # row 3, 4 commands
]
```

### 使用方法

你可以在 `config.json` 中创建自己的键盘：

``` json
"telegram": {
      "enabled": true,
      "token": "your_telegram_token",
      "chat_id": "your_telegram_chat_id",
      "keyboard": [
          ["/daily", "/stats", "/balance", "/profit"],
          ["/status table", "/performance"],
          ["/reload_config", "/count", "/logs"]
      ]
   },
```

!!! Note "支持的命令"
    只有以下命令是允许的。不支持命令参数！

    `/start`, `/pause`, `/stop`, `/status`, `/status table`, `/trades`, `/profit`, `/performance`, `/daily`, `/stats`, `/count`, `/locks`, `/balance`, `/stopentry`, `/reload_config`, `/show_config`, `/logs`, `/whitelist`, `/blacklist`, `/help`, `/version`, `/marketdir`

## Telegram 命令

默认情况下，Telegram 机器人会显示预定义的命令。某些命令只能通过发送给机器人的消息来使用。下表列出了官方命令。你可以随时通过 `/help` 请求帮助。

|  命令 | 描述 |
|----------|-------------|
| **系统命令**
| `/start` | 启动交易机器人
| `/pause | /stopentry | /stopbuy` | 暂停交易机器人。根据规则优雅地处理未平仓交易。不再开立新仓位。
| `/stop` | 停止交易机器人
| `/reload_config` | 重新加载配置文件
| `/show_config` | 显示当前配置中与运行相关的部分设置
| `/logs [limit]` | 显示最近的日志消息
| `/help` | 显示帮助消息
| `/version` | 显示版本
| **状态** |
| `/status` | 列出所有未平仓交易
| `/status <trade_id>` | 列出一个或多个特定交易。多个 <trade_id> 之间用空格分隔。
| `/status table` | 以表格格式列出所有未平仓交易。待处理的买入订单用星号 (*) 标记。待处理的卖出订单用双星号 (**) 标记。
| `/order <trade_id>` | 列出一个或多个特定交易的订单。多个 <trade_id> 之间用空格分隔。
| `/trades [limit]` | 以表格格式列出所有最近平仓的交易。
| `/count` | 显示已使用的交易数量和可用数量
| `/locks` | 显示当前锁定的交易对。
| `/unlock <pair or lock_id>` | 移除此交易对（或此锁 ID）的锁定。
| `/marketdir [long | short | even | none]` | 更新代表当前市场方向的用户管理变量。如果未提供方向，将显示当前设置的方向。
| `/list_custom_data <trade_id> [key]` | 列出指定交易 ID 和键组合的 custom_data。如果未提供键，将列出该交易 ID 的所有键值对。
| **修改交易状态** |
| `/forceexit <trade_id> | /fx <tradeid>` | 立即退出指定交易（忽略 `minimum_roi`）。
| `/forceexit all | /fx all` | 立即退出所有未平仓交易（忽略 `minimum_roi`）。
| `/fx` | `/forceexit` 的别名
| `/forcelong <pair> [rate]` | 立即买入指定交易对。价格为可选项，仅适用于限价单。（`force_entry_enable` 必须设置为 True）
| `/forceshort <pair> [rate]` | 立即做空指定交易对。价格为可选项，仅适用于限价单。这仅适用于非现货市场。（`force_entry_enable` 必须设置为 True）
| `/delete <trade_id>` | 从数据库中删除特定交易。尝试关闭未完成的订单。需要在交易所上手动处理此交易。
| `/reload_trade <trade_id>` | 从交易所重新加载交易。仅在实盘模式下有效，可能有助于恢复在交易所上手动卖出的交易。
| `/cancel_open_order <trade_id> | /coo <trade_id>` | 取消交易的未完成订单。
| **指标** |
| `/profit [<n>]` | 显示过去 n 天（默认为所有交易）的已平仓交易盈亏摘要和一些性能统计
| `/profit_[long|short] [<n>]` | 显示过去 n 天（默认为所有交易）的单一方向已平仓交易盈亏摘要和一些性能统计
| `/performance` | 显示按交易对分组的每笔已完成交易的表现
| `/balance` | 显示机器人管理的各币种余额
| `/balance full` | 显示账户各币种余额
| `/daily <n>` | 显示过去 n 天（n 默认为 7）的每日盈亏
| `/weekly <n>` | 显示过去 n 周（n 默认为 8）的每周盈亏
| `/monthly <n>` | 显示过去 n 个月（n 默认为 6）的每月盈亏
| `/stats` | 显示按退出原因分组的盈亏次数以及买入和卖出的平均持仓时间
| `/exits` | 显示按退出原因分组的盈亏次数以及买入和卖出的平均持仓时间
| `/entries` | 显示按退出原因分组的盈亏次数以及买入和卖出的平均持仓时间
| `/whitelist [sorted] [baseonly]` | 显示当前白名单。可选择按字母顺序显示和/或仅显示每个配对的基础货币。
| `/blacklist [pair]` | 显示当前黑名单，或将一个交易对添加到黑名单。

## Telegram 命令演示

以下是每个命令你将收到的 Telegram 消息示例。

### /start

> **Status:** `running`

### /pause | /stopentry | /stopbuy

> **Status:** `paused, no more entries will occur from now. Run /start to enable entries.`

通过将状态更改为 `paused` 来阻止机器人开立新交易。
未平仓交易将继续根据其常规规则（ROI/退出信号、止损等）进行管理。
请注意，仓位调整仍然有效，但仅限于减仓方面 — 这意味着当机器人处于 `paused` 状态时，它只能减少未平仓交易的仓位大小。

此后，给机器人时间来关闭未平仓交易（可通过 `/status table` 检查）。
一旦所有仓位都已平仓，运行 `/stop` 以完全停止机器人。

使用 `/start` 将机器人恢复到 `running` 状态，允许其开立新仓位。

!!! Warning
    暂停/停止入场信号仅在机器人运行期间有效，并且不会以任何方式持久化，因此重启机器人将导致此状态重置。

### /stop

> `Stopping trader ...`
> **Status:** `stopped`

### /status

对于每笔未平仓交易，机器人将向你发送以下消息。
入场标签可通过策略进行配置。

> **Trade ID:** `123` `(since 1 days ago)`
> **Current Pair:** CVC/BTC
> **Direction:** Long
> **Leverage:** 1.0
> **Amount:** `26.64180098`
> **Enter Tag:** Awesome Long Signal
> **Open Rate:** `0.00007489`
> **Current Rate:** `0.00007489`
> **Unrealized Profit:** `12.95%`
> **Stoploss:** `0.00007389 (-0.02%)`

### /status table

以表格格式返回所有未平仓交易的状态。

```
ID L/S    Pair     Since   Profit
----    --------  -------  --------
  67 L   SC/BTC    1 d      13.33%
 123 S   CVC/BTC   1 h      12.95%
```

### /count

返回已使用的交易数量和可用数量。

```
current    max
---------  -----
     2     10
```

### /profit

也可通过 `/profit_long` 和 `/profit_short` 使用，以仅显示多头或空头交易的利润。

返回你的盈亏和表现摘要。

> **ROI:** Close trades
>   ∙ `0.00485701 BTC (2.2%) (15.2 Σ%)`
>   ∙ `62.968 USD`
> **ROI:** All trades
>   ∙ `0.00255280 BTC (1.5%) (6.43 Σ%)`
>   ∙ `33.095 EUR`
>
> **Total Trade Count:** `138`
> **Bot started:** `2022-07-11 18:40:44`
> **First Trade opened:** `3 days ago`
> **Latest Trade opened:** `2 minutes ago`
> **Avg. Duration:** `2:33:45`
> **Best Performing:** `PAY/BTC: 50.23%`
> **Trading volume:** `0.5 BTC`
> **Profit factor:** `1.04`
> **Win / Loss:** `102 / 36`
> **Winrate:** `73.91%`
> **Expectancy (Ratio):** `4.87 (1.66)`
> **Max Drawdown:** `9.23% (0.01255 BTC)`

相对利润 `1.2%` 是每笔交易的平均利润。
相对利润 `15.2 Σ%` 是基于起始资本计算的 — 因此在本例中，起始资本为 `0.00485701 * 1.152 = 0.00738 BTC`。
**起始资本(**) 取自 `available_capital` 设置，或通过使用当前钱包大小减去利润来计算。
**Profit Factor** 计算为总利润 / 总亏损 — 应作为策略的整体指标。
**Expectancy** 对应于每单位风险货币的平均回报，即胜率和风险回报比（盈利交易的平均收益与亏损交易的平均损失之比）。
**Expectancy Ratio** 是基于所有过去交易的表现对后续交易的预期盈亏。
**Max drawdown** 对应于回测指标 `Absolute Drawdown (Account)` — 计算方式为 `(Absolute Drawdown) / (DrawdownHigh + startingBalance)`。
**Bot started date** 指的是机器人首次启动的日期。对于较旧的机器人，这将默认为第一笔交易的开仓日期。

### /forceexit <trade_id>

> **BINANCE:** Exiting BTC/LTC with limit `0.01650000 (profit: ~-4.07%, -0.00008168)`

!!! Tip
    你可以通过不带参数调用 `/forceexit` 来获取所有未平仓交易的列表，该列表将显示一个按钮列表，以便简单地退出交易。
    此命令有别名 `/fx` — 具有相同的功能，但在 "紧急" 情况下输入更快。

### /forcelong <pair> [rate] | /forceshort <pair> [rate]

`/forcebuy <pair> [rate]` 也支持多头交易，但应被视为已弃用。

> **BINANCE:** Long ETH/BTC with limit `0.03400000` (`1.000000 ETH`, `225.290 USD`)

省略交易对将打开一个查询，要求输入要交易的交易对（基于当前白名单）。
通过 `/forcelong` 创建的交易将带有买入标签 `force_entry`。

![Telegram 强制买入截图](../assets/telegram_forcebuy.png)

请注意，要使此功能正常工作，`force_entry_enable` 需要设置为 true。

[更多详情](configuration.md#understand-force_entry_enable)

### /performance

返回机器人已卖出的每种加密货币的表现。
> Performance:
> 1. `RCN/BTC 0.003 BTC (57.77%) (1)`
> 2. `PAY/BTC 0.0012 BTC (56.91%) (1)`
> 3. `VIB/BTC 0.0011 BTC (47.07%) (1)`
> 4. `SALT/BTC 0.0010 BTC (30.24%) (1)`
> 5. `STORJ/BTC 0.0009 BTC (27.24%) (1)`
> ...

相对表现是根据该币种的总投资计算的，汇总该币种所有已成交的入场订单。

### /balance

返回你在交易所上所有加密货币的余额。

> **Currency:** BTC
> **Available:** 3.05890234
> **Balance:** 3.05890234
> **Pending:** 0.0
>
> **Currency:** CVC
> **Available:** 86.64180098
> **Balance:** 86.64180098
> **Pending:** 0.0

### /daily <n>

默认情况下 `/daily` 将返回最近 7 天。以下是 `/daily 3` 的示例：

> **Daily Profit over the last 3 days:**

```
Day (count)     USDT          USD         Profit %
--------------  ------------  ----------  ----------
2022-06-11 (1)  -0.746 USDT   -0.75 USD   -0.08%
2022-06-10 (0)  0 USDT        0.00 USD    0.00%
2022-06-09 (5)  20 USDT       20.10 USD   5.00%
```

### /weekly <n>

默认情况下 `/weekly` 将返回最近 8 周，包括当前周。每周从周一开始。以下是 `/weekly 3` 的示例：

> **Weekly Profit over the last 3 weeks (starting from Monday):**

```
Monday (count)  Profit BTC      Profit USD   Profit %
-------------  --------------  ------------    ----------
2018-01-03 (5)  0.00224175 BTC  29,142 USD   4.98%
2017-12-27 (1)  0.00033131 BTC   4,307 USD   0.00%
2017-12-20 (4)  0.00269130 BTC  34.986 USD   5.12%
```

### /monthly <n>

默认情况下 `/monthly` 将返回最近 6 个月，包括当前月。以下是 `/monthly 3` 的示例：

> **Monthly Profit over the last 3 months:**
```
Month (count)  Profit BTC      Profit USD    Profit %
-------------  --------------  ------------    ----------
2018-01 (20)    0.00224175 BTC  29,142 USD  4.98%
2017-12 (5)    0.00033131 BTC   4,307 USD   0.00%
2017-11 (10)    0.00269130 BTC  34.986 USD  5.10%
```

### /whitelist

显示当前白名单

> Using whitelist `StaticPairList` with 22 pairs
> `IOTA/BTC, NEO/BTC, TRX/BTC, VET/BTC, ADA/BTC, ETC/BTC, NCASH/BTC, DASH/BTC, XRP/BTC, XVG/BTC, EOS/BTC, LTC/BTC, OMG/BTC, BTG/BTC, LSK/BTC, ZEC/BTC, HOT/BTC, IOTX/BTC, XMR/BTC, AST/BTC, XLM/BTC, NANO/BTC`

### /blacklist [pair]

显示当前黑名单。
如果设置了交易对，则该交易对将被添加到交易对列表中。
也支持多个交易对，用空格分隔。
使用 `/reload_config` 重置黑名单。

> Using blacklist `StaticPairList` with 2 pairs
>`DODGE/BTC`, `HOT/BTC`.

### /version

> **Version:** `0.14.3`

### /marketdir

如果提供了市场方向，该命令将更新代表当前市场方向的用户管理变量。
此变量在机器人启动时不会设置为任何有效的市场方向，必须由用户设置。以下是 `/marketdir long` 的示例：

```
Successfully updated marketdirection from none to long.
```

如果未提供市场方向，该命令将输出当前设置的市场方向。以下是 `/marketdir` 的示例：

```
Currently set marketdirection: even
```

你可以在策略中通过 `self.market_direction` 使用市场方向。

!!! Warning "机器人重启"
    请注意，市场方向不会被持久化，并且在机器人重启/重新加载后将被重置。

!!! Danger "回测"
    由于此值/变量旨在手动在模拟/实盘交易中更改。
    使用 `market_direction` 的策略可能不会产生可靠、可重现的结果（此变量的更改不会在回测中反映）。请自行承担风险。