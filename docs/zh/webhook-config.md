<!-- 本文件为中文翻译版，由 AI 根据 docs/webhook-config.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# Webhook 用法

!!! Tip "新用户推荐"
    Freqtrade 现在包含一个新的 `message_stream` 类型，当启用时会将所有消息广播到消息流中。这使得在 Freqtrade 内部消费消息变得比以往更容易，是大多数用例的推荐方式。如果你仍希望将消息转发到外部接收者，请继续阅读。

## 配置

使用 Webhook 支持，Freqtrade 能够将交易状态（以及发生的消息）以 json 格式发送到指定的 webhook URL 或消息流。启用此功能需要设置以下配置值：

```json
  "webhook": {
        "enabled": true,
        "url": "https://<your_webhook_url>",
        "entry": {
            "value1": "Buy alert",
            "value2": "Some Other value"
        },
        "entry_cancel": {
            "value1": "Cancelled Entry Order"
        },
        "exit": {
            "value1": "Sell alert",
            "value2": "Some Other value"
        },
        "exit_cancel": {
            "value1": "Cancelled Exit Order"
        },
        "exit_fill": {
            "value1": "Exit Fill alert"
        },
        "status": {
            "value1": "Status message"
        },
        "protection_trigger": {
            "value1": "Protection Triggered"
        },
        "protection_trigger_global": {
            "value1": "Global Protection Triggered"
        }
    }
```

Webhook 通过 json 回调发送事件。详细的 json 格式说明可以在 Webhook 配置的下一部分找到。

### 配置 Webhook 的 URL

如果你的 webhook 提供商需要某些内容来始终存在于数据中（如密钥/令牌），你可以在启动时通过环境变量 `FREQTRADE__WEBHOOK__URL` 或 `FREQTRADE__WEBHOOK__<some_prefix>__URL`（例如 `FREQTRADE__WEBHOOK__ENTRY__URL`）来设置 URL。

```bash
export FREQTRADE__WEBHOOK__ENTRY__URL="https://<your_webhook_url>?apikey=xxxx"
```

这会将 URL 设置为 `https://<your_webhook_url>?apikey=xxxx`，适用于 webhook 进入消息。

### 配置 Webhook 条目

可以配置不同的消息类型，允许你将不同的事件发送到不同的 URL。

可以在 `webhook` 配置中为每种消息类型设置不同的 URL：

- `webhookentry`（默认按顺序回退到 `webhookentryfill`、`webhookurl`）
- `webhookentrycancel`（默认按顺序回退到 `webhookentryfill`、`webhookurl`）
- `webhookentryfill`（默认按顺序回退到 `webhookurl`）
- `webhookexit`（默认按顺序回退到 `webhookexitcancel`、`webhookexitfill`、`webhookurl`）
- `webhookexitcancel`（默认按顺序回退到 `webhookexitfill`、`webhookurl`）
- `webhookexitfill`（默认回退到 `webhookurl`）
- `webhookstatus`（默认回退到 `webhookurl`）
- `webhookprotectiontrigger`（默认回退到 `webhookurl`）
- `webhookprotectiontriggerglobal`（默认回退到 `webhookurl`）
- `webhookurl`

这意味着如果你为 `webhookurl` 设置了 URL，所有的消息类型都将被发送到这个 URL。当某些消息类型有不同的 URL 时，这些特定的消息将发送到各自的 URL，而其余的消息仍然发送到 `webhookurl`。

如果一种消息类型没有指定 URL，它会按照列表中的顺序回退（从具体到通用），直到找到设置了 URL 的消息类型。

#### 使用环境变量配置

你可以通过前缀 `FREQTRADE__WEBHOOK__` 来设置 webhook URL 环境变量，将 url 附加到消息类型之后。

例如：

```bash
export FREQTRADE__WEBHOOK__URL="https://example.com/webhook"
export FREQTRADE__WEBHOOK__ENTRY__URL="https://entry.example.com/webhook"
export FREQTRADE__WEBHOOK__EXIT__URL="https://exit.example.com/webhook"
```

这会将所有的 webhook 发送到 `https://example.com/webhook`，但进入消息发送到 `https://entry.example.com/webhook`，退出消息发送到 `https://exit.example.com/webhook`。

## 特定消息格式

### Entry（进入）

进入消息在订单被放置时发送。如果订单由于交易所返回拒绝、余额不足等原因被取消，则会发送 `entry_cancel` 消息。

```json
{
    "type": "entry",
    "trade_id": 120,
    "exchange": "binance",
    "pair": "ETH/USDT",
    "leverage": 1.0,
    "direction": "Long",
    "limit": 2400,
    "open_rate": 2400,
    "order_type": "limit",
    "stake_amount": 60.0,
    "stake_currency": "USDT",
    "base_currency": "ETH",
    "quote_currency": "USDT",
    "fiat_currency": "USD",
    "amount": 0.025,
    "open_date": "2022-01-01 12:00:00",
    "current_rate": 2400
}
```

### Entry Fill（进入成交）

当订单被成交时发送。

```json
{
    "type": "entry_fill",
    "trade_id": 120,
    "exchange": "binance",
    "pair": "ETH/USDT",
    "leverage": 1.0,
    "direction": "Long",
    "limit": 2400,
    "open_rate": 2400,
    "order_type": "limit",
    "stake_amount": 60.0,
    "stake_currency": "USDT",
    "base_currency": "ETH",
    "quote_currency": "USDT",
    "fiat_currency": "USD",
    "amount": 0.025,
    "open_date": "2022-01-01 12:00:00",
    "current_rate": 2400,
    "cumulative_entry_filled": 0.025,
    "cumulative_entry_total": 0.025
}
```

### Entry Cancel（进入取消）

当进入订单在交易所被取消时发送。这通常在订单由于余额不足或其他错误条件被交易所取消时发生。

```json
{
    "type": "entry_cancel",
    "trade_id": 120,
    "exchange": "binance",
    "pair": "ETH/USDT",
    "leverage": 1.0,
    "direction": "Long",
    "limit": 2400,
    "open_rate": 2400,
    "order_type": "limit",
    "stake_amount": 60.0,
    "stake_currency": "USDT",
    "base_currency": "ETH",
    "quote_currency": "USDT",
    "fiat_currency": "USD",
    "amount": 0.025,
    "open_date": "2022-01-01 12:00:00",
    "current_rate": 2400,
    "reason": "cancelled_exchange"
}
```

### Exit（退出）

退出消息在退出订单被放置时发送。

```json
{
    "type": "exit",
    "trade_id": 120,
    "exchange": "binance",
    "pair": "ETH/USDT",
    "leverage": 1.0,
    "direction": "Long",
    "gain": "profit",
    "limit": 2450,
    "open_rate": 2400,
    "order_type": "limit",
    "amount": 0.025,
    "open_date": "2022-01-01 12:00:00",
    "current_rate": 2450,
    "profit_amount": 1.25,
    "profit_ratio": 0.0208,
    "stake_currency": "USDT",
    "base_currency": "ETH",
    "quote_currency": "USDT",
    "fiat_currency": "USD",
    "enter_tag": "buy_signal_01",
    "exit_reason": "exit_signal",
    "cumulative_entry_filled": 0.025,
    "cumulative_entry_total": 0.025
}
```

### Exit Cancel（退出取消）

当退出订单在交易所被取消时发送。

```json
{
    "type": "exit_cancel",
    "trade_id": 120,
    "exchange": "binance",
    "pair": "ETH/USDT",
    "leverage": 1.0,
    "direction": "Long",
    "gain": "profit",
    "limit": 2450,
    "open_rate": 2400,
    "order_type": "limit",
    "amount": 0.025,
    "open_date": "2022-01-01 12:00:00",
    "current_rate": 2450,
    "profit_amount": 1.25,
    "profit_ratio": 0.0208,
    "stake_currency": "USDT",
    "base_currency": "ETH",
    "quote_currency": "USDT",
    "fiat_currency": "USD",
    "enter_tag": "buy_signal_01",
    "exit_reason": "exit_signal",
    "cumulative_entry_filled": 0.025,
    "cumulative_entry_total": 0.025,
    "reason": "cancelled_exchange"
}
```

### Exit Fill（退出成交）

当退出订单被成交时发送。

```json
{
    "type": "exit_fill",
    "trade_id": 120,
    "exchange": "binance",
    "pair": "ETH/USDT",
    "leverage": 1.0,
    "direction": "Long",
    "gain": "profit",
    "limit": 2450,
    "open_rate": 2400,
    "order_type": "limit",
    "amount": 0.025,
    "open_date": "2022-01-01 12:00:00",
    "current_rate": 2450,
    "profit_amount": 1.25,
    "profit_ratio": 0.0208,
    "stake_currency": "USDT",
    "base_currency": "ETH",
    "quote_currency": "USDT",
    "fiat_currency": "USD",
    "enter_tag": "buy_signal_01",
    "exit_reason": "exit_signal",
    "cumulative_entry_filled": 0.025,
    "cumulative_entry_total": 0.025
}
```

### Status（状态）

状态消息在特定的间隔发送，包含机器人状态信息。可以在[配置](configuration.md)中使用 `"webhookstatus"` 键进行配置。

```json
{
    "type": "status",
    "status": "running"
}
```

status 的值可以是 `running`、`stopped`、`paused` 或 `stopped_buy`。

### Protection Trigger（保护触发）

当触发保护时发送。

```json
{
    "type": "protection_trigger",
    "protection_name": "StoplossGuard",
    "pair": "ETH/USDT"
}
```

### Protection Trigger Global（全局保护触发）

当触发全局保护时发送。

```json
{
    "type": "protection_trigger_global",
    "protection_name": "StoplossGuard"
}
```

## 自定义消息

每个消息都可以根据你的需要进行自定义。你可以为每种消息类型包含额外的值，以及覆盖每个消息中包含的默认值。

```json
  "webhook": {
        "enabled": true,
        "url": "https://<your_webhook_url>",
        "entry": {
            "value1": "Buy alert",
            "value2": "Some Other value"
    },
```

这样，"value1" 和 "value2" 将始终出现在 entry 消息中。你可以使用 `{value}` 占位符来替换动态值（如下所述）。

## 使用占位符自定义消息

你可以在 `webhook` 消息中使用 [消息占位符](#消息占位符) 来自定义消息内容。占位符是会被替换为实际值的动态变量。

例如，你可以这样做：

``` json
"webhook": {
    "enabled": true,
    "url": "https://hooks.slack.com/services/your/slack/webhook/url",
    "entry": {
        "text": "进入 {pair} 于 {current_rate}"
    }
},
```

`{pair}` 和 `{current_rate}` 将被替换为实际值。

### 消息占位符

占位符使用 `{placeholder}` 格式，支持哪些占位符取决于消息类型。

#### 所有消息通用的占位符

| 占位符 | 描述 |
|-----------|-------------|
| `{trade_id}` | 交易 ID |
| `{exchange}` | 交易所 |
| `{pair}` | 交易对 |
| `{leverage}` | 杠杆 |
| `{direction}` | 交易方向 (Long/Short) |
| `{limit}` | 订单限价 |
| `{open_rate}` | 开仓价 |
| `{order_type}` | 订单类型 |
| `{stake_amount}` | 本金金额 |
| `{stake_currency}` | 本金货币 |
| `{base_currency}` | 基础货币 |
| `{quote_currency}` | 计价货币 |
| `{fiat_currency}` | 法币 |
| `{amount}` | 数量 |
| `{open_date}` | 开仓日期 |
| `{current_rate}` | 当前价格 |

#### Entry Fill 特定占位符

| 占位符 | 描述 |
|-----------|-------------|
| `{cumulative_entry_filled}` | 累计已成交的进入数量 |
| `{cumulative_entry_total}` | 累计进入总数量 |

#### Exit 特定占位符

| 占位符 | 描述 |
|-----------|-------------|
| `{gain}` | 收益 (profit/loss) |
| `{profit_amount}` | 利润金额 |
| `{profit_ratio}` | 利润率 |
| `{enter_tag}` | 进入标签 |
| `{exit_reason}` | 退出原因 |
| `{cumulative_entry_filled}` | 累计已成交的进入数量 |
| `{cumulative_entry_total}` | 累计进入总数量 |

#### Exit Fill 特定占位符

| 占位符 | 描述 |
|-----------|-------------|
| `{gain}` | 收益 (profit/loss) |
| `{profit_amount}` | 利润金额 |
| `{profit_ratio}` | 利润率 |
| `{enter_tag}` | 进入标签 |
| `{exit_reason}` | 退出原因 |
| `{cumulative_entry_filled}` | 累计已成交的进入数量 |
| `{cumulative_entry_total}` | 累计进入总数量 |
