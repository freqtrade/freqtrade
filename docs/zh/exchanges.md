# 交易所特定说明（Exchange-specific Notes）

本页合并了常见的陷阱以及特定于交易所、且很可能不适用于其他交易所的信息。

## 支持的交易所功能快速概览

--8<-- "includes/exchange-features.md"

## 交易所配置

Freqtrade 基于 [CCXT 库](https://github.com/ccxt/ccxt)，该库支持超过 100 个加密货币交易所市场和交易 API。完整的、最新的列表可以在 [CCXT 仓库主页](https://github.com/ccxt/ccxt/tree/master/python) 找到。但是，开发团队仅对少数几个交易所进行了测试。这些交易所的当前列表可以在本文档的“主页”部分找到。

欢迎测试其他交易所并提交你的反馈或 PR 以改进机器人，或确认可完美运行的交易所。

某些交易所需要特殊配置，可以在下面找到。

### 交易所配置示例

"binance" 的交易所配置如下所示：

```json
"exchange": {
    "name": "binance",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret",
    "ccxt_config": {},
    "ccxt_async_config": {},
    // ... 
```

### 设置速率限制

通常，CCXT 设置的速率限制是可靠的并且工作良好。在出现与速率限制相关的问题时（通常日志中会出现 DDOS 异常），很容易将 rateLimit 设置更改为其他值。

```json
"exchange": {
    "name": "kraken",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret",
    "ccxt_config": {"enableRateLimit": true},
    "ccxt_async_config": {
        "enableRateLimit": true,
        "rateLimit": 3100
    },
```

此配置启用了 kraken，以及速率限制以避免被交易所封禁。`"rateLimit": 3100` 定义了每次调用之间的等待事件为 3.1 秒。也可以通过将 `"enableRateLimit"` 设置为 false 来完全禁用它。

!!! Note
    速率限制的最佳设置取决于交易所和白名单的大小，因此理想参数会因许多其他设置而异。
    我们尽可能尝试为每个交易所提供合理的默认值，如果你遇到封禁，请确保 `"enableRateLimit"` 已启用，并逐步增加 `"rateLimit"` 参数。

## Binance

!!! Warning "服务器位置和地理 IP 限制"
    请注意，Binance 根据服务器所在国家限制 API 访问。当前（非详尽）被封锁的国家包括加拿大、马来西亚、荷兰和美国。请前往 [binance 条款 > b. 资格](https://www.binance.com/en/terms) 查找最新的列表。

Binance 支持 [time_in_force](configuration.md#understand-order_time_in_force)。

!!! Tip "交易所止损"
    Binance 支持 `stoploss_on_exchange` 并使用 `stop-loss-limit` 订单。它提供了巨大的优势，因此我们建议通过启用交易所止损来从中受益。
    在合约（futures）上，Binance 同时支持 `stop-limit` 和 `stop-market` 订单。你可以在 `order_types.stoploss` 配置设置中使用 `"limit"` 或 `"market"` 来决定使用哪种类型。

### Binance 黑名单建议

对于 Binance，建议将 `"BNB/<STAKE>"` 添加到你的黑名单以避免问题，除非你愿意在账户中维持足够的额外 `BNB`，或者愿意禁用使用 `BNB` 支付手续费。
Binance 账户可能会使用 `BNB` 支付手续费，如果一笔交易恰好发生在 `BNB` 上，进一步的交易可能会消耗这个仓位，并使初始的 BNB 交易无法卖出，因为预期的数量已经不存在了。

如果没有足够的 `BNB` 来支付交易手续费，那么手续费将不会被 `BNB` 覆盖，也不会发生手续费减免。Freqtrade 永远不会买入 BNB 来支付手续费。为此，BNB 需要被手动买入和监控。

### Binance 站点

Binance 已拆分为 2 个，用户必须使用正确的 ccxt 交易所 ID 来对应他们的交易所，否则 API 密钥将无法被识别。

* [binance.com](https://www.binance.com/) - 国际用户。使用交易所 id：`binance`。
* [binance.us](https://www.binance.us/) - 美国用户。使用交易所 id：`binanceus`。

### Binance RSA 密钥

Freqtrade 支持 binance RSA API 密钥。

我们建议将它们用作环境变量。

``` bash
export FREQTRADE__EXCHANGE__SECRET="$(cat ./rsa_binance.private)"
```

但是，它们也可以通过配置文件进行配置。由于 json 不支持多行字符串，你将必须将所有的换行符替换为 `\n` 以拥有有效的 json 文件。

``` json
// ...
 "key": "<someapikey>",
 "secret": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBABACAFQA<...>s8KX8=\n-----END PRIVATE KEY-----"
// ...
```

### Binance Futures

Binance 有特定的（不幸地复杂的）[合约交易量化规则](https://www.binance.com/en/support/faq/4f462ebe6ff445d4a170be7d9e897272)，需要被遵守，其中包括禁止过多订单使用过低的 stake-amount。违反这些规则将导致交易限制。

在 Binance Futures 市场交易时，必须使用 orderbook，因为合约没有价格 ticker 数据。

``` jsonc
  "entry_pricing": {
      "use_order_book": true,
      "order_book_top": 1,
      "check_depth_of_market": {
          "enabled": false,
          "bids_to_ask_delta": 1
      }
  },
  "exit_pricing": {
      "use_order_book": true,
      "order_book_top": 1
  },
```

#### Binance 隔离合约设置

用户还必须将合约设置中的 "Position Mode" 设置为 "One-way Mode"，并将 "Asset Mode" 设置为 "Single-Asset Mode"。这些设置将在启动时进行检查，如果此设置错误，freqtrade 将显示错误。

![Binance futures settings](../assets/binance_futures_settings.png)

Freqtrade 不会尝试更改这些设置。

#### Binance BNFCR 合约

BNFCR 模式是 Binance 上一种特殊类型的合约模式，用于规避欧洲的监管问题。
要使用 BNFCR 合约，你必须拥有以下设置组合：

``` jsonc
{
    // ...
    "trading_mode": "futures",
    "margin_mode": "cross",
    "proxy_coin": "BNFCR",
    "stake_currency": "USDT" // 或 "USDC"
    // ...
}
```

`stake_currency` 设置定义了机器人将操作的货币市场。这个选择实际上是任意的。

在交易所上，你将必须使用 "Multi-asset Mode" - 并且 "Position Mode 设置为 "One-way Mode"。
Freqtrade 将在启动时检查这些设置，但不会尝试更改它们。

## Bingx

BingX 支持 [time_in_force](configuration.md#understand-order_time_in_force)，设置为 "GTC"（good till cancelled，撤销前有效）、"IOC"（immediate-or-cancel，立即成交或取消）和 "PO"（Post only，只挂单）。

!!! Tip "交易所止损"
    Bingx 支持 `stoploss_on_exchange` 并且可以同时使用 stop-limit 和 stop-market 订单。它提供了巨大的优势，因此我们建议通过启用交易所止损来从中受益。

## Kraken

Kraken 支持 [time_in_force](configuration.md#understand-order_time_in_force)，设置为 "GTC"（good till cancelled）、"IOC"（immediate-or-cancel）和 "PO"（Post only）。

!!! Tip "交易所止损"
    Kraken 支持 `stoploss_on_exchange` 并且可以同时使用 stop-loss-market 和 stop-loss-limit 订单。它提供了巨大的优势，因此我们建议从中受益。
    你可以在 `order_types.stoploss` 配置设置中使用 `"limit"` 或 `"market"` 来决定使用哪种类型。

### Kraken 历史数据

Kraken API 仅提供 720 根历史蜡烛，这对于 Freqtrade dry-run 和 live trade 模式是足够的，但对于回测来说是个问题。要下载 Kraken 交易所的数据，使用 `--dl-trades` 是强制性的，否则机器人将一遍又一遍地下载相同的 720 根蜡烛，你将没有足够的回测数据。

为了加快下载速度，你可以下载 Kraken 提供的 [trades zip 文件](https://support.kraken.com/hc/en-us/articles/360047543791-Downloadable-historical-market-data-time-and-sales-)。这些文件通常每季度更新一次。Freqtrade 期望这些文件被放置在 `user_data/data/kraken/trades_csv` 中。

如果使用增量文件，以下结构可能是有意义的，一个目录中存放“完整”历史记录，不同的目录中存放增量文件。这种模式的前提是数据被下载并解压，保持文件名不变。重复的内容将被忽略（基于时间戳）——尽管前提是数据中没有间隙。

这意味着，如果你的“完整”历史记录结束于 2022 年 Q4 - 那么 Q1 2023 和 Q2 2023 的增量更新都是可用的。没有这些将导致数据不完整，从而在使用数据时产生无效结果。

```
└── trades_csv
    ├── Kraken_full_history
    │   ├── BCHEUR.csv
    │   └── XBTEUR.csv
    ├── Kraken_Trading_History_Q1_2023
    │   ├── BCHEUR.csv
    │   └── XBTEUR.csv
    └── Kraken_Trading_History_Q2_2023
        ├── BCHEUR.csv
        └── XBTEUR.csv
```

你可以将这些文件转换为 freqtrade 文件：

``` bash
freqtrade convert-trade-data --exchange kraken --format-from kraken_csv --format-to feather
# 将交易数据转换为不同的 ohlcv 时间周期
freqtrade trades-to-ohlcv -p BTC/EUR BCH/EUR --exchange kraken -t 1m 5m 15m 1h
```

转换后的数据也使下载数据成为可能，并将在加载的最新交易之后开始下载。

``` bash
freqtrade download-data --exchange kraken --dl-trades -p BTC/EUR BCH/EUR 
```

!!! Warning "从 kraken 下载数据"
    下载 kraken 数据将比其他任何交易所需要显著更多的内存（RAM），因为交易数据需要在你的机器上转换为蜡烛。
    这也将花费很长时间，因为 freqtrade 将需要为币对 / timerange 组合下载交易所发生的每一笔交易，因此请耐心等待。

!!! Warning "rateLimit 调优"
    请注意，rateLimit 配置项保存的是请求之间的延迟（以毫秒为单位），而不是每秒请求速率。
    因此，为了缓解 Kraken API 的 "Rate limit exceeded"（超出速率限制）异常，应该增加此配置，而不是减少。

## Kraken Futures

Kraken Futures 使用交易所 id `krakenfutures`，并支持隔离合约模式。

```jsonc
"exchange": {
    "name": "krakenfutures",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret"
},
"trading_mode": "futures",
"margin_mode": "isolated",
"stake_currency": "USD"
```

!!! Tip "交易所止损"
    Kraken Futures 支持 `stoploss_on_exchange`，同时使用 `limit` 和 `market` 止损订单。
    使用 `order_types.stoploss_price_type` 来选择触发价格来源（`mark`、`last` 或 `index`）。

!!! Note "抵押品"
    Kraken Futures 以 USD 结算。使用 USD 作为你的 stake 货币。

!!! Note "Flex（多抵押品）账户"
    Kraken Futures flex 账户允许以多种货币作为抵押品，而交易仍然以 USD 结算。
    Freqtrade 从 Kraken 保证金字段推导出 `USD` 余额，因此请将 `stake_currency` 设置为 `USD`。

## Kucoin

Kucoin 要求每个 api 密钥有一个 passphrase，因此你将需要将此密钥添加到配置中，使你的交易所部分看起来如下：

```json
"exchange": {
    "name": "kucoin",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret",
    "password": "your_exchange_api_key_password",
    // ...
}
```

Kucoin 支持 [time_in_force](configuration.md#understand-order_time_in_force)，设置为 "GTC"（good till cancelled）、"FOK"（full-or-cancel，全数成交或取消）和 "IOC"（immediate-or-cancel）。

!!! Tip "交易所止损"
    Kucoin 支持 `stoploss_on_exchange` 并且可以同时使用 stop-loss-market 和 stop-loss-limit 订单。它提供了巨大的优势，因此我们建议从中受益。
    你可以在 `order_types.stoploss` 配置设置中使用 `"limit"` 或 `"market"` 来决定应使用哪种类型的止损。

### Kucoin 黑名单

对于 Kucoin，建议将 `"KCS/<STAKE>"` 添加到你的黑名单以避免问题，除非你愿意在账户中维持足够的额外 `KCS`，或者愿意禁用使用 `KCS` 支付手续费。Kucoin 账户可能会使用 `KCS` 支付手续费，如果一笔交易恰好发生在 `KCS` 上，进一步的交易可能会消耗这个仓位，并使初始的 `KCS` 交易无法卖出，因为预期的数量已经不存在了。

## HTX

!!! Tip "交易所止损"
    HTX 支持 `stoploss_on_exchange` 并使用 `stop-limit` 订单。它提供了巨大的优势，因此我们建议通过启用交易所止损来从中受益。

## OKX

!!! Tip "交易所止损"
    OKX 支持 `stoploss_on_exchange`，在现货和合约市场上同时使用 stop-limit 和 stop-market 订单。你可以在 `order_types.stoploss` 配置设置中使用 `"limit"` 或 `"market"` 来选择止损订单类型。

OKX 要求每个 api 密钥有一个 passphrase，因此你将需要将此密钥添加到配置中，使你的交易所部分看起来如下：

```json
"exchange": {
    "name": "okx",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret",
    "password": "your_exchange_api_key_password",
    // ...
}
```

如果你在主机 my.okx.com（OKX EAA）上注册 - 你将需要使用 `"myokx"` 作为交易所名称。使用错误的交易所将导致错误 "OKX Error 50119: API key doesn't exist" - 因为它们是两个独立的实体。

!!! Warning
    OKX 每次 API 调用仅提供 100 根蜡烛。因此，在回测模式下，策略将只有相当少量的数据可用。

!!! Warning "合约"
    OKX Futures 有 "position mode"（仓位模式）的概念 - 可以是 "Buy/Sell" 或 long/short（对冲模式）。
    Freqtrade 支持两种模式（我们推荐使用 Buy/Sell 模式）- 但在交易中途更改模式不受支持，并将导致异常和无法下单。
    OKX 也只为过去约 3 个月提供 MARK 蜡烛。在此之前回测合约将因此导致轻微偏差，因为如果没有这些数据，资金费率（funding-fees）无法被正确计算。

## Gate.io

!!! Tip "交易所止损"
    Gate.io 支持 `stoploss_on_exchange` 并使用 `stop-loss-limit` 订单。它提供了巨大的优势，因此我们建议通过启用交易所止损来从中受益。

Gate.io 支持 [time_in_force](configuration.md#understand-order_time_in_force)，设置为 "GTC"（good till cancelled）和 "IOC"（immediate-or-cancel）。

Gate.io 允许使用 `POINT` 支付手续费。由于这不是一种可交易的货币（没有常规市场可用），自动手续费计算将失败（并默认为 0 手续费）。
配置参数 `exchange.unknown_fee_rate` 可用于指定 Point 与 stake 货币之间的汇率。显然，更改 stake-currency 也将需要更改此值。

Gate API 密钥在你要交易的市场的类型之上需要以下权限：

* "Spot Trade"（现货交易）_或_ "Perpetual Futures"（永续合约）（读和写）（选择两者，或匹配你要交易的市场的一个）
* "Wallet"（钱包）（只读）
* "Account"（账户）（只读）

没有这些权限，机器人将无法正确启动并显示诸如 "permission missing"（缺少权限）之类的错误。

## Bybit

!!! Tip "交易所止损"
    Bybit（仅限合约）支持 `stoploss_on_exchange` 并使用 `stop-loss-limit` 订单。它提供了巨大的优势，因此我们建议通过启用交易所止损来从中受益。
    在合约上，Bybit 同时支持 `stop-limit` 和 `stop-market` 订单。你可以在 `order_types.stoploss` 配置设置中使用 `"limit"` 或 `"market"` 来决定使用哪种类型。

Bybit 支持 [time_in_force](configuration.md#understand-order_time_in_force)，设置为 "GTC"（good till cancelled）、"FOK"（full-or-cancel）、"IOC"（immediate-or-cancel）和 "PO"（Post only）。

!!! Warning "统一账户"
    Freqtrade 假设账户专用于机器人。
    因此我们建议每个机器人使用一个子账户。这在使用统一账户时尤其重要。
    其他配置（一个账户上多个机器人、在机器人账户上手动进行非机器人交易）不受支持，并可能导致意外行为。

### Bybit Futures

Bybit 上的合约交易支持隔离合约模式。

在启动时，freqtrade 将为整个（子）账户将仓位模式设置为 "One-way Mode"。这避免了反复进行此调用（减慢机器人操作），但意味着对此设置的手动更改可能导致异常和错误。

由于 bybit 不提供资金费率历史，dry-run 计算也用于 live 交易。

实时合约交易的 API 密钥必须具有以下权限：

* 读写（Read-write）
* Contract - Orders（合约 - 订单）
* Contract - Positions（合约 - 仓位）

我们强烈建议将所有 API 密钥限制为你将从中使用它的 IP。

### Bybit 演示模式

Bybit 有一个 [演示模式](https://learn.bybit.com/en/bybit-guide/how-to-use-bybit-demo-trading) - 可以通过在配置中将 `exchange.demo_trading` 设置为 `true` 来激活。Bybit 使用 live 市场来模拟你的交易（没有市场影响）- 使其工作方式与 freqtrade 的 dry-run 模式非常相似。

你将需要为演示交易使用单独的 API 密钥，你可以在 bybit 的演示页面上创建它们。

演示模式与 dry-run 不兼容。

## Bitmart

Bitmart 要求 API 密钥 Memo（你给 API 密钥起的名字）与交易所密钥和 secret 一起提供。因此还需要传递 UID。

```json
"exchange": {
    "name": "bitmart",
    "uid": "your_bitmart_api_key_memo",
    "secret": "your_exchange_secret",
    "password": "your_exchange_api_key_password",
    // ...
}
```

!!! Warning "必要的验证"
    Bitmart 需要通过 Lvl2 验证才能在现货市场上通过 API 成功交易 - 即使仅通过 UI 交易在 Lvl1 验证下也能正常工作。

## Bitget

Bitget 要求每个 api 密钥有一个 passphrase，因此你将需要将此密钥添加到配置中，使你的交易所部分看起来如下：

```json
"exchange": {
    "name": "bitget",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret",
    "password": "your_exchange_api_key_password",
    // ...
}
```

Bitget 支持 [time_in_force](configuration.md#understand-order_time_in_force)，设置为 "GTC"（good till cancelled）、"FOK"（full-or-cancel）、"IOC"（immediate-or-cancel）和 "PO"（Post only）。

!!! Tip "交易所止损"
    Bitget 支持 `stoploss_on_exchange` 并且可以同时使用 stop-loss-market 和 stop-loss-limit 订单。它提供了巨大的优势，因此我们建议从中受益。
    你可以在 `order_types.stoploss` 配置设置中使用 `"limit"` 或 `"market"` 来决定应使用哪种类型的止损。

### Bitget Futures

Bitget 上的合约交易支持隔离合约模式。

在启动时，freqtrade 将为整个（子）账户将仓位模式设置为 "One-way Mode"。这避免了反复进行此调用（减慢机器人操作），但意味着对此设置的手动更改可能导致异常和错误。

## Hyperliquid

!!! Tip "交易所止损"
    Hyperliquid 支持 `stoploss_on_exchange` 并使用 `stop-loss-limit` 订单。它提供了巨大的优势，因此我们建议从中受益。

!!! Warning "统一账户"
    支持 Hyperliquid 统一账户 - 尽管这依赖于 freqtrade 对“拥有”该账户的假设，并且是唯一在该账户上交易的人（在这种情况下，扩展到现货和合约）。
    因此我们建议在可能的情况下使用子账户，并避免在该机器人运行时在同一账户上手动交易。
    Freqtrade 将在启动时尝试检测账户类型 - 在交易中更改账户类型不受支持，并可能导致异常和错误。

Hyperliquid 是一个去中心化交易所（DEX）。去中心化交易所与普通交易所的工作方式略有不同。不是使用 API 密钥对私有 API 调用进行身份验证，私有 API 调用需要用你的钱包私钥签名（我们建议使用 API 钱包来执行此操作，可以在 Hyperliquid 上或你选择的钱包中生成）。这需要这样配置：

```json
"exchange": {
    "name": "hyperliquid",
    "walletAddress": "your_eth_wallet_address",  // 这不应该是你的 API 钱包地址！
    "privateKey": "your_api_private_key",
    // ...
}
```

* 十六进制格式的 walletAddress：`0x<40 个十六进制字符>` - 可以轻松地从你的钱包复制 - 并且应该是你的主钱包地址，而不是你的 API 钱包地址。
* 十六进制格式的 privateKey：`0x<64 个十六进制字符>` - 使用 API 钱包在创建时显示的密钥。

Hyperliquid 在 Arbitrum One 链上处理存款和取款，Arbitrum One 是构建在以太坊之上的 Layer 2 扩展解决方案。Hyperliquid 使用 USDC 作为报价 / 抵押品。在 Hyperliquid 上存入 USDC 的过程需要几个步骤，有关所需步骤的详细信息，请参阅 [如何开始交易](https://hyperliquid.gitbook.io/hyperliquid-docs/onboarding/how-to-start-trading)。

!!! Note "Hyperliquid 一般使用说明"
    Hyperliquid 不支持市价订单，但是 ccxt 将通过放置最大滑点为 5% 的限价订单来模拟市价订单。
    不幸的是，hyperliquid 仅提供 5000 根历史蜡烛，因此回测将需要历史地构建蜡烛（通过等待并随着时间的推移增量下载数据）- 或者将限制为最后 5000 根蜡烛。

!!! Info "一些一般最佳实践（非详尽）"
    * 警惕供应链攻击，如 pip 包投毒等。每当你使用你的私钥时，请确保你的环境是安全的。
    * 不要使用你的实际钱包私钥进行交易。使用 Hyperliquid [API 生成器](https://app.hyperliquid.xyz/API) 创建一个单独的 API 钱包。
    * 不要将你的实际钱包私钥存储在你用于 freqtrade 的服务器上。改为使用 API 钱包私钥。这个密钥不允许取款，只能交易。
    * 始终将你的助记词和私钥保密。
    * 不要使用与你初始化硬件钱包时必须备份的助记词相同的助记词，使用相同的助记词基本上会删除你硬件钱包的安全性。
    * 创建一个不同的软件钱包，只将你想交易的资金转移到该钱包，并使用该钱包在 Hyperliquid 上交易。
    * 如果你有不想用于交易的资金（例如获利后），将它们转回你的硬件钱包。

!!! Warning "Vaults 和子账户"
    你只能使用 vault 或子账户 - 不能同时两者。

### Hyperliquid 子账户

Hyperliquid 允许你在有足够的先前交易量的前提下创建子账户。
要将子账户与 Freqtrade 一起使用，你将需要使用以下配置模式：

``` json
"exchange": {
    "name": "hyperliquid",
    "walletAddress": "your_master_wallet_address", // 你的主钱包地址（不是 API 钱包或 vault 地址 - 也不是子账户地址）。
    "privateKey": "your_api_private_key", // API 钱包私钥（参见 https://app.hyperliquid.xyz/API）。你只需要私钥。
    "ccxt_config": {
        "options": {
            "subAccountAddress": "your_subaccount_address" // 如果你想使用子账户，则必需。
        }
    },
    // ...
}
```

你的余额和交易现在将使用你的子账户 - 而不再使用你的主账户。

### Hyperliquid Vault

Hyperliquid 允许你创建 vaults。要将 vaults 与 Freqtrade 一起使用，你将需要使用以下配置模式：

``` json
"exchange": {
    "name": "hyperliquid",
    "walletAddress": "your_vault_address", // 你的 vault 钱包地址（还必须在下面的 ccxt_config.options.vaultAddress 字段中添加）
    "privateKey": "your_api_private_key", // API 钱包私钥（参见 https://app.hyperliquid.xyz/API）。你只需要私钥。
    "ccxt_config": {
        "options": {
            "vaultAddress": "your_vault_address", // 可选，仅当你想使用 vault 时...（vault 地址也必须添加到 walletAdress）
        }
    },
    // ...
}
```

你的余额和交易现在将使用你的 vault - 而不再使用你的主账户。

### Hyperliquid 历史数据

Hyperliquid API 不提供超出单次调用以获取当前数据之外的历史数据，因此无法下载数据，因为下载的数据不会构成适当的历史数据。

### HIP-3 DEXes

Hyperliquid 支持 HIP-3 去中心化交易所（DEXes），这些是在 Hyperliquid 基础设施之上构建的独立交易所。这些 DEXes 的运行方式与主 Hyperliquid 交易所类似，但由社区创建和管理。

要使用 Freqtrade 在 HIP-3 DEXes 上交易，你需要使用 `hip3_dexes` 参数将它们添加到你的配置中：

```json
"exchange": {
    "name": "hyperliquid",
    "walletAddress": "your_master_wallet_address",
    "privateKey": "your_api_private_key",
    "hip3_dexes": ["dex_name_1", "dex_name_2"]
}
```

将 `"dex_name_1"` 和 `"dex_name_2"` 替换为你想要交易的 HIP-3 DEXes 的实际名称（例如 `vntl` 和 `xyz`）。

!!! Warning "性能和速率限制影响"
    你添加的每一个 HIP-3 DEX 都会显著影响机器人性能和速率限制。

    * **额外的 API 调用**：对于配置的每一个 HIP-3 DEX，Freqtrade 都需要进行额外的 API 调用。
    * **速率限制压力**：额外的 API 调用会加剧 Hyperliquid 严格的速率限制。使用多个 DEX，你可能会更快地达到速率限制，或者更确切地说，由于强制延迟而减慢机器人操作。

    请只添加你实际交易的 HIP-3 DEXes。监控你的日志以查找速率限制警告或操作减慢的迹象，并相应地调整你的配置。
    不同的 HIP-3 DEXes 也可能使用不同的报价货币 - 因此请确保只添加与你的 stake 货币兼容的 DEXes，以避免不必要的延迟。

!!! Note
    HIP-3 DEXes 与你的主 Hyperliquid 账户共享相同的钱包和免费抵押品金额。在不同 DEXes 上的交易将影响你的整体账户余额和保证金。

    HIP-3 币对的币对名称将与非 HIP-3 币对略有不同。请使用 `list-pairs` 子命令来获取指定 dexes 的所有币对的正确币对命名。

## Bitvavo

如果你的账户需要使用 operatorId，你可以按如下方式在配置文件中设置它：

``` json
"exchange": {
        "name": "bitvavo",
        "key": "",
        "secret": "",
        "ccxt_config": {
            "options": {
                "operatorId": "123567"
            }
        },
   }
```

Bitvavo 期望 `operatorId` 是一个整数。

## 所有交易所

如果你遇到与 Nonce 相关的持续错误（如 `InvalidNonce`），最好重新生成 API 密钥。重置 Nonce 很困难，通常重新生成 API 密钥更容易。

## 其他交易所的随机说明

* The Ocean（交易所 id：`theocean`）交易所使用 Web3 功能，需要安装 `web3` python 包：

```shell
pip3 install web3
```

### 获取最新价格 / 不完整的蜡烛

大多数交易所通过其 OHLCV/klines API 接口返回当前不完整的蜡烛。默认情况下，Freqtrade 假设从交易所获取了不完整的蜡烛，并删除最后一根蜡烛，假设它是不完整的蜡烛。

你的交易所是否返回不完整的蜡烛可以通过使用贡献者文档中的[辅助脚本](developer.md#incomplete-candles)来检查。

由于 repainting（重绘）的危险，Freqtrade 不允许你使用这根不完整的蜡烛。

但是，如果这是基于你的策略需要最新价格的需求 - 那么这个需求可以通过策略内的[数据提供者](strategy-customization.md#possible-options-for-dataprovider)来获取。

### 高级 Freqtrade 交易所配置

可以使用 `_ft_has_params` 设置来配置高级选项，该设置将覆盖默认值和特定于交易所的行为。

可用选项在 exchange 类中列为 `_ft_has_default`。

例如，要测试 Kraken 的订单类型 `FOK`，并将蜡烛限制修改为 200（因此你每次 API 调用只获得 200 根蜡烛）：

```json
"exchange": {
    "name": "kraken",
    "_ft_has_params": {
        "order_time_in_force": ["GTC", "FOK"],
        "ohlcv_candle_limit": 200
        }
    //...
}
```

!!! Warning
    在修改这些设置之前，请确保完全理解它们的影响。
    使用 `_ft_has_params` 覆盖可能会导致意外行为，甚至可能破坏你的机器人。
    对于由 `_ft_has_params` 中的自定义设置引起的问题，我们将无法提供支持。
