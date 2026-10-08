<!-- 本文件为中文翻译版，由 AI 根据 docs/data-download.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# 数据下载

## 获取用于回测和 hyperopt 的数据

要下载回测和超参数优化所需的数据（蜡烛 / OHLCV），请使用 `freqtrade download-data` 命令。

如果未指定额外参数，freqtrade 会下载最近 30 天 `"1m"` 和 `"5m"` 时间周期的数据。
交易所和交易对来自 `config.json`（如果使用 `-c/--config` 指定）。
在未提供配置的情况下，`--exchange` 变为必填项。

你可以使用相对时间范围（`--days 20`）或绝对起点（`--timerange 20200101-`）。对于增量下载，应该使用相对方式。

!!! Tip "提示：更新现有数据"
    如果你的数据目录中已经有可用的回测数据，并且想将这份数据刷新到今天，freqtrade 会自动为现有交易对计算缺失的时间范围，下载会从最新的可用点进行直到"现在"，既不需要 `--days` 也不需要 `--timerange` 参数。Freqtrade 会保留可用数据，只下载缺失的数据。
    如果你在插入了还没有数据的交易对之后更新现有数据，请使用 `--new-pairs-days xx` 参数。指定的天数会为新的交易对下载，而旧的交易对只会用缺失的数据更新。

### 用法

--8<-- "../commands/download-data.md"

!!! Tip "下载某个报价货币的所有数据"
    通常，你会想下载某个特定报价货币的所有交易对的数据。在这种情况下，你可以使用以下简写：
    `freqtrade download-data --exchange binance --pairs ".*/USDT" <...>`。提供的 "pairs" 字符串会被扩展为包含交易所上所有活跃的交易对。
    要同时下载不活跃（已下架）交易对的数据，请在命令中添加 `--include-inactive-pairs`。

!!! Note "启动期（Startup period）"
    `download-data` 是一个与策略无关的命令。其思路是一次性下载一大块数据，然后迭代地增加存储的数据量。

    因此，`download-data` 不会关心策略中定义的 "startup-period"。如果用户希望回测从某个特定时间点开始（同时尊重启动期），需要由用户自己下载额外的天数。

### 开始下载

一个非常简单的命令（假设有一个可用的 `config.json` 文件）可以如下所示。

```bash
freqtrade download-data --exchange binance
```

这会为配置中定义的所有货币对下载历史蜡烛（OHLCV）数据。

或者，直接指定交易对

```bash
freqtrade download-data --exchange binance --pairs ETH/USDT XRP/USDT BTC/USDT
```

或者以正则形式（此处为下载所有活跃的 USDT 交易对）

```bash
freqtrade download-data --exchange binance --pairs ".*/USDT"
```

### 其他说明

* 要使用不同于交易所特定默认值的目录，使用 `--datadir user_data/data/some_directory`。
* 要更改用于下载历史数据的交易所，可以使用 `--exchange <exchange>`，或指定一个不同的配置文件。
* 要使用来自其他目录的 `pairs.json`，使用 `--pairs-file some_other_dir/pairs.json`。
* 要只下载 10 天的历史蜡烛（OHLCV）数据，使用 `--days 10`（默认为 30 天）。
* 要从固定的起点下载历史蜡烛（OHLCV）数据，使用 `--timerange 20200101-`——这将下载从 2020 年 1 月 1 日起的所有数据。
* 如果数据已可用，则给定的起点会被忽略，只下载到今天为止的缺失数据。
* 使用 `--timeframes` 指定要为哪个时间周期下载历史蜡烛（OHLCV）数据。默认为 `--timeframes 1m 5m`，将下载 1 分钟和 5 分钟数据。
* 要使用配置文件中定义的交易所、时间周期和交易对列表，使用 `-c/--config` 选项。这样，脚本会使用配置中定义的白名单作为要下载数据的货币对列表，并且不需要 pairs.json 文件。你可以将 `-c/--config` 与大多数其他选项组合使用。
* 当下载合约数据（`--trading-mode futures` 或指定了合约模式的配置）时，freqtrade 会自动下载必要的蜡烛类型（例如 `mark` 和 `funding_rate` 蜡烛），除非通过 `--candle-types` 另行指定。

??? Note "权限被拒绝错误"
    如果你的配置目录 `user_data` 是由 docker 创建的，你可能会遇到以下错误：

    ```
    cp: cannot create regular file 'user_data/data/binance/pairs.json': Permission denied
    ```

    你可以按如下方式修复用户数据目录的权限：

    ```
    sudo chown -R $UID:$GID user_data
    ```

### 在当前时间范围之前下载额外数据

假设你已经下载了 2022 年的所有数据（`--timerange 20220101-`）——但你现在还想用更早的数据进行回测。
你可以通过使用 `--prepend` 标志，结合 `--timerange`——指定一个结束日期。

``` bash
freqtrade download-data --exchange binance --pairs ETH/USDT XRP/USDT BTC/USDT --prepend --timerange 20210101-20220101
```

!!! Note
    在此模式下，如果数据可用，freqtrade 会忽略结束日期，将结束日期更新为现有数据的起点。

### 数据格式

Freqtrade 目前支持以下数据格式：

* `feather` - 基于 Apache Arrow 的数据格式
* `json` - 纯文本 json 文件
* `jsongz` - json 文件的 gzip 压缩版本
* `parquet` - 列式数据存储（仅 OHLCV）

默认情况下，OHLCV 数据和 trades 数据都以 `feather` 格式存储。

这可以通过 `--data-format-ohlcv` 和 `--data-format-trades` 命令行参数分别更改。
要持久化此更改，你还应该将以下片段添加到你的配置中，这样你就不必每次都插入上述参数：

``` jsonc
    // ...
    "dataformat_ohlcv": "feather",
    "dataformat_trades": "feather",
    // ...
```

如果在下载期间更改了默认数据格式，则配置文件中的 `dataformat_ohlcv` 和 `dataformat_trades` 键也需要调整为所选的数据格式。

!!! Note
    你可以使用 [convert-data](#sub-command-convert-data) 和 [convert-trade-data](#sub-command-convert-trade-data) 方法在多种数据格式之间转换。

#### 数据格式比较

以下比较使用了以下数据，并通过 linux 的 `time` 命令进行。

```
                       Found 6 pair / timeframe combinations.
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃     Pair ┃ Timeframe ┃ Type ┃                From ┃                  To ┃ Candles ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ BTC/USDT │        1m │ spot │ 2017-08-17 04:00:00 │ 2026-07-15 04:42:00 │ 4677171 │
│ BTC/USDT │        5m │ spot │ 2017-08-17 04:00:00 │ 2026-07-15 04:35:00 │  935445 │
│ ETH/USDT │        1m │ spot │ 2017-08-17 04:00:00 │ 2026-07-15 04:43:00 │ 4677172 │
│ ETH/USDT │        5m │ spot │ 2017-08-17 04:00:00 │ 2026-07-15 04:35:00 │  935445 │
│ XRP/USDT │        1m │ spot │ 2018-05-04 08:11:00 │ 2026-07-15 04:44:00 │ 4305252 │
│ XRP/USDT │        5m │ spot │ 2018-05-04 08:10:00 │ 2026-07-15 04:40:00 │  861055 │
└──────────┴───────────┴──────┴─────────────────────┴─────────────────────┴─────────┘
```

计时是以一种不太科学的方式，通过以下命令（强制将数据读入内存）取得的。

``` bash
time freqtrade list-data --show-timerange --data-format-ohlcv <dataformat>
```

|  格式 | 大小 | 耗时 |
|------------|-------------|-------------|
| `feather` | 115Mb | 1.6s |
| `json` | 265Mb | 17.1s |
| `jsongz` | 83Mb | 23.6s |
| `parquet` | 149Mb | 2.5s |

大小取自上述时间范围内 BTC/USDT 1m spot 组合。

为了获得性能/大小的最佳组合，我们推荐使用默认的 **feather** 格式。

### 交易对文件（Pairs file）

作为 `config.json` 中白名单的替代方案，可以使用 `pairs.json` 文件。
例如，如果你使用的是 Binance：

* 创建一个目录 `user_data/data/binance`，并在该目录中复制或创建 `pairs.json` 文件。
* 更新 `pairs.json` 文件，使其包含你感兴趣的交易对。

```bash
mkdir -p user_data/data/binance
touch user_data/data/binance/pairs.json
```

`pairs.json` 文件的格式是一个简单的 json 列表。
该文件允许混合不同的 stake 计价货币，因为它仅用于下载。

``` json
[
    "ETH/BTC",
    "ETH/USDT",
    "BTC/USDT",
    "XRP/ETH"
]
```

!!! Note
    `pairs.json` 文件仅在未加载任何配置时（通过命名隐式加载，或通过 `--config` 标志）使用。
    你可以通过 `--pairs-file pairs.json` 强制使用该文件——但我们建议使用配置中的 pairlist，无论是通过 `exchange.pair_whitelist` 还是配置中的 `pairs` 设置。

## 子命令 convert data

--8<-- "../commands/convert-data.md"

### 转换数据示例

以下命令将把 `~/.freqtrade/data/binance` 中所有可用的蜡烛（OHLCV）数据从 json 转换为 jsongz，在此过程中节省磁盘空间。
它还会移除原始的 json 数据文件（`--erase` 参数）。

``` bash
freqtrade convert-data --format-from json --format-to jsongz --datadir ~/.freqtrade/data/binance -t 5m 15m --erase
```

## 子命令 convert trade data

--8<-- "../commands/convert-trade-data.md"

### 转换 trades 示例

以下命令将把 `~/.freqtrade/data/kraken` 中所有可用的 trade-data 从 jsongz 转换为 json。
它还会移除原始的 jsongz 数据文件（`--erase` 参数）。

``` bash
freqtrade convert-trade-data --format-from jsongz --format-to json --datadir ~/.freqtrade/data/kraken --erase
```

## 子命令 trades to ohlcv

当你需要使用 `--dl-trades`（仅 kraken）下载数据时，将 trades 数据转换为 ohlcv 数据是最后一步。
此命令将允许你为额外的时间周期重复这最后一步，而无需重新下载数据。

--8<-- "../commands/trades-to-ohlcv.md"

### trade-to-ohlcv 转换示例

``` bash
freqtrade trades-to-ohlcv --exchange kraken -t 5m 1h 1d --pairs BTC/EUR ETH/EUR
```

## 子命令 list-data

你可以使用 `list-data` 子命令获取已下载数据的列表。

--8<-- "../commands/list-data.md"

### list-data 示例

```bash
> freqtrade list-data --userdir ~/.freqtrade/user_data/

              Found 33 pair / timeframe combinations.
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┓
┃          Pair ┃                                 Timeframe ┃ Type ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━┩
│       ADA/BTC │     5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d │ spot │
│       ADA/ETH │     5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d │ spot │
│       ETH/BTC │     5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d │ spot │
│      ETH/USDT │                  5m, 15m, 30m, 1h, 2h, 4h │ spot │
└───────────────┴───────────────────────────────────────────┴──────┘

```

显示所有 trades 数据，包括起止时间范围

``` bash
> freqtrade list-data --show --trades
                     Found trades data for 1 pair.                     
┏━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃    Pair ┃ Type ┃                From ┃                  To ┃ Trades ┃
┡━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ XRP/ETH │ spot │ 2019-10-11 00:00:11 │ 2019-10-13 11:19:28 │  12477 │
└─────────┴──────┴─────────────────────┴─────────────────────┴────────┘

```

## Trades（tick）数据

默认情况下，`download-data` 子命令下载蜡烛（OHLCV）数据。大多数交易所也通过它们的 API 提供历史 trade-data。
如果你需要许多不同的时间周期，这些数据会很有用，因为它只下载一次，然后在本地重采样（resample）为所需的时间周期。

由于默认情况下这些数据很大，文件默认使用 feather 文件格式。它们以 `<pair>-trades.feather`（`ETH_BTC-trades.feather`）的命名约定存储在你的数据目录中。也支持增量模式，就像历史 OHLCV 数据一样，因此每周使用 `--days 8` 下载一次数据就会创建一个增量数据仓库。

要使用此模式，只需在调用中添加 `--dl-trades`。这会将下载方法切换为下载 trades。
如果同时还提供了 `--convert`，重采样步骤会自动发生，并覆盖给定交易对/时间周期组合的现有 OHLCV 数据。

!!! Warning "请勿使用"
    除非你是 kraken 用户（Kraken 不提供历史 OHLCV 数据），否则不应使用此方式。
    大多数其他交易所提供具有足够历史记录的 OHLCV 数据，因此通过该方式下载多个时间周期仍然被证明比下载 trades 数据快得多。

!!! Note "Kraken 用户"
    Kraken 用户在开始下载数据之前应阅读[此内容](exchanges.md#historic-kraken-data)。

    Kraken Futures 使用标准的 OHLCV 下载，不需要 `--dl-trades`。

示例调用：

```bash
freqtrade download-data --exchange kraken --pairs XRP/EUR ETH/EUR --days 20 --dl-trades
```

!!! Note
    虽然此方法使用了异步调用，但它会很慢，因为它需要上一次调用的结果来生成对交易所的下一次请求。

## 下一步

很好，你现在已经下载了一些数据，因此你可以开始[回测](backtesting.md)你的策略了。
