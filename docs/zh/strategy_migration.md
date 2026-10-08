<!-- 本文件为中文翻译版，由 AI 根据 docs/strategy_migration.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# V2 到 V3 的策略迁移

为了支持新的市场和交易类型（即做空交易/杠杆交易），接口方面做了一些调整。
如果您打算使用现货市场以外的市场，请将您的策略迁移到新格式。

我们在保持与现有策略的兼容性方面投入了大量精力，因此如果您只想继续在**现货市场**上使用 freqtrade，目前无需做任何更改。

您可以将快速摘要作为检查清单使用。完整的迁移详情请参阅下面的详细章节。

## 快速摘要 / 迁移检查清单

注意：`forcesell`、`forcebuy`、`emergencysell` 已分别更改为 `force_exit`、`force_enter`、`emergency_exit`。

* 策略方法：
  * [`populate_buy_trend()` -> `populate_entry_trend()`](#populate_buy_trend)
  * [`populate_sell_trend()` -> `populate_exit_trend()`](#populate_sell_trend)
  * [`custom_sell()` -> `custom_exit()`](#custom_sell)
  * [`check_buy_timeout()` -> `check_entry_timeout()`](#custom_entry_timeout)
  * [`check_sell_timeout()` -> `check_exit_timeout()`](#custom_entry_timeout)
  * 不含交易对象的回调函数新增 `side` 参数
    * [`custom_stake_amount`](#custom_stake_amount)
    * [`confirm_trade_entry`](#confirm_trade_entry)
    * [`custom_entry_price`](#custom_entry_price)
  * [`confirm_trade_exit` 中参数名已更改](#confirm_trade_exit)
* 数据框列名：
  * [`buy` -> `enter_long`](#populate_buy_trend)
  * [`sell` -> `exit_long`](#populate_sell_trend)
  * [`buy_tag` -> `enter_tag`（用于多头和空头交易）](#populate_buy_trend)
  * [新增列 `enter_short` 及对应的新列 `exit_short`](#populate_sell_trend)
* 交易对象现在具有以下新属性：
  * `is_short`
  * `entry_side`
  * `exit_side`
  * `trade_direction`
  * 重命名：`sell_reason` -> `exit_reason`
* [`trade.nr_of_successful_buys` 已重命名为 `trade.nr_of_successful_entries`（主要与 `adjust_trade_position()` 相关）](#adjust-trade-position-changes)
* 引入了新的 [`leverage` 回调](strategy-callbacks.md#leverage-callback)。
* 信息对现在可以在元组中传递第三个元素，用于定义 K 线类型。
* `@informative` 装饰器现在接受可选的 `candle_type` 参数。
* [辅助方法](#helper-methods) `stoploss_from_open` 和 `stoploss_from_absolute` 现在新增了 `is_short` 参数。
* `INTERFACE_VERSION` 应设置为 3。
* [策略/配置设置](#strategyconfiguration-settings)。
  * `order_time_in_force` 中 buy -> entry，sell -> exit。
  * `order_types` 中 buy -> entry，sell -> exit。
  * `unfilledtimeout` 中 buy -> entry，sell -> exit。
  * `ignore_buying_expired_candle_after` -> 从 "ask_strategy/exit_pricing" 移至根级别
* 术语变更
  * 卖出原因已更改为使用 "exit" 而非 "sell"。如果您的策略中使用了 `exit_reason` 检查，请注意并及时更新策略。
    * `sell_signal` -> `exit_signal`
    * `custom_sell` -> `custom_exit`
    * `force_sell` -> `force_exit`
    * `emergency_sell` -> `emergency_exit`
  * 订单定价
    * `bid_strategy` -> `entry_pricing`
    * `ask_strategy` -> `exit_pricing`
    * `ask_last_balance` -> `price_last_balance`
    * `bid_last_balance` -> `price_last_balance`
  * Webhook 术语从 "sell" 改为 "exit"，从 "buy" 改为 "entry"
    * `webhookbuy` -> `entry`
    * `webhookbuyfill` -> `entry_fill`
    * `webhookbuycancel` -> `entry_cancel`
    * `webhooksell` -> `exit`
    * `webhooksellfill` -> `exit_fill`
    * `webhooksellcancel` -> `exit_cancel`
  * Telegram 通知设置
    * `buy` -> `entry`
    * `buy_fill` -> `entry_fill`
    * `buy_cancel` -> `entry_cancel`
    * `sell` -> `exit`
    * `sell_fill` -> `exit_fill`
    * `sell_cancel` -> `exit_cancel`
  * 策略/配置设置：
    * `use_sell_signal` -> `use_exit_signal`
    * `sell_profit_only` -> `exit_profit_only`
    * `sell_profit_offset` -> `exit_profit_offset`
    * `ignore_roi_if_buy_signal` -> `ignore_roi_if_entry_signal`
    * `forcebuy_enable` -> `force_entry_enable`

## 详细说明

### `populate_buy_trend`

在 `populate_buy_trend()` 中 - 您需要将分配的列从 `'buy'` 更改为 `'enter_long'`，同时将方法名从 `populate_buy_trend` 更改为 `populate_entry_trend`。

```python hl_lines="1 9"
def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (qtpylib.crossed_above(dataframe['rsi'], 30)) &  # 信号：RSI 上穿 30
            (dataframe['tema'] <= dataframe['bb_middleband']) &  # 保护条件
            (dataframe['tema'] > dataframe['tema'].shift(1)) &  # 保护条件
            (dataframe['volume'] > 0)  # 确保成交量不为 0
        ),
        ['buy', 'buy_tag']] = (1, 'rsi_cross')

    return dataframe
```

迁移后：

```python hl_lines="1 9"
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (qtpylib.crossed_above(dataframe['rsi'], 30)) &  # 信号：RSI 上穿 30
            (dataframe['tema'] <= dataframe['bb_middleband']) &  # 保护条件
            (dataframe['tema'] > dataframe['tema'].shift(1)) &  # 保护条件
            (dataframe['volume'] > 0)  # 确保成交量不为 0
        ),
        ['enter_long', 'enter_tag']] = (1, 'rsi_cross')

    return dataframe
```

有关如何开仓和平仓空头交易，请参阅[策略文档](strategy-customization.md#entry-signal-rules)。

### `populate_sell_trend`

与 `populate_buy_trend` 类似，`populate_sell_trend()` 将被重命名为 `populate_exit_trend()`。
同时将列名从 `'sell'` 更改为 `'exit_long'`。

``` python hl_lines="1 9"
def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (qtpylib.crossed_above(dataframe['rsi'], 70)) &  # 信号：RSI 上穿 70
            (dataframe['tema'] > dataframe['bb_middleband']) &  # 保护条件
            (dataframe['tema'] < dataframe['tema'].shift(1)) &  # 保护条件
            (dataframe['volume'] > 0)  # 确保成交量不为 0
        ),
        ['sell', 'exit_tag']] = (1, 'some_exit_tag')
    return dataframe
```

迁移后

``` python hl_lines="1 9"
def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            (qtpylib.crossed_above(dataframe['rsi'], 70)) &  # 信号：RSI 上穿 70
            (dataframe['tema'] > dataframe['bb_middleband']) &  # 保护条件
            (dataframe['tema'] < dataframe['tema'].shift(1)) &  # 保护条件
            (dataframe['volume'] > 0)  # 确保成交量不为 0
        ),
        ['exit_long', 'exit_tag']] = (1, 'some_exit_tag')
    return dataframe
```

有关如何开仓和平仓空头交易，请参阅[策略文档](strategy-customization.md#exit-signal-rules)。

### `custom_sell`

`custom_sell` 已重命名为 `custom_exit`。
现在它会在每次迭代时被调用，不再受当前利润和 `exit_profit_only` 设置的影响。

``` python hl_lines="2"
class AwesomeStrategy(IStrategy):
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        # ...
```

``` python hl_lines="2"
class AwesomeStrategy(IStrategy):
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        # ...
```

### `custom_entry_timeout`

`check_buy_timeout()` 已重命名为 `check_entry_timeout()`，`check_sell_timeout()` 已重命名为 `check_exit_timeout()`。

``` python hl_lines="2 6"
class AwesomeStrategy(IStrategy):
    def check_buy_timeout(self, pair: str, trade: 'Trade', order: dict, 
                            current_time: datetime, **kwargs) -> bool:
        return False

    def check_sell_timeout(self, pair: str, trade: 'Trade', order: dict, 
                            current_time: datetime, **kwargs) -> bool:
        return False 
```

``` python hl_lines="2 6"
class AwesomeStrategy(IStrategy):
    def check_entry_timeout(self, pair: str, trade: 'Trade', order: 'Order', 
                            current_time: datetime, **kwargs) -> bool:
        return False

    def check_exit_timeout(self, pair: str, trade: 'Trade', order: 'Order', 
                            current_time: datetime, **kwargs) -> bool:
        return False 
```

### `custom_stake_amount`

新增字符串参数 `side` - 可以是 `"long"` 或 `"short"`。

``` python hl_lines="4"
class AwesomeStrategy(IStrategy):
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            entry_tag: Optional[str], **kwargs) -> float:
        # ... 
        return proposed_stake
```

``` python hl_lines="4"
class AwesomeStrategy(IStrategy):
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            entry_tag: str | None, side: str, **kwargs) -> float:
        # ... 
        return proposed_stake
```

### `confirm_trade_entry`

新增字符串参数 `side` - 可以是 `"long"` 或 `"short"`。

``` python hl_lines="4"
class AwesomeStrategy(IStrategy):
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str], 
                            **kwargs) -> bool:
      return True
```

迁移后：

``` python hl_lines="4"
class AwesomeStrategy(IStrategy):
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str | None, 
                            side: str, **kwargs) -> bool:
      return True
```

### `confirm_trade_exit`

参数 `sell_reason` 已更改为 `exit_reason`。
为了兼容性，`sell_reason` 在一段时间内仍将被保留提供。

``` python hl_lines="3"
class AwesomeStrategy(IStrategy):
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:
    return True
```

迁移后：

``` python hl_lines="3"
class AwesomeStrategy(IStrategy):
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
    return True
```

### `custom_entry_price`

新增字符串参数 `side` - 可以是 `"long"` 或 `"short"`。

``` python hl_lines="3"
class AwesomeStrategy(IStrategy):
    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], **kwargs) -> float:
      return proposed_rate
```

迁移后：

``` python hl_lines="3"
class AwesomeStrategy(IStrategy):
    def custom_entry_price(self, pair: str, trade: Trade | None, current_time: datetime, proposed_rate: float,
                           entry_tag: str | None, side: str, **kwargs) -> float:
      return proposed_rate
```

### adjust_trade_position 相关变更

虽然 adjust_trade_position 本身没有变化，但您不应再使用 `trade.nr_of_successful_buys` - 而应使用 `trade.nr_of_successful_entries`，它也会包含空头开仓记录。

### 辅助方法

`stoploss_from_open` 和 `stoploss_from_absolute` 新增了 `is_short` 参数。
该参数应传入 `trade.is_short` 的值。

``` python hl_lines="5 7"
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # 当利润超过 10% 时，将止损保持在开盘价上方 7%
        if current_profit > 0.10:
            return stoploss_from_open(0.07, current_profit)

        return stoploss_from_absolute(current_rate - (candle['atr'] * 2), current_rate)

        return 1

```

迁移后：

``` python hl_lines="5 7"
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool, 
                        **kwargs) -> float | None:
        # 当利润超过 10% 时，将止损保持在开盘价上方 7%
        if current_profit > 0.10:
            return stoploss_from_open(0.07, current_profit, is_short=trade.is_short)

        return stoploss_from_absolute(current_rate - (candle['atr'] * 2), current_rate, is_short=trade.is_short, leverage=trade.leverage)


```

### 策略/配置设置

#### `order_time_in_force`

`order_time_in_force` 属性从 `"buy"` 改为 `"entry"`，从 `"sell"` 改为 `"exit"`。

``` python
    order_time_in_force: dict = {
        "buy": "gtc",
        "sell": "gtc",
    }
```

迁移后：

``` python hl_lines="2 3"
    order_time_in_force: dict = {
        "entry": "GTC",
        "exit": "GTC",
    }
```

#### `order_types`

`order_types` 中所有用词已从 `buy` 改为 `entry`，从 `sell` 改为 `exit`。
且两个单词之间用 `_` 连接。

``` python hl_lines="2-6"
    order_types = {
        "buy": "limit",
        "sell": "limit",
        "emergencysell": "market",
        "forcesell": "market",
        "forcebuy": "market",
        "stoploss": "market",
        "stoploss_on_exchange": false,
        "stoploss_on_exchange_interval": 60
    }
```

迁移后：

``` python hl_lines="2-6"
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "force_exit": "market",
        "force_entry": "market",
        "stoploss": "market",
        "stoploss_on_exchange": false,
        "stoploss_on_exchange_interval": 60
    }
```

#### 策略级别设置

* `use_sell_signal` -> `use_exit_signal`
* `sell_profit_only` -> `exit_profit_only`
* `sell_profit_offset` -> `exit_profit_offset`
* `ignore_roi_if_buy_signal` -> `ignore_roi_if_entry_signal`

``` python hl_lines="2-5"
    # 这些值可以在配置中被覆盖。
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset: 0.01
    ignore_roi_if_buy_signal = False
```

迁移后：

``` python hl_lines="2-5"
    # 这些值可以在配置中被覆盖。
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset: 0.01
    ignore_roi_if_entry_signal = False
```

#### `unfilledtimeout`

`unfilledtimeout` 中所有用词已从 `buy` 改为 `entry`，从 `sell` 改为 `exit`。

``` python hl_lines="2-3"
unfilledtimeout = {
        "buy": 10,
        "sell": 10,
        "exit_timeout_count": 0,
        "unit": "minutes"
    }
```

迁移后：

``` python hl_lines="2-3"
unfilledtimeout = {
        "entry": 10,
        "exit": 10,
        "exit_timeout_count": 0,
        "unit": "minutes"
    }
```

#### 订单定价

订单定价在两方面发生了变化。`bid_strategy` 被重命名为 `entry_pricing`，`ask_strategy` 被重命名为 `exit_pricing`。
属性 `ask_last_balance` -> `price_last_balance` 和 `bid_last_balance` -> `price_last_balance` 也被重命名。
此外，价格方面现在可以定义为 `ask`、`bid`、`same` 或 `other`。
更多信息请参阅[定价文档](configuration.md#prices-used-for-orders)。

``` json hl_lines="2-3 6 12-13 16"
{
    "bid_strategy": {
        "price_side": "bid",
        "use_order_book": true,
        "order_book_top": 1,
        "ask_last_balance": 0.0,
        "check_depth_of_market": {
            "enabled": false,
            "bids_to_ask_delta": 1
        }
    },
    "ask_strategy":{
        "price_side": "ask",
        "use_order_book": true,
        "order_book_top": 1,
        "bid_last_balance": 0.0
        "ignore_buying_expired_candle_after": 120
    }
}
```

迁移后：

``` json  hl_lines="2-3 6 12-13 16"
{
    "entry_pricing": {
        "price_side": "same",
        "use_order_book": true,
        "order_book_top": 1,
        "price_last_balance": 0.0,
        "check_depth_of_market": {
            "enabled": false,
            "bids_to_ask_delta": 1
        }
    },
    "exit_pricing":{
        "price_side": "same",
        "use_order_book": true,
        "order_book_top": 1,
        "price_last_balance": 0.0
    },
    "ignore_buying_expired_candle_after": 120
}
```

## FreqAI 策略

`populate_any_indicators()` 方法已被拆分为 `feature_engineering_expand_all()`、`feature_engineering_expand_basic()`、`feature_engineering_standard()` 和 `set_freqai_targets()`。

对于每个新函数，交易对（以及在必要时的时间周期）将自动添加到列名中。
因此，使用新逻辑定义特征变得更加简单。

有关每个方法的完整说明，请前往相应的 [FreqAI 文档页面](freqai-feature-engineering.md#defining-the-features)

``` python linenums="1" hl_lines="12-37 39-42 63-65 67-75"

def populate_any_indicators(
        self, pair, df, tf, informative=None, set_generalized_indicators=False
    ):

        if informative is None:
            informative = self.dp.get_pair_dataframe(pair, tf)

        # 第一个循环自动为不同时间周期复制指标
        for t in self.freqai_info["feature_parameters"]["indicator_periods_candles"]:

            t = int(t)
            informative[f"%-{pair}rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)
            informative[f"%-{pair}mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)
            informative[f"%-{pair}adx-period_{t}"] = ta.ADX(informative, timeperiod=t)
            informative[f"%-{pair}sma-period_{t}"] = ta.SMA(informative, timeperiod=t)
            informative[f"%-{pair}ema-period_{t}"] = ta.EMA(informative, timeperiod=t)

            bollinger = qtpylib.bollinger_bands(
                qtpylib.typical_price(informative), window=t, stds=2.2
            )
            informative[f"{pair}bb_lowerband-period_{t}"] = bollinger["lower"]
            informative[f"{pair}bb_middleband-period_{t}"] = bollinger["mid"]
            informative[f"{pair}bb_upperband-period_{t}"] = bollinger["upper"]

            informative[f"%-{pair}bb_width-period_{t}"] = (
                informative[f"{pair}bb_upperband-period_{t}"]
                - informative[f"{pair}bb_lowerband-period_{t}"]
            ) / informative[f"{pair}bb_middleband-period_{t}"]
            informative[f"%-{pair}close-bb_lower-period_{t}"] = (
                informative["close"] / informative[f"{pair}bb_lowerband-period_{t}"]
            )

            informative[f"%-{pair}roc-period_{t}"] = ta.ROC(informative, timeperiod=t)

            informative[f"%-{pair}relative_volume-period_{t}"] = (
                informative["volume"] / informative["volume"].rolling(t).mean()
            ) # (1)

        informative[f"%-{pair}pct-change"] = informative["close"].pct_change()
        informative[f"%-{pair}raw_volume"] = informative["volume"]
        informative[f"%-{pair}raw_price"] = informative["close"]
        # (2)

        indicators = [col for col in informative if col.startswith("%")]
        # 此循环复制并偏移所有指标，为数据添加近期性
        for n in range(self.freqai_info["feature_parameters"]["include_shifted_candles"] + 1):
            if n == 0:
                continue
            informative_shift = informative[indicators].shift(n)
            informative_shift = informative_shift.add_suffix("_shift-" + str(n))
            informative = pd.concat((informative, informative_shift), axis=1)

        df = merge_informative_pair(df, informative, self.config["timeframe"], tf, ffill=True)
        skip_columns = [
            (s + "_" + tf) for s in ["date", "open", "high", "low", "close", "volume"]
        ]
        df = df.drop(columns=skip_columns)

        # 在此处添加通用指标（因为在实盘中，训练期间会调用此函数来填充指标）。
        # 注意确保不重复添加
        if set_generalized_indicators:
            df["%-day_of_week"] = (df["date"].dt.dayofweek + 1) / 7
            df["%-hour_of_day"] = (df["date"].dt.hour + 1) / 25
            # (3)

            # 用户在此添加目标，以 &- 为前缀（参见下面的约定）
            df["&-s_close"] = (
                df["close"]
                .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
                .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
                .mean()
                / df["close"]
                - 1
            )  # (4)

        return df
```

1. 特征 - 移至 `feature_engineering_expand_all`
2. 基础特征，不在 `indicator_periods_candles` 上展开 - 移至 `feature_engineering_expand_basic()`
3. 不应展开的标准特征 - 移至 `feature_engineering_standard()`
4. 目标 - 将此部分移至 `set_freqai_targets()`

### FreqAI - 展开所有特征工程

特征现在将自动展开。因此，展开循环以及 `{pair}` / `{timeframe}` 部分需要被移除。

``` python linenums="1"
    def feature_engineering_expand_all(self, dataframe, period, **kwargs) -> DataFrame::
        """
        *仅在启用 FreqAI 的策略中有效*
        此函数将根据配置定义的 `indicator_periods_candles`、`include_timeframes`、
        `include_shifted_candles` 和 `include_corr_pairs` 自动展开所定义的特征。
        换言之，在此函数中定义的单个特征将自动扩展为总计
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` 个特征添加到模型中。

        所有特征必须以 `%` 为前缀才能被 FreqAI 内部识别。

        有关这些配置定义参数如何加速特征工程的更多详情，请参阅文档：

        https://www.freqtrade.io/en/stable/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/stable/freqai-feature-engineering/#defining-the-features

        :param df: 接收特征的策略数据框
        :param period: 指标周期 - 用法示例：
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """

        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]

        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"]
            - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = (
            dataframe["close"] / dataframe["bb_lowerband-period"]
        )

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        return dataframe

```

### FreqAI - 基础特征工程

基础特征。请确保从特征中移除 `{pair}` 部分。

``` python linenums="1"
    def feature_engineering_expand_basic(self, dataframe: DataFrame, **kwargs) -> DataFrame::
        """
        *仅在启用 FreqAI 的策略中有效*
        此函数将根据配置定义的 `include_timeframes`、`include_shifted_candles`
        和 `include_corr_pairs` 自动展开所定义的特征。
        换言之，在此函数中定义的单个特征将自动扩展为总计
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`
        个特征添加到模型中。

        此处定义的特征*不会*在用户定义的 `indicator_periods_candles` 上自动复制。

        所有特征必须以 `%` 为前缀才能被 FreqAI 内部识别。

        有关这些配置定义参数如何加速特征工程的更多详情，请参阅文档：

        https://www.freqtrade.io/en/stable/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/stable/freqai-feature-engineering/#defining-the-features

        :param df: 接收特征的策略数据框
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe
```

### FreqAI - 标准特征工程

``` python linenums="1"
    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        """
        *仅在启用 FreqAI 的策略中有效*
        此可选函数将使用基础时间周期的数据框调用一次。
        这是最后一个被调用的函数，这意味着进入此函数的数据框将包含所有其他
        freqai_feature_engineering_* 函数创建的特征和列。

        此函数适合进行自定义特殊特征提取（例如 tsfresh）。
        此函数适合放置任何不应自动展开的特征（例如星期几）。

        所有特征必须以 `%` 为前缀才能被 FreqAI 内部识别。

        有关特征工程的更多详情，请参阅：

        https://www.freqtrade.io/en/stable/freqai-feature-engineering

        :param df: 接收特征的策略数据框
        用法示例：dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        """
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe
```

### FreqAI - 设置目标

目标现在拥有自己专用的方法。

``` python linenums="1"
    def set_freqai_targets(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        """
        *仅在启用 FreqAI 的策略中有效*
        设置模型目标的必需函数。
        所有目标必须以 `&` 为前缀才能被 FreqAI 内部识别。

        有关特征工程的更多详情，请参阅：

        https://www.freqtrade.io/en/stable/freqai-feature-engineering

        :param df: 接收目标的策略数据框
        用法示例：dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .mean()
            / dataframe["close"]
            - 1
            )

        return dataframe
```

### FreqAI - 新数据管道

如果您创建了自定义的 `IFreqaiModel` 并使用了自定义的 `train()`/`predict()` 函数，*并且*仍然依赖 `data_cleaning_train/predict()`，那么您将需要迁移到新的管道。如果您的模型*不*依赖 `data_cleaning_train/predict()`，则无需担心此迁移。这意味着本迁移指南仅适用于极少数高级用户。如果您是无意中看到此指南的，欢迎在 Freqtrade Discord 服务器上详细咨询您的问题。

转换过程包括首先移除 `data_cleaning_train/predict()`，并在您的 `IFreqaiModel` 类中替换为 `define_data_pipeline()` 和 `define_label_pipeline()` 函数：

```python  linenums="1" hl_lines="11-14 47-49 55-57"
class MyCoolFreqaiModel(BaseRegressionModel):
    """
    您在 Freqtrade 2023.6 版本之前创建的某个很酷的自定义 IFreqaiModel
    """
    def train(
        self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:

        # ... 您的自定义代码

        # 移除以下行
        # data_dictionary = dk.make_train_test_datasets(features_filtered, labels_filtered)
        # self.data_cleaning_train(dk)
        # data_dictionary = dk.normalize_data(data_dictionary)
        # (1)

        # 添加以下行。现在我们自行控制管道的拟合/转换
        dd = dk.make_train_test_datasets(features_filtered, labels_filtered)
        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)
        dk.label_pipeline = self.define_label_pipeline(threads=dk.thread_count)

        (dd["train_features"],
         dd["train_labels"],
         dd["train_weights"]) = dk.feature_pipeline.fit_transform(dd["train_features"],
                                                                  dd["train_labels"],
                                                                  dd["train_weights"])

        (dd["test_features"],
         dd["test_labels"],
         dd["test_weights"]) = dk.feature_pipeline.transform(dd["test_features"],
                                                             dd["test_labels"],
                                                             dd["test_weights"])

        dd["train_labels"], _, _ = dk.label_pipeline.fit_transform(dd["train_labels"])
        dd["test_labels"], _, _ = dk.label_pipeline.transform(dd["test_labels"])

        # ... 您的自定义代码

        return model

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, npt.NDArray[np.int_]]:

        # ... 您的自定义代码

        # 移除以下行：
        # self.data_cleaning_predict(dk)
        # (2)

        # 添加以下行：
        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True)

        # 移除此行
        # pred_df = dk.denormalize_labels_from_metadata(pred_df)
        # (3)

        # 替换为以下行
        pred_df, _, _ = dk.label_pipeline.inverse_transform(pred_df)
        if self.freqai_info.get("DI_threshold", 0) > 0:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers

        # ... 您的自定义代码
        return (pred_df, dk.do_predict)
```

1. 数据标准化和清洗现在通过新的管道定义统一处理。这是在新的 `define_data_pipeline()` 和 `define_label_pipeline()` 函数中创建的。`data_cleaning_train()` 和 `data_cleaning_predict()` 函数不再使用。您可以根据需要覆盖 `define_data_pipeline()` 来创建自定义管道。
2. 数据标准化和清洗现在通过新的管道定义统一处理。这是在新的 `define_data_pipeline()` 和 `define_label_pipeline()` 函数中创建的。`data_cleaning_train()` 和 `data_cleaning_predict()` 函数不再使用。您可以根据需要覆盖 `define_data_pipeline()` 来创建自定义管道。
3. 数据反标准化通过新管道完成。请用下面的代码替换。