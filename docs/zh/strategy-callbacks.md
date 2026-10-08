<!-- 本文件为中文翻译版，由 AI 根据 docs/strategy-callbacks.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件（如 exchanges.md），这些文件将在逐步翻译过程中补齐。 -->
<!-- 图片与 includes 引用使用 ../ 指向英文原文档资源，以保证显示正常。 -->

# 策略回调

虽然主要的策略函数（`populate_indicators()`、`populate_entry_trend()`、`populate_exit_trend()`）应以向量化方式使用，且在回测中[仅调用一次](bot-basics.md#backtesting-hyperopt-execution-logic)，但回调函数会在"需要时"被调用。

因此，您应避免在回调中执行繁重的计算，以防止操作过程中出现延迟。
根据所使用的回调不同，它们可能在开仓/平仓时被调用，也可能在整个交易持续期间被调用。

当前可用的回调：

* [`bot_start()`](#bot-start)
* [`bot_loop_start()`](#bot-loop-start)
* [`custom_stake_amount()`](#stake-size-management)
* [`custom_exit()`](#custom-exit-signal)
* [`custom_stoploss()`](#custom-stoploss)
* [`custom_roi()`](#custom-roi)
* [`custom_entry_price()` 和 `custom_exit_price()`](#custom-order-price-rules)
* [`check_entry_timeout()` 和 `check_exit_timeout()`](#custom-order-timeout-rules)
* [`confirm_trade_entry()`](#trade-entry-buy-order-confirmation)
* [`confirm_trade_exit()`](#trade-exit-sell-order-confirmation)
* [`adjust_trade_position()`](#adjust-trade-position)
* [`adjust_entry_price()`](#adjust-entry-price)
* [`leverage()`](#leverage-callback)
* [`order_filled()`](#order-filled-callback)

!!! Tip "回调调用顺序"
    您可以在 [bot-basics](bot-basics.md#bot-execution-logic) 中找到回调的调用顺序。

--8<-- "includes/strategy-imports.md"

--8<-- "includes/strategy-exit-comparisons.md"


## Bot start

一个简单的回调，在策略加载时仅调用一次。
可用于执行只需要执行一次的操作，运行在 dataprovider 和 wallet 设置完成之后。

``` python
import requests

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def bot_start(self, **kwargs) -> None:
        """
        Called only once after bot instantiation.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        if self.config["runmode"].value in ("live", "dry_run"):
            # Assign this to the class by using self.*
            # can then be used by populate_* methods
            self.custom_remote_data = requests.get("https://some_remote_source.example.com")

```

在超参数优化期间，此回调仅在启动时运行一次。

## Bot loop start

一个简单的回调，在实盘/模拟盘模式下每次 bot 节流迭代开始时调用一次（大约每 5 秒一次，除非配置不同），或在回测/超参数优化模式下每根 K 线调用一次。
可用于执行与交易对无关的计算（适用于所有交易对）、加载外部数据等。

``` python
# Default imports
import requests

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Called at the start of the bot iteration (one loop).
        Might be used to perform pair-independent tasks
        (e.g. gather some remote resource for comparison)
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        if self.config["runmode"].value in ("live", "dry_run"):
            # Assign this to the class by using self.*
            # can then be used by populate_* methods
            self.remote_data = requests.get("https://some_remote_source.example.com")

```

## 仓位大小管理

在开仓之前调用，可以在下新单时管理您的仓位大小。

```python
# Default imports

class AwesomeStrategy(IStrategy):
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_candle["fastk_rsi_1h"] > current_candle["fastd_rsi_1h"]:
            if self.config["stake_amount"] == "unlimited":
                # Use entire available wallet during favorable conditions when in compounding mode.
                return max_stake
            else:
                # Compound profits during favorable conditions instead of using a static stake.
                return self.wallets.get_total_stake_amount() / self.config["max_open_trades"]

        # Use default stake amount.
        return proposed_stake
```

如果您的代码抛出异常，Freqtrade 将回退到 `proposed_stake` 值。异常本身会被记录到日志中。

!!! Tip
    您_不必_确保 `min_stake <= returned_value <= max_stake`。交易会成功，因为返回值将被限制在支持的范围内，此操作会被记录到日志中。

!!! Tip
    返回 `0` 或 `None` 将阻止下单。

## 自定义退出信号

对未平仓交易，在每次节流迭代时（大约每 5 秒）调用，直到交易平仓。

允许定义自定义退出信号，指示应关闭指定仓位（全部退出）。当我们需要为每笔交易自定义退出条件，或需要交易数据来做退出决策时，这非常有用。

例如，您可以使用 `custom_exit()` 实现 1:2 风险回报比的 ROI。

但是，使用 `custom_exit()` 信号来代替止损是*不推荐的*。在这方面，使用 `custom_stoploss()` 是更优的方法——它还允许您将止损挂在交易所上。

!!! Note
    从此方法返回一个（非空的）`string` 或 `True` 等同于在指定时间的 K 线上设置退出信号。如果已经设置了退出信号，或退出信号被禁用（`use_exit_signal=False`），则不会调用此方法。`string` 最大长度为 64 个字符。超出此限制将导致消息被截断为 64 个字符。
    `custom_exit()` 会忽略 `exit_profit_only`，并且除非 `use_exit_signal=False`，否则始终会被调用，即使有新的入场信号。

以下示例展示了如何根据当前利润使用不同的指标，以及如何退出持仓超过一天的交易：

``` python
# Default imports

class AwesomeStrategy(IStrategy):
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Above 20% profit, sell when rsi < 80
        if current_profit > 0.2:
            if last_candle["rsi"] < 80:
                return "rsi_below_80"

        # Between 2% and 10%, sell if EMA-long above EMA-short
        if 0.02 < current_profit < 0.1:
            if last_candle["emalong"] > last_candle["emashort"]:
                return "ema_long_below_80"

        # Sell any positions at a loss if they are held for more than one day.
        if current_profit < 0.0 and (current_time - trade.open_date_utc).days >= 1:
            return "unclog"
```

有关在策略回调中使用 DataFrame 的更多信息，请参阅 [DataFrame 访问](strategy-advanced.md#dataframe-access)。

## 自定义止损

对未平仓交易，在每次迭代时（大约每 5 秒）调用，直到交易平仓。

必须通过在策略对象上设置 `use_custom_stoploss=True` 来启用自定义止损方法。

止损价格只能向上移动——如果 `custom_stoploss` 返回的止损值会导致止损价格低于之前设置的价格，它将被忽略。传统的 `stoploss` 值作为绝对下限，并将作为初始止损（在该方法首次被调用之前），且仍然是必需的。
由于自定义止损充当常规的、变化的止损，它的行为类似于 `trailing_stop`——因此退出的交易的退出原因为 `"trailing_stop_loss"`。

该方法必须返回一个止损值（浮点数/数字），表示为当前价格的百分比。
例如，如果 `current_rate` 为 200 USD，则返回 `0.02` 将止损价格设置为低于 2%，即 196 USD。
在回测期间，`current_rate`（和 `current_profit`）是根据 K 线的最高价（做空交易则为最低价）提供的——而结果止损是根据 K 线的最低价（做空交易则为最高价）来评估的。

使用返回值的绝对值（符号被忽略），因此返回 `0.05` 或 `-0.05` 结果相同，都是低于当前价格 5% 的止损。
返回 `None` 将被解释为"不想更改"，这是您不想修改止损时唯一安全的返回方式。
`NaN` 和 `inf` 值被视为无效，将被忽略（与 `None` 相同）。

交易所止损的工作方式类似于 `trailing_stop`，交易所止损按照 `stoploss_on_exchange_interval` 的配置进行更新（[更多关于交易所止损的详情](stoploss.md#stop-loss-on-exchangefreqtrade)）。

如果您在期货市场交易，请注意[止损和杠杆](stoploss.md#stoploss-and-leverage)部分，因为 `custom_stoploss` 返回的止损值是此交易的风险——而不是相对价格变动。

!!! Note "日期的使用"
    所有基于时间的计算都应基于 `current_time` 完成——不建议使用 `datetime.now()` 或 `datetime.utcnow()`，因为这会破坏回测支持。

!!! Tip "追踪止损"
    使用自定义止损值时，建议禁用 `trailing_stop`。两者可以同时工作，但您可能会遇到追踪止损将价格抬高而您的自定义函数不希望这样做的情况，导致行为冲突。

### 仓位调整后的止损调整

根据您的策略，您可能需要在[仓位调整](#adjust-trade-position)后双向调整止损。
为此，Freqtrade 会在订单成交后进行一次额外的调用，参数 `after_fill=True`，这将允许策略在任意方向移动止损（也包括扩大止损与当前价格之间的差距，这在其他情况下是被禁止的）。

!!! Note "向后兼容"
    只有当 `after_fill` 参数是您的 `custom_stoploss` 函数定义的一部分时，才会进行此调用。
    因此，这不会影响（也不会意外影响）现有的、正在运行的策略。

### 自定义止损示例

下一节将展示一些关于自定义止损函数可以实现什么的示例。
当然，还可以实现更多功能，所有示例可以自由组合。

#### 通过自定义止损实现追踪止损

要模拟常规的 4% 追踪止损（在达到的最高价格后追踪 4%），您可以使用以下非常简单的方法：

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool, 
                        **kwargs) -> float | None:
        """
        Custom stoploss logic, returning the new distance relative to current_rate (as ratio).
        e.g. returning -0.05 would create a stoploss 5% below current_rate.
        The custom stoploss can never be below self.stoploss, which serves as a hard maximum loss.

        For full documentation please go to https://www.freqtrade.io/en/stable/strategy-advanced/

        When not implemented by a strategy, returns the initial stoploss value.
        Only called when use_custom_stoploss is set to True.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param after_fill: True if the stoploss is called after the order was filled.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New stoploss value, relative to the current_rate
        """
        return -0.04 * trade.leverage
```

#### 基于时间的追踪止损

前 60 分钟使用初始止损，之后切换到 10% 追踪止损，2 小时（120 分钟）后使用 5% 追踪止损。

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool, 
                        **kwargs) -> float | None:

        # Make sure you have the longest interval first - these conditions are evaluated from top to bottom.
        if current_time - timedelta(minutes=120) > trade.open_date_utc:
            return -0.05 * trade.leverage
        elif current_time - timedelta(minutes=60) > trade.open_date_utc:
            return -0.10 * trade.leverage
        return None
```

#### 带成交后调整的基于时间的追踪止损

前 60 分钟使用初始止损，之后切换到 10% 追踪止损，2 小时（120 分钟）后使用 5% 追踪止损。
如果额外订单成交，将止损设置为低于新 `open_rate` 的 -10%（[所有入场均价](#position-adjust-calculations)）。

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool, 
                        **kwargs) -> float | None:

        if after_fill: 
            # After an additional order, start with a stoploss of 10% below the new open rate
            return stoploss_from_open(0.10, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        # Make sure you have the longest interval first - these conditions are evaluated from top to bottom.
        if current_time - timedelta(minutes=120) > trade.open_date_utc:
            return -0.05 * trade.leverage
        elif current_time - timedelta(minutes=60) > trade.open_date_utc:
            return -0.10 * trade.leverage
        return None
```

#### 不同交易对使用不同止损

根据不同交易对使用不同的止损。
在此示例中，`ETH/BTC` 和 `XRP/BTC` 使用 10% 追踪止损跟踪最高价，`LTC/BTC` 使用 5% 追踪止损，其他所有交易对使用 15% 追踪止损。

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> float | None:

        if pair in ("ETH/BTC", "XRP/BTC"):
            return -0.10 * trade.leverage
        elif pair in ("LTC/BTC"):
            return -0.05 * trade.leverage
        return -0.15 * trade.leverage
```

#### 带正向偏移的追踪止损

在利润超过 4% 之前使用初始止损，之后使用当前利润的 50% 作为追踪止损，最小 2.5%，最大 5%。

请注意，止损只能增加，低于当前止损的值将被忽略。

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> float | None:

        if current_profit < 0.04:
            return None # return None to keep using the initial stoploss

        # After reaching the desired offset, allow the stoploss to trail by half the profit
        desired_stoploss = current_profit / 2

        # Use a minimum of 2.5% and a maximum of 5%
        return max(min(desired_stoploss, 0.05), 0.025) * trade.leverage
```

#### 阶梯止损

此示例不是持续追踪当前价格，而是根据当前利润设置固定的止损价格水平。

* 利润达到 20% 之前使用常规止损
* 利润超过 20% 时——止损设置为高于开仓价格 7%。
* 利润超过 25% 时——止损设置为高于开仓价格 15%。
* 利润超过 40% 时——止损设置为高于开仓价格 25%。

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> float | None:

        # evaluate highest to lowest, so that highest possible stop is used
        if current_profit > 0.40:
            return stoploss_from_open(0.25, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        elif current_profit > 0.25:
            return stoploss_from_open(0.15, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        elif current_profit > 0.20:
            return stoploss_from_open(0.07, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        # return maximum stoploss value, keeping current stoploss price unchanged
        return None
```

#### 使用 DataFrame 中指标的自定义止损示例

绝对止损值可以从存储在 DataFrame 中的指标推导。此示例使用抛物线 SAR 低于价格作为止损。

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # <...>
        dataframe["sar"] = ta.SAR(dataframe)

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> float | None:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Use parabolic sar as absolute stoploss price
        stoploss_price = last_candle["sar"]

        # Convert absolute price to percentage relative to current_rate
        if stoploss_price < current_rate:
            return stoploss_from_absolute(stoploss_price, current_rate, is_short=trade.is_short)

        # return maximum stoploss value, keeping current stoploss price unchanged
        return None
```

有关在策略回调中使用 DataFrame 的更多信息，请参阅 [DataFrame 访问](strategy-advanced.md#dataframe-access)。

### 止损计算常用辅助函数

#### 相对于开仓价格的止损

`custom_stoploss()` 返回的止损值必须指定相对于 `current_rate` 的百分比，但有时您可能希望指定相对于_入场_价格的止损。
`stoploss_from_open()` 是一个辅助函数，用于计算可以从 `custom_stoploss` 返回的止损值，该值等同于入场点之上期望的交易利润。

??? Example "从自定义止损函数返回相对于开仓价格的止损"

    假设开仓价格为 $100，`current_price` 为 $121（`current_profit` 将为 `0.21`）。

    如果我们想要止损价高于开仓价格 7%，可以调用 `stoploss_from_open(0.07, current_profit, False)`，将返回 `0.1157024793`。$121 的 11.57% 低于 $107，这与高于 $100 的 7% 相同。

    此函数会考虑杠杆——因此在 10 倍杠杆下，实际止损将高于 $100 的 0.7%（0.7% * 10x = 7%）。


    ``` python
    # Default imports

    class AwesomeStrategy(IStrategy):

        # ... populate_* methods

        use_custom_stoploss = True

        def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float, after_fill: bool,
                            **kwargs) -> float | None:

            # once the profit has risen above 10%, keep the stoploss at 7% above the open price
            if current_profit > 0.10:
                return stoploss_from_open(0.07, current_profit, is_short=trade.is_short, leverage=trade.leverage)

            return 1

    ```

    完整示例可以在文档的[自定义止损](strategy-callbacks.md#custom-stoploss)部分找到。

!!! Note
    向 `stoploss_from_open()` 提供无效输入可能会产生"CustomStoploss function did not return valid stoploss"警告。
    当 `current_profit` 参数低于指定的 `open_relative_stop` 时可能会发生这种情况。当 `confirm_trade_exit()` 方法阻止平仓时可能会出现此类情况。
    可以通过在 `confirm_trade_exit()` 中检查 `exit_reason` 来永远不阻止止损卖出，或使用 `return stoploss_from_open(...) or 1` 惯用法来解决警告，当 `current_profit < open_relative_stop` 时将请求不更改止损。

#### 从绝对价格计算止损百分比

`custom_stoploss()` 返回的止损值始终指定相对于 `current_rate` 的百分比。为了在指定的绝对价格水平设置止损，我们需要使用 `stop_rate` 来计算相对于 `current_rate` 的百分比，使其产生与从开仓价格指定百分比相同的结果。

辅助函数 `stoploss_from_absolute()` 可用于从绝对价格转换为当前价格相对止损，该值可以从 `custom_stoploss()` 返回。

??? Example "从自定义止损函数返回使用绝对价格的止损"

    如果我们想要追踪低于当前价格 2 倍 ATR 的止损价，可以调用 `stoploss_from_absolute(current_rate + (side * candle["atr"] * 2), current_rate=current_rate, is_short=trade.is_short, leverage=trade.leverage)`。
    对于期货，我们需要调整方向（向上或向下），以及调整杠杆，因为 [`custom_stoploss`](strategy-callbacks.md#custom-stoploss) 回调返回的是["此交易的风险"](stoploss.md#stoploss-and-leverage)——而不是相对价格变动。

    ``` python
    # Default imports

    class AwesomeStrategy(IStrategy):

        use_custom_stoploss = True

        def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
            return dataframe

        def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float, after_fill: bool,
                            **kwargs) -> float | None:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            candle = dataframe.iloc[-1].squeeze()
            side = 1 if trade.is_short else -1
            return stoploss_from_absolute(current_rate + (side * candle["atr"] * 2), 
                                          current_rate=current_rate, 
                                          is_short=trade.is_short,
                                          leverage=trade.leverage)

    ```

---

## 自定义 ROI

对未平仓交易，在每次迭代时（大约每 5 秒）调用，直到交易平仓。

必须通过在策略对象上设置 `use_custom_roi=True` 来启用自定义 ROI 方法。

此方法允许您定义自定义的最小 ROI 阈值来退出交易，以比率表示（例如 `0.05` 表示 5% 利润）。如果同时定义了 `minimal_roi` 和 `custom_roi`，则较低的阈值将触发退出。例如，如果 `minimal_roi` 设置为 `{"0": 0.10}`（0 分钟时 10%），而 `custom_roi` 返回 `0.05`，则当利润达到 5% 时交易将退出。同样，如果 `custom_roi` 返回 `0.10` 而 `minimal_roi` 设置为 `{"0": 0.05}`（0 分钟时 5%），则当利润达到 5% 时交易将被关闭。

该方法必须返回一个浮点数，表示新的 ROI 阈值（比率），或返回 `None` 以回退到 `minimal_roi` 逻辑。返回 `NaN` 或 `inf` 值被视为无效，将被当作 `None` 处理，导致 bot 使用 `minimal_roi` 配置。

### 自定义 ROI 示例

以下示例说明了如何使用 `custom_roi` 函数实现不同的 ROI 逻辑。

#### 按方向的自定义 ROI

根据 `side` 使用不同的 ROI 阈值。在此示例中，做多入场为 5%，做空入场为 2%。

```python
# Default imports

class AwesomeStrategy(IStrategy):

    use_custom_roi = True

    # ... populate_* methods

    def custom_roi(self, pair: str, trade: Trade, current_time: datetime, trade_duration: int,
                   entry_tag: str | None, side: str, **kwargs) -> float | None:
        """
        Custom ROI logic, returns a new minimum ROI threshold (as a ratio, e.g., 0.05 for +5%).
        Only called when use_custom_roi is set to True.

        If used at the same time as minimal_roi, an exit will be triggered when the lower
        threshold is reached. Example: If minimal_roi = {"0": 0.01} and custom_roi returns 0.05,
        an exit will be triggered if profit reaches 5%.

        :param pair: Pair that's currently analyzed.
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime.
        :param trade_duration: Current trade duration in minutes.
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the current trade.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New ROI value as a ratio, or None to fall back to minimal_roi logic.
        """
        return 0.05 if side == "long" else 0.02
```

#### 按交易对的自定义 ROI

根据 `pair` 使用不同的 ROI 阈值。

```python
# Default imports

class AwesomeStrategy(IStrategy):

    use_custom_roi = True

    # ... populate_* methods

    def custom_roi(self, pair: str, trade: Trade, current_time: datetime, trade_duration: int,
                   entry_tag: str | None, side: str, **kwargs) -> float | None:

        stake = trade.stake_currency
        roi_map = {
            f"BTC/{stake}": 0.02, # 2% for BTC
            f"ETH/{stake}": 0.03, # 3% for ETH
            f"XRP/{stake}": 0.04, # 4% for XRP
        }

        return roi_map.get(pair, 0.01) # 1% for any other pair
```

#### 按入场标签的自定义 ROI

根据买入信号提供的 `entry_tag` 使用不同的 ROI 阈值。

```python
# Default imports

class AwesomeStrategy(IStrategy):

    use_custom_roi = True

    # ... populate_* methods

    def custom_roi(self, pair: str, trade: Trade, current_time: datetime, trade_duration: int,
                   entry_tag: str | None, side: str, **kwargs) -> float | None:

        roi_by_tag = {
            "breakout": 0.08,       # 8% if tag is "breakout"
            "rsi_overbought": 0.05, # 5% if tag is "rsi_overbought"
            "mean_reversion": 0.03, # 3% if tag is "mean_reversion"
        }

        return roi_by_tag.get(entry_tag, 0.01)  # 1% if tag is unknown
```

#### 基于 ATR 的自定义 ROI

ROI 值可以从存储在 DataFrame 中的指标推导。此示例使用 ATR 比率作为 ROI。

``` python
# Default imports
# <...>
import talib.abstract as ta

class AwesomeStrategy(IStrategy):

    use_custom_roi = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # <...>
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=10)

    def custom_roi(self, pair: str, trade: Trade, current_time: datetime, trade_duration: int,
                   entry_tag: str | None, side: str, **kwargs) -> float | None:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        atr_ratio = last_candle["atr"] / last_candle["close"]

        return atr_ratio # Returns the ATR value as ratio
```

---

## 自定义订单价格规则

默认情况下，Freqtrade 使用订单簿自动设置订单价格（[相关文档](configuration.md#prices-used-for-orders)），您也可以选择根据策略创建自定义订单价格。

您可以通过在策略文件中创建 `custom_entry_price()` 函数来自定义入场价格，创建 `custom_exit_price()` 函数来自定义退出价格。

每个方法在交易所下单之前被调用。

!!! Note
    如果您的自定义定价函数返回 None 或无效值，价格将回退到 `proposed_rate`，该值基于常规定价配置。

!!! Note
    使用 `custom_entry_price()` 时，Trade 对象将在与该交易关联的第一个入场订单创建后立即可用，首次入场时 `trade` 参数值将为 `None`。

### 自定义订单入场和退出价格示例

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def custom_entry_price(self, pair: str, trade: Trade | None, current_time: datetime, proposed_rate: float,
                           entry_tag: str | None, side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)
        new_entryprice = dataframe["bollinger_10_lowerband"].iat[-1]

        return new_entryprice

    def custom_exit_price(self, pair: str, trade: Trade,
                          current_time: datetime, proposed_rate: float,
                          current_profit: float, exit_tag: str | None, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)
        new_exitprice = dataframe["bollinger_10_upperband"].iat[-1]

        return new_exitprice

```

!!! Warning
    修改入场和退出价格仅对限价单有效。根据所选价格，这可能导致大量未成交订单。默认情况下，当前价格与自定义价格之间允许的最大距离为 2%，此值可通过配置中的 `custom_price_max_distance_ratio` 参数更改。
    **示例**：
    如果 new_entryprice 为 97，proposed_rate 为 100，且 `custom_price_max_distance_ratio` 设置为 2%，则保留的有效自定义入场价格将为 98，即低于当前（建议）价格 2%。

!!! Warning "回测"
    回测中支持自定义价格（从 2021.12 开始），如果价格在 K 线的最低/最高范围内，订单将成交。
    未立即成交的订单将按照常规超时处理，每个（详情）K 线处理一次。
    `custom_exit_price()` 仅对 exit_signal 类型的卖出、自定义退出和部分退出调用。所有其他退出类型将使用常规回测价格。

## 自定义订单超时规则

简单的、基于时间的订单超时可以通过策略或配置中的 `unfilledtimeout` 部分进行配置。

然而，Freqtrade 还为两种订单类型提供了自定义回调，允许您根据自定义标准判断订单是否超时。

!!! Note
    回测中，如果订单价格在 K 线的最低/最高范围内，订单将被成交。
    对于未立即成交的订单（使用自定义定价的订单），以下回调每个（详情）K 线调用一次。

!!! Tip "替换订单"
    如果您希望以不同的价格替换订单而不仅仅是取消它，您可能需要查看 [`adjust_order_price()`](#adjust-order-price)，它允许您既取消订单，又以新价格替换。

### 自定义订单超时示例

对每个未结订单调用，直到该订单成交或被取消。
`check_entry_timeout()` 对交易入场调用，而 `check_exit_timeout()` 对交易退出订单调用。

以下是一个简单示例，根据资产价格应用不同的未成交超时。
对高价格资产应用较紧的超时，而对便宜的币允许更多时间来成交。

函数必须返回 `True`（取消订单）或 `False`（保持订单存活）。

``` python
    # Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    # Set unfilledtimeout to 25 hours, since the maximum timeout from below is 24 hours.
    unfilledtimeout = {
        "entry": 60 * 25,
        "exit": 60 * 25
    }

    def check_entry_timeout(self, pair: str, trade: Trade, order: Order,
                            current_time: datetime, **kwargs) -> bool:
        if trade.open_rate > 100 and trade.open_date_utc < current_time - timedelta(minutes=5):
            return True
        elif trade.open_rate > 10 and trade.open_date_utc < current_time - timedelta(minutes=3):
            return True
        elif trade.open_rate < 1 and trade.open_date_utc < current_time - timedelta(hours=24):
           return True
        return False


    def check_exit_timeout(self, pair: str, trade: Trade, order: Order,
                           current_time: datetime, **kwargs) -> bool:
        if trade.open_rate > 100 and trade.open_date_utc < current_time - timedelta(minutes=5):
            return True
        elif trade.open_rate > 10 and trade.open_date_utc < current_time - timedelta(minutes=3):
            return True
        elif trade.open_rate < 1 and trade.open_date_utc < current_time - timedelta(hours=24):
           return True
        return False
```

!!! Note
    对于上述示例，`unfilledtimeout` 必须设置为大于 24 小时的值，否则该类型的超时将先被应用。

### 自定义订单超时示例（使用额外数据）

``` python
    # Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    # Set unfilledtimeout to 25 hours, since the maximum timeout from below is 24 hours.
    unfilledtimeout = {
        "entry": 60 * 25,
        "exit": 60 * 25
    }

    def check_entry_timeout(self, pair: str, trade: Trade, order: Order,
                            current_time: datetime, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob["bids"][0][0]
        # Cancel buy order if price is more than 2% above the order.
        if current_price > order.price * 1.02:
            return True
        return False


    def check_exit_timeout(self, pair: str, trade: Trade, order: Order,
                           current_time: datetime, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob["asks"][0][0]
        # Cancel sell order if price is more than 2% below the order.
        if current_price < order.price * 0.98:
            return True
        return False
```

---

## Bot 订单确认

确认交易入场/退出。
这些是在下单之前最后调用的方法。

### 交易入场（买单）确认

`confirm_trade_entry()` 可用于在最后一刻中止交易入场（可能因为价格不是我们预期的）。

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str | None,
                            side: str, **kwargs) -> bool:
        """
        Called right before placing a entry order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/stable/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be bought/shorted.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in target (base) currency that's going to be traded.
        :param rate: Rate that's going to be used when using limit orders 
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param current_time: datetime object, containing the current datetime
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the buy-order is placed on the exchange.
            False aborts the process
        """
        return True

```

### 交易退出（卖单）确认

`confirm_trade_exit()` 可用于在最后一刻中止交易退出（卖出）（可能因为价格不是我们预期的）。

`confirm_trade_exit()` 可能在一次迭代中对同一交易被多次调用，如果存在不同的退出原因。
退出原因（如果适用）将按以下顺序：

* `exit_signal` / `custom_exit`
* `stop_loss`
* `roi`
* `trailing_stop_loss`

``` python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing a regular exit order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/stable/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair for trade that's about to be exited.
        :param trade: trade object.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in base currency.
        :param rate: Rate that's going to be used when using limit orders
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param exit_reason: Exit reason.
            Can be any of ["roi", "stop_loss", "stoploss_on_exchange", "trailing_stop_loss",
                           "exit_signal", "force_exit", "emergency_exit"]
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True, then the exit-order is placed on the exchange.
            False aborts the process
        """
        if exit_reason == "force_exit" and trade.calc_profit_ratio(rate) < 0:
            # Reject force-sells with negative profit
            # This is just a sample, please adjust to your needs
            # (this does not necessarily make sense, assuming you know when you're force-selling)
            return False
        return True

```

!!! Warning
    `confirm_trade_exit()` 可以阻止止损退出，从而导致重大损失，因为这将忽略止损退出。
    `confirm_trade_exit()` 不会为强制平仓调用——因为强制平仓是由交易所强制执行的，因此无法拒绝。

## 调整交易仓位

`position_adjustment_enable` 策略属性启用了策略中 `adjust_trade_position()` 回调的使用。
出于性能原因，默认情况下它是禁用的，如果启用，Freqtrade 将在启动时显示警告消息。
`adjust_trade_position()` 可用于执行额外的订单，例如使用 DCA（美元成本平均法）管理风险或增加或减少仓位。

额外的订单也会产生额外的费用，且这些订单不计入 `max_open_trades`。

当存在未结订单（买入或卖出）等待执行时，也会调用此回调——如果金额、价格或方向不同，将取消现有的未结订单以下新订单。部分成交的订单也将被取消，并替换为回调返回的新金额。

`adjust_trade_position()` 在交易持续期间非常频繁地被调用，因此您必须尽可能保持实现的高效性。

仓位调整始终在交易方向上应用，因此正值始终会增加您的仓位（负值会减少您的仓位），无论是做多还是做空交易。
可以通过返回一个 2 元素的 Tuple 来为调整订单分配标签，第一个元素是调整金额，第二个元素是标签（例如 `return 250, "increase_favorable_conditions"`）。

无法修改杠杆，返回的 stake-amount 被认为是应用杠杆之前的金额。

当前分配给仓位的合并 stake 金额保存在 `trade.stake_amount` 中。因此 `trade.stake_amount` 将在每次通过 `adjust_trade_position()` 进行的额外入场和部分退出时更新。

!!! Danger "宽松逻辑"
    在模拟盘和实盘运行中，此函数将每 `throttle_process_secs`（默认为 5 秒）调用一次。如果您有宽松的逻辑（例如，如果最后一根 K 线的 RSI 低于 30 则增加仓位），您的 bot 将每 5 秒进行一次额外的重新入场，直到资金耗尽、达到 `max_position_adjustment` 限制或新的 RSI 超过 30 的 K 线到来。

    部分退出也会发生同样的事情。
    因此请确保有严格的逻辑和/或检查最后成交的订单以及是否已有未结订单。

!!! Warning "多次仓位调整的性能"
    仓位调整可能是增加策略产出的好方法——但如果大量使用此功能也可能有缺点。
    每个订单在交易持续期间都会附加到交易对象上——因此会增加内存使用。
    因此不建议交易持续时间长且有 10 次甚至 100 次仓位调整的情况，应定期关闭以不影响性能。

!!! Warning "回测"
    在回测期间，此回调对 `timeframe` 或 `timeframe_detail` 中的每根 K 线调用，因此运行时性能会受到影响。
    这也可能导致实盘和回测之间的结果偏差，因为回测每根 K 线只能调整交易一次，而实盘每根 K 线可以多次调整交易。

### 增加仓位

当需要下额外的入场订单时（仓位增加 -> 做多交易的买单，做空交易的卖单），策略应返回一个正的 **stake_amount**（以 stake 货币计），介于 `min_stake` 和 `max_stake` 之间。

如果钱包中没有足够的资金（返回值超过 `max_stake`），信号将被忽略。
`max_entry_position_adjustment` 属性用于限制每笔交易（在第一个入场订单之上）bot 可以执行的额外入场次数。默认情况下，该值为 -1，表示 bot 对调整入场次数没有限制。

一旦达到您在 `max_entry_position_adjustment` 上设置的最大额外入场数量，额外的入场将被忽略，但无论如何都会调用回调以寻找部分退出。

!!! Note "关于仓位大小"
    使用固定仓位大小意味着它将是第一个订单使用的金额，就像没有仓位调整一样。
    如果您希望通过 DCA 购买额外订单，请确保在钱包中留有足够的资金。
    在 DCA 订单中使用 `"unlimited"` stake 金额还需要您实现 `custom_stake_amount()` 回调，以避免将所有资金分配给初始订单。

### 减少仓位

策略应返回一个负的 stake_amount（以 stake 货币计）用于部分退出。
返回当时拥有的全部 stake（`-trade.stake_amount`）将导致全部退出。
返回超过上述值（即剩余的 stake_amount 将变为负数）将导致 bot 忽略该信号。

对于部分退出，重要的是要知道用于计算部分退出订单币数量的公式是 `部分退出数量 = negative_stake_amount * trade.amount / trade.stake_amount`，其中 `negative_stake_amount` 是从 `adjust_trade_position` 函数返回的值。如公式所示，该公式不关心仓位的当前盈亏。它只关心 `trade.amount` 和 `trade.stake_amount`，这两个值完全不受价格变动的影响。

例如，假设您以 50 的开仓价格购买了 2 个 SHITCOIN/USDT，这意味着交易的 stake 金额为 100 USDT。现在价格上涨到 200，您想卖出一半。在这种情况下，您必须返回 `trade.stake_amount` 的 -50%（0.5 * 100 USDT），即 -50。Bot 将计算需要卖出的数量，即 `50 * 2 / 100`，等于 1 个 SHITCOIN/USDT。如果您返回 -200（2 * 200 的 50%），bot 将忽略它，因为 `trade.stake_amount` 只有 100 USDT，但您要求卖出 200 USDT，这意味着您要求卖出 4 个 SHITCOIN/USDT。

回到上面的例子，由于当前价格为 200，您交易的当前 USDT 价值现在为 400 USDT。假设您想部分卖出 100 USDT 以取出初始投资并将利润留在交易中，希望价格继续上涨。在这种情况下，您需要采用不同的方法。首先，您需要计算需要卖出的确切数量。在这种情况下，由于您想基于当前价格卖出价值 100 USDT 的币，您需要部分卖出的确切数量是 `100 * 2 / 400`，等于 0.5 个 SHITCOIN/USDT。由于我们现在知道要卖出的确切数量（0.5），您需要在 `adjust_trade_position` 函数中返回的值是 `-部分退出数量 * trade.stake_amount / trade.amount`，等于 -25。Bot 将卖出 0.5 个 SHITCOIN/USDT，保留 1.5 个在交易中。您将从部分退出中获得 100 USDT。

!!! Warning "止损计算"
    止损仍然从初始开仓价格计算，而不是均价。
    常规止损规则仍然适用（不能向下移动）。

    虽然 `/stopentry` 命令阻止 bot 进入新交易，但仓位调整功能将继续在现有交易上购买新订单。

``` python
# Default imports

class DigDeeperStrategy(IStrategy):

    position_adjustment_enable = True

    # Attempts to handle large drops with DCA. High stoploss is required.
    stoploss = -0.30

    # ... populate_* methods

    # Example specific variables
    max_entry_position_adjustment = 3
    # This number is explained a bit further down
    max_dca_multiplier = 5.5

    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:

        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        return proposed_stake / self.max_dca_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs
                              ) -> float | None | tuple[float | None, str | None]:
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be
        increased or decreased.
        This means extra entry or exit orders with additional fees.
        Only called when `position_adjustment_enable` is set to True.

        For full documentation please go to https://www.freqtrade.io/en/stable/strategy-advanced/

        When not implemented by a strategy, returns None

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current entry rate (same as current_entry_profit)
        :param current_profit: Current profit (as ratio), calculated based on current_rate 
                               (same as current_entry_profit).
        :param min_stake: Minimal stake size allowed by exchange (for both entries and exits)
        :param max_stake: Maximum stake allowed (either through balance, or by exchange limits).
        :param current_entry_rate: Current rate using entry pricing.
        :param current_exit_rate: Current rate using exit pricing.
        :param current_entry_profit: Current profit using entry pricing.
        :param current_exit_profit: Current profit using exit pricing.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade,
                       Positive values to increase position, Negative values to decrease position.
                       Return None for no action.
                       Optionally, return a tuple with a 2nd element with an order reason
        """
        if trade.has_open_orders:
            # Only act if no orders are open
            return

        if current_profit > 0.05 and trade.nr_of_successful_exits == 0:
            # Take half of the profit at +5%
            return -(trade.stake_amount / 2), "half_profit_5%"

        if current_profit > -0.05:
            return None

        # Obtain pair dataframe (just to show how to access it)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        # Only buy when not actively falling price.
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        if last_candle["close"] < previous_candle["close"]:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        # Allow up to 3 additional increasingly larger buys (4 in total)
        # Initial buy is 1x
        # If that falls to -5% profit, we buy 1.25x more, average profit should increase to roughly -2.2%
        # If that falls down to -5% again, we buy 1.5x more
        # If that falls once again down to -5%, we buy 1.75x more
        # Total stake for this trade would be 1 + 1.25 + 1.5 + 1.75 = 5.5x of the initial allowed stake.
        # That is why max_dca_multiplier is 5.5
        # Hope you have a deep wallet!
        try:
            # This returns first order stake size
            stake_amount = filled_entries[0].stake_amount_filled
            # This then calculates current safety order size
            stake_amount = stake_amount * (1 + (count_of_entries * 0.25))
            return stake_amount, "1/3rd_increase"
        except Exception as exception:
            return None

        return None

```

### 仓位调整计算

* 入场价格使用加权平均法计算。
* 退出不会影响平均入场价格。
* 部分退出的相对利润是相对于此时的平均入场价格。
* 最终退出的相对利润是基于总投入资金计算的。（参见下面的示例）

??? example "计算示例"
    *此示例为简化起见假设手续费为 0，且为某虚拟币的多头仓位。*

    * 买入 100@8\$
    * 买入 100@9\$ -> 均价: 8.5\$
    * 卖出 100@10\$ -> 均价: 8.5\$, 已实现利润 150\$, 17.65%
    * 买入 150@11\$ -> 均价: 10\$, 已实现利润 150\$, 17.65%
    * 卖出 100@12\$ -> 均价: 10\$, 总已实现利润 350\$, 20%
    * 卖出 150@14\$ -> 均价: 10\$, 总已实现利润 950\$, 40%  <- *这将是最后一条"退出"消息*

    此交易的总利润为 950$，投资为 3350$（`100@8$ + 100@9$ + 150@11$`）。因此——最终相对利润为 28.35%（`950 / 3350`）。

## 调整订单价格

`adjust_order_price()` 回调可被策略开发者用于在新 K 线到来时刷新/替换限价单。
此回调在每次迭代中调用一次，除非订单在当前 K 线内已被（重新）下单——将每个订单的最大（重新）下单次数限制为每根 K 线一次。
这也意味着第一次调用将在初始订单下单后的下一根 K 线开始时。

请注意，`custom_entry_price()`/`custom_exit_price()` 仍然是在信号产生时决定初始限价单目标价格的函数。

可以通过返回 `None` 从此回调中取消订单。

返回 `current_order_rate` 将保持交易所上的订单"原样"。
返回任何其他价格将取消现有订单，并用新订单替换。

如果原始订单的取消失败，则订单不会被替换——尽管订单很可能已在交易所上被取消。如果这发生在初始入场时，将导致订单被删除，而在仓位调整订单时，将导致交易大小保持不变。
如果订单已部分成交，订单将不会被替换。但是，您可以使用 [`adjust_trade_position()`](#adjust-trade-position) 将交易大小调整到预期的仓位大小（如果需要/希望的话）。

!!! Warning "常规超时"
    入场 `unfilledtimeout` 机制（以及 `check_entry_timeout()`/`check_exit_timeout()`）优先于此回调。
    通过上述方法取消的订单不会调用此回调。请务必更新超时值以匹配您的期望。

```python
# Default imports

class AwesomeStrategy(IStrategy):

    # ... populate_* methods

    def adjust_order_price(
        self,
        trade: Trade,
        order: Order | None,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        current_order_rate: float,
        entry_tag: str | None,
        side: str,
        is_entry: bool,
        **kwargs,
    ) -> float | None:
        """
        Exit and entry order price re-adjustment logic, returning the user desired limit price.
        This only executes when a order was already placed, still open (unfilled fully or partially)
        and not timed out on subsequent candles after entry trigger.

        For full documentation please go to https://www.freqtrade.io/en/stable/strategy-callbacks/

        When not implemented by a strategy, returns current_order_rate as default.
        If current_order_rate is returned then the existing order is maintained.
        If None is returned then order gets canceled but not replaced by a new one.

        :param pair: Pair that's currently analyzed
        :param trade: Trade object.
        :param order: Order object
        :param current_time: datetime object, containing the current datetime
        :param proposed_rate: Rate, calculated based on pricing settings in entry_pricing.
        :param current_order_rate: Rate of the existing order in place.
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :param is_entry: True if the order is an entry order, False if it's an exit order.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float or None: New entry price value if provided
        """

        # Limit entry orders to use and follow SMA200 as price target for the first 10 minutes since entry trigger for BTC/USDT pair.
        if (
            is_entry
            and pair == "BTC/USDT" 
            and entry_tag == "long_sma200" 
            and side == "long" 
            and (current_time - timedelta(minutes=10)) <= trade.open_date_utc
        ):
            # just cancel the order if it has been filled more than half of the amount
            if order.filled > order.remaining:
                return None
            else:
                dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
                current_candle = dataframe.iloc[-1].squeeze()
                # desired price
                return current_candle["sma_200"]
        # default: maintain existing order
        return current_order_rate
```

!!! danger "与 `adjust_*_price()` 的不兼容性"
    如果您同时实现了 `adjust_order_price()` 和 `adjust_entry_price()`/`adjust_exit_price()`，则只使用 `adjust_order_price()`。
    如果您需要调整入场/退出价格，您可以在 `adjust_order_price()` 中实现逻辑，或使用拆分的 `adjust_entry_price()` / `adjust_exit_price()` 回调，但不能同时使用两者。
    混合使用这些是不支持的，会在 bot 启动时引发错误。

### 调整入场价格

`adjust_entry_price()` 回调可被策略开发者用于在到来时刷新/替换入场限价单。
它是 `adjust_order_price()` 的子集，仅对入场订单调用。
所有其余行为与 `adjust_order_price()` 相同。

交易开仓日期（`trade.open_date_utc`）将保持在第一个订单下单的时间。
请注意这一点——并最终在其他回调中调整您的逻辑以考虑这一点，使用第一个成交订单的日期代替。

### 调整退出价格

`adjust_exit_price()` 回调可被策略开发者用于在到来时刷新/替换退出限价单。
它是 `adjust_order_price()` 的子集，仅对退出订单调用。
所有其余行为与 `adjust_order_price()` 相同。

## 杠杆回调

在允许杠杆的市场中交易时，此方法必须返回所需的杠杆（默认为 1 -> 无杠杆）。

假设资金为 500USDT，杠杆为 3 的交易将产生 500 x 3 = 1500 USDT 的仓位。

超过 `max_leverage` 的值将被调整为 `max_leverage`。
对于不支持杠杆的市场/交易所，此方法将被忽略。

``` python
# Default imports

class AwesomeStrategy(IStrategy):
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 1.0
```

所有利润计算都包含杠杆。止损/ROI 的计算也包含杠杆。
在 10 倍杠杆下定义 10% 的止损，将在价格下跌 1% 时触发止损。

## 订单成交回调

`order_filled()` 回调可用于在订单成交后根据当前交易状态执行特定操作。
它将独立于订单类型（入场、退出、止损或仓位调整）被调用。

假设您的策略需要存储交易入场时 K 线的最高价，可以使用此回调实现，如以下示例所示。

``` python
# Default imports

class AwesomeStrategy(IStrategy):
    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> None:
        """
        Called right after an order fills. 
        Will be called for all order types (entry, exit, stoploss, position adjustment).
        :param pair: Pair for trade
        :param trade: trade object.
        :param order: Order object.
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        # Obtain pair dataframe (just to show how to access it)
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        if (trade.nr_of_successful_entries == 1) and (order.ft_order_side == trade.entry_side):
            trade.set_custom_data(key="entry_candle_high", value=last_candle["high"])

        return None

```

!!! Tip "了解更多关于存储数据的信息"
    您可以在[存储自定义交易数据](strategy-advanced.md#storing-information-persistent)部分了解更多关于存储数据的信息。
    请注意，这被认为是高级用法，应谨慎使用。

## 图表标注回调

图表标注回调在 freqUI 请求数据显示图表时调用。
此回调在交易周期上下文中没有意义，仅用于图表目的。

策略可以返回一个 `AnnotationType` 对象列表以显示在图表上。
根据返回的内容——图表可以显示水平区域、垂直区域、矩形或线条。

### 标注类型

目前支持两种类型的标注：`area` 和 `line`。

#### 区域

``` json
{
    "type": "area", // Type of the annotation, currently only "area" is supported
    "start": "2024-01-01 15:00:00", // Start date of the area
    "end": "2024-01-01 16:00:00",  // End date of the area
    "y_start": 94000.2,  // Price / y axis value
    "y_end": 98000, // Price / y axis value
    "color": "",
    "z_level": 5, // z-level, higher values are drawn on top of lower values. Positions relative to the Chart elements need to be set in freqUI.
    "label": "some label"
}
```

#### 线条

``` json
{
    "type": "line", // Type of the annotation, currently only "line" is supported
    "start": "2024-01-01 15:00:00", // Start date of the line
    "end": "2024-01-01 16:00:00",  // End date of the line
    "y_start": 94000.2,  // Price / y axis value
    "y_end": 98000, // Price / y axis value
    "color": "",
    "z_level": 5, // z-level, higher values are drawn on top of lower values. Positions relative to the Chart elements need to be set in freqUI.
    "label": "some label",
    "width": 2, // Optional, line width in pixels. Defaults to 1
    "line_style": "dashed", // Optional, can be "solid", "dashed" or "dotted". Defaults to "solid"

}
```

#### 点

``` json
{
    "type": "point", // Type of the annotation, currently only "point" is supported
    "x": "2024-01-01 15:00:00", // Start date of the point
    "y": 94000.2,  // Price / y axis value
    "color": "",
    "z_level": 5, // z-level, higher values are drawn on top of lower values. Positions relative to the Chart elements need to be set in freqUI.
    "label": "some label",
    "size": 2, // Optional, line width in pixels. Defaults to 10
    "shape": "circle", // Optional, can be "circle", "rect", "roundRect", "triangle", "pin", "arrow", "none".
    "rotate": 0, // Optional, rotation of the shape/symbol in degrees. Defaults to 0

}
```

以下示例将以灰色区域标记 8 点和 15 点的图表，突出显示市场开盘和收盘时间。
这显然是一个非常基础的示例。

``` python
# Default imports

class AwesomeStrategy(IStrategy):
    def plot_annotations(
        self, pair: str, start_date: datetime, end_date: datetime, dataframe: DataFrame, **kwargs
    ) -> list[AnnotationType]:
        """
        Retrieve area annotations for a chart.
        Must be returned as array, with type, label, color, start, end, y_start, y_end.
        All settings except for type are optional - though it usually makes sense to include either
        "start and end" or "y_start and y_end" for either horizontal or vertical plots
        (or all 4 for boxes).
        :param pair: Pair that's currently analyzed
        :param start_date: Start date of the chart data being requested
        :param end_date: End date of the chart data being requested
        :param dataframe: DataFrame with the analyzed data for the chart
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return: List of AnnotationType objects
        """
        annotations = []
        while start_dt < end_date:
            start_dt += timedelta(hours=1)
            if start_dt.hour in (8, 15):
                annotations.append(
                    {
                        "type": "area",
                        "label": "Trade open and close hours",
                        "start": start_dt,
                        "end": start_dt + timedelta(hours=1),
                        # Omitting y_start and y_end will result in a vertical area spanning the whole height of the main Chart
                        "color": "rgba(133, 133, 133, 0.4)",
                    }
                )

        return annotations

```

条目将被验证，如果不符合预期的架构将不会传递给 UI，如果不符合将记录错误。

!!! Warning "大量标注"
    使用过多标注可能导致 UI 卡顿，尤其是在绘制大量历史数据时。
    请谨慎使用标注功能。

### 图示标注示例

![FreqUI - plot Annotations](../assets/freqUI-chart-annotations-dark.png#only-dark)
![FreqUI - plot Annotations](../assets/freqUI-chart-annotations-light.png#only-light)

??? Info "上图使用的代码"
    这是一个示例代码，应仅作为示例参考。

    ``` python
    # Default imports

    class AwesomeStrategy(IStrategy):
        def plot_annotations(
            self, pair: str, start_date: datetime, end_date: datetime, dataframe: DataFrame, **kwargs
        ) -> list[AnnotationType]:
            annotations = []
            while start_dt < end_date:
                start_dt += timedelta(hours=1)
                if (start_dt.hour % 4) == 0:
                    annotations.append(
                        {
                            "type": "area",
                            "label": "4h",
                            "start": start_dt,
                            "end": start_dt + timedelta(hours=1),
                            "color": "rgba(133, 133, 133, 0.4)",
                        }
                    )
                elif (start_dt.hour % 2) == 0:
                price = dataframe.loc[dataframe["date"] == start_dt, "close"].mean()
                    annotations.append(
                        {
                            "type": "area",
                            "label": "2h",
                            "start": start_dt,
                            "end": start_dt + timedelta(hours=1),
                            "y_end": price * 1.01,
                            "y_start": price * 0.99,
                            "color": "rgba(0, 255, 0, 0.4)",
                            "z_level": 5,
                        }
                    )

            return annotations

    ```