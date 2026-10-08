<!-- 本文件为中文翻译版，由 AI 根据 docs/producer-consumer.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# 生产者 / 消费者模式

freqtrade 提供了一种机制，使得一个实例（也称为 `consumer`）可以通过消息 websocket 监听来自上游 freqtrade 实例（也称为 `producer`）的消息，主要是 `analyzed_df` 和 `whitelist` 消息。这允许在多个机器人之间复用已计算的指标（和信号），而无需重复计算。

有关设置消息 websocket 的 `api_server` 配置（这将是你的生产者），请参阅 Rest API 文档中的 [Message Websocket](rest-api.md#message-websocket)。

!!! Note
    我们强烈建议将 `ws_token` 设置为仅自己知道的随机值，以避免对机器人的未授权访问。

## 配置

通过在消费者的配置文件中添加 `external_message_consumer` 部分来启用对实例的订阅。

```json
{
    //...
   "external_message_consumer": {
        "enabled": true,
        "producers": [
            {
                "name": "default", // This can be any name you'd like, default is "default"
                "host": "127.0.0.1", // The host from your producer's api_server config
                "port": 8080, // The port from your producer's api_server config
                "secure": false, // Use a secure websockets connection, default false
                "ws_token": "sercet_Ws_t0ken" // The ws_token from your producer's api_server config
            }
        ],
        // The following configurations are optional, and usually not required
        // "wait_timeout": 300,
        // "ping_timeout": 10,
        // "sleep_time": 10,
        // "remove_entry_exit_signals": false,
        // "message_size_limit": 8
    }
    //...
}
```

| 参数 | 描述 |
|------------|-------------|
| `enabled` | **必填。** 启用消费者模式。如果设置为 false，则忽略此部分中的所有其他设置。<br>*默认为 `false`。*<br> **数据类型：** boolean。
| `producers` | **必填。** 生产者列表。<br> **数据类型：** Array。
| `producers.name` | **必填。** 此生产者的名称。如果使用多个生产者，则在调用 `get_producer_pairs()` 和 `get_producer_df()` 时必须使用此名称。<br> **数据类型：** string
| `producers.host` | **必填。** 生产者的主机名或 IP 地址。<br> **数据类型：** string
| `producers.port` | **必填。** 与上述主机匹配的端口。<br>*默认为 `8080`。*<br> **数据类型：** Integer
| `producers.secure` | **可选。** 在 websocket 连接中使用 SSL。默认为 False。<br> **数据类型：** string
| `producers.ws_token` | **必填。** 在生产者上配置的 `ws_token`。<br> **数据类型：** string
| | **可选设置**
| `wait_timeout` | 如果未收到消息，再次发送 ping 之前的超时时间。<br>*默认为 `300`。*<br> **数据类型：** Integer - 单位为秒。
| `ping_timeout` | Ping 超时时间。<br>*默认为 `10`。*<br> **数据类型：** Integer - 单位为秒。
| `sleep_time` | 重试连接前的休眠时间。<br>*默认为 `10`。*<br> **数据类型：** Integer - 单位为秒。
| `remove_entry_exit_signals` | 在接收到数据框时移除数据框中的信号列（将它们设为 0）。<br>*默认为 `false`。*<br> **数据类型：** Boolean。
| `initial_candle_limit` | 期望从生产者获取的初始 K 线数量。<br>*默认为 `1500`。*<br> **数据类型：** Integer - K 线数量。
| `message_size_limit` | 每条消息的大小限制。<br>*默认为 `8`。*<br> **数据类型：** Integer - 兆字节。

跟随实例无需（或也可以同时）在 `populate_indicators()` 中计算指标，而是监听与生产者实例消息的连接（或在高级配置中监听多个生产者实例），并请求生产者针对活跃白名单中每个交易对的最新分析数据框。

消费者实例随后将拥有一份完整的已分析数据框副本，而无需自行计算。

## 示例

### 示例 - 生产者策略

一个包含多个指标的简单策略。策略本身不需要任何特殊考虑。

```py
class ProducerStrategy(IStrategy):
    #...
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate indicators in the standard freqtrade way which can then be broadcast to other instances
        """
        dataframe['rsi'] = ta.RSI(dataframe)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates the entry signal for the given dataframe
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], self.buy_rsi.value)) &
                (dataframe['tema'] <= dataframe['bb_middleband']) &
                (dataframe['tema'] > dataframe['tema'].shift(1)) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe
```

!!! Tip "FreqAI"
    你可以使用此功能在一台性能强大的机器上设置 [FreqAI](freqai.md)，同时在像树莓派这样的简单机器上运行消费者，消费者可以以不同的方式解释生产者生成的信号。


### 示例 - 消费者策略

一个逻辑上等价的策略，它本身不计算任何指标，但将拥有相同的已分析数据框，可以基于生产者中计算的指标进行交易决策。在此示例中，消费者具有相同的入场条件，但这不是必须的。消费者可以使用不同的逻辑来开仓/平仓交易，仅使用指定的指标。

```py
class ConsumerStrategy(IStrategy):
    #...
    process_only_new_candles = False # required for consumers

    _columns_to_expect = ['rsi_default', 'tema_default', 'bb_middleband_default']

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Use the websocket api to get pre-populated indicators from another freqtrade instance.
        Use `self.dp.get_producer_df(pair)` to get the dataframe
        """
        pair = metadata['pair']
        timeframe = self.timeframe

        producer_pairs = self.dp.get_producer_pairs()
        # You can specify which producer to get pairs from via:
        # self.dp.get_producer_pairs("my_other_producer")

        # This func returns the analyzed dataframe, and when it was analyzed
        producer_dataframe, _ = self.dp.get_producer_df(pair)
        # You can get other data if the producer makes it available:
        # self.dp.get_producer_df(
        #   pair,
        #   timeframe="1h",
        #   candle_type=CandleType.SPOT,
        #   producer_name="my_other_producer"
        # )

        if not producer_dataframe.empty:
            # If you plan on passing the producer's entry/exit signal directly,
            # specify ffill=False or it will have unintended results
            merged_dataframe = merge_informative_pair(dataframe, producer_dataframe,
                                                      timeframe, timeframe,
                                                      append_timeframe=False,
                                                      suffix="default")
            return merged_dataframe
        else:
            dataframe[self._columns_to_expect] = 0

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates the entry signal for the given dataframe
        """
        # Use the dataframe columns as if we calculated them ourselves
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi_default'], self.buy_rsi.value)) &
                (dataframe['tema_default'] <= dataframe['bb_middleband_default']) &
                (dataframe['tema_default'] > dataframe['tema_default'].shift(1)) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe
```

!!! Tip "使用上游信号"
    通过设置 `remove_entry_exit_signals=false`，你也可以直接使用生产者的信号。它们应以 `enter_long_default` 的形式可用（假设使用了 `suffix="default"`）——可以直接作为信号使用，也可以作为额外的指标。