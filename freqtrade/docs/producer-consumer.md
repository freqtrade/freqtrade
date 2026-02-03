# Producer / Consumer mode

freqtrade provides a mechanism whereby an instance (also called `consumer`) may listen to messages from an upstream freqtrade instance (also called `producer`) using the message websocket. Mainly, `analyzed_df` and `whitelist` messages. This allows the reuse of computed indicators (and signals) for pairs in multiple bots without needing to compute them multiple times.

See [Message Websocket](rest-api.md#message-websocket) in the Rest API docs for setting up the `api_server` configuration for your message websocket (this will be your producer).

!!! Note
    We strongly recommend to set `ws_token` to something random and known only to yourself to avoid unauthorized access to your bot.

## Configuration

Enable subscribing to an instance by adding the `external_message_consumer` section to the consumer's config file.

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

|  Parameter | Description |
|------------|-------------|
| `enabled` | **Required.** Enable consumer mode. If set to false, all other settings in this section are ignored.<br>*Defaults to `false`.*<br> **Datatype:** boolean .
| `producers` | **Required.** List of producers <br> **Datatype:** Array.
| `producers.name` | **Required.** Name of this producer. This name must be used in calls to `get_producer_pairs()` and `get_producer_df()` if more than one producer is used.<br> **Datatype:** string
| `producers.host` | **Required.** The hostname or IP address from your producer.<br> **Datatype:** string
| `producers.port` | **Required.** The port matching the above host.<br>*Defaults to `8080`.*<br> **Datatype:** Integer
| `producers.secure` | **Optional.**  Use ssl in websockets connection. Default False.<br> **Datatype:** string
| `producers.ws_token` | **Required.**  `ws_token` as configured on the producer.<br> **Datatype:** string
| | **Optional settings**
| `wait_timeout` | Timeout until we ping again if no message is received. <br>*Defaults to `300`.*<br> **Datatype:** Integer - in seconds.
| `ping_timeout` | Ping timeout <br>*Defaults to `10`.*<br> **Datatype:** Integer - in seconds.
| `sleep_time` | Sleep time before retrying to connect.<br>*Defaults to `10`.*<br> **Datatype:** Integer - in seconds.
| `remove_entry_exit_signals` | Remove signal columns from the dataframe (set them to 0) on dataframe receipt.<br>*Defaults to `false`.*<br> **Datatype:** Boolean.
| `initial_candle_limit` | Initial candles to expect from the Producer.<br>*Defaults to `1500`.*<br> **Datatype:** Integer - Number of candles.
| `message_size_limit` | Size limit per message<br>*Defaults to `8`.*<br> **Datatype:** Integer - Megabytes.

Instead of (or as well as) calculating indicators in `populate_indicators()` the follower instance listens on the connection to a producer instance's messages (or multiple producer instances in advanced configurations) and requests the producer's most recently analyzed dataframes for each pair in the active whitelist.

A consumer instance will then have a full copy of the analyzed dataframes without the need to calculate them itself.

## Examples

### Example - Producer Strategy

A simple strategy with multiple indicators. No special considerations are required in the strategy itself.

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
    You can use this to setup [FreqAI](freqai.md) on a powerful machine, while you run consumers on simple machines like raspberries, which can interpret the signals generated from the producer in different ways.


### Example - Consumer Strategy

A logically equivalent strategy which calculates no indicators itself, but will have the same analyzed dataframes available to make trading decisions based on the indicators calculated in the producer. In this example the consumer has the same entry criteria, however this is not necessary. The consumer may use different logic to enter/exit trades, and only use the indicators as specified.

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

!!! Tip "Using upstream signals"
    By setting `remove_entry_exit_signals=false`, you can also use the producer's signals directly. They should be available as `enter_long_default` (assuming `suffix="default"` was used) - and can be used as either signal directly, or as additional indicator.
