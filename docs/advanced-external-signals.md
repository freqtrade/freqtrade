# External Signals

freqtrade provides a mechanism whereby an instance may listen to messages from an upstream freqtrade instance using the message websocket. Mainly, `analyzed_df` and `whitelist` messages. This allows the reuse of computed indicators (and signals) for pairs in multiple bots without needing to compute them multiple times.

See [Message Websocket](rest-api.md#message-websocket) in the Rest API docs for setting up the `api_server` configuration for your message websocket.

!!! Note
    We strongly recommend to also set `ws_token` to something random and known only to yourself to avoid unauthorized access to your bot.

## Configuration

Enable subscribing to an instance by adding the `external_message_consumer` section to the follower's config file.

```json
{
    //...
   "external_message_consumer": {
        "enabled": true,
        "producers": [
            {
                "name": "default", // This can be any name you'd like, default is "default"
                "host": "127.0.0.1", // The host from your leader's api_server config
                "port": 8080, // The port from your leader's api_server config
                "ws_token": "mysecretapitoken" // The ws_token from your leader's api_server config
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
| `enabled` | **Required.** Enable follower mode. If set to false, all other settings in this section are ignored.<br>*Defaults to `false`.*<br> **Datatype:** boolean .
| `producers` | **Required.** List of producers <br> **Datatype:** Array.
| `producers.name` | **Required.** Name of this producer. This name must be used in calls to `get_producer_pairs()` and `get_external_df()` if more than one producer is used.<br> **Datatype:** string
| `producers.host` | **Required.** The hostname or IP address from your leader.<br> **Datatype:** string
| `producers.port` | **Required.** The port matching the above host.<br> **Datatype:** string
| `producers.ws_token` | **Required.**  `ws_token` as configured on the leader.<br> **Datatype:** string
| | **Optional settings**
| `wait_timeout` | Timeout until we ping again if no message is received. <br>*Defaults to `300`.*<br> **Datatype:** Integer - in seconds.
| `wait_timeout` | Ping timeout <br>*Defaults to `10`.*<br> **Datatype:** Integer - in seconds.
| `sleep_time` | Sleep time before retrying to connect.<br>*Defaults to `10`.*<br> **Datatype:** Integer - in seconds.
| `remove_entry_exit_signals` | Remove signal columns from the dataframe (set them to 0) on dataframe receipt.<br>*Defaults to `10`.*<br> **Datatype:** Integer - in seconds.
| `message_size_limit` | Size limit per message<br>*Defaults to `8`.*<br> **Datatype:** Integer - Megabytes.

Instead of (or as well as) calculating indicators in `populate_indicators()` the follower instance listens on the connection to a leader instance's messages (or multiple leader instances in advanced configurations) and requests the leader's most recently analyzed dataframes for each pair in the active whitelist.

A follower instance will then have a full copy of the analyzed dataframes without the need to calculate them itself.

## Examples

### Example - Leader Strategy

A simple strategy with multiple indicators. No special considerations are required in the strategy itself.

```py
class LeaderStrategy(IStrategy):
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

### Example - Follower Strategy

A logically equivalent strategy which calculates no indicators itself, but will have the same analyzed dataframes available to make trading decisions based on the indicators calculated in the leader. In this example the follower has the same entry criteria, however this is not necessary. The follower may use different logic to enter/exit trades, and only use the indicators as specified.

```py
class FollowerStrategy(IStrategy):
    #...
    process_only_new_candles = False # required for followers

    _columns_to_expect = ['rsi_default', 'tema_default', 'bb_middleband_default']

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Use the websocket api to get pre-populated indicators from another freqtrade instance.
        Use `self.dp.get_external_df(pair)` to get the dataframe
        """
        pair = metadata['pair']
        timeframe = self.timeframe

        leader_pairs = self.dp.get_producer_pairs()
        # You can specify which producer to get pairs from via:
        # self.dp.get_producer_pairs("my_other_producer")

        # This func returns the analyzed dataframe, and when it was analyzed
        leader_dataframe, _ = self.dp.get_external_df(pair)
        # You can get other data if your leader makes it available:
        # self.dp.get_external_df(
        #   pair,
        #   timeframe="1h",
        #   candle_type=CandleType.SPOT,
        #   producer_name="my_other_producer"
        # )

        if not leader_dataframe.empty:
            # If you plan on passing the leader's entry/exit signal directly,
            # specify ffill=False or it will have unintended results
            merged_dataframe = merge_informative_pair(dataframe, leader_dataframe,
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
