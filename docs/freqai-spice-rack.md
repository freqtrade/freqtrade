# Using the `spice_rack`

!!! Note:
    `spice_rack` indicators should not be used exclusively for entries and exits, the following example is just a demonstration of syntax. `spice_rack` indicators should **always** be used to support existing strategies.

The `spice_rack` is aimed at users who do not wish to deal with setting up `FreqAI` confgs, but instead prefer to interact with `FreqAI` similar to a `talib` indicator. In this case, the user can instead simply add two keys to their config:

```json
    "freqai_spice_rack": true, 
    "freqai_identifier": "spicey-id",
```

Which tells `FreqAI` to set up a pre-set `FreqAI` instance automatically under the hood with preset parameters. Now the user can access a suite of custom `FreqAI` supercharged indicators inside their strategy by placing the following code into `populate_indicators`:

```python
        dataframe['dissimilarity_index'] = self.freqai.spice_rack(
            'DI_values', dataframe, metadata, self)
        dataframe['extrema'] = self.freqai.spice_rack(
            '&s-extrema', dataframe, metadata, self)
        self.freqai.close_spice_rack()  # user must close the spicerack
```

Users can then use these columns in concert with all their own additional indicators added to `populate_indicators` in their entry/exit criteria and strategy callback methods the same way as any typical indicator. For example:

```python
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['dissimilarity_index'] < 1) &
                (df['extrema'] < -0.1)
            ),
            'enter_long'] = 1

        df.loc[
            (
                (df['dissimilarity_index'] < 1) &
                (df['extrema'] > 0.1)
            ),
            'enter_short'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['dissimilarity_index'] < 1) &
                (df['extrema'] > 0.1) 
            ),

            'exit_long'] = 1

        df.loc[
            (

                (df['dissimilarity_index'] < 1) &
                (df['extrema'] < -0.1)
            ),
            'exit_short'] = 1

        return df
```


## Available indicators

|  Parameter | Description |
|------------|-------------|
| `DI_values` | **Required.** <br> The dissimilarity index of the current candle to the recent candles. More information available [here](freqai-feature-engineering.md#identifying-outliers-with-the-dissimilarity-index-di) <br> **Datatype:** Floats.
| `extrema` | **Required.** <br> A continuous prediction from FreqAI which aims to help predict if the current candle is a maxima or a minma. FreqAI aims for 1 to be a maxima and -1 to be a minima - but the values should typically hover between -0.2 and 0.2. <br> **Datatype:** Floats.
