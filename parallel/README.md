1. install parallel:

```unzip pp-1.6.4.4.zip```

```cd pp-1.6.4.4```

```python3.6 setup.py install```

2. Move Generator.py into the parent folder, or your "main freqtrade folder."

```mv Generator.py ../```

3. Move the default strategy over the default strategy in freqtrade/strategies folder.

4. Move backtesting.py over the backtesting.py in freqtrade/optimize folder.

5. Optinlally install modded bittrex.py

6. Install dependencies:

```
sudo add-apt-repository ppa:jonathonf/python-3.6

sudo apt-get update

sudo apt-get install python3.6 python3.6-venv python3.6-dev build-essential autoconf libtool pkg-config make wget git

```


7. Install ta-lib:

```
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

tar xvzf ta-lib-0.4.0-src.tar.gz

cd ta-lib

./configure --prefix=/usr

make

make install

cd ..

rm -rf ./ta-lib*

```

8. Install freqtrade:

```
cd ~/freqtrade && pip3.6 install -r requirements.txt && python3.6 setup.py install && pip3.6 install -e .
```

9. Run generator.py:

```python3.6 generator.py```

10. Wait for results.

11. Implement these results into your default_strategy:

***
You will need to read the if statements and populate_buy_signal and populate_sell_signal in this file carefully.

Once implemented, remove the if statements in the populate_buy_trend.

***

For ease of use, here is an example:

The if statements that run the random generator are:

```
    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:

        conditions = []
        # GUARDS AND TRENDS
        if 'uptrend_long_ema' in str(self.params):
            conditions.append(dataframe['ema50'] > dataframe['ema100'])
        if 'macd_below_zero' in str(self.params):
            conditions.append(dataframe['macd'] < 0)
        if 'uptrend_short_ema' in str(self.params):
            conditions.append(dataframe['ema5'] > dataframe['ema10'])
        if 'mfi' in str(self.params):

            conditions.append(dataframe['mfi'] < self.valm)
        if 'fastd' in str(self.params):

            conditions.append(dataframe['fastd'] < self.valfast)
        if 'adx' in str(self.params):

            conditions.append(dataframe['adx'] > self.valadx)
        if 'rsi' in str(self.params):

            conditions.append(dataframe['rsi'] < self.valrsi)
        if 'over_sar' in str(self.params):
            conditions.append(dataframe['close'] > dataframe['sar'])
        if 'green_candle' in str(self.params):
            conditions.append(dataframe['close'] > dataframe['open'])
        if 'uptrend_sma' in str(self.params):
            prevsma = dataframe['sma'].shift(1)
            conditions.append(dataframe['sma'] > prevsma)
        if 'closebb' in str(self.params):
            conditions.append(dataframe['close'] < dataframe['bb_lowerband'])
        if 'temabb' in str(self.params):
            conditions.append(dataframe['tema'] < dataframe['bb_lowerband'])
        if 'fastdt' in str(self.params):
            conditions.append(qtpylib.crossed_above(dataframe['fastd'], 10.0))
        if 'ao' in str(self.params):
            conditions.append(qtpylib.crossed_above(dataframe['ao'], 0.0))
        if 'ema3' in str(self.params):
            conditions.append(qtpylib.crossed_above(dataframe['ema3'], dataframe['ema10']))
        if 'macd' in str(self.params):
            conditions.append(qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']))
        if 'closesar' in str(self.params):
            conditions.append(qtpylib.crossed_above(dataframe['close'], dataframe['sar']))
        if 'htsine' in str(self.params):
            conditions.append(qtpylib.crossed_above(dataframe['htleadsine'], dataframe['htsine']))
        if 'has' in str(self.params):
            conditions.append((qtpylib.crossed_above(dataframe['ha_close'], dataframe['ha_open'])) & (dataframe['ha_low'] == dataframe['ha_open']))
        if 'plusdi' in str(self.params):
            conditions.append(qtpylib.crossed_above(dataframe['plus_di'], dataframe['minus_di']))

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'buy'] = 1

        return dataframe
```


So of you get MFI as a option runnning generator.py, and it's option in output is 91, look at the if statements above at mfi, the populate_buy_trend will now look like this:


```
    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
            (dataframe['mfi'] < 91)
            ),
            'buy'] = 1

        return dataframe
```

It's as simple as that.
