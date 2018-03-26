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

***```
You will need to read the if statements and populate_buy_signal and populate_sell_signal in this file carefully.

Once implemented, remove the if statements in the populate_buy_trend.

```***

