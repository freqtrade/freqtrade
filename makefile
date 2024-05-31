trade:
	freqtrade trade --config /workspaces/freqtrade/user_data/config.json --freqaimodel XGBoostRegressor --strategy FreqaiExampleStrategy

load_data:
	freqtrade download-data --exchange binance --config /workspaces/freqtrade/user_data/config.json --pairs BTC/USDT:USDT --timeframe 3m 5m 15m 1h --timerange 20231201-20240331 --data-format-ohlcv json


# ================
load_data_case:
	freqtrade download-data --exchange binance --config /workspaces/freqtrade/user_data/config.json --pairs BTC/USDT:USDT ETH/USDT:USDT ADA/USDT:USDT LTC/USDT:USDT DOGE/USDT:USDT XRP/USDT:USDT XLM/USDT:USDT XMR/USDT:USDT --timeframe 3m 5m 15m 1h --timerange 20230101-20230501 --data-format-ohlcv json

gaps:
	python3 /workspaces/freqtrade/user_data/scripts/meke_gaps_in_data.py
	python3 /workspaces/freqtrade/user_data/scripts/meke_gaps_in_data.py
	python3 /workspaces/freqtrade/user_data/scripts/meke_gaps_in_data.py

backtests:
	freqtrade backtesting --timerange 20230101-20230201 --strategy FreqaiExampleStrategy --freqaimodel XGBoostRegressor --config /workspaces/freqtrade/user_data/config.json --data-format-ohlcv json --logfile /workspaces/freqtrade/user_data/backtest01.log
	make clean
	freqtrade backtesting --timerange 20230201-20230301 --strategy FreqaiExampleStrategy --freqaimodel XGBoostRegressor --config /workspaces/freqtrade/user_data/config.json --data-format-ohlcv json --logfile /workspaces/freqtrade/user_data/backtest02.log
	make clean
	freqtrade backtesting --timerange 20230301-20230401 --strategy FreqaiExampleStrategy --freqaimodel XGBoostRegressor --config /workspaces/freqtrade/user_data/config.json --data-format-ohlcv json --logfile /workspaces/freqtrade/user_data/backtest03.log
	make clean
	freqtrade backtesting --timerange 20230415-20230501 --strategy FreqaiExampleStrategy --freqaimodel XGBoostRegressor --config /workspaces/freqtrade/user_data/config.json --data-format-ohlcv json --logfile /workspaces/freqtrade/user_data/backtest04.log
	make clean
	freqtrade backtesting --timerange 20230101-20230601 --strategy FreqaiExampleStrategy --freqaimodel XGBoostRegressor --config /workspaces/freqtrade/user_data/config.json --data-format-ohlcv json --logfile /workspaces/freqtrade/user_data/backtest05.log
	make clean

before_merge:
	# make load_data_case
	# make gaps
	make backtests

pre_deploy:
	pytest
	ruff check .
	pre-commit run -a
	mypy freqtrade
	isort .
	

# ================

backtest_issue:
	freqtrade backtesting --timerange 20231201-20240331 --strategy FreqaiExampleStrategy --freqaimodel XGBoostRegressor --config /workspaces/freqtrade/user_data/config.json --data-format-ohlcv json

test:
	pytest /workspaces/freqtrade/tests/freqai/test_freqai_backtesting.py

clean:
	rm -rf /workspaces/freqtrade/user_data/models/unique-id
	cd /workspaces/freqtrade/user_data/backtest_results/ && rm .last_result.json && rm *.json
		
clean_data:
	rm -rf /workspaces/freqtrade/user_data/data/

format:
	ruff check . --fix