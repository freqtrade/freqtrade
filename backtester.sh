#!/bin/bash

strategy_name="KalmanFilterStrategy"
fee=0.001
start_time="20240101"
timeframe="5m"
config_file="user_data/config.json"
max_open_trades=3
pairs=(BTC/USDT ADA/USDT ETH/USDT BNB/USDT)

echo -e "\ndownloading data: \n"
freqtrade download-data -p $pairs --timerange ${start_time}- -t $timeframe -c $config_file

echo -e "\nrunning backtest: \n"
freqtrade backtesting -i $timeframe --fee $fee -s $strategy_name --max-open-trades $max_open_trades -p $pairs --timerange ${start_time}- -c $config_file 
echo -e "\n****************************************************\n"

echo -e "\nploting backtest results: \n"
freqtrade plot-dataframe -s $strategy_name -p $pairs --timerange ${start_time}- -c $config_file
