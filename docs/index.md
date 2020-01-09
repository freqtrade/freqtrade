# Freqtrade
[![Build Status](https://travis-ci.org/freqtrade/freqtrade.svg?branch=develop)](https://travis-ci.org/freqtrade/freqtrade)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

<!-- Place this tag where you want the button to render. -->
<a class="github-button" href="https://github.com/freqtrade/freqtrade" data-icon="octicon-star" data-size="large" aria-label="Star freqtrade/freqtrade on GitHub">Star</a>
<!-- Place this tag where you want the button to render. -->
<a class="github-button" href="https://github.com/freqtrade/freqtrade/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork freqtrade/freqtrade on GitHub">Fork</a>
<!-- Place this tag where you want the button to render. -->
<a class="github-button" href="https://github.com/freqtrade/freqtrade/archive/master.zip" data-icon="octicon-cloud-download" data-size="large" aria-label="Download freqtrade/freqtrade on GitHub">Download</a>
<!-- Place this tag where you want the button to render. -->
<a class="github-button" href="https://github.com/freqtrade" data-size="large" aria-label="Follow @freqtrade on GitHub">Follow @freqtrade</a>

## Introduction

Freqtrade is a crypto-currency algorithmic trading software developed in python (3.6+) and supported on Windows, macOS and Linux.

!!! Danger "DISCLAIMER"
    This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

    Always start by running a trading bot in Dry-run and do not engage money before you understand how it works and what profit/loss you should expect.

    We strongly recommend you to have basic coding skills and Python knowledge. Do not hesitate to read the source code and understand the mechanisms of this bot, algorithms and techniques implemented in it.

## Features

- Run: Run the bot on exchange with simulated money (Dry-Run mode) or with real money (Live-Trade mode).
- Select markets: Create your static list or use an automatic one based on top traded volumes and/or prices (not available during backtesting). You can also explicitly blacklist markets you don't want to trade.
- Download market data: Download historical data of the exchange and the markets your may want to trade with. The historical data can be based on [OHLCV](https://en.wikipedia.org/wiki/Open-high-low-close_chart) candles or be trade ticks (for exchanges that support this).
- Strategy: Write your strategy in python, using [pandas](https://pandas.pydata.org/). Example strategies to inspire you are available in the [strategy repository](https://github.com/freqtrade/freqtrade-strategies).
- Backtest: Test your strategy on downloaded historical data.
- Optimize: Find the best parameters for your strategy using hyperoptimization which employs machining learning methods. You can optimize buy, sell, take profit (ROI), stop-loss and trailing stop-loss parameters for your strategy.
- Run using Edge (optional module): The concept is to find the best historical [trade expectancy](edge.md#expectancy) by markets based on variation of the stop-loss and then allow/reject markets to trade. The sizing of the trade is based on a risk of a percentage of your capital.
- Control/Monitor: Use Telegram or a REST API (start/stop the bot, show profit/loss, daily summary, current open trades results, etc.).
- Analyse: Further analysis can be possibilities on either Backtesting data or Freqtrade trading history (SQL database), including automated standard plots, and methods to load the data into [interactive environments](data-analysis.md).

## Requirements

### Up to date clock

The clock on the system running the bot must be accurate, synchronized to a NTP server frequently enough to avoid problems with communication to the exchanges.

### Hardware requirements

To run this bot we recommend you a cloud instance with a minimum of:

- 2GB RAM
- 1GB disk space
- 2vCPU

### Software requirements

- Python 3.6.x
- pip (pip3)
- git
- TA-Lib
- virtualenv (Recommended)
- Docker (Recommended)

## Support

### Help / Slack
For any questions not covered by the documentation or for further information about the bot, we encourage you to join our passionate  Slack community.

Click [here](https://join.slack.com/t/highfrequencybot/shared_invite/enQtNjU5ODcwNjI1MDU3LTU1MTgxMjkzNmYxNWE1MDEzYzQ3YmU4N2MwZjUyNjJjODRkMDVkNjg4YTAyZGYzYzlhOTZiMTE4ZjQ4YzM0OGE) to join the Freqtrade Slack channel.

## Ready to try?

Begin by reading our installation guide [here](installation).
