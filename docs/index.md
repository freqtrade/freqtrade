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
Freqtrade is a crypto-currency algorithmic trading software develop in python (3.6+) supported on windows, macOs and Linux.
!!! Danger "DISCLAIMER"
    This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

    Always start by running a trading bot in Dry-run and do not engage money before you understand how it works and what profit/loss you should expect.

    We strongly recommend you to have basic coding skills and Python knowledge. Do not hesitate to read the source code and understand the mechanisms of this bot, algorithms and techniques implemented in it.

## Features

 1. Download markets datas : download historical datas of the exchange and the markets your may want to trade with. 
 2. Select markets : create your list or use an automatic one based on top traded volume (not available during backtesting). You can blacklist markets you don't want to trade.
 3. Backtest : Test your strategy on past datas (based on [ohcl](https://en.wikipedia.org/wiki/Open-high-low-close_chart) candles).
 4. Optimize : Find the best parameters for your strategy (use machine leaning)
 5. Run : Run the bot on exchange without playing money (dry-run) or with money (live).
 6. Run using edge (optionnal module) : the concept is to find the best historical [trade expectancy](https://www.freqtrade.io/en/latest/edge/#expectancy) by markets based on variation of the stop-loss and then allow/reject markets to trade (the sizing of the trade is based on a risk of a percentage of your capital)
 7. Control/Monitor/Analyse : use Telegram or a REST API (start/stop the bot, profit/loss, daily summary, current open trades results...). Futher analysis can be done as trades are saved (SQLite database)

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

Help / Slack
For any questions not covered by the documentation or for further information about the bot, we encourage you to join our Slack channel.

Click [here](https://join.slack.com/t/highfrequencybot/shared_invite/enQtNjU5ODcwNjI1MDU3LTU1MTgxMjkzNmYxNWE1MDEzYzQ3YmU4N2MwZjUyNjJjODRkMDVkNjg4YTAyZGYzYzlhOTZiMTE4ZjQ4YzM0OGE) to join Slack channel.

## Ready to try?

Begin by reading our installation guide [here](installation).
