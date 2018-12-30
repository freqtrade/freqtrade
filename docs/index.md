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
Freqtrade is a cryptocurrency trading bot written in Python.

!!! Danger "DISCLAIMER"
    This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

    Always start by running a trading bot in Dry-run and do not engage money before you understand how it works and what profit/loss you should expect.

    We strongly recommend you to have coding and Python knowledge. Do not hesitate to read the source code and understand the mechanism of this bot.


## Features
 - Based on Python 3.6+: For botting on any operating system - Windows, macOS and Linux
 - Persistence: Persistence is achieved through sqlite
 - Dry-run: Run the bot without playing money.
 - Backtesting: Run a simulation of your buy/sell strategy.
 - Strategy Optimization by machine learning: Use machine learning to optimize your buy/sell strategy parameters with real exchange data.
 - Edge position sizing Calculate your win rate, risk reward ratio, the best stoploss and adjust your position size before taking a position for each specific market. Learn more
 - Whitelist crypto-currencies: Select which crypto-currency you want to trade or use dynamic whitelists.
 - Blacklist crypto-currencies: Select which crypto-currency you want to avoid.
 - Manageable via Telegram: Manage the bot with Telegram
 - Display profit/loss in fiat: Display your profit/loss in 33 fiat.
 - Daily summary of profit/loss: Provide a daily summary of your profit/loss.
 - Performance status report: Provide a performance status of your current trades.


## Requirements
### Uptodate clock
The clock must be accurate, syncronized to a NTP server very frequently to avoid problems with communication to the exchanges.

### Hardware requirements
To run this bot we recommend you a cloud instance with a minimum of:

- 2GB RAM
- 1GB disk space
- 2vCPU

### Software requirements
- Python 3.6.x
- pip
- git
- TA-Lib
- virtualenv (Recommended)
- Docker (Recommended)


## Support
Help / Slack
For any questions not covered by the documentation or for further information about the bot, we encourage you to join our slack channel.

Click [here](https://join.slack.com/t/highfrequencybot/shared_invite/enQtMjQ5NTM0OTYzMzY3LWMxYzE3M2MxNDdjMGM3ZTYwNzFjMGIwZGRjNTc3ZGU3MGE3NzdmZGMwNmU3NDM5ZTNmM2Y3NjRiNzk4NmM4OGE) to join Slack channel.

## Ready to try?
Begin buy reading our installation guide [here](pre-requisite).