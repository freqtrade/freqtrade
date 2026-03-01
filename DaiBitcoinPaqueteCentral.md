

Your account has been flagged.Because of that, your profile is hidden from the public. If you believe this is a mistake, contact support to have your account status reviewed.

DAIDAOXING/freqtradePublic

forked from freqtrade/freqtrade

Free, open source crypto trading bot

www.freqtrade.io

License

 GPL-3.0 license

 0 stars  4.4k forks

 Star

 Watch 

CodePull requestsActionsProjectsWikiSecurityInsightsSettings

DAIDAOXING/runing trading
 WWW.DAIBITCOIN… 

This branch is up to date with freqtrade/freqtrade:develop.

 Contribute  Sync fork 

Latest commit

xmatthias

…

1 hours ago

Git stats

Files

View code

README.md

    

Freqtrade is a free and open source crypto trading bot written in Python. It is designed to support all major exchanges and be controlled via Telegram or webUI. It contains backtesting, plotting and money management tools as well as strategy optimization by machine learning.

Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by running a trading bot in Dry-run and do not engage money before you understand how it works and what profit/loss you should expect.

We strongly recommend you to have coding and Python knowledge. Do not hesitate to read the source code and understand the mechanism of this bot.

Supported Exchange marketplaces

Please read the exchange specific notes to learn about eventual, special configurations needed for each exchange.

 Binance Bittrex Gate.io Huobi Kraken OKX (Former OKEX) potentially many others. (We cannot guarantee they will work)Supported Futures Exchanges (experimental) Binance Gate.io OKX Bybit

Please make sure to read the exchange specific notes, as well as the trading with leverage documentation before diving in.

Community tested

Exchanges confirmed working by the community:

 Bitvavo KucoinDocumentation

We invite you to read the bot documentation to ensure you understand how the bot is working.

Please find the complete documentation on the freqtrade website.

Features Based on Python 3.8+: For botting on any operating system - Windows, macOS and Linux. Persistence: Persistence is achieved through sqlite. Dry-run: Run the bot without paying money. Backtesting: Run a simulation of your buy/sell strategy. Strategy Optimization by machine learning: Use machine learning to optimize your buy/sell strategy parameters with real exchange data. Adaptive prediction modeling: Build a smart strategy with FreqAI that self-trains to the market via adaptive machine learning methods. Learn more Edge position sizing Calculate your win rate, risk reward ratio, the best stoploss and adjust your position size before taking a position for each specific market. Learn more. Whitelist crypto-currencies: Select which crypto-currency you want to trade or use dynamic whitelists. Blacklist crypto-currencies: Select which crypto-currency you want to avoid. Builtin WebUI: Builtin web UI to manage your bot. Manageable via Telegram: Manage the bot with Telegram. Display profit/loss in fiat: Display your profit/loss in fiat currency. Performance status report: Provide a performance status of your current trades.Quick start

Please refer to the Docker Quickstart documentation on how to get started quickly.

For further (native) installation methods, please refer to the Installation documentation page.

Basic UsageBot commands

usage: freqtrade [-h] [-V] {trade,create-userdir,new-config,new-strategy,download-data,convert-data,convert-trade-data,list-data,backtesting,edge,hyperopt,hyperopt-list,hyperopt-show,list-exchanges,list-hyperopts,list-markets,list-pairs,list-strategies,list-timeframes,show-trades,test-pairlist,install-ui,plot-dataframe,plot-profit,webserver} ... Free, open source crypto trading bot positional arguments: {trade,create-userdir,new-config,new-strategy,download-data,convert-data,convert-trade-data,list-data,backtesting,edge,hyperopt,hyperopt-list,hyperopt-show,list-exchanges,list-hyperopts,list-markets,list-pairs,list-strategies,list-timeframes,show-trades,test-pairlist,install-ui,plot-dataframe,plot-profit,webserver} trade Trade module. create-userdir Create user-data directory. new-config Create new config new-strategy Create new strategy download-data Download backtesting data. convert-data Convert candle (OHLCV) data from one format to another. convert-trade-data Convert trade data from one format to another. list-data List downloaded data. backtesting Backtesting module. edge Edge module. hyperopt Hyperopt module. hyperopt-list List Hyperopt results hyperopt-show Show details of Hyperopt results list-exchanges Print available exchanges. list-hyperopts Print available hyperopt classes. list-markets Print markets on exchange. list-pairs Print pairs on exchange. list-strategies Print available strategies. list-timeframes Print available timeframes for the exchange. show-trades Show trades. test-pairlist Test your pairlist configuration. install-ui Install FreqUI plot-dataframe Plot candles with indicators. plot-profit Generate plot showing profits. webserver Webserver module. optional arguments: -h, --help show this help message and exit -V, --version show program's version number and exit 

Telegram RPC commands

Telegram is not mandatory. However, this is a great way to control your bot. More details and the full command list on the documentation

/start: Starts the trader./stop: Stops the trader./stopentry: Stop entering new trades./status <trade_id>|[table]: Lists all or specific open trades./profit [<n>]: Lists cumulative profit from all finished trades, over the last n days./forceexit <trade_id>|all: Instantly exits the given trade (Ignoring minimum_roi)./fx <trade_id>|all: Alias to /forceexit/performance: Show performance of each finished trade grouped by pair/balance: Show account balance per currency./daily <n>: Shows profit or loss per day, over the last n days./help: Show help message./version: Show version.Development branches

The project is currently setup in two main branches:

develop - This branch has often new features, but might also contain breaking changes. We try hard to keep this branch as stable as possible.stable - This branch contains the latest stable release. This branch is generally well tested.feat/* - These are feature branches, which are being worked on heavily. Please don't use these unless you want to test a specific feature.SupportHelp / Discord

For any questions not covered by the documentation or for further information about the bot, or to simply engage with like-minded individuals, we encourage you to join the Freqtrade discord server.

Bugs / Issues

If you discover a bug in the bot, please search the issue tracker first. If it hasn't been reported, please create a new issue and ensure you follow the template guide so that the team can assist you as quickly as possible.

For every issue created, kindly follow up and mark satisfaction or reminder to close issue when equilibrium ground is reached.

--Maintain github's community policy--

Feature Requests

Have you a great idea to improve the bot you want to share? Please, first search if this feature was not already discussed. If it hasn't been requested, please create a new request and ensure you follow the template guide so that it does not get lost in the bug reports.

Pull Requests www.coingecko.com
Ingreao.total.trasfer.auto.por.coingecko
Feel like the bot is missing a feature? We welcome your pull requests!
Unico propietario CEO DAOXINGDAI
UNICO TELEPHON CONECTING FOR CUENTA PERSONAL  ACCEDER DIRECTOR CEO UNICO CORREO ELECTRONICO daoxingdaiunico@gmail.com la empresa www.daibitcoin.org sin trabajador unico personal para manager and actually ingresos 100% en efectivo gas table por pripietario on trade 
