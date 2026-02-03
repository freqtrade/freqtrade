``` output
Possible commands:

available_pairs
	Return available pair (backtest data) based on timeframe / stake_currency selection

:param timeframe: Only pairs with this timeframe available.
:param stake_currency: Only pairs that include this stake currency.

balance
	Get the account balance.

blacklist
	Show the current blacklist.

:param add: List of coins to add (example: "BNB/BTC")

cancel_open_order
	Cancel open order for trade.

:param trade_id: Cancels open orders for this trade.

count
	Return the amount of open trades.

daily
	Return the profits for each day, and amount of trades.

delete_lock
	Delete (disable) lock from the database.

:param lock_id: ID for the lock to delete

delete_trade
	Delete trade from the database.
Tries to close open orders. Requires manual handling of this asset on the exchange.

:param trade_id: Deletes the trade with this ID from the database.

entries
	Returns List of dicts containing all Trades, based on buy tag performance
Can either be average for all pairs or a specific pair provided

exits
	Returns List of dicts containing all Trades, based on exit reason performance
Can either be average for all pairs or a specific pair provided

forcebuy
	Buy an asset.

:param pair: Pair to buy (ETH/BTC)
:param price: Optional - price to buy

forceenter
	Force entering a trade

:param pair: Pair to buy (ETH/BTC)
:param side: 'long' or 'short'
:param price: Optional - price to buy
:param order_type: Optional keyword argument - 'limit' or 'market'
:param stake_amount: Optional keyword argument - stake amount (as float)
:param leverage: Optional keyword argument - leverage (as float)
:param enter_tag: Optional keyword argument - entry tag (as string, default: 'force_enter')

forceexit
	Force-exit a trade.

:param tradeid: Id of the trade (can be received via status command)
:param ordertype: Order type to use (must be market or limit)
:param amount: Amount to sell. Full sell if not given

health
	Provides a quick health check of the running bot.

list_custom_data
	List custom-data of the running bot for a specific trade.

:param trade_id: ID of the trade
:param key: str, optional - Key of the custom-data

list_open_trades_custom_data
	List open trades custom-data of the running bot.

:param key: str, optional - Key of the custom-data
:param limit: limit of trades
:param offset: trades offset for pagination

lock_add
	Lock pair

:param pair: Pair to lock
:param until: Lock until this date (format "2024-03-30 16:00:00Z")
:param side: Side to lock (long, short, *)
:param reason: Reason for the lock

locks
	Return current locks

logs
	Show latest logs.

:param limit: Limits log messages to the last <limit> logs. No limit to get the entire log.

mix_tags
	Returns List of dicts containing all Trades, based on entry_tag + exit_reason performance
Can either be average for all pairs or a specific pair provided

monthly
	Return the profits for each month, and amount of trades.

pair_candles
	Return live dataframe for <pair><timeframe>.

:param pair: Pair to get data for
:param timeframe: Only pairs with this timeframe available.
:param limit: Limit result to the last n candles.
:param columns: List of dataframe columns to return. Empty list will return OHLCV.

pair_history
	Return historic, analyzed dataframe

:param pair: Pair to get data for
:param timeframe: Only pairs with this timeframe available.
:param strategy: Strategy to analyze and get values for
:param freqaimodel: FreqAI model to use for analysis
:param timerange: Timerange to get data for (same format than --timerange endpoints)

pairlists_available
	Lists available pairlist providers

performance
	Return the performance of the different coins.

ping
	simple ping

plot_config
	Return plot configuration if the strategy defines one.

profit
	Return the profit summary.

reload_config
	Reload configuration.

show_config
	Returns part of the configuration, relevant for trading operations.

start
	Start the bot if it's in the stopped state.

stats
	Return the stats report (durations, sell-reasons).

status
	Get the status of open trades.

stop
	Stop the bot. Use `start` to restart.

stopbuy
	Stop buying (but handle sells gracefully). Use `reload_config` to reset.

strategies
	Lists available strategies

strategy
	Get strategy details

:param strategy: Strategy class name

sysinfo
	Provides system information (CPU, RAM usage)

trade
	Return specific trade

:param trade_id: Specify which trade to get.

trades
	Return trades history, sorted by id (or by latest timestamp if order_by_id=False)

:param limit: Limits trades to the X last trades. Max 500 trades.
:param offset: Offset by this amount of trades.
:param order_by_id: Sort trades by id (default: True). If False, sorts by latest timestamp.

version
	Return the version of the bot.

weekly
	Return the profits for each week, and amount of trades.

whitelist
	Show the current whitelist.


```
