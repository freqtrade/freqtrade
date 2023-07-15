# REST API & FreqUI

## FreqUI

Freqtrade provides a builtin webserver, which can serve [FreqUI](https://github.com/freqtrade/frequi), the freqtrade UI.

By default, the UI is not included in the installation (except for docker images), and must be installed explicitly with `freqtrade install-ui`.
This same command can also be used to update freqUI, should there be a new release.

Once the bot is started in trade / dry-run mode (with `freqtrade trade`) - the UI will be available under the configured port below (usually `http://127.0.0.1:8080`).

!!! Note "developers"
    Developers should not use this method, but instead use the method described in the [freqUI repository](https://github.com/freqtrade/frequi) to get the source-code of freqUI.

## Configuration

Enable the rest API by adding the api_server section to your configuration and setting `api_server.enabled` to `true`.

Sample configuration:

``` json
    "api_server": {
        "enabled": true,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "verbosity": "error",
        "enable_openapi": false,
        "jwt_secret_key": "somethingrandom",
        "CORS_origins": [],
        "username": "Freqtrader",
        "password": "SuperSecret1!",
        "ws_token": "sercet_Ws_t0ken"
    },
```

!!! Danger "Security warning"
    By default, the configuration listens on localhost only (so it's not reachable from other systems). We strongly recommend to not expose this API to the internet and choose a strong, unique password, since others will potentially be able to control your bot.

??? Note "API/UI Access on a remote servers"
    If you're running on a VPS, you should consider using either a ssh tunnel, or setup a VPN (openVPN, wireguard) to connect to your bot.
    This will ensure that freqUI is not directly exposed to the internet, which is not recommended for security reasons (freqUI does not support https out of the box).
    Setup of these tools is not part of this tutorial, however many good tutorials can be found on the internet.

You can then access the API by going to `http://127.0.0.1:8080/api/v1/ping` in a browser to check if the API is running correctly.
This should return the response:

``` output
{"status":"pong"}
```

All other endpoints return sensitive info and require authentication and are therefore not available through a web browser.

### Security

To generate a secure password, best use a password manager, or use the below code.

``` python
import secrets
secrets.token_hex()
```

!!! Hint "JWT token"
    Use the same method to also generate a JWT secret key (`jwt_secret_key`).

!!! Danger "Password selection"
    Please make sure to select a very strong, unique password to protect your bot from unauthorized access.
    Also change `jwt_secret_key` to something random (no need to remember this, but it'll be used to encrypt your session, so it better be something unique!).

### Configuration with docker

If you run your bot using docker, you'll need to have the bot listen to incoming connections. The security is then handled by docker.

``` json
    "api_server": {
        "enabled": true,
        "listen_ip_address": "0.0.0.0",
        "listen_port": 8080,
        "username": "Freqtrader",
        "password": "SuperSecret1!",
        //...
    },
```

Make sure that the following 2 lines are available in your docker-compose file:

```yml
    ports:
      - "127.0.0.1:8080:8080"
```

!!! Danger "Security warning"
    By using `8080:8080` in the docker port mapping, the API will be available to everyone connecting to the server under the correct port, so others may be able to control your bot.

## Rest API

### Consuming the API

You can consume the API by using the script `scripts/rest_client.py`.
The client script only requires the `requests` module, so Freqtrade does not need to be installed on the system.

``` bash
python3 scripts/rest_client.py <command> [optional parameters]
```

By default, the script assumes `127.0.0.1` (localhost) and port `8080` to be used, however you can specify a configuration file to override this behaviour.

#### Minimalistic client config

``` json
{
    "api_server": {
        "enabled": true,
        "listen_ip_address": "0.0.0.0",
        "listen_port": 8080,
        "username": "Freqtrader",
        "password": "SuperSecret1!",
        //...
    }
}
```

``` bash
python3 scripts/rest_client.py --config rest_config.json <command> [optional parameters]
```

### Available endpoints

|  Command | Description |
|----------|-------------|
| `ping` | Simple command testing the API Readiness - requires no authentication.
| `start` | Starts the trader.
| `stop` | Stops the trader.
| `stopbuy` | Stops the trader from opening new trades. Gracefully closes open trades according to their rules.
| `reload_config` | Reloads the configuration file.
| `trades` | List last trades. Limited to 500 trades per call.
| `trade/<tradeid>` | Get specific trade.
| `trade/<tradeid>` | DELETE - Remove trade from the database. Tries to close open orders. Requires manual handling of this trade on the exchange.
| `trade/<tradeid>/open-order` | DELETE - Cancel open order for this trade.
| `trade/<tradeid>/reload` | GET - Reload a trade from the Exchange. Only works in live, and can potentially help recover a trade that was manually sold on the exchange.
| `show_config` | Shows part of the current configuration with relevant settings to operation.
| `logs` | Shows last log messages.
| `status` | Lists all open trades.
| `count` | Displays number of trades used and available.
| `locks` | Displays currently locked pairs.
| `delete_lock <lock_id>` | Deletes (disables) the lock by id.
| `profit` | Display a summary of your profit/loss from close trades and some stats about your performance.
| `forceexit <trade_id>` | Instantly exits the given trade  (Ignoring `minimum_roi`).
| `forceexit all` | Instantly exits all open trades (Ignoring `minimum_roi`).
| `forceenter <pair> [rate]` | Instantly enters the given pair. Rate is optional. (`force_entry_enable` must be set to True)
| `forceenter <pair> <side> [rate]` | Instantly longs or shorts the given pair. Rate is optional. (`force_entry_enable` must be set to True)
| `performance` | Show performance of each finished trade grouped by pair.
| `balance` | Show account balance per currency.
| `daily <n>` | Shows profit or loss per day, over the last n days (n defaults to 7).
| `stats` | Display a summary of profit / loss reasons as well as average holding times.
| `whitelist` | Show the current whitelist.
| `blacklist [pair]` | Show the current blacklist, or adds a pair to the blacklist.
| `edge` | Show validated pairs by Edge if it is enabled.
| `pair_candles` | Returns dataframe for a pair / timeframe combination while the bot is running. **Alpha**
| `pair_history` | Returns an analyzed dataframe for a given timerange, analyzed by a given strategy. **Alpha**
| `plot_config` | Get plot config from the strategy (or nothing if not configured). **Alpha**
| `strategies` | List strategies in strategy directory. **Alpha**
| `strategy <strategy>` | Get specific Strategy content. **Alpha**
| `available_pairs` | List available backtest data. **Alpha**
| `version` | Show version.
| `sysinfo` | Show information about the system load.
| `health` | Show bot health (last bot loop).

!!! Warning "Alpha status"
    Endpoints labeled with *Alpha status* above may change at any time without notice.

Possible commands can be listed from the rest-client script using the `help` command.

``` bash
python3 scripts/rest_client.py help
```

``` output
Possible commands:

available_pairs
	Return available pair (backtest data) based on timeframe / stake_currency selection

        :param timeframe: Only pairs with this timeframe available.
        :param stake_currency: Only pairs that include this timeframe

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

edge
	Return information about edge.

forcebuy
	Buy an asset.

        :param pair: Pair to buy (ETH/BTC)
        :param price: Optional - price to buy

forceenter
	Force entering a trade

        :param pair: Pair to buy (ETH/BTC)
        :param side: 'long' or 'short'
        :param price: Optional - price to buy

forceexit
	Force-exit a trade.

        :param tradeid: Id of the trade (can be received via status command)
        :param ordertype: Order type to use (must be market or limit)
        :param amount: Amount to sell. Full sell if not given

health
	Provides a quick health check of the running bot.

locks
	Return current locks

logs
	Show latest logs.

        :param limit: Limits log messages to the last <limit> logs. No limit to get the entire log.

pair_candles
	Return live dataframe for <pair><timeframe>.

        :param pair: Pair to get data for
        :param timeframe: Only pairs with this timeframe available.
        :param limit: Limit result to the last n candles.

pair_history
	Return historic, analyzed dataframe

        :param pair: Pair to get data for
        :param timeframe: Only pairs with this timeframe available.
        :param strategy: Strategy to analyze and get values for
        :param timerange: Timerange to get data for (same format than --timerange endpoints)

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
	Return trades history, sorted by id

        :param limit: Limits trades to the X last trades. Max 500 trades.
        :param offset: Offset by this amount of trades.

version
	Return the version of the bot.

whitelist
	Show the current whitelist.


```

### Message WebSocket

The API Server includes a websocket endpoint for subscribing to RPC messages from the freqtrade Bot.
This can be used to consume real-time data from your bot, such as entry/exit fill messages, whitelist changes, populated indicators for pairs, and more.

This is also used to setup [Producer/Consumer mode](producer-consumer.md) in Freqtrade.

Assuming your rest API is set to `127.0.0.1` on port `8080`, the endpoint is available at `http://localhost:8080/api/v1/message/ws`.

To access the websocket endpoint, the `ws_token` is required as a query parameter in the endpoint URL.

To generate a safe `ws_token` you can run the following code:

``` python
>>> import secrets
>>> secrets.token_urlsafe(25)
'hZ-y58LXyX_HZ8O1cJzVyN6ePWrLpNQv4Q'
```

You would then add that token under `ws_token` in your `api_server` config. Like so:

``` json
"api_server": {
    "enabled": true,
    "listen_ip_address": "127.0.0.1",
    "listen_port": 8080,
    "verbosity": "error",
    "enable_openapi": false,
    "jwt_secret_key": "somethingrandom",
    "CORS_origins": [],
    "username": "Freqtrader",
    "password": "SuperSecret1!",
    "ws_token": "hZ-y58LXyX_HZ8O1cJzVyN6ePWrLpNQv4Q" // <-----
},
```

You can now connect to the endpoint at `http://localhost:8080/api/v1/message/ws?token=hZ-y58LXyX_HZ8O1cJzVyN6ePWrLpNQv4Q`.

!!! Danger "Reuse of example tokens"
    Please do not use the above example token. To make sure you are secure, generate a completely new token.

#### Using the WebSocket

Once connected to the WebSocket, the bot will broadcast RPC messages to anyone who is subscribed to them. To subscribe to a list of messages, you must send a JSON request through the WebSocket like the one below. The `data` key must be a list of message type strings.

``` json
{
  "type": "subscribe",
  "data": ["whitelist", "analyzed_df"] // A list of string message types
}
```

For a list of message types, please refer to the RPCMessageType enum in `freqtrade/enums/rpcmessagetype.py`

Now anytime those types of RPC messages are sent in the bot, you will receive them through the WebSocket as long as the connection is active. They typically take the same form as the request:

``` json
{
  "type": "analyzed_df",
  "data": {
      "key": ["NEO/BTC", "5m", "spot"],
      "df": {}, // The dataframe
      "la": "2022-09-08 22:14:41.457786+00:00"
  }
}
```

#### Reverse Proxy setup

When using [Nginx](https://nginx.org/en/docs/), the following configuration is required for WebSockets to work (Note this configuration is incomplete, it's missing some information and can not be used as is):

Please make sure to replace `<freqtrade_listen_ip>` (and the subsequent port) with the IP and Port matching your configuration/setup.

```
http {
    map $http_upgrade $connection_upgrade {
        default upgrade;
        '' close;
    }

    #...

    server {
        #...

        location / {
            proxy_http_version 1.1;
            proxy_pass http://<freqtrade_listen_ip>:8080;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection $connection_upgrade;
            proxy_set_header Host $host;
        }
    }
}
```

To properly configure your reverse proxy (securely), please consult it's documentation for proxying websockets.

- **Traefik**: Traefik supports websockets out of the box, see the [documentation](https://doc.traefik.io/traefik/)
- **Caddy**: Caddy v2 supports websockets out of the box, see the [documentation](https://caddyserver.com/docs/v2-upgrade#proxy)

!!! Tip "SSL certificates"
    You can use tools like certbot to setup ssl certificates to access your bot's UI through encrypted connection by using any fo the above reverse proxies.
    While this will protect your data in transit, we do not recommend to run the freqtrade API outside of your private network (VPN, SSH tunnel).

### OpenAPI interface

To enable the builtin openAPI interface (Swagger UI), specify `"enable_openapi": true` in the api_server configuration.
This will enable the Swagger UI at the `/docs` endpoint. By default, that's running at http://localhost:8080/docs - but it'll depend on your settings.

### Advanced API usage using JWT tokens

!!! Note
    The below should be done in an application (a Freqtrade REST API client, which fetches info via API), and is not intended to be used on a regular basis.

Freqtrade's REST API also offers JWT (JSON Web Tokens).
You can login using the following command, and subsequently use the resulting access_token.

``` bash
> curl -X POST --user Freqtrader http://localhost:8080/api/v1/token/login
{"access_token":"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE1ODkxMTk2ODEsIm5iZiI6MTU4OTExOTY4MSwianRpIjoiMmEwYmY0NWUtMjhmOS00YTUzLTlmNzItMmM5ZWVlYThkNzc2IiwiZXhwIjoxNTg5MTIwNTgxLCJpZGVudGl0eSI6eyJ1IjoiRnJlcXRyYWRlciJ9LCJmcmVzaCI6ZmFsc2UsInR5cGUiOiJhY2Nlc3MifQ.qt6MAXYIa-l556OM7arBvYJ0SDI9J8bIk3_glDujF5g","refresh_token":"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE1ODkxMTk2ODEsIm5iZiI6MTU4OTExOTY4MSwianRpIjoiZWQ1ZWI3YjAtYjMwMy00YzAyLTg2N2MtNWViMjIxNWQ2YTMxIiwiZXhwIjoxNTkxNzExNjgxLCJpZGVudGl0eSI6eyJ1IjoiRnJlcXRyYWRlciJ9LCJ0eXBlIjoicmVmcmVzaCJ9.d1AT_jYICyTAjD0fiQAr52rkRqtxCjUGEMwlNuuzgNQ"}

> access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE1ODkxMTk2ODEsIm5iZiI6MTU4OTExOTY4MSwianRpIjoiMmEwYmY0NWUtMjhmOS00YTUzLTlmNzItMmM5ZWVlYThkNzc2IiwiZXhwIjoxNTg5MTIwNTgxLCJpZGVudGl0eSI6eyJ1IjoiRnJlcXRyYWRlciJ9LCJmcmVzaCI6ZmFsc2UsInR5cGUiOiJhY2Nlc3MifQ.qt6MAXYIa-l556OM7arBvYJ0SDI9J8bIk3_glDujF5g"
# Use access_token for authentication
> curl -X GET --header "Authorization: Bearer ${access_token}" http://localhost:8080/api/v1/count

```

Since the access token has a short timeout (15 min) - the `token/refresh` request should be used periodically to get a fresh access token:

``` bash
> curl -X POST --header "Authorization: Bearer ${refresh_token}"http://localhost:8080/api/v1/token/refresh
{"access_token":"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE1ODkxMTk5NzQsIm5iZiI6MTU4OTExOTk3NCwianRpIjoiMDBjNTlhMWUtMjBmYS00ZTk0LTliZjAtNWQwNTg2MTdiZDIyIiwiZXhwIjoxNTg5MTIwODc0LCJpZGVudGl0eSI6eyJ1IjoiRnJlcXRyYWRlciJ9LCJmcmVzaCI6ZmFsc2UsInR5cGUiOiJhY2Nlc3MifQ.1seHlII3WprjjclY6DpRhen0rqdF4j6jbvxIhUFaSbs"}
```

### CORS

This whole section is only necessary in cross-origin cases (where you multiple bot API's running on `localhost:8081`, `localhost:8082`, ...), and want to combine them into one FreqUI instance.

??? info "Technical explanation"
    All web-based front-ends are subject to [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) - Cross-Origin Resource Sharing.
    Since most of the requests to the Freqtrade API must be authenticated, a proper CORS policy is key to avoid security problems.
    Also, the standard disallows `*` CORS policies for requests with credentials, so this setting must be set appropriately.

Users can allow access from different origin URL's to the bot API via the `CORS_origins` configuration setting.
It consists of a list of allowed URL's that are allowed to consume resources from the bot's API.

Assuming your application is deployed as `https://frequi.freqtrade.io/home/` - this would mean that the following configuration becomes necessary:

```jsonc
{
    //...
    "jwt_secret_key": "somethingrandom",
    "CORS_origins": ["https://frequi.freqtrade.io"],
    //...
}
```

In the following (pretty common) case, FreqUI is accessible on `http://localhost:8080/trade` (this is what you see in your navbar when navigating to freqUI).
![freqUI url](assets/frequi_url.png)

The correct configuration for this case is `http://localhost:8080` - the main part of the URL including the port.

```jsonc
{
    //...
    "jwt_secret_key": "somethingrandom",
    "CORS_origins": ["http://localhost:8080"],
    //...
}
```

!!! Note
    We strongly recommend to also set `jwt_secret_key` to something random and known only to yourself to avoid unauthorized access to your bot.
