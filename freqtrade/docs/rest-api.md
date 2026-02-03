# REST API

## FreqUI

FreqUI now has it's own dedicated [documentation section](freq-ui.md) - please refer to that section for all information regarding the FreqUI.

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
    By using `"8080:8080"` (or `"0.0.0.0:8080:8080"`) in the docker port mapping, the API will be available to everyone connecting to the server under the correct port, so others may be able to control your bot.
    This **may** be safe if you're running the bot in a secure environment (like your home network), but it's not recommended to expose the API to the internet.

## Rest API

### Consuming the API

We advise consuming the API by using the supported `freqtrade-client` package (also available as `scripts/rest_client.py`).

This command can be installed independent of any running freqtrade bot by using `pip install freqtrade-client`.

This module is designed to be lightweight, and only depends on the `requests` and `python-rapidjson` modules, skipping all heavy dependencies freqtrade otherwise needs.

``` bash
freqtrade-client <command> [optional parameters]
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
freqtrade-client --config rest_config.json <command> [optional parameters]
```

Commands with many arguments may require keyword arguments (for clarity) - which can be provided as follows:

``` bash
freqtrade-client --config rest_config.json forceenter BTC/USDT long enter_tag=GutFeeling
```

This method will work for all arguments - check the "show" command for a list of available parameters.

??? Note "Programmatic use"
    The `freqtrade-client` package (installable independent of freqtrade) can be used in your own scripts to interact with the freqtrade API.
    to do so, please use the following:

    ``` python
    from freqtrade_client import FtRestClient
    

    client = FtRestClient(server_url, username, password)

    # Get the status of the bot
    ping = client.ping()
    print(ping)

    # Add pairs to blacklist
    client.blacklist("BTC/USDT", "ETH/USDT")
    # Add pairs to blacklist by supplying a list
    client.blacklist(*listPairs)
    # ... 
    ```

    For a full list of available commands, please refer to the list below.

#### Freqtrade client- available commands

Possible commands can be listed from the rest-client script using the `help` command.

``` bash
freqtrade-client help
```

--8<-- "commands/freqtrade-client.md"


### Available endpoints

If you wish to call the REST API manually via another route, e.g. directly via `curl`, the table below shows the relevant URL endpoints and parameters.
All endpoints in the below table need to be prefixed with the base URL of the API, e.g. `http://127.0.0.1:8080/api/v1/` - so the command becomes `http://127.0.0.1:8080/api/v1/<command>`.

|  Endpoint | Method | Description / Parameters |
|-----------|--------|--------------------------|
| `/ping` | GET | Simple command testing the API Readiness - requires no authentication.
| `/start` | POST | Starts the trader.
| `/pause` | POST | Pause the trader. Gracefully handle open trades according to their rules. Do not enter new positions.
| `/stop` | POST | Stops the trader.
| `/stopbuy` | POST | Stops the trader from opening new trades. Gracefully closes open trades according to their rules.
| `/reload_config` | POST | Reloads the configuration file.
| `/trades` | GET | List last trades. Limited to 500 trades per call.
| `/trade/<tradeid>` | GET | Get specific trade.<br/>*Params:*<br/>- `tradeid` (`int`)
| `/trades/<tradeid>` | DELETE | Remove trade from the database. Tries to close open orders. Requires manual handling of this trade on the exchange.<br/>*Params:*<br/>- `tradeid` (`int`)
| `/trades/<tradeid>/open-order` | DELETE | Cancel open order for this trade.<br/>*Params:*<br/>- `tradeid` (`int`)
| `/trades/<tradeid>/reload` | POST | Reload a trade from the Exchange. Only works in live, and can potentially help recover a trade that was manually sold on the exchange.<br/>*Params:*<br/>- `tradeid` (`int`)
| `/show_config` | GET | Shows part of the current configuration with relevant settings to operation.
| `/logs` | GET | Shows last log messages.
| `/status` | GET | Lists all open trades.
| `/count` | GET | Displays number of trades used and available.
| `/entries` | GET | Shows profit statistics for each enter tags for given pair (or all pairs if pair isn't given). Pair is optional.<br/>*Params:*<br/>- `pair` (`str`)
| `/exits` | GET | Shows profit statistics for each exit reasons for given pair (or all pairs if pair isn't given). Pair is optional.<br/>*Params:*<br/>- `pair` (`str`)
| `/mix_tags` | GET | Shows profit statistics for each combinations of enter tag + exit reasons for given pair (or all pairs if pair isn't given). Pair is optional.<br/>*Params:*<br/>- `pair` (`str`)
| `/locks` | GET | Displays currently locked pairs.
| `/locks` | POST | Locks a pair until "until". (Until will be rounded up to the nearest timeframe). Side is optional and is either `long` or `short` (default is `long`). Reason is optional.<br/>*Params:*<br/>- `<pair>` (`str`)<br/>- `<until>` (`datetime`)<br/>- `[side]` (`str`)<br/>- `[reason]` (`str`)
| `/locks/<lockid>` | DELETE | Deletes (disables) the lock by id.<br/>*Params:*<br/>- `lockid` (`int`)
| `/profit` | GET | Display a summary of your profit/loss from close trades and some stats about your performance.
| `/forceexit` | POST | Instantly exits the given trade (ignoring `minimum_roi`), using the given order type ("market" or "limit", uses your config setting if not specified), and the chosen amount (full sell if not specified). If `all` is supplied as the `tradeid`, then all currently open trades will be forced to exit.<br/>*Params:*<br/>- `<tradeid>` (`int` or `str`)<br/>- `<ordertype>` (`str`)<br/>- `[amount]` (`float`)
| `/forceenter` | POST | Instantly enters the given pair. Side is optional and is either `long` or `short` (default is `long`). Price, stake amount, entry tag and leverage are optional. Order type is optional and is either `market` or `long` (default using the value set in config). (`force_entry_enable` must be set to True)<br/>*Params:*<br/>- `<pair>` (`str`)<br/>- `<side>` (`str`)<br/>- `[price]` (`float`)<br/>- `[ordertype]` (`str`)<br/>- `[stakeamount]` (`float`)<br/>- `[entry_tag]` (`str`)<br/>- `[leverage]` (`float`)
| `/performance` | GET | Show performance of each finished trade grouped by pair.
| `/balance` | GET | Show account balance per currency.
| `/daily` | GET | Shows profit or loss per day, over the last n days (n defaults to 7).<br/>*Params:*<br/>- `timescale` (`int`)
| `/weekly` | GET | Shows profit or loss per week, over the last n days (n defaults to 4).<br/>*Params:*<br/>- `timescale` (`int`)
| `/monthly` | GET | Shows profit or loss per month, over the last n days (n defaults to 3).<br/>*Params:*<br/>- `timescale` (`int`)
| `/stats` | GET | Display a summary of profit / loss reasons as well as average holding times.
| `/whitelist` | GET | Show the current whitelist.
| `/blacklist` | GET | Show the current blacklist.
| `/blacklist` | POST | Adds the specified pair to the blacklist.<br/>*Params:*<br/>- `blacklist` (`str`)
| `/blacklist` | DELETE | Deletes the specified list of pairs from the blacklist.<br/>*Params:*<br/>- `[pair,pair]` (`list[str]`)
| `/pair_candles` | GET | Returns dataframe for a pair / timeframe combination while the bot is running. **Alpha**
| `/pair_candles` | POST | Returns dataframe for a pair / timeframe combination while the bot is running, filtered by a provided list of columns to return. **Alpha**<br/>*Params:*<br/>- `<column_list>` (`list[str]`)
| `/pair_history` | GET | Returns an analyzed dataframe for a given timerange, analyzed by a given strategy. **Alpha**
| `/pair_history` | POST | Returns an analyzed dataframe for a given timerange, analyzed by a given strategy, filtered by a provided list of columns to return. **Alpha**<br/>*Params:*<br/>- `<column_list>` (`list[str]`)
| `/plot_config` | GET | Get plot config from the strategy (or nothing if not configured). **Alpha**
| `/strategies` | GET | List strategies in strategy directory. **Alpha**
| `/strategy/<strategy>` | GET | Get specific Strategy content by strategy class name. **Alpha**<br/>*Params:*<br/>- `<strategy>` (`str`)
| `/available_pairs` | GET | List available backtest data. **Alpha**
| `/version` | GET | Show version.
| `/sysinfo` | GET | Show information about the system load.
| `/health` | GET | Show bot health (last bot loop).

!!! Warning "Alpha status"
    Endpoints labeled with *Alpha status* above may change at any time without notice.

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
    You can use tools like certbot to setup ssl certificates to access your bot's UI through encrypted connection by using any of the above reverse proxies.
    While this will protect your data in transit, we do not recommend to run the freqtrade API outside of your private network (VPN, SSH tunnel).

### OpenAPI interface

To enable the builtin openAPI interface (Swagger UI), specify `"enable_openapi": true` in the api_server configuration.
This will enable the Swagger UI at the `/docs` endpoint. By default, that's running at <http://localhost:8080/docs> - but it'll depend on your settings.

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

--8<-- "includes/cors.md"
