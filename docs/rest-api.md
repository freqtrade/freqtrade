# REST API Usage

## Configuration

Enable the rest API by adding the api_server section to your configuration and setting `api_server.enabled` to `true`.

Sample configuration:

``` json
    "api_server": {
        "enabled": true,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "verbosity": "info",
        "jwt_secret_key": "somethingrandom",
        "CORS_origins": [],
        "username": "Freqtrader",
        "password": "SuperSecret1!"
    },
```

!!! Danger "Security warning"
    By default, the configuration listens on localhost only (so it's not reachable from other systems). We strongly recommend to not expose this API to the internet and choose a strong, unique password, since others will potentially be able to control your bot.

!!! Danger "Password selection"
    Please make sure to select a very strong, unique password to protect your bot from unauthorized access.

You can then access the API by going to `http://127.0.0.1:8080/api/v1/ping` in a browser to check if the API is running correctly.
This should return the response:

``` output
{"status":"pong"}
```

All other endpoints return sensitive info and require authentication and are therefore not available through a web browser.

To generate a secure password, either use a password manager, or use the below code snipped.

``` python
import secrets
secrets.token_hex()
```

!!! Hint
    Use the same method to also generate a JWT secret key (`jwt_secret_key`).

### Configuration with docker

If you run your bot using docker, you'll need to have the bot listen to incomming connections. The security is then handled by docker.

``` json
    "api_server": {
        "enabled": true,
        "listen_ip_address": "0.0.0.0",
        "listen_port": 8080
    },
```

Add the following to your docker command:

``` bash
  -p 127.0.0.1:8080:8080
```

A complete sample-command may then look as follows:

```bash
docker run -d \
  --name freqtrade \
  -v ~/.freqtrade/config.json:/freqtrade/config.json \
  -v ~/.freqtrade/user_data/:/freqtrade/user_data \
  -v ~/.freqtrade/tradesv3.sqlite:/freqtrade/tradesv3.sqlite \
  -p 127.0.0.1:8080:8080 \
  freqtrade trade --db-url sqlite:///tradesv3.sqlite --strategy MyAwesomeStrategy
```

!!! Danger "Security warning"
    By using `-p 8080:8080` the API is available to everyone connecting to the server under the correct port, so others may be able to control your bot.

## Consuming the API

You can consume the API by using the script `scripts/rest_client.py`.
The client script only requires the `requests` module, so Freqtrade does not need to be installed on the system.

``` bash
python3 scripts/rest_client.py <command> [optional parameters]
```

By default, the script assumes `127.0.0.1` (localhost) and port `8080` to be used, however you can specify a configuration file to override this behaviour.

### Minimalistic client config

``` json
{
    "api_server": {
        "enabled": true,
        "listen_ip_address": "0.0.0.0",
        "listen_port": 8080
    }
}
```

``` bash
python3 scripts/rest_client.py --config rest_config.json <command> [optional parameters]
```

## Available commands

|  Command | Default | Description |
|----------|---------|-------------|
| `start` | | Starts the trader
| `stop` | | Stops the trader
| `stopbuy` | | Stops the trader from opening new trades. Gracefully closes open trades according to their rules.
| `reload_config` | | Reloads the configuration file
| `show_config` | | Shows part of the current configuration with relevant settings to operation
| `status` | | Lists all open trades
| `count` | | Displays number of trades used and available
| `profit` | | Display a summary of your profit/loss from close trades and some stats about your performance
| `forcesell <trade_id>` | | Instantly sells the given trade  (Ignoring `minimum_roi`).
| `forcesell all` | | Instantly sells all open trades (Ignoring `minimum_roi`).
| `forcebuy <pair> [rate]` | | Instantly buys the given pair. Rate is optional. (`forcebuy_enable` must be set to True)
| `performance` | | Show performance of each finished trade grouped by pair
| `balance` | | Show account balance per currency
| `daily <n>` | 7 | Shows profit or loss per day, over the last n days
| `whitelist` | | Show the current whitelist
| `blacklist [pair]` | | Show the current blacklist, or adds a pair to the blacklist.
| `edge` | | Show validated pairs by Edge if it is enabled.
| `version` | | Show version

Possible commands can be listed from the rest-client script using the `help` command.

``` bash
python3 scripts/rest_client.py help
```

``` output
Possible commands:
balance
        Get the account balance
        :returns: json object

blacklist
        Show the current blacklist
        :param add: List of coins to add (example: "BNB/BTC")
        :returns: json object

count
        Returns the amount of open trades
        :returns: json object

daily
        Returns the amount of open trades
        :returns: json object

edge
        Returns information about edge
        :returns: json object

forcebuy
        Buy an asset
        :param pair: Pair to buy (ETH/BTC)
        :param price: Optional - price to buy
        :returns: json object of the trade

forcesell
        Force-sell a trade
        :param tradeid: Id of the trade (can be received via status command)
        :returns: json object

performance
        Returns the performance of the different coins
        :returns: json object

profit
        Returns the profit summary
        :returns: json object

reload_config
        Reload configuration
        :returns: json object

show_config
        Returns part of the configuration, relevant for trading operations.
        :return: json object containing the version

start
        Start the bot if it's in stopped state.
        :returns: json object

status
        Get the status of open trades
        :returns: json object

stop
        Stop the bot. Use start to restart
        :returns: json object

stopbuy
        Stop buying (but handle sells gracefully).
        use reload_config to reset
        :returns: json object

version
        Returns the version of the bot
        :returns: json object containing the version

whitelist
        Show the current whitelist
        :returns: json object
```

## Advanced API usage using JWT tokens

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

## CORS

All web-based frontends are subject to [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) - Cross-Origin Resource Sharing.
Since most of the requests to the Freqtrade API must be authenticated, a proper CORS policy is key to avoid security problems.
Also, the standard disallows `*` CORS policies for requests with credentials, so this setting must be set appropriately.

Users can configure this themselves via the `CORS_origins` configuration setting.
It consists of a list of allowed sites that are allowed to consume resources from the bot's API.

Assuming your application is deployed as `https://frequi.freqtrade.io/home/` - this would mean that the following configuration becomes necessary:

```jsonc
{
    //...
    "jwt_secret_key": "somethingrandom",
    "CORS_origins": ["https://frequi.freqtrade.io"],
    //...
}
```

!!! Note
    We strongly recommend to also set `jwt_secret_key` to something random and known only to yourself to avoid unauthorized access to your bot.
