# REST API Usage

## Configuration

Enable the rest API by adding the api_server section to your configuration and setting `api_server.enabled` to `true`.

Sample configuration:

``` json
    "api_server": {
        "enabled": true,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
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

All other endpoints return sensitive info and require authentication, so are not available through a webrowser.

To generate a secure password, either use a password manager, or use the below code snipped.

``` python
import secrets
secrets.token_hex()
```

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
  freqtrade --db-url sqlite:///tradesv3.sqlite --strategy MyAwesomeStrategy
```

!!! Danger "Security warning"
    By using `-p 8080:8080` the API is available to everyone connecting to the server under the correct port, so others may be able to control your bot.

## Consuming the API

You can consume the API by using the script `scripts/rest_client.py`.
The client script only requires the `requests` module, so FreqTrade does not need to be installed on the system.

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
| `reload_conf` | | Reloads the configuration file
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

reload_conf
        Reload configuration
        :returns: json object

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
        use reload_conf to reset
        :returns: json object

version
        Returns the version of the bot
        :returns: json object containing the version

whitelist
        Show the current whitelist
        :returns: json object
```
