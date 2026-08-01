# FreqUI

Freqtrade provides a builtin webserver, which can serve [FreqUI](https://github.com/freqtrade/frequi), the freqtrade frontend.

By default, the UI is automatically installed as part of the installation (script, docker).
freqUI can also be manually installed by using the `freqtrade install-ui` command.
This same command can also be used to update freqUI to new releases.

Once the bot is started in trade / dry-run mode (with `freqtrade trade`) - the UI will be available under the configured API port (by default `http://127.0.0.1:8080`).

??? Note "Looking to contribute to freqUI?"
    Developers should not use this method, but instead clone the corresponding use the method described in the [freqUI repository](https://github.com/freqtrade/frequi) to get the source-code of freqUI. A working installation of node will be required to build the frontend.

!!! tip "freqUI is not required to run freqtrade"
    freqUI is an optional component of freqtrade, and is not required to run the bot.
    It is a frontend that can be used to monitor the bot and to interact with it - but freqtrade itself will work perfectly fine without it.

## Configuration

FreqUI does not have it's own configuration file - but assumes a working setup for the [rest-api](rest-api.md) is available.
Please refer to the corresponding documentation page to get setup with freqUI

## UI

FreqUI is a modern, responsive web application that can be used to monitor and interact with your bot.

FreqUI provides a light, as well as a dark theme.
Themes can be easily switched via a prominent button at the top of the page.
The theme of the screenshots on this page will adapt to the selected documentation Theme, so to see the dark (or light) version, please switch the theme of the Documentation.

### Login

The below screenshot shows the login screen of freqUI.

![FreqUI - login](assets/frequi-login-CORS.png#only-dark)
![FreqUI - login](assets/frequi-login-CORS-light.png#only-light)

!!! Hint "CORS"
    The Cors error shown in this screenshot is due to the fact that the UI is running on a different port than the API, and [CORS](#cors) has not been setup correctly yet.

### Trade view

The trade view allows you to visualize the trades that the bot is making and to interact with the bot.
On this page, you can also interact with the bot by starting and stopping it and - if configured - force trade entries and exits.

![FreqUI - trade view](assets/freqUI-trade-pane-dark.png#only-dark)
![FreqUI - trade view](assets/freqUI-trade-pane-light.png#only-light)

### Dashboard

The dashboard view provides an overview of the bot's performance and status.
If multiple bots are connected, the dashboard will show an overview of all connected bots, allowing you to easily switch between them or show just a subset of available bots.

#### Wallet Balance

New in freqtrade 2026.4: This shows the balance of the bot over time.

Compared to the "cumulative Profit" chart, this chart will show the actual balance of the bot over time, including unrealized profit and losses, as well as deposits and withdrawals.

Historic data has re-populated based on available exchange data - however is assumed to be best-effort and may not be 100% accurate.
More specifically, it won't cover deposits and withdrawals, and will assume a starting balance of current balance - profit/losses.

For clarity - a "Capture start" marker line is shown on the chart, which indicates the point at which the migration to the new wallet balance tracking system happened.
Only beyond this point, the wallet balance is expected to be accurate.

### Plot Configurator

FreqUI Plots can be configured either via a `plot_config` configuration object in the strategy (which can be loaded via "from strategy" button) or via the UI.
Multiple plot configurations can be created and switched at will - allowing for flexible, different views into your charts.

The plot configuration can be accessed via the "Plot Configurator" (Cog icon) button in the top right corner of the trade view.

![FreqUI - plot configuration](assets/freqUI-plot-configurator-dark.png#only-dark)
![FreqUI - plot configuration](assets/freqUI-plot-configurator-light.png#only-light)

### Settings

Several UI related settings can be changed by accessing the settings page.

Things you can change (among others):

* Timezone of the UI
* Visualization of open trades as part of the favicon (browser tab)
* Candle colors (up/down -> red/green)
* Enable / disable in-app notification types

![FreqUI - Settings view](assets/frequi-settings-dark.png#only-dark)
![FreqUI - Settings view](assets/frequi-settings-light.png#only-light)

## Webserver mode

when freqtrade is started in [webserver mode](utils.md#webserver-mode) (freqtrade started with `freqtrade webserver`), the webserver will start in a special mode allowing for additional features, for example:

* Downloading data
* Testing pairlists
* [Backtesting strategies](#backtesting)
* ... to be expanded

### Backtesting

When freqtrade is started in [webserver mode](utils.md#webserver-mode) (freqtrade started with `freqtrade webserver`), the backtesting view becomes available.
This view allows you to backtest strategies and visualize the results.

You can also load and visualize previous backtest results, as well as compare the results with each other.

![FreqUI - Backtesting](assets/freqUI-backtesting-dark.png#only-dark)
![FreqUI - Backtesting](assets/freqUI-backtesting-light.png#only-light)


--8<-- "includes/cors.md"
