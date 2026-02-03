## CORS

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

!!! Tip "trailing Slash"
    The trailing slash is not allowed in the `CORS_origins` configuration (e.g. `"http://localhots:8080/"`).
    Such a configuration will not take effect, and the cors errors will remain.

!!! Note
    We strongly recommend to also set `jwt_secret_key` to something random and known only to yourself to avoid unauthorized access to your bot.
