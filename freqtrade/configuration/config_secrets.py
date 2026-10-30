from copy import deepcopy

from freqtrade.constants import Config, ExchangeConfig


_SENSITIVE_KEYS = [
    "exchange.key",
    "exchange.api_key",
    "exchange.apiKey",
    "exchange.secret",
    "exchange.password",
    "exchange.uid",
    "exchange.account_id",
    "exchange.accountId",
    "exchange.wallet_address",
    "exchange.walletAddress",
    "exchange.private_key",
    "exchange.privateKey",
    "telegram.token",
    "telegram.chat_id",
    "discord.webhook_url",
    "api_server.password",
    "api_server.jwt_secret_key",
    "api_server.ws_token",
    "webhook.url",
    "coingecko.api_key",
    # ccxt credentials (requiredCredentials) - ccxt also accepts snake_case variants
    *[
        f"exchange.{ccxt_section}.{key}"
        for ccxt_section in ("ccxt_config", "ccxt_sync_config", "ccxt_async_config")
        for key in (
            "apiKey",
            "api_key",
            "secret",
            "uid",
            "accountId",
            "account_id",
            "login",
            "password",
            "twofa",
            "privateKey",
            "private_key",
            "walletAddress",
            "wallet_address",
            "token",
        )
    ],
]


def sanitize_config(config: Config, *, show_sensitive: bool = False) -> Config:
    """
    Remove sensitive information from the config.
    :param config: Configuration
    :param show_sensitive: Show sensitive information
    :return: Configuration
    """
    if show_sensitive:
        return config
    config = deepcopy(config)
    for key in _SENSITIVE_KEYS:
        if "." in key:
            nested_keys = key.split(".")
            nested_config = config
            for nested_key in nested_keys[:-1]:
                nested_config = nested_config.get(nested_key, {})
            if nested_config.get(nested_keys[-1]):
                nested_config[nested_keys[-1]] = "REDACTED"
        else:
            if config.get(key):
                config[key] = "REDACTED"

    for producer in config.get("external_message_consumer", {}).get("producers", []):
        if producer.get("ws_token"):
            producer["ws_token"] = "REDACTED"  # noqa: S105
    for pairlist in config.get("pairlists", []):
        if pairlist.get("bearer_token"):
            pairlist["bearer_token"] = "REDACTED"  # noqa: S105

    return config


def remove_exchange_credentials(exchange_config: ExchangeConfig, dry_run: bool) -> None:
    """
    Removes exchange keys from the configuration and specifies dry-run
    Used for backtesting / hyperopt and utils.
    Modifies the input dict!
    :param exchange_config: Exchange configuration
    :param dry_run: If True, remove sensitive keys from the exchange configuration
    """
    if not dry_run:
        return

    for key in [k for k in _SENSITIVE_KEYS if k.startswith("exchange.")]:
        nested_keys = key.removeprefix("exchange.").split(".")
        nested_config = exchange_config
        for nested_key in nested_keys[:-1]:
            nested_config = nested_config.get(nested_key, {})
        if nested_keys[-1] in nested_config:
            nested_config[nested_keys[-1]] = None
