import logging
from collections import Counter
from copy import deepcopy
from typing import Any

from jsonschema import Draft4Validator, validators
from jsonschema.exceptions import ValidationError, best_match

from freqtrade.configuration.config_schema import (
    CONF_SCHEMA,
    SCHEMA_BACKTEST_REQUIRED,
    SCHEMA_BACKTEST_REQUIRED_FINAL,
    SCHEMA_MINIMAL_REQUIRED,
    SCHEMA_MINIMAL_WEBSERVER,
    SCHEMA_TRADE_REQUIRED,
)
from freqtrade.configuration.deprecated_settings import process_deprecated_setting
from freqtrade.constants import UNLIMITED_STAKE_AMOUNT
from freqtrade.enums import RunMode, TradingMode
from freqtrade.exceptions import ConfigurationError


logger = logging.getLogger(__name__)


def _extend_validator(validator_class):
    """
    Extended validator for the Freqtrade configuration JSON Schema.
    Currently it only handles defaults for subschemas.
    """
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for prop, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(prop, subschema["default"])

        yield from validate_properties(validator, properties, instance, schema)

    return validators.extend(validator_class, {"properties": set_defaults})


FreqtradeValidator = _extend_validator(Draft4Validator)


def validate_config_schema(conf: dict[str, Any], preliminary: bool = False) -> dict[str, Any]:
    """
    Validate the configuration follow the Config Schema
    :param conf: Config in JSON format
    :return: Returns the config if valid, otherwise throw an exception
    """
    conf_schema = deepcopy(CONF_SCHEMA)
    if conf.get("runmode", RunMode.OTHER) in (RunMode.DRY_RUN, RunMode.LIVE):
        conf_schema["required"] = SCHEMA_TRADE_REQUIRED
    elif conf.get("runmode", RunMode.OTHER) in (RunMode.BACKTEST, RunMode.HYPEROPT):
        if preliminary:
            conf_schema["required"] = SCHEMA_BACKTEST_REQUIRED
        else:
            conf_schema["required"] = SCHEMA_BACKTEST_REQUIRED_FINAL
    elif conf.get("runmode", RunMode.OTHER) == RunMode.WEBSERVER:
        conf_schema["required"] = SCHEMA_MINIMAL_WEBSERVER
    else:
        conf_schema["required"] = SCHEMA_MINIMAL_REQUIRED
    try:
        FreqtradeValidator(conf_schema).validate(conf)
        return conf
    except ValidationError as e:
        logger.critical(f"Invalid configuration. Reason: {e}")
        raise ValidationError(best_match(Draft4Validator(conf_schema).iter_errors(conf)).message)


def validate_config_consistency(conf: dict[str, Any], *, preliminary: bool = False) -> None:
    """
    Validate the configuration consistency.
    Should be ran after loading both configuration and strategy,
    since strategies can set certain configuration settings too.
    :param conf: Config in JSON format
    :return: Returns None if everything is ok, otherwise throw an ConfigurationError
    """

    # validating trailing stoploss
    _validate_trailing_stoploss(conf)
    _validate_price_config(conf)
    _validate_edge(conf)
    _validate_whitelist(conf)
    _validate_unlimited_amount(conf)
    _validate_ask_orderbook(conf)
    _validate_freqai_hyperopt(conf)
    _validate_freqai_backtest(conf)
    _validate_freqai_include_timeframes(conf, preliminary=preliminary)
    _validate_consumers(conf)
    validate_migrated_strategy_settings(conf)
    _validate_orderflow(conf)

    # validate configuration before returning
    logger.info("Validating configuration ...")
    validate_config_schema(conf, preliminary=preliminary)


def _validate_unlimited_amount(conf: dict[str, Any]) -> None:
    """
    If edge is disabled, either max_open_trades or stake_amount need to be set.
    :raise: ConfigurationError if config validation failed
    """
    if (
        not conf.get("edge", {}).get("enabled")
        and conf.get("max_open_trades") == float("inf")
        and conf.get("stake_amount") == UNLIMITED_STAKE_AMOUNT
    ):
        raise ConfigurationError("`max_open_trades` and `stake_amount` cannot both be unlimited.")


def _validate_price_config(conf: dict[str, Any]) -> None:
    """
    When using market orders, price sides must be using the "other" side of the price
    """
    # TODO: The below could be an enforced setting when using market orders
    if conf.get("order_types", {}).get("entry") == "market" and conf.get("entry_pricing", {}).get(
        "price_side"
    ) not in ("ask", "other"):
        raise ConfigurationError('Market entry orders require entry_pricing.price_side = "other".')

    if conf.get("order_types", {}).get("exit") == "market" and conf.get("exit_pricing", {}).get(
        "price_side"
    ) not in ("bid", "other"):
        raise ConfigurationError('Market exit orders require exit_pricing.price_side = "other".')


def _validate_trailing_stoploss(conf: dict[str, Any]) -> None:
    if conf.get("stoploss") == 0.0:
        raise ConfigurationError(
            "The config stoploss needs to be different from 0 to avoid problems with sell orders."
        )
    # Skip if trailing stoploss is not activated
    if not conf.get("trailing_stop", False):
        return

    tsl_positive = float(conf.get("trailing_stop_positive", 0))
    tsl_offset = float(conf.get("trailing_stop_positive_offset", 0))
    tsl_only_offset = conf.get("trailing_only_offset_is_reached", False)

    if tsl_only_offset:
        if tsl_positive == 0.0:
            raise ConfigurationError(
                "The config trailing_only_offset_is_reached needs "
                "trailing_stop_positive_offset to be more than 0 in your config."
            )
    if tsl_positive > 0 and 0 < tsl_offset <= tsl_positive:
        raise ConfigurationError(
            "The config trailing_stop_positive_offset needs "
            "to be greater than trailing_stop_positive in your config."
        )

    # Fetch again without default
    if "trailing_stop_positive" in conf and float(conf["trailing_stop_positive"]) == 0.0:
        raise ConfigurationError(
            "The config trailing_stop_positive needs to be different from 0 "
            "to avoid problems with sell orders."
        )


def _validate_edge(conf: dict[str, Any]) -> None:
    """
    Edge and Dynamic whitelist should not both be enabled, since edge overrides dynamic whitelists.
    """

    if not conf.get("edge", {}).get("enabled"):
        return

    if not conf.get("use_exit_signal", True):
        raise ConfigurationError(
            "Edge requires `use_exit_signal` to be True, otherwise no sells will happen."
        )


def _validate_whitelist(conf: dict[str, Any]) -> None:
    """
    Dynamic whitelist does not require pair_whitelist to be set - however StaticWhitelist does.
    """
    if conf.get("runmode", RunMode.OTHER) in [
        RunMode.OTHER,
        RunMode.PLOT,
        RunMode.UTIL_NO_EXCHANGE,
        RunMode.UTIL_EXCHANGE,
    ]:
        return

    for pl in conf.get("pairlists", [{"method": "StaticPairList"}]):
        if (
            isinstance(pl, dict)
            and pl.get("method") == "StaticPairList"
            and not conf.get("exchange", {}).get("pair_whitelist")
        ):
            raise ConfigurationError("StaticPairList requires pair_whitelist to be set.")


def _validate_ask_orderbook(conf: dict[str, Any]) -> None:
    ask_strategy = conf.get("exit_pricing", {})
    ob_min = ask_strategy.get("order_book_min")
    ob_max = ask_strategy.get("order_book_max")
    if ob_min is not None and ob_max is not None and ask_strategy.get("use_order_book"):
        if ob_min != ob_max:
            raise ConfigurationError(
                "Using order_book_max != order_book_min in exit_pricing is no longer supported."
                "Please pick one value and use `order_book_top` in the future."
            )
        else:
            # Move value to order_book_top
            ask_strategy["order_book_top"] = ob_min
            logger.warning(
                "DEPRECATED: "
                "Please use `order_book_top` instead of `order_book_min` and `order_book_max` "
                "for your `exit_pricing` configuration."
            )


def validate_migrated_strategy_settings(conf: dict[str, Any]) -> None:
    _validate_time_in_force(conf)
    _validate_order_types(conf)
    _validate_unfilledtimeout(conf)
    _validate_pricing_rules(conf)
    _strategy_settings(conf)


def _validate_time_in_force(conf: dict[str, Any]) -> None:
    time_in_force = conf.get("order_time_in_force", {})
    if "buy" in time_in_force or "sell" in time_in_force:
        if conf.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT:
            raise ConfigurationError(
                "Please migrate your time_in_force settings to use 'entry' and 'exit'."
            )
        else:
            logger.warning(
                "DEPRECATED: Using 'buy' and 'sell' for time_in_force is deprecated."
                "Please migrate your time_in_force settings to use 'entry' and 'exit'."
            )
            process_deprecated_setting(
                conf, "order_time_in_force", "buy", "order_time_in_force", "entry"
            )

            process_deprecated_setting(
                conf, "order_time_in_force", "sell", "order_time_in_force", "exit"
            )


def _validate_order_types(conf: dict[str, Any]) -> None:
    order_types = conf.get("order_types", {})
    old_order_types = [
        "buy",
        "sell",
        "emergencysell",
        "forcebuy",
        "forcesell",
        "emergencyexit",
        "forceexit",
        "forceentry",
    ]
    if any(x in order_types for x in old_order_types):
        if conf.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT:
            raise ConfigurationError(
                "Please migrate your order_types settings to use the new wording."
            )
        else:
            logger.warning(
                "DEPRECATED: Using 'buy' and 'sell' for order_types is deprecated."
                "Please migrate your order_types settings to use 'entry' and 'exit' wording."
            )
            for o, n in [
                ("buy", "entry"),
                ("sell", "exit"),
                ("emergencysell", "emergency_exit"),
                ("forcesell", "force_exit"),
                ("forcebuy", "force_entry"),
                ("emergencyexit", "emergency_exit"),
                ("forceexit", "force_exit"),
                ("forceentry", "force_entry"),
            ]:
                process_deprecated_setting(conf, "order_types", o, "order_types", n)


def _validate_unfilledtimeout(conf: dict[str, Any]) -> None:
    unfilledtimeout = conf.get("unfilledtimeout", {})
    if any(x in unfilledtimeout for x in ["buy", "sell"]):
        if conf.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT:
            raise ConfigurationError(
                "Please migrate your unfilledtimeout settings to use the new wording."
            )
        else:
            logger.warning(
                "DEPRECATED: Using 'buy' and 'sell' for unfilledtimeout is deprecated."
                "Please migrate your unfilledtimeout settings to use 'entry' and 'exit' wording."
            )
            for o, n in [
                ("buy", "entry"),
                ("sell", "exit"),
            ]:
                process_deprecated_setting(conf, "unfilledtimeout", o, "unfilledtimeout", n)


def _validate_pricing_rules(conf: dict[str, Any]) -> None:
    if conf.get("ask_strategy") or conf.get("bid_strategy"):
        if conf.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT:
            raise ConfigurationError("Please migrate your pricing settings to use the new wording.")
        else:
            logger.warning(
                "DEPRECATED: Using 'ask_strategy' and 'bid_strategy' is deprecated."
                "Please migrate your settings to use 'entry_pricing' and 'exit_pricing'."
            )
            conf["entry_pricing"] = {}
            for obj in list(conf.get("bid_strategy", {}).keys()):
                if obj == "ask_last_balance":
                    process_deprecated_setting(
                        conf, "bid_strategy", obj, "entry_pricing", "price_last_balance"
                    )
                else:
                    process_deprecated_setting(conf, "bid_strategy", obj, "entry_pricing", obj)
            del conf["bid_strategy"]

            conf["exit_pricing"] = {}
            for obj in list(conf.get("ask_strategy", {}).keys()):
                if obj == "bid_last_balance":
                    process_deprecated_setting(
                        conf, "ask_strategy", obj, "exit_pricing", "price_last_balance"
                    )
                else:
                    process_deprecated_setting(conf, "ask_strategy", obj, "exit_pricing", obj)
            del conf["ask_strategy"]


def _validate_freqai_hyperopt(conf: dict[str, Any]) -> None:
    freqai_enabled = conf.get("freqai", {}).get("enabled", False)
    analyze_per_epoch = conf.get("analyze_per_epoch", False)
    if analyze_per_epoch and freqai_enabled:
        raise ConfigurationError(
            "Using analyze-per-epoch parameter is not supported with a FreqAI strategy."
        )


def _validate_freqai_include_timeframes(conf: dict[str, Any], preliminary: bool) -> None:
    freqai_enabled = conf.get("freqai", {}).get("enabled", False)
    if freqai_enabled:
        main_tf = conf.get("timeframe", "5m")
        freqai_include_timeframes = (
            conf.get("freqai", {}).get("feature_parameters", {}).get("include_timeframes", [])
        )

        from freqtrade.exchange import timeframe_to_seconds

        main_tf_s = timeframe_to_seconds(main_tf)
        offending_lines = []
        for tf in freqai_include_timeframes:
            tf_s = timeframe_to_seconds(tf)
            if tf_s < main_tf_s:
                offending_lines.append(tf)
        if offending_lines:
            raise ConfigurationError(
                f"Main timeframe of {main_tf} must be smaller or equal to FreqAI "
                f"`include_timeframes`.Offending include-timeframes: {', '.join(offending_lines)}"
            )

        # Ensure that the base timeframe is included in the include_timeframes list
        if not preliminary and main_tf not in freqai_include_timeframes:
            feature_parameters = conf.get("freqai", {}).get("feature_parameters", {})
            include_timeframes = [main_tf] + freqai_include_timeframes
            conf.get("freqai", {}).get("feature_parameters", {}).update(
                {**feature_parameters, "include_timeframes": include_timeframes}
            )


def _validate_freqai_backtest(conf: dict[str, Any]) -> None:
    if conf.get("runmode", RunMode.OTHER) == RunMode.BACKTEST:
        freqai_enabled = conf.get("freqai", {}).get("enabled", False)
        timerange = conf.get("timerange")
        freqai_backtest_live_models = conf.get("freqai_backtest_live_models", False)
        if freqai_backtest_live_models and freqai_enabled and timerange:
            raise ConfigurationError(
                "Using timerange parameter is not supported with "
                "--freqai-backtest-live-models parameter."
            )

        if freqai_backtest_live_models and not freqai_enabled:
            raise ConfigurationError(
                "Using --freqai-backtest-live-models parameter is only "
                "supported with a FreqAI strategy."
            )

        if freqai_enabled and not freqai_backtest_live_models and not timerange:
            raise ConfigurationError(
                "Please pass --timerange if you intend to use FreqAI for backtesting."
            )


def _validate_consumers(conf: dict[str, Any]) -> None:
    emc_conf = conf.get("external_message_consumer", {})
    if emc_conf.get("enabled", False):
        if len(emc_conf.get("producers", [])) < 1:
            raise ConfigurationError("You must specify at least 1 Producer to connect to.")

        producer_names = [p["name"] for p in emc_conf.get("producers", [])]
        duplicates = [item for item, count in Counter(producer_names).items() if count > 1]
        if duplicates:
            raise ConfigurationError(
                f"Producer names must be unique. Duplicate: {', '.join(duplicates)}"
            )
        if conf.get("process_only_new_candles", True):
            # Warning here or require it?
            logger.warning(
                "To receive best performance with external data, "
                "please set `process_only_new_candles` to False"
            )


def _validate_orderflow(conf: dict[str, Any]) -> None:
    if conf.get("exchange", {}).get("use_public_trades"):
        if "orderflow" not in conf:
            raise ConfigurationError(
                "Orderflow is a required configuration key when using public trades."
            )


def _strategy_settings(conf: dict[str, Any]) -> None:
    process_deprecated_setting(conf, None, "use_sell_signal", None, "use_exit_signal")
    process_deprecated_setting(conf, None, "sell_profit_only", None, "exit_profit_only")
    process_deprecated_setting(conf, None, "sell_profit_offset", None, "exit_profit_offset")
    process_deprecated_setting(
        conf, None, "ignore_roi_if_buy_signal", None, "ignore_roi_if_entry_signal"
    )
