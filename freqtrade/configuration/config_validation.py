import logging
from copy import deepcopy
from typing import Any, Dict

from jsonschema import Draft4Validator, validators
from jsonschema.exceptions import ValidationError, best_match

from freqtrade import constants
from freqtrade.exceptions import OperationalException
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


def _extend_validator(validator_class):
    """
    Extended validator for the Freqtrade configuration JSON Schema.
    Currently it only handles defaults for subschemas.
    """
    validate_properties = validator_class.VALIDATORS['properties']

    def set_defaults(validator, properties, instance, schema):
        for prop, subschema in properties.items():
            if 'default' in subschema:
                instance.setdefault(prop, subschema['default'])

        for error in validate_properties(
            validator, properties, instance, schema,
        ):
            yield error

    return validators.extend(
        validator_class, {'properties': set_defaults}
    )


FreqtradeValidator = _extend_validator(Draft4Validator)


def validate_config_schema(conf: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the configuration follow the Config Schema
    :param conf: Config in JSON format
    :return: Returns the config if valid, otherwise throw an exception
    """
    conf_schema = deepcopy(constants.CONF_SCHEMA)
    if conf.get('runmode', RunMode.OTHER) in (RunMode.DRY_RUN, RunMode.LIVE):
        conf_schema['required'] = constants.SCHEMA_TRADE_REQUIRED
    else:
        conf_schema['required'] = constants.SCHEMA_MINIMAL_REQUIRED
        # Dynamically allow empty stake-currency
        # Since the minimal config specifies this too.
        # It's not allowed for Dry-run or live modes
        conf_schema['properties']['stake_currency']['enum'] += ['']  # type: ignore

    try:
        FreqtradeValidator(conf_schema).validate(conf)
        return conf
    except ValidationError as e:
        logger.critical(
            f"Invalid configuration. See config.json.example. Reason: {e}"
        )
        raise ValidationError(
            best_match(Draft4Validator(conf_schema).iter_errors(conf)).message
        )


def validate_config_consistency(conf: Dict[str, Any]) -> None:
    """
    Validate the configuration consistency.
    Should be ran after loading both configuration and strategy,
    since strategies can set certain configuration settings too.
    :param conf: Config in JSON format
    :return: Returns None if everything is ok, otherwise throw an OperationalException
    """

    # validating trailing stoploss
    _validate_trailing_stoploss(conf)
    _validate_edge(conf)
    _validate_whitelist(conf)

    # validate configuration before returning
    logger.info('Validating configuration ...')
    validate_config_schema(conf)


def _validate_trailing_stoploss(conf: Dict[str, Any]) -> None:

    if conf.get('stoploss') == 0.0:
        raise OperationalException(
            'The config stoploss needs to be different from 0 to avoid problems with sell orders.'
            )
    # Skip if trailing stoploss is not activated
    if not conf.get('trailing_stop', False):
        return

    tsl_positive = float(conf.get('trailing_stop_positive', 0))
    tsl_offset = float(conf.get('trailing_stop_positive_offset', 0))
    tsl_only_offset = conf.get('trailing_only_offset_is_reached', False)

    if tsl_only_offset:
        if tsl_positive == 0.0:
            raise OperationalException(
                'The config trailing_only_offset_is_reached needs '
                'trailing_stop_positive_offset to be more than 0 in your config.')
    if tsl_positive > 0 and 0 < tsl_offset <= tsl_positive:
        raise OperationalException(
            'The config trailing_stop_positive_offset needs '
            'to be greater than trailing_stop_positive in your config.')

    # Fetch again without default
    if 'trailing_stop_positive' in conf and float(conf['trailing_stop_positive']) == 0.0:
        raise OperationalException(
            'The config trailing_stop_positive needs to be different from 0 '
            'to avoid problems with sell orders.'
        )


def _validate_edge(conf: Dict[str, Any]) -> None:
    """
    Edge and Dynamic whitelist should not both be enabled, since edge overrides dynamic whitelists.
    """

    if not conf.get('edge', {}).get('enabled'):
        return

    if conf.get('pairlist', {}).get('method') == 'VolumePairList':
        raise OperationalException(
            "Edge and VolumePairList are incompatible, "
            "Edge will override whatever pairs VolumePairlist selects."
        )


def _validate_whitelist(conf: Dict[str, Any]) -> None:
    """
    Dynamic whitelist does not require pair_whitelist to be set - however StaticWhitelist does.
    """
    if conf.get('runmode', RunMode.OTHER) in [RunMode.OTHER, RunMode.PLOT,
                                              RunMode.UTIL_NO_EXCHANGE, RunMode.UTIL_EXCHANGE]:
        return

    for pl in conf.get('pairlists', [{'method': 'StaticPairList'}]):
        if (pl.get('method') == 'StaticPairList'
                and not conf.get('exchange', {}).get('pair_whitelist')):
            raise OperationalException("StaticPairList requires pair_whitelist to be set.")

        if pl.get('method') == 'StaticPairList':
            stake = conf['stake_currency']
            invalid_pairs = []
            for pair in conf['exchange'].get('pair_whitelist'):
                if not pair.endswith(f'/{stake}'):
                    invalid_pairs.append(pair)

            if invalid_pairs:
                raise OperationalException(
                    f"Stake-currency '{stake}' not compatible with pair-whitelist. "
                    f"Please remove the following pairs: {invalid_pairs}")
