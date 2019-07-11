import logging
from typing import Any, Dict

from jsonschema import Draft4Validator, validators
from jsonschema.exceptions import ValidationError, best_match

from freqtrade import constants


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
    try:
        FreqtradeValidator(constants.CONF_SCHEMA).validate(conf)
        return conf
    except ValidationError as e:
        logger.critical(
            f"Invalid configuration. See config.json.example. Reason: {e}"
        )
        raise ValidationError(
            best_match(Draft4Validator(constants.CONF_SCHEMA).iter_errors(conf)).message
        )
