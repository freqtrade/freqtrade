
# Required json-schema for user specified config
CONF_SCHEMA = {
    'type': 'object',
    'properties': {
        'max_open_trades': {'type': 'integer', 'minimum': 1},
        'stake_currency': {'type': 'string', 'enum': ['BTC', 'ETH', 'USDT']},
        'stake_amount': {'type': 'number', 'minimum': 0.0005},
        'dry_run': {'type': 'boolean'},
        'minimal_roi': {
            'type': 'object',
            'patternProperties': {
                '^[0-9.]+$': {'type': 'number'}
            },
            'minProperties': 1
        },
        'stoploss': {'type': 'number', 'maximum': 0, 'exclusiveMaximum': True},
        'poloniex': {'$ref': '#/definitions/exchange'},
        'bittrex': {'$ref': '#/definitions/exchange'},
        'telegram': {
            'type': 'object',
            'properties': {
                'enabled': {'type': 'boolean'},
                'token': {'type': 'string'},
                'chat_id': {'type': 'string'},
            },
            'required': ['enabled', 'token', 'chat_id']
        }
    },
    'definitions': {
        'exchange': {
            'type': 'object',
            'properties': {
                'enabled': {'type': 'boolean'},
                'key': {'type': 'string'},
                'secret': {'type': 'string'},
                'pair_whitelist': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'uniqueItems': True
                }
            },
            'required': ['enabled', 'key', 'secret', 'pair_whitelist']
        }
    },
    'anyOf': [
        {'required': ['poloniex']},
        {'required': ['bittrex']}
    ],
    'required': [
        'max_open_trades',
        'stake_currency',
        'stake_amount',
        'dry_run',
        'minimal_roi',
        'telegram'
    ]
}
