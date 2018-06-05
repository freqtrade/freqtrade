from freqtrade.aws.trade import store, submit
from freqtrade.aws.tables import get_trade_table
import simplejson as json
from boto3.dynamodb.conditions import Key, Attr


def test_store(lambda_context):
    store({
        "Records": [
            {
                "Sns": {
                    "Subject": "trade",
                    "Message": json.dumps(
                        {
                            'id': 'GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TG.MyFancyTestStrategy:BTC/USDT:test',
                            'trade': '2018-05-05 14:15:00 to 2018-05-18 00:40:00',
                            'pair': 'BTC/USDT',
                            'duration': 625,
                            'profit_percent': -0.20453928,
                            'profit_stake': -0.20514198,
                            'entry_date': '2018-05-05 14:15:00',
                            'exit_date': '2018-05-18 00:40:00'
                        }
                    )
                }
            }]
    }
        , {})

    # trade table should not have 1 item in it, with our given key

    table = get_trade_table()
    response = table.query(
        KeyConditionExpression=Key('id')
            .eq('GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TG.MyFancyTestStrategy:BTC/USDT:test')
    )

    print(response)
    assert 'Items' in response
    assert len(response['Items']) == 1
