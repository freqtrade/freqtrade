import boto3
import simplejson as json
import os
from freqtrade.aws.tables import get_trade_table
from boto3.dynamodb.conditions import Key, Attr


def store(event, context):
    """
    stores the received data in the internal database
    :param data:
    :return:
    """
    if 'Records' in event:
        for x in event['Records']:
            if 'Sns' in x and 'Message' in x['Sns']:
                data = json.loads(x['Sns']['Message'], use_decimal=True)
                get_trade_table().put_item(Item=data)


def submit(event, context):
    """
        submits a new trade to be registered in the internal queue system
    :param event:
    :param context:
    :return:
    """

    print(event)
    data = json.loads(event['body'])
    client = boto3.client('sns')
    topic_arn = client.create_topic(Name=os.environ['tradeTopic'])['TopicArn']

    result = client.publish(
        TopicArn=topic_arn,
        Message=json.dumps({'default': data}),
        Subject="persist data",
        MessageStructure='json'
    )

    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }


def get_aggregated_trades(event, context):
    """
        returns the aggregated trades for the given key combination
    :param event:
    :param context:
    :return:
    """

    assert 'pathParameters' in event
    assert 'ticker' in event['pathParameters']
    assert 'days' in event['pathParameters']

    table = get_trade_table()

    response = table.query(
        KeyConditionExpression=Key('id').eq(
            "aggregate:{}:{}:{}:test".format(
                "TOTAL",
                event['pathParameters']['ticker'],
                event['pathParameters']['days']
            )
        )
    )

    if "Items" in response and len(response['Items']) > 0:

        # preparation for pagination
        # TODO include in parameters an optional
        # start key ExclusiveStartKey=response['LastEvaluatedKey']

        data = {
            "result": response['Items'],
            "paginationKey": response.get('LastEvaluatedKey')
        }

        return {
            "statusCode": response['ResponseMetadata']['HTTPStatusCode'],
            "body": json.dumps(data)
        }

    else:
        return {
            "statusCode": 404,
            "body": json.dumps({
                "error": "sorry this query did not produce any results",
                "event": event
            })
        }


def get_trades(event, context):
    """
        this function returns all the known trades for a user, strategy and pair
    :param event:
    :param context:
    :return:
    """

    assert 'pathParameters' in event
    assert 'user' in event['pathParameters']
    assert 'name' in event['pathParameters']
    assert 'stake' in event['pathParameters']
    assert 'asset' in event['pathParameters']

    table = get_trade_table()

    response = table.query(
        KeyConditionExpression=Key('id').eq(
            "{}.{}:{}/{}".format(
                event['pathParameters']['user'],
                event['pathParameters']['name'],
                event['pathParameters']['asset'].upper(),
                event['pathParameters']['stake'].upper()
            )
        )
    )

    if "Items" in response and len(response['Items']) > 0:

        # preparation for pagination
        # TODO include in parameters an optional
        # start key ExclusiveStartKey=response['LastEvaluatedKey']

        data = {
            "result": response['Items'],
            "paginationKey": response.get('LastEvaluatedKey')
        }

        return {
            "statusCode": response['ResponseMetadata']['HTTPStatusCode'],
            "body": json.dumps(data)
        }

    else:
        return {
            "statusCode": 404,
            "body": json.dumps({
                "error": "sorry this query did not produce any results",
                "event": event
            })
        }
