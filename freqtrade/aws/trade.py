import boto3
import simplejson as json
import os
from freqtrade.aws.tables import get_trade_table


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

    data = json.loads(event['body'])
    client = boto3.client('sns')
    topic_arn = client.create_topic(Name=os.environ['tradeTopic'])['TopicArn']

    result = client.publish(
        TopicArn=topic_arn,
        Message=json.dumps({'default': data}),
        Subject="persist data",
        MessageStructure='json'
    )
