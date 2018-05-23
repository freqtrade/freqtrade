import os

import boto3
from boto.dynamodb2.exceptions import ResourceInUseException

db = boto3.resource('dynamodb')


def get_trade_table():
    """
        provides access to the trade table and if it doesn't exists
        creates it for us
    :return:
    """

    table_name = os.environ['tradeTable']
    existing_tables = boto3.client('dynamodb').list_tables()['TableNames']
    if table_name not in existing_tables:
        try:
            db.create_table(
                TableName=table_name,
                KeySchema=[
                    {
                        'AttributeName': 'id',
                        'KeyType': 'HASH'
                    },
                    {
                        'AttributeName': 'trade',
                        'KeyType': 'RANGE'
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'id',
                        'AttributeType': 'S'
                    }, {
                        'AttributeName': 'trade',
                        'AttributeType': 'S'
                    }
                ],

                ProvisionedThroughput={
                    'ReadCapacityUnits': 1,
                    'WriteCapacityUnits': 1
                }
            )
        except ResourceInUseException as e:
            print("table already exist {}".format(e))

    return db.Table(table_name)


def get_strategy_table():
    """
        provides us access to the strategy table and if it doesn't exists creates it for us
    :return:
    """
    table_name = os.environ['strategyTable']
    existing_tables = boto3.client('dynamodb').list_tables()['TableNames']

    existing_tables = boto3.client('dynamodb').list_tables()['TableNames']
    if table_name not in existing_tables:
        try:
            db.create_table(
                TableName=os.environ[table_name],
                KeySchema=[
                    {
                        'AttributeName': 'user',
                        'KeyType': 'HASH'
                    },
                    {
                        'AttributeName': 'name',
                        'KeyType': 'RANGE'
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'user',
                        'AttributeType': 'S'
                    }, {
                        'AttributeName': 'name',
                        'AttributeType': 'S'
                    }
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 1,
                    'WriteCapacityUnits': 1
                }
            )
        except ResourceInUseException as e:
            print("table already exist {}".format(e))

    return db.Table(table_name)
