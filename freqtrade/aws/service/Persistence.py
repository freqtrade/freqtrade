import boto3
import simplejson as json
import decimal

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            if o % 1 > 0:
                return float(o)
            else:
                return int(o)
        return super(DecimalEncoder, self).default(o)


class Persistence:
    """
        simplistic persistence framework
    """

    def __init__(self, table):
        """
            creates a new object with the associated table
        :param table:
        """

        self.table = table
        self.db = boto3.resource('dynamodb')

    def list(self):
        table = self.db.Table(self.table)

        response = table.scan()
        result = []

        while 'LastEvaluatedKey' in response:
            response = table.scan(
                ExclusiveStartKey=response['LastEvaluatedKey']
            )

            for i in response['Items']:
                result.append(i)

        return result

    def load(self, sample):
        """
            loads a given object from the database storage
        :param sample:
        :return:
        """

        table = self.db.Table(self.table)
        result = table.get_item(
            Key={
                'id': sample
            }
        )

        if 'Item' in result:
            return result['Item']
        else:
            return None

    def save(self, object):
        """

        saves and object to the database storage with the specific key

        :param object:
        :return:
        """

        table = self.db.Table(self.table)

        # force serialization to deal with decimal number tag
        data = json.dumps(object, use_decimal=True)
        data = json.loads(data, use_decimal=True)
        print(data)
        return table.put_item(Item=data)
