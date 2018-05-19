import simplejson as json
import os
import boto3


class Queue:
    """
    abstraction of the underlaying queuing system to schedule a message to the backend for processing
    """

    def submit(self, object, routingKey):
        """
            submits the given object to the queue associated with the
            routing key.
            The routing lambda function will than make sure it will be delivered to the right destination

        :param object:
        :param routingKey:
        :return:
        """

        # get topic refrence
        client = boto3.client('sns')

        # if topic exists, we just reuse it
        topic_arn = client.create_topic(Name=os.environ['topic'])['TopicArn']

        serialized = json.dumps(object, use_decimal=True)
        # submit item to queue for routing to the correct persistence

        result = client.publish(
            TopicArn=topic_arn,
            Message=json.dumps({'default': serialized}),
            Subject="route:" + routingKey,
            MessageStructure='json',
            MessageAttributes={
                'route': {
                    'DataType': 'String',
                    'StringValue': routingKey
                }
            },
        )

        return {
            "statusCode": result['ResponseMetadata']['HTTPStatusCode'],
            "body": serialized
        }
