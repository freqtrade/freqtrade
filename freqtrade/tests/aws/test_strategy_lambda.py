import simplejson as json
from base64 import urlsafe_b64encode
import freqtrade.aws.strategy as aws
import responses


def test_strategy(lambda_context):
    """
        very ugly long test

    :param lambda_context:
    :return:
    """
    content = """# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from hyperopt import hp
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class TestStrategy(IStrategy):
    minimal_roi = {
        "0": 0.5
    }
    stoploss = -0.2
    ticker_interval = '5m'

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        macd = ta.MACD(dataframe)
        dataframe['maShort'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['maMedium'] = ta.EMA(dataframe, timeperiod=21)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['maShort'], dataframe['maMedium'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['maMedium'], dataframe['maShort'])
            ),
            'sell'] = 1
        return dataframe


        """

    request = {
        "user": "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TG",
        "description": "simple test strategy",
        "name": "TestStrategy",
        "content": urlsafe_b64encode(content.encode('utf-8')),
        "public": False
    }

    # db should be empty
    assert (len(json.loads(aws.names({}, {})['body'])) == 0)
    # now we add an entry
    aws.submit({
        "body": json.dumps(request)
    }, {})

    # now we should have items
    assert (len(json.loads(aws.names({}, {})['body'])) == 1)

    # able to add a second strategy with the sample name, but different user

    request = {
        "user": "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TH",
        "description": "simple test strategy",
        "name": "TestStrategy",
        "content": urlsafe_b64encode(content.encode('utf-8')),
        "public": True
    }

    aws.submit({
        "body": json.dumps(request)
    }, {})

    assert (len(json.loads(aws.names({}, {})['body'])) == 2)

    # able to add a duplicated strategy, which should overwrite the existing strategy

    request = {
        "user": "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TH",
        "description": "simple test strategy",
        "name": "TestStrategy",
        "content": urlsafe_b64encode(content.encode('utf-8')),
        "public": True
    }

    aws.submit({
        "body": json.dumps(request)
    }, {})

    assert (len(json.loads(aws.names({}, {})['body'])) == 2)

    # we need to be able to get a strategy ( code cannot be included )
    strategy = aws.get({'pathParameters': {
        "name": "TestStrategy",
        "user": "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TH"
    }}, {})

    print(strategy)
    strategy = json.loads(strategy['body'])

    assert "content" not in strategy
    assert "user" in strategy
    assert "name" in strategy
    assert "description" in strategy
    assert "public" in strategy
    assert "content" not in strategy

    # we need to be able to get the code of the strategy
    code = aws.code({'pathParameters': {
        "name": "TestStrategy",
        "user": "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TH"
    }}, {})

    # code should equal our initial content
    #assert code == content

    # we are not allowed to load a private strategy
    code = aws.code({'pathParameters': {
        "name": "TestStrategy",
        "user": "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TG"
    }}, {})

    # code should equal our initial content
    assert code['statusCode'] == 403
    assert json.loads(code['body']) == {"success": False, "reason": "Denied"}


def test_strategy_submit_github(lambda_context):
    event = {'resource': '/strategies/submit/github', 'path': '/strategies/submit/github', 'httpMethod': 'POST',
             'headers': {'Accept': '*/*', 'CloudFront-Forwarded-Proto': 'https', 'CloudFront-Is-Desktop-Viewer': 'true',
                         'CloudFront-Is-Mobile-Viewer': 'false', 'CloudFront-Is-SmartTV-Viewer': 'false',
                         'CloudFront-Is-Tablet-Viewer': 'false', 'CloudFront-Viewer-Country': 'US',
                         'content-type': 'application/json', 'Host': '887c8k0tui.execute-api.us-east-2.amazonaws.com',
                         'User-Agent': 'GitHub-Hookshot/419cd30',
                         'Via': '1.1 fd885dc16612d4e9d70f328fd0542052.cloudfront.net (CloudFront)',
                         'X-Amz-Cf-Id': 'l8qrc32exLsdGHyWDr5i1WtmlJIQZKo7cqOElKrEEDGRgOm7PPxoKA==',
                         'X-Amzn-Trace-Id': 'Root=1-5b035d39-de61ead01e4729f073a67480',
                         'X-Forwarded-For': '192.30.252.39, 54.182.230.5', 'X-Forwarded-Port': '443',
                         'X-Forwarded-Proto': 'https', 'X-GitHub-Delivery': 'e7baca80-5d52-11e8-86c9-f183bfa87d9b',
                         'X-GitHub-Event': 'ping', 'X-Hub-Signature': 'sha1=d7d4cd82a5e7e4357e0f4df8d032c474c26b6d61'},
             'queryStringParameters': None, 'pathParameters': None, 'stageVariables': None,
             'requestContext': {'resourceId': 'dmek8c', 'resourcePath': '/strategies/submit/github',
                                'httpMethod': 'POST', 'extendedRequestId': 'HQuA9EbLiYcFr3A=',
                                'requestTime': '21/May/2018:23:58:49 +0000', 'path': '/dev/strategies/submit/github',
                                'accountId': '905951628980', 'protocol': 'HTTP/1.1', 'stage': 'dev',
                                'requestTimeEpoch': 1526947129330, 'requestId': 'e7d99de1-5d52-11e8-a559-fb527c3a0860',
                                'identity': {'cognitoIdentityPoolId': None, 'accountId': None,
                                             'cognitoIdentityId': None, 'caller': None, 'sourceIp': '192.30.252.39',
                                             'accessKey': None, 'cognitoAuthenticationType': None,
                                             'cognitoAuthenticationProvider': None, 'userArn': None,
                                             'userAgent': 'GitHub-Hookshot/419cd30', 'user': None},
                                'apiId': '887c8k0tui'},
             'body': '{"zen":"Mind your words, they are important.","hook_id":30374368,"hook":{"type":"Repository",'
                     '"id":30374368,"name":"web","active":true,"events":["push"],"config":{"content_type":"json",'
                     '"insecure_ssl":"0","secret":"********","url":"https://887c8k0tui'
                     '.execute-api.us-east-2.amazonaws.com/dev/strategies/submit/github"},"updated_at":"2018-05'
                     '-21T23:58:49Z","created_at":"2018'
                     '-05-21T23:58:49Z","url":"https://api.'
                     'github.com/repos/'
                     'berlinguyinca/freqtrade-trading-strategies/hooks/30374368","test_url":"https://api'
                     '.github.com/repos/berlinguyinca/freqtrade-trading-strategies/hooks/30374368/test","ping_url'
                     '":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/hooks/30374368/pings'
                     '","last_response":{"code":null,"status":"unused","message":null}},"repository":{"id":130613180,"'
                     'name":"freqtrade-trading-strategies","full_name":"berlinguyinca/freqtrade-trading-strategies",'
                     '"owner":{"login":"berlinguyinca","id":16364,"avatar_url":"https://avatars2.githubusercontent.com'
                     '/u/16364?v=4","gravatar_id":"","url":"https://api.github.com/users/berlinguyinca","html_url":"'
                     'https://github.com/berlinguyinca","followers_url":"https://api.github.com/users/berlinguyinca/'
                     'followers","following_url":"https://api.github.com/users/berlinguyinca/following{/other_user}",'
                     '"gists_url":"https://api.github.com/users/berlinguyinca/gists{/gist_id}","'
                     'starred_url":"https://api.github.com/users/berlinguyinca/starred{/owner}{/repo}","subscriptions_url'
                     '":"https://api.github.com/users/berlinguyinca/subscriptions","organizations_url":"'
                     'https://api.github.com/users/berlinguyinca/orgs","repos_url":"https://api.github.com/users'
                     '/berlinguyinca/repos","events_url":"https://api.github.com/users/berlinguyinca/events{/privacy}'
                     '","received_events_url":"https://api.github.com/users/berlinguyinca/received_events","type":"Us'
                     'er","site_admin":false},"private":false,"html_url":"https://github.com/berlinguyinca/freqtrade-'
                     'trading-strategies","description":"contains strategies for using freqtrade","fork":false,"url":"'
                     'https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies","forks_url":"https://a'
                     'pi.github.com/repos/berlinguyinca/freqtrade-trading-strategies/forks","keys_url":"https://api.gi'
                     'thub.com/repos/berlinguyinca/freqtrade-trading-strategies/keys{/key_id}","collaborators_url":"htt'
                     'ps://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/collaborators{/collaborator}'
                     '","teams_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/teams","ho'
                     'oks_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/hooks","issue_e'
                     'vents_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/issues/events'
                     '{/number}","events_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/'
                     'events","assignees_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/'
                     'assignees{/user}","branches_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-st'
                     'rategies/branches{/branch}","tags_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trad'
                     'ing-strategies/tags","blobs_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-st'
                     'rategies/git/blobs{/sha}","git_tags_url":"https://api.github.com/repos/berlinguyinca/freqtrade-tr'
                     'ading-strategies/git/tags{/sha}","git_refs_url":"https://api.github.com/repos/berlinguyinca/freqtr'
                     'ade-trading-strategies/git/refs{/sha}","trees_url":"https://api.github.com/repos/berlinguyinca/fre'
                     'qtrade-trading-strategies/git/trees{/sha}","statuses_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/statuses/{sha}","languages_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/languages","stargazers_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/stargazers","contributors_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/contributors","subscribers_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/subscribers","subscription_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/subscription","commits_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/commits{/sha}","git_commits_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/git/commits{/sha}","comments_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/comments{/number}","issue_comment_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/issues/comments{/number}","contents_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/contents/{+path}","compare_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/compare/{base}...{head}","merges_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/merges","archive_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/{archive_format}{/ref}","downloads_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/downloads","issues_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/issues{/number}","pulls_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/pulls{/number}","milestones_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/milestones{/number}","notifications_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/notifications{?since,all,participating}","labels_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/labels{/name}","releases_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/releases{/id}","deployments_url":"https://api.github.com/repos/berlinguyinca/freqtrade-trading-strategies/deployments","created_at":"2018-04-22T22:31:25Z","updated_at":"2018-05-21T05:46:21Z","pushed_at":"2018-05-16T07:53:59Z","git_url":"git://github.com/berlinguyinca/freqtrade-trading-strategies.git","ssh_url":"git@github.com:berlinguyinca/freqtrade-trading-strategies.git","clone_url":"https://github.com/berlinguyinca/freqtrade-trading-strategies.git","svn_url":"https://github.com/berlinguyinca/freqtrade-trading-strategies","homepage":null,"size":67,"stargazers_count":11,"watchers_count":11,"language":"Python","has_issues":true,"has_projects":true,"has_downloads":true,"has_wiki":true,"has_pages":false,"forks_count":3,"mirror_url":null,"archived":false,"open_issues_count":1,"license":{"key":"mit","name":"MIT License","spdx_id":"MIT","url":"https://api.github.com/licenses/mit"},"forks":3,"open_issues":1,"watchers":11,"default_branch":"master"},"sender":{"login":"berlinguyinca","id":16364,"avatar_url":"https://avatars2.githubusercontent.com/u/16364?v=4","gravatar_id":"","url":"https://api.github.com/users/berlinguyinca","html_url":"https://github.com/berlinguyinca","followers_url":"https://api.github.com/users/berlinguyinca/followers","following_url":"https://api.github.com/users/berlinguyinca/following{/other_user}","gists_url":"https://api.github.com/users/berlinguyinca/gists{/gist_id}","starred_url":"https://api.github.com/users/berlinguyinca/starred{/owner}{/repo}","subscriptions_url":"https://api.github.com/users/berlinguyinca/subscriptions","organizations_url":"https://api.github.com/users/berlinguyinca/orgs","repos_url":"https://api.github.com/users/berlinguyinca/repos","events_url":"https://api.github.com/users/berlinguyinca/events{/privacy}","received_events_url":"https://api.github.com/users/berlinguyinca/received_events","type":"User","site_admin":false}}',
             'isBase64Encoded': False}

    aws.submit_github(event, {})
