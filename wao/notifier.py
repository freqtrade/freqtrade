import sys
from wao.brain_config import BrainConfig
import requests

from execution.config import Config
from execution.keys import Keys
from execution._429_file_util import delete_429_file, write_to_429_file

TELEGRAM_RESPONSE_200 = "<Response [200]>"
TELEGRAM_RESPONSE_429 = "<Response [429]>"


def send_start_deliminator_message(brain, month, year):
    print("notifier: send_start_deliminator_message: ")
    text = "========" + str(brain) + " " + str(month) + " " + str(year) + "=======>"

    post_request(text, brain=brain)


def send_stop_bot_message(reason, brain):
    text = "[STOP_BOT_SCRIPT] Bot Stopped! Positions Closed! Reason: " + reason
    print("Notifier: send_stop_bot_message: " + str(text))
    post_request(text, brain=brain)


def post_request(text, is_from_429_watcher=False, brain=BrainConfig.BRAIN, is_from_error_handler=False):
    text.replace("#", "_")
    # if Config.TELEGRAM_LOG_ENABLED:
    #     print("post_request: " + text + " ---------------------")

    if Config.NOTIFIER_ENABLED:
        result = requests.post('https://api.telegram.org/bot' + get_telegram_bot_api_token(brain, is_from_429_watcher) +
                               '/sendMessage?chat_id=' + get_telegram_channel_id(brain) +
                               '&text=' + text.replace("_", "-") + '&parse_mode=Markdown')

        if not is_from_error_handler:
            print(str(result))

        if is_from_429_watcher:
            if str(result) == TELEGRAM_RESPONSE_429:
                delete_429_file(text)
                write_to_429_file(text)
            elif str(result) == TELEGRAM_RESPONSE_200:
                delete_429_file(text)


def get_telegram_bot_api_token(brain, is_from_429_watcher):
    if BrainConfig.MODE == "test":
        if Config.IS_BACKTEST:
            return Keys.NOTIFIER_TELEGRAM_BOT_API_TOKEN_429 if is_from_429_watcher else Keys.NOTIFIER_TELEGRAM_BOT_API_TOKEN_BACKTEST
        elif Config.BRAIN == "lstm":
            return Keys.NOTIFIER_TELEGRAM_BOT_API_TOKEN_LSTM_TEST
        else:
            return Keys.NOTIFIER_TELEGRAM_BOT_API_TOKEN[brain]
    else:
        if Config.BRAIN == "lstm":
            return Keys.NOTIFIER_TELEGRAM_BOT_API_TOKEN_LSTM_PROD
        else:
            return Keys.NOTIFIER_TELEGRAM_BOT_API_TOKEN[brain]


def get_telegram_channel_id(brain):
    if BrainConfig.MODE == "test":
        if Config.IS_BACKTEST:
            return Keys.NOTIFIER_TELEGRAM_CHANNEL_ID_BACKTEST
        elif Config.BRAIN == "lstm":
            return Keys.NOTIFIER_TELEGRAM_CHANNEL_ID_LSTM_TEST
        else:
            return Keys.NOTIFIER_TELEGRAM_CHANNEL_ID[brain]
    else:
        if Config.BRAIN == "lstm":
            return Keys.NOTIFIER_TELEGRAM_CHANNEL_ID_LSTM_PROD
        else:
            return Keys.NOTIFIER_TELEGRAM_CHANNEL_ID[brain]
