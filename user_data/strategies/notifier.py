EXECUTION_PATH = '/root/workspace2/execution'  # do not move this to config.py

import requests
import sys
import time
from user_data.strategies._429_file_util import delete_429_file, write_to_429_file

sys.path.append(EXECUTION_PATH)
from config import Config as executionConfig


def send_start_deliminator_message(brain, coin, month, year, dup, max_counter_dup):
    print("notifier: send_start_deliminator_message: ")
    text = "========" + str(brain) + " DUP = " + str(dup) + " MAX_COUNTER_DUP = " + max_counter_dup + " " + str(
        coin) + " " + str(month) + " " + str(year) + "=======>"

    post_request(text)


def post_request(text, is_from_429_watcher=False):
    # time.sleep(1)  # safer for parallel execution
    print("post_request: " + text)
    result = requests.post('https://api.telegram.org/bot' + executionConfig.NOTIFIER_TELEGRAM_BOT_API_TOKEN_429 +
                           '/sendMessage?chat_id=' + executionConfig.NOTIFIER_TELEGRAM_CHANNEL_ID_BACKTEST + '&text=' + text + '&parse_mode=Markdown')
    print(str(result))
    if str(result) == "<Response [429]>":
        write_to_429_file(text)
    elif str(result) == "<Response [200]>" and is_from_429_watcher:
        delete_429_file(text)
