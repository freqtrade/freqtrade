EXECUTION_PATH = '/root/workspace2/execution'  # do not move this to config.py

import requests
import sys
sys.path.append(EXECUTION_PATH)
from config import Config as ExecutionConfig


def send_start_deliminator_message(brain, coin, month, year, dup, max_counter_dup):
    print("notifier: send_start_deliminator_message: ")
    text = "========" + str(brain) + " DUP = " + str(dup) + " MAX_COUNTER_DUP = " + max_counter_dup + " " + str(coin) + " " + str(month) + " " + str(year) + "=======>"
    if ExecutionConfig.NOTIFIER_ENABLE:
        result = requests.post('https://api.telegram.org/bot' + ExecutionConfig.NOTIFIER_TELEGRAM_BOT_API_TOKEN_BACKTEST +
                                   '/sendMessage?chat_id=' + ExecutionConfig.NOTIFIER_TELEGRAM_CHANNEL_ID_BACKTEST + '&text=' + text + '&parse_mode=Markdown')
        print(str(result))