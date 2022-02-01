import requests
from user_data.strategies.config import Config


def send_backtest_start_deliminator_message(action, coin, month, year):
    print("Notifier: send_backtest_start_deliminator_message: action= " + str(action))
    text = "=========" + str(coin) + " " + str(month) + " " + str(year) + "=========>"

    result = requests.post('https://api.telegram.org/bot' + Config.NOTIFIER_TELEGRAM_BOT_API_TOKEN_BACKTEST +
                                   '/sendMessage?chat_id=' + Config.NOTIFIER_TELEGRAM_CHANNEL_ID_BACKTEST + '&text=' + text + '&parse_mode=Markdown')
    print(str(result))