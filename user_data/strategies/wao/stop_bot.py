import subprocess
import sys

from execution.config import Config
from execution.broker.binance_future_ccxt_broker import Binance_Future_Ccxt_broker
from execution.broker.binance_spot_broker import Binance_Spot_Broker
from execution.notifier import Notifier

if len(sys.argv) != 5:
    exit("""Incorrect number of arguments. 
    python3 stop_bot.py [mode:test/prod] [brain] [coin] [reason]
    """)
else:
    is_test_mode = sys.argv[1] == "test" or sys.argv[1] == "True"
    brain = str(sys.argv[2])
    Config.BRAIN = brain if brain.split("_")[0] == "Freq" else "Freq_" + brain
    Config.COIN = sys.argv[3]
    reason = str(sys.argv[4]).replace("_", "#") + " " + Config.COIN

    notifier = Notifier(is_test_mode)
    print("stop_bot: is_test_mode="+str(is_test_mode) +  ", brain="+ Config.BRAIN + ", coin="+ Config.COIN + ", reason=" + str(reason))

    if Config.IS_BINANCE_FUTURE_MODE:
        broker = Binance_Future_Ccxt_broker(notifier)
    else:
        broker = Binance_Spot_Broker(notifier)

    broker.close_all_open_positions_by_coin()

    notifier.send_stop_bot_message(reason)

    subprocess.Popen(["killall screen"],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, shell=True, executable='/bin/bash')

