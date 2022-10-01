import subprocess
import sys

from execution.config import Config
from execution.broker.binance_future_broker import Binance_Future_Broker
from execution.broker.binance_spot_broker import Binance_Spot_Broker
from execution.notifier import Notifier


if len(sys.argv) < 3:
    exit("""Incorrect number of arguments. 
    python3 stop_bot.py [mode:test/prod] [brain] [reason]
    """)
else:
    Config.BRAIN = str(sys.argv[2]) if str(sys.argv[2]).split("_")[0] == "Freq" else "Freq_" + str(sys.argv[2])
    reason = str(sys.argv[3]).replace("_", "#")
    is_test_mode = sys.argv[1] == "test" or sys.argv[1] == "True"

    notifier = Notifier(is_test_mode)

    if Config.IS_BINANCE_FUTURE_MODE:
        broker = Binance_Future_Broker(notifier)
    else:
        broker = Binance_Spot_Broker(notifier)
    broker.close_all_open_positions()

    notifier.send_stop_bot_message(reason)

    subprocess.Popen(["killall screen"],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, shell=True, executable='/bin/bash')

