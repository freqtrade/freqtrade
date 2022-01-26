import subprocess
import threading
from user_data.strategies.config import Config

def execute(mode, coin, brain):
    if Config.IS_PARRALER_EXECUTION:
        threading.Thread(target=_perform_execute, args=(mode, coin, brain)).start()
    else:
        subprocess.call("python3 "+Config.EXECUTION_PATH+"executeer.py " + mode + " " + coin + " " + brain, shell=True)

def _perform_execute(mode, coin, brain):
        subprocess.call("python3 "+Config.EXECUTION_PATH+"executeer.py " + mode + " " + coin + " " + brain, shell=True)

def _perform_back_test(date_time, coin, brain):
    date = str(date_time)
    date = date.replace(" ", "#")
    subprocess.call("python3 "+ Config.EXECUTION_PATH + "back_tester.py " + date + " " + coin + " " + brain + " 0.45 3", shell=True)

def back_test(date_time, coin, brain):
    if Config.IS_PARRALER_EXECUTION:
        threading.Thread(target=_perform_back_test, args=(coin, brain, date_time)).start()
    else:
        date = str(date_time)
        date = date.replace(" ", "#")
        subprocess.call("python3 "+ Config.EXECUTION_PATH + "back_tester.py " + date + " " + coin + " " + brain + " 0.45 3", shell=True)     
