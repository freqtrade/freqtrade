import subprocess
import threading
from user_data.strategies.config import Config

def execute(mode, coin, brain):
    if Config.IS_PARALLEL_EXECUTION:
        threading.Thread(target=_perform_execute, args=(mode, coin, brain)).start()
    else:
        _perform_execute(mode, coin, brain)

def _perform_execute(mode, coin, brain):
        subprocess.call("python3 "+Config.EXECUTION_PATH+"launcher.py " + mode + " " + coin + " " + brain, shell=True)

def _perform_back_test(date_time, coin, brain):
    date = str(date_time)
    date = date.replace(" ", "#")
    subprocess.call("python3 "+ Config.EXECUTION_PATH + "back_tester.py " + date + " " + coin + " " + brain + " 0.45 3", shell=True)

def back_test(date_time, coin, brain):
    if Config.IS_PARALLEL_EXECUTION:
        threading.Thread(target=_perform_back_test, args=(date_time, coin, brain)).start()
    else:
        _perform_back_test(date_time, coin, brain)