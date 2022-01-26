import subprocess
import threading
from user_data.strategies.config import Config

def launcher(mode, coin, brain, date_time):
    if Config.IS_BACKTEST:
            if Config.IS_PARRALER_EXECUTION:
                 threading.Thread(target=back_tester, args=(mode, coin, brain, date_time)).start()
            else:
                date = str(date_time)
                date = date.replace(" ", "#")
                subprocess.call("python3 "+ Config.EXECUTION_PATH + "back_tester.py " + date + " " + coin + " " + brain + " 0.45 3", shell=True)     
    else:
        if Config.IS_PARRALER_EXECUTION:
            threading.Thread(target=_perform_launcher, args=(mode, coin, brain)).start()
        else:
            subprocess.call("python3 "+Config.EXECUTION_PATH+"launcher.py " + mode + " " + coin + " " + brain, shell=True)

def _perform_launcher(mode, coin, brain):
        subprocess.call("python3 "+Config.EXECUTION_PATH+"launcher.py " + mode + " " + coin + " " + brain, shell=True)

def back_tester(date_time, coin, brain):
    date = str(date_time)
    date = date.replace(" ", "#")
    subprocess.call("python3 "+ Config.EXECUTION_PATH + "back_tester.py " + date + " " + coin + " " + brain + " 0.45 3", shell=True)
