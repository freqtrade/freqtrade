import subprocess

CURRENT_YEAR = 2020
CURRENT_MONTH = 10
IS_BACKTEST = False
WORKSPACE_PATH = "workspace2" if IS_BACKTEST else "workspace"
EXECUTION_PATH = "/root/" + WORKSPACE_PATH + "/execution/"

def launcher(mode, coin, brain):
    subprocess.call("python3 "+EXECUTION_PATH+"launcher.py " + mode + " " + coin + " " + brain, shell=True)

def back_tester(date_time, coin, brain):
    date = str(date_time)
    date = date.replace(" ", "#")
    subprocess.call("python3 "+ EXECUTION_PATH + "back_tester.py " + date + " " + coin + " " + brain + " 0.45 3", shell=True)
