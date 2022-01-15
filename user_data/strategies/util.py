import subprocess

IS_BACKTEST = False

EXECUTION_PATH = "/root/workspace2/execution/"

def launcher(mode, coin):
    subprocess.call("python3 "+EXECUTION_PATH+"launcher.py " + mode + " " + coin, shell=True)

def back_tester(date_time, coin):
    subprocess.call("python3 "+EXECUTION_PATH+"back_tester.py " + date_time + " " + coin, shell=True)
