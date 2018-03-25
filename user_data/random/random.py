#!/usr/bin/env python3
import os
import multiprocessing
import subprocess
import re
PROC_COUNT = multiprocessing.cpu_count() - 1
cwd = os.getcwd()
print(cwd)
WORK_DIR = os.path.join(
    os.path.sep,
    os.path.abspath(os.path.dirname(__file__)),
    '..', 'freqtrade', 'main.py'
)

# Spawn workers
command = [
    'python3.6',
     WORK_DIR,
    'backtesting',
]
global current
global procs
current = 0
procs = 0
DEVNULL = open(os.devnull, 'wb')

while True:
    while procs < PROC_COUNT:
        try:
            procs + 1
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            wait = proc.communicate()
            string = str(wait)
            params = re.search(r'~~~~(.*)~~~~', string).group(1)
            mfi = re.search(r'MFI Value(.*)XXX', string)
            fastd = re.search(r'FASTD Value(.*)XXX', string)
            adx = re.search(r'ADX Value(.*)XXX', string)
            rsi = re.search(r'RSI Value(.*)XXX', string)
            tot = re.search(r'TOTAL(.*)', string).group(1)
            total = re.search(r'[-+]?([0-9]*\.[0-9]+|[0-9]+)', tot).group(1)
            if total and (float(total) > float(current)):
                current = total
                print('total better profit paremeters:  ')
                print(total)
                if params:
                    print(params)
                    print('~~~~~~')
                    print('Only enable the above settings, not all settings below are used!')
                    print('~~~~~~')
                if mfi:
                    print('~~~MFI~~~')
                    print(mfi.group(1))
                if fastd:
                    print('~~~FASTD~~~')
                    print(fastd.group(1))
                if adx:
                    print('~~~ADX~~~')
                    print(adx.group(1))
                if rsi:
                    print('~~~RSI~~~')
                    print(rsi.group(1))
            procs - 1
        except Exception as e:
            print(e)
