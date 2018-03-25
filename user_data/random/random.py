#!/usr/bin/env python3
import os
import multiprocessing
from itertools import zip_longest
import subprocess
import re
PROC_COUNT = multiprocessing.cpu_count() - 1
cwd = os.getcwd()
print(cwd)


limit = multiprocessing.cpu_count() - 1
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
current = 0

DEVNULL = open(os.devnull, 'wb')

while True:
    def Run(command):
        global current
        processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(PROC_COUNT)]
        for proc in processes:
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
                    print(mfi.group(1))
                if fastd:
                    print(fastd.group(1))
                if adx:
                    print(adx.group(1))
                if rsi:
                    print(rsi.group(1))
    data = Run(command)
