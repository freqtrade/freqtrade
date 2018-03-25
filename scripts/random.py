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

while True:
    def Run(command):
        global current
        processes = [subprocess.Popen(command) for i in range(PROC_COUNT)]
        for proc in processes:
            wait = proc.communicate()
    data = Run(command)
