#!/usr/bin/env python3
import multiprocessing
import os
import subprocess

PROC_COUNT = multiprocessing.cpu_count() - 1
DB_NAME = 'freqtrade_hyperopt'
WORK_DIR = os.path.join(
    os.path.sep,
    os.path.abspath(os.path.dirname(__file__)),
    '..', '.hyperopt', 'worker'
)
if not os.path.exists(WORK_DIR):
    os.makedirs(WORK_DIR)

# Spawn workers
command = [
    'hyperopt-mongo-worker',
    '--mongo=127.0.0.1:1234/{}'.format(DB_NAME),
    '--poll-interval=0.1',
    '--workdir={}'.format(WORK_DIR),
]
processes = [subprocess.Popen(command) for i in range(PROC_COUNT)]

# Join all workers
for proc in processes:
    proc.wait()
