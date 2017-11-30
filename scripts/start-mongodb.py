#!/usr/bin/env python3

import os
import subprocess


DB_PATH = os.path.join(
    os.path.sep,
    os.path.abspath(os.path.dirname(__file__)),
    '..', '.hyperopt', 'mongodb'
)
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

subprocess.Popen([
    'mongod',
    '--bind_ip=127.0.0.1',
    '--port=1234',
    '--nohttpinterface',
    '--dbpath={}'.format(DB_PATH),
]).wait()
