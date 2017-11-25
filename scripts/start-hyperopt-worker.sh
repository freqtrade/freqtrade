#!/bin/bash

DB_NAME=freqtrade_hyperopt

hyperopt-mongo-worker --mongo=127.0.0.1:1234/${DB_NAME} --poll-interval=0.1
