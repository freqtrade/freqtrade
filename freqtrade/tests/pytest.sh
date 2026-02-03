#!/bin/bash

echo "Running Unit tests"

pytest --random-order --cov=freqtrade --cov-config=.coveragerc tests/
