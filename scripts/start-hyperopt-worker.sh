#!/bin/bash -e

DB_NAME=freqtrade_hyperopt

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORK_DIR="${DIR}/../.hyperopt/worker"

mkdir -p "${WORK_DIR}"
hyperopt-mongo-worker \
    --mongo="127.0.0.1:1234/${DB_NAME}" \
    --poll-interval=0.1 \
    --workdir="${WORK_DIR}"