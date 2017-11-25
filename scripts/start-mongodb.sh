#!/bin/bash -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DB_PATH="${DIR}/../.mongodb"

mkdir -p ${DB_PATH}
mongod --dbpath ${DB_PATH} \
    --bind_ip 127.0.0.1 \
    --port 1234 \
    --directoryperdb \
    --journal \
    --nohttpinterface
