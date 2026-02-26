#!/bin/bash

DB_ARG=""
if [ -n "$DATABASE_URL" ]; then
    DB_URL=$(echo "$DATABASE_URL" | sed 's|^postgres://|postgresql://|')
    DB_ARG="--db-url $DB_URL"
    echo "✅ PostgreSQL connected"
fi

exec freqtrade trade \
    --config /freqtrade/config_5m.json \
    --strategy TrendRider5mStrategy \
    --userdir /freqtrade/user_data \
    $DB_ARG
