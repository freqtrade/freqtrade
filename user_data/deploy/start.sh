#!/bin/bash

# Build the db-url argument from Railway's DATABASE_URL if available
DB_ARG=""
if [ -n "$DATABASE_URL" ]; then
    # Convert postgres:// to postgresql:// for SQLAlchemy
    DB_URL=$(echo "$DATABASE_URL" | sed 's|^postgres://|postgresql://|')
    DB_ARG="--db-url $DB_URL"
    echo "Using PostgreSQL database for trade tracking"
else
    echo "No DATABASE_URL found, using SQLite"
fi

exec freqtrade trade \
    --config /freqtrade/config_mexc.json \
    --strategy TrendRiderStrategy \
    --userdir /freqtrade/user_data \
    $DB_ARG
