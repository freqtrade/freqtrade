#!/bin/bash
# P18 Paper Ledger Report Utility

LEDGER_DIR="user_data/generated/paper_ledger"
TRADES_FILE="$LEDGER_DIR/paper_trades.csv"
DAILY_FILE="$LEDGER_DIR/paper_daily_summary.csv"

echo "======================================"
echo "      PAPER FORWARD TEST REPORT       "
echo "======================================"
echo "Timestamp: $(date)"
echo "Directory: $LEDGER_DIR"
echo ""

if [ ! -d "$LEDGER_DIR" ]; then
    echo "No paper ledger directory found."
    exit 0
fi

if [ -f "$DAILY_FILE" ]; then
    echo "--- Daily Summary (Last 5 Days) ---"
    # Show header + last 5 lines
    head -n 1 "$DAILY_FILE" | column -t -s,
    tail -n 5 "$DAILY_FILE" | column -t -s,
    echo ""
else
    echo "No daily summary file found."
fi

if [ -f "$TRADES_FILE" ]; then
    echo "--- Recent Trades (Last 10) ---"
    # Show header + last 10 lines. Use cut to limit width if needed, but let's just show it all.
    # Columns: utc_ts,local_ts_ist,order_id,symbol,side,amount,avg_price,base_price,slippage_bps,fee,notional
    head -n 1 "$TRADES_FILE"
    tail -n 10 "$TRADES_FILE"
    echo ""
else
    echo "No trades file found."
fi
