#!/usr/bin/env bash
# ============================================================================
# GA Server Runner — tmux-based session management
# ============================================================================
# Creates a named tmux session with the GA running in the main pane
# and a log tail in a split pane below.
#
# Usage:
#   ./start_server_run.sh                    # Standard single-population run
#   ./start_server_run.sh island             # Island model run
#   ./start_server_run.sh nsga2              # NSGA-II run
#   ./start_server_run.sh resume <path>      # Resume from checkpoint
#   ./start_server_run.sh dashboard          # Start web dashboard only
#
# Monitor from remote:
#   ssh user@server 'tmux attach -t ga_evolution'
#
# ============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
SESSION_NAME="ga_evolution"

# ── Configuration mapping ──
MODE="${1:-standard}"
RESUME_PATH="${2:-}"

case "$MODE" in
    standard)
        CONFIG="${REPO_DIR}/genetic_algorithm/config/ga_config_server_6core.yaml"
        LOG_FILE="${REPO_DIR}/genetic_algorithm/logs/ga_server.log"
        DESC="Standard single-population (server-optimized)"
        ;;
    island)
        CONFIG="${REPO_DIR}/genetic_algorithm/config/ga_config_server_island.yaml"
        LOG_FILE="${REPO_DIR}/genetic_algorithm/logs/ga_island_server.log"
        DESC="Island model with parallel islands"
        ;;
    nsga2)
        CONFIG="${REPO_DIR}/genetic_algorithm/config/ga_config_server_nsga2.yaml"
        LOG_FILE="${REPO_DIR}/genetic_algorithm/logs/ga_server_nsga2.log"
        DESC="NSGA-II multi-objective"
        ;;
    resume)
        if [[ -z "$RESUME_PATH" ]]; then
            echo "Usage: $0 resume <checkpoint_path>"
            exit 1
        fi
        CONFIG="${REPO_DIR}/genetic_algorithm/config/ga_config_server_6core.yaml"
        LOG_FILE="${REPO_DIR}/genetic_algorithm/logs/ga_server.log"
        DESC="Resume from checkpoint: $RESUME_PATH"
        ;;
    dashboard)
        CONFIG=""
        LOG_FILE=""
        DESC="Web dashboard only"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Options: standard, island, nsga2, resume <path>, dashboard"
        exit 1
        ;;
esac

# ── Preflight checks ──
if ! command -v tmux &>/dev/null; then
    echo "ERROR: tmux is not installed. Install with: sudo apt install tmux"
    exit 1
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "ERROR: Virtual environment not found at ${VENV_DIR}"
    exit 1
fi

# Kill existing session if present
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists."
    echo "  Attach: tmux attach -t $SESSION_NAME"
    echo "  Kill:   tmux kill-session -t $SESSION_NAME"
    echo ""
    read -p "Kill existing session and start new one? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION_NAME"
    else
        echo "Attaching to existing session..."
        tmux attach -t "$SESSION_NAME"
        exit 0
    fi
fi

# Ensure log directory exists
mkdir -p "${REPO_DIR}/genetic_algorithm/logs"

# ── Build GA command ──
if [[ "$MODE" == "dashboard" ]]; then
    GA_CMD="cd ${REPO_DIR} && source ${VENV_DIR}/bin/activate && python -m genetic_algorithm.web.server"
    TAIL_CMD="echo 'Dashboard running at http://0.0.0.0:8501'"
elif [[ "$MODE" == "resume" ]]; then
    GA_CMD="cd ${REPO_DIR} && source ${VENV_DIR}/bin/activate && python genetic_algorithm/run_ga.py --config ${CONFIG} --no-monitor --yes --resume ${RESUME_PATH} 2>&1 | tee -a ${LOG_FILE}"
    TAIL_CMD="tail -f ${LOG_FILE}"
else
    GA_CMD="cd ${REPO_DIR} && source ${VENV_DIR}/bin/activate && python genetic_algorithm/run_ga.py --config ${CONFIG} --no-monitor --yes 2>&1 | tee -a ${LOG_FILE}"
    TAIL_CMD="tail -f ${LOG_FILE}"
fi

# ── Create tmux session ──
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  GA Server Runner                                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Mode:    $DESC"
echo "  Config:  $CONFIG"
echo "  Log:     $LOG_FILE"
echo "  Session: $SESSION_NAME"
echo ""

# Create session with GA running in main pane
tmux new-session -d -s "$SESSION_NAME" -n "ga" "$GA_CMD"

# Add split pane with log tail (only if there's a log to tail)
if [[ -n "$LOG_FILE" ]]; then
    sleep 1  # Give the GA a moment to create the log file
    touch "$LOG_FILE"
    tmux split-window -t "$SESSION_NAME:ga" -v -p 30 "$TAIL_CMD"
    tmux select-pane -t "$SESSION_NAME:ga.0"  # Focus back on GA pane
fi

echo "Session created! Attaching..."
echo ""
echo "  Detach:  Ctrl+B, then D"
echo "  Re-attach from remote:  tmux attach -t $SESSION_NAME"
echo ""

tmux attach -t "$SESSION_NAME"
