#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_FILE="${HOME}/.freqtrade_telegram_env"
CONFIG_FILE="${ROOT_DIR}/user_data/config_futures_dryrun.json"
PID_FILE="${ROOT_DIR}/user_data/run/freqtrade_futures_dryrun_a1.pid"
LOG_FILE="${ROOT_DIR}/user_data/logs/freqtrade_futures_dryrun_a1.log"
NOHUP_LOG="${ROOT_DIR}/user_data/logs/freqtrade_futures_dryrun_a1.nohup.log"
PREFLIGHT="${ROOT_DIR}/user_data/strategy_research/runtime/preflight_futures_runtime.py"
RISK_PREFLIGHT="${ROOT_DIR}/user_data/strategy_research/dryrun_strategy_risk_preflight.py"
ACTION="${1:-start}"

mkdir -p "${ROOT_DIR}/user_data/run" "${ROOT_DIR}/user_data/logs"

current_pid() {
    if [[ -f "${PID_FILE}" ]]; then
        tr -d '[:space:]' < "${PID_FILE}" || true
    fi
}

is_running() {
    local pid="$1"
    [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1
}

stop_bot() {
    local pid
    pid="$(current_pid)"
    if is_running "${pid}"; then
        kill "${pid}"
        for _ in {1..10}; do
            sleep 1
            if ! is_running "${pid}"; then
                break
            fi
        done
        if is_running "${pid}"; then
            echo "dry-run did not stop after SIGTERM; sending SIGKILL: pid=${pid}" >&2
            kill -KILL "${pid}"
            sleep 1
        fi
        if is_running "${pid}"; then
            echo "failed to stop dry-run after SIGKILL: pid=${pid}" >&2
            exit 1
        fi
        rm -f "${PID_FILE}"
        echo "stopped futures dry-run: pid=${pid}"
    else
        echo "futures dry-run is not running"
        rm -f "${PID_FILE}"
    fi
}

preflight_telegram() {
    "${ROOT_DIR}/.venv/bin/python" - <<'PY'
import json
import os
import sys
import urllib.parse
import urllib.request

token = os.environ.get("FREQTRADE__TELEGRAM__TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")
chat_id = os.environ.get("FREQTRADE__TELEGRAM__CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID")

if not token and not chat_id:
    print("telegram preflight skipped: token/chat_id not configured")
    raise SystemExit(0)
if not token or not chat_id:
    print("telegram preflight failed: token or chat_id missing", file=sys.stderr)
    raise SystemExit(1)

base = f"https://api.telegram.org/bot{token}"


def fetch(method: str, params: dict[str, str] | None = None) -> dict:
    url = f"{base}/{method}"
    data = None
    if params:
        data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(url, data=data, timeout=15) as response:
        return json.loads(response.read().decode())


try:
    me = fetch("getMe")
    chat = fetch("getChat", {"chat_id": chat_id})
except Exception as exc:
    print(f"telegram preflight failed: {type(exc).__name__}: {str(exc)[:240]}", file=sys.stderr)
    raise SystemExit(1)

if not me.get("ok") or not chat.get("ok"):
    print("telegram preflight failed: Telegram API returned ok=false", file=sys.stderr)
    raise SystemExit(1)

username = me.get("result", {}).get("username", "<unknown>")
chat_type = chat.get("result", {}).get("type", "<unknown>")
print(f"telegram preflight ok: bot=@{username}, chat_type={chat_type}")
PY
}

case "${ACTION}" in
    stop)
        stop_bot
        exit 0
        ;;
    status)
        pid="$(current_pid)"
        if is_running "${pid}"; then
            ps -p "${pid}" -o pid,ppid,stat,lstart,command
        else
            echo "futures dry-run is not running"
            exit 1
        fi
        exit 0
        ;;
    restart)
        stop_bot
        ;;
    start)
        ;;
    *)
        echo "usage: $0 [start|stop|restart|status]" >&2
        exit 2
        ;;
esac

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
else
    echo "warning: ${ENV_FILE} not found; starting without Telegram env" >&2
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "missing ${CONFIG_FILE}; copy runtime/config_futures_dryrun.template.json first" >&2
    exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(current_pid)"
    if is_running "${old_pid}"; then
        echo "dry-run already running: pid=${old_pid}"
        exit 0
    fi
fi

export PYTHONPATH="user_data/offline_exchange${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT_DIR}"
if [[ ! -f "${PREFLIGHT}" ]]; then
    echo "missing runtime preflight: ${PREFLIGHT}" >&2
    exit 1
fi
if [[ ! -f "${RISK_PREFLIGHT}" ]]; then
    echo "missing dry-run risk preflight: ${RISK_PREFLIGHT}" >&2
    exit 1
fi
"${ROOT_DIR}/.venv/bin/python" "${RISK_PREFLIGHT}" --config "${CONFIG_FILE}"
"${ROOT_DIR}/.venv/bin/python" "${PREFLIGHT}" --pair "BTC/USDT:USDT" --timeframe "15m"
preflight_telegram

"${ROOT_DIR}/.venv/bin/python" - "${ROOT_DIR}" "${PID_FILE}" "${NOHUP_LOG}" <<'PY'
import os
import pathlib
import subprocess
import sys
import time

root = pathlib.Path(sys.argv[1])
pid_file = pathlib.Path(sys.argv[2])
nohup_log = pathlib.Path(sys.argv[3])

stdout = open(nohup_log, "ab", buffering=0)
proc = subprocess.Popen(
    [
        "./.venv/bin/freqtrade",
        "trade",
        "-c",
        "user_data/config_futures_dryrun.json",
        "--logfile",
        "user_data/logs/freqtrade_futures_dryrun_a1.log",
    ],
    cwd=str(root),
    env=os.environ.copy(),
    stdout=stdout,
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True,
)
pid_file.write_text(f"{proc.pid}\n")
time.sleep(1)
if proc.poll() is not None:
    raise SystemExit(f"freqtrade exited early with code {proc.returncode}")
print(proc.pid)
PY

pid="$(tr -d '[:space:]' < "${PID_FILE}")"
echo "started futures dry-run: pid=${pid}"
echo "log: ${LOG_FILE}"
