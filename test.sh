#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON:-python3}
VENV=".venv"
PLATFORM="$(uname -s)-$(uname -m)"

echo "[info] Python: $(${PY} -V)  Platform: ${PLATFORM}"

# 1) venv
if [ ! -d "${VENV}" ]; then
  ${PY} -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

echo "[step] Install dev requirements (this pulls freqai/hyperopt/docs extras)"
set +e
python -m pip install -r requirements-dev.txt
RC=$?
set -e

if [ $RC -ne 0 ]; then
  echo "[warn] requirements-dev.txt failed (RC=${RC}). Applying fallbacks..."

  # Common heavy deps fallbacks (Mac/arm64で失敗しやすい項目)
  # 1. torch (freqai-rl 由来) の事前導入（CPU版）
  if [[ "${PLATFORM}" == "Darwin-arm64" || "${PLATFORM}" == "Darwin-arm64e" ]]; then
    echo "[fallback] Installing torch (CPU) for macOS arm64"
    python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio || true
  else
    echo "[fallback] Installing torch (CPU) generic"
    python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio || true
  fi

  # 2. 科学計算系のホイールを先に入れる（ビルド回避）
  echo "[fallback] Preinstall scientific wheels"
  python -m pip install --only-binary=:all: "numpy>=1.26" "scipy>=1.11" "pandas>=2.2" "pyarrow>=15" || true

  # 3. もう一度 dev 要件にトライ
  echo "[retry] pip install -r requirements-dev.txt"
  python -m pip install -r requirements-dev.txt
fi

echo "[step] Install extension (free) extras"
if [ -f requirements-ext.txt ]; then
  python -m pip install -r requirements-ext.txt
fi

echo "[ok] Setup complete. To activate: source .venv/bin/activate"sh
