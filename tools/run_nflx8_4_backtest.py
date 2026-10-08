"""NFlX8_4（NostalgiaForInfinityX8_8020）月度回测执行脚本。

用法（工作区根目录或任意位置均可，脚本会自动切到仓库根目录）：
    $env:PYTHONUTF8="1"; py -3.14 tools/run_nflx8_4_backtest.py 202401
    py -3.14 tools/run_nflx8_4_backtest.py 202401 202402 202403      # 多个月
    py -3.14 tools/run_nflx8_4_backtest.py all                       # 全部月份

输出：
    - 回测结果  -> user_data/backtest_results/NFlX8/
    - 运行日志  -> user_data/backtest_results/NFlX8/logs/<YYYYMM>.log
"""

import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

STRATEGY = "NostalgiaForInfinityX8_8020"
STRATEGY_PATH = "user_data/strategies/NFlX8"
CFG_DIR = "user_data/configs/binance/NFIX8Backtest"
CONFIGS = [
    "proxy-binance.json",
    "trading_mode-futures.json",
    "exampleconfig.json",
    "blacklist-binance.json",
    "pairlist-backtest-static-binance-futures-usdt.json",
]

OUT_DIR = ROOT / "user_data" / "backtest_results" / "NFlX8"
LOG_DIR = OUT_DIR / "logs"

START_YEAR, START_MONTH = 2024, 1
END_YEAR, END_MONTH = 2026, 8


def all_months():
    y, m = START_YEAR, START_MONTH
    out = []
    while (y, m) <= (END_YEAR, END_MONTH):
        out.append(f"{y}{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def timerange_of(ym: str):
    y, m = int(ym[:4]), int(ym[4:])
    start = f"{y}{m:02d}01"
    nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    return start, nxt.strftime("%Y%m%d")


def build_cmd(ym: str):
    start, end = timerange_of(ym)
    export = (OUT_DIR / f"binance_nflx8_4_full343_{ym}").as_posix()
    cmd = [sys.executable, "-m", "freqtrade", "backtesting"]
    cmd += ["--strategy", STRATEGY, "--strategy-path", STRATEGY_PATH]
    for name in CONFIGS:
        cmd += ["--config", f"{CFG_DIR}/{name}"]
    cmd += ["--timerange", f"{start}-{end}"]
    cmd += ["--export", "trades"]
    cmd += ["--export-filename", export]
    return cmd


def archive_result(ym: str, since_ts: float):
    """freqtrade 新版忽略 --export-filename 的路径，结果落在默认目录，这里归档到 NFlX8/。"""
    root = ROOT / "user_data" / "backtest_results"
    moved = []
    for pattern, suffix in (("backtest-result-*.zip", ".zip"), ("backtest-result-*.meta.json", ".meta.json")):
        cands = [
            p for p in root.glob(pattern)
            if p.is_file() and p.stat().st_mtime >= since_ts - 5
        ]
        if not cands:
            continue
        newest = max(cands, key=lambda p: p.stat().st_mtime)
        target = OUT_DIR / f"binance_nflx8_4_full343_{ym}{suffix}"
        if target.exists():
            target.unlink()
        newest.replace(target)
        moved.append(target.name)
    if moved:
        print(f"[{ym}] 已归档: {', '.join(moved)}")
    else:
        print(f"[{ym}] 警告：未找到新生成的结果文件（可能回测失败）")


def run_one(ym: str) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(ym)
    print(f"[{ym}] 开始回测  {timerange_of(ym)[0]}-{timerange_of(ym)[1]}")
    print(f"[{ym}] 命令: {' '.join(cmd)}", flush=True)
    started = time.time()

    log_path = LOG_DIR / f"{ym}.log"
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        proc.wait()

    print(f"[{ym}] 退出码: {proc.returncode}")
    print(f"[{ym}] 日志: {log_path}")
    if proc.returncode == 0:
        archive_result(ym, started)
    return proc.returncode


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    if args[0].lower() == "all":
        months = all_months()
    else:
        months = args

    print(f"待回测月份: {months}")
    codes = {}
    for ym in months:
        codes[ym] = run_one(ym)
    print("\n===== 汇总 =====")
    for ym, code in codes.items():
        print(f"  {ym}: {'成功' if code == 0 else '失败 code=' + str(code)}")
    return 0 if all(c == 0 for c in codes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
