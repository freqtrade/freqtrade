"""补跑 NFlX8_4 尚未完成（缺结果包）的月份，并自动回填文档 + 重新汇总。

用法：
    $env:PYTHONUTF8="1"; py -3.14 tools/run_pending.py                 # 立即补跑
    $env:PYTHONUTF8="1"; py -3.14 tools/run_pending.py --wait 1800      # 先等 1800 秒（等 Binance 解封）再跑
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "user_data" / "backtest_results" / "NFlX8"
TOOLS = ROOT / "tools"

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


def pending_months():
    return [ym for ym in all_months() if not (OUT_DIR / f"binance_nflx8_4_full343_{ym}.zip").exists()]


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.run([sys.executable] + cmd, cwd=ROOT).returncode


def main():
    args = sys.argv[1:]
    if "--wait" in args:
        idx = args.index("--wait")
        secs = int(args[idx + 1])
        print(f"等待 {secs} 秒后开始（用于等 Binance 限流解封）...")
        time.sleep(secs)

    todo = pending_months()
    if not todo:
        print("所有月份均已完成，无需补跑。")
        return 0
    print(f"待补跑月份: {todo}")

    code = run(["tools/run_nflx8_4_backtest.py", *todo])
    done = [ym for ym in todo if (OUT_DIR / f"binance_nflx8_4_full343_{ym}.zip").exists()]
    print(f"\n已完成: {done}")

    for ym in done:
        run(["tools/update_docs_from_result.py", ym])

    run(["tools/summarize_all.py"])
    print("\n补跑与回填结束。")
    return code


if __name__ == "__main__":
    sys.exit(main())
