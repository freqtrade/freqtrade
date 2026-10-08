"""扫描 user_data/data/binance 下的数据文件，统计数量与时间范围。

只读脚本，不做任何修改。运行方式（工作区根目录）：
    $env:PYTHONUTF8="1"; py -3.14 tools/check_data_range.py
"""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.feather as feather


def fmt(ts):
    if ts is None:
        return "-"
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M")
    return str(ts)


def scan_dir(base: Path, label: str):
    if not base.is_dir():
        print(f"[{label}] 目录不存在: {base}")
        return

    files = sorted(base.glob("*.feather"))
    kinds = Counter()
    tfs = Counter()
    for f in files:
        name = f.name
        if name.endswith("-futures.feather"):
            kinds["futures"] += 1
            tfs[name.split("-")[-2]] += 1
        elif name.endswith("-mark.feather"):
            kinds["mark"] += 1
            tfs[name.split("-")[-2]] += 1
        elif name.endswith("-funding_rate.feather"):
            kinds["funding_rate"] += 1
            tfs[name.split("-")[-2]] += 1
        else:
            kinds["spot/other"] += 1
            tfs[name.split("-")[-1].replace(".feather", "")] += 1

    print(f"\n=== {label} ===")
    print(f"文件总数: {len(files)}")
    print(f"类型分布: {dict(kinds)}")
    print(f"周期分布: {dict(sorted(tfs.items()))}")

    ohlcv = sorted(base.glob("*-5m-futures.feather")) or sorted(base.glob("*-5m.feather"))
    print(f"5m 主周期文件数: {len(ohlcv)}")

    rows = []
    for f in ohlcv:
        try:
            table = feather.read_table(f, columns=["date"])
            dates = table.column("date").to_pylist()
        except Exception as exc:  # noqa: BLE001
            print(f"  读取失败 {f.name}: {exc}")
            continue
        if not dates:
            continue
        rows.append((f.name, dates[0], dates[-1], len(dates)))

    if not rows:
        print("没有可读取的 5m 数据")
        return

    g_start = min(r[1] for r in rows)
    g_end = max(r[2] for r in rows)
    print(f"全局起始: {fmt(g_start)}")
    print(f"全局截止: {fmt(g_end)}")

    end_counter = Counter()
    for _n, _s, e, _c in rows:
        if isinstance(e, datetime):
            end_counter[e.strftime("%Y-%m")] += 1
    print("\n各币数据截止月份分布（前 15）:")
    for month, cnt in sorted(end_counter.items())[:15]:
        print(f"  {month}: {cnt} 个币")

    print("\n按起始月份分布（前 15）:")
    start_counter = Counter()
    for _n, s, _e, _c in rows:
        if isinstance(s, datetime):
            start_counter[s.strftime("%Y-%m")] += 1
    for month, cnt in sorted(start_counter.items())[:15]:
        print(f"  {month}: {cnt} 个币")

    print("\n示例（前 8 个币，按文件名排序）:")
    print(f"  {'交易对':<28}{'起始':<18}{'截止':<18}{'K线数'}")
    for name, s, e, c in rows[:8]:
        print(f"  {name:<28}{fmt(s):<18}{fmt(e):<18}{c}")


def main():
    root = Path("user_data/data/binance")
    scan_dir(root / "futures", "合约数据 futures/")
    scan_dir(root, "现货数据 binance/")

    now = datetime.now(timezone.utc)
    print(f"\n当前 UTC 时间: {fmt(now)}")


if __name__ == "__main__":
    main()
