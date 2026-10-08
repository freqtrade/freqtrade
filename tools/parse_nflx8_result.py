"""解析 NFlX8_4 回测结果包（zip），输出月度汇总指标与逐笔明细。

用法：
    $env:PYTHONUTF8="1"; py -3.14 tools/parse_nflx8_result.py 202401

输入：user_data/backtest_results/NFlX8/binance_nflx8_4_full343_<YYYYMM>.zip
输出：
  - 控制台打印汇总指标 + 逐笔 markdown 表行
  - user_data/backtest_results/NFlX8/<YYYYMM>_summary.json
  - user_data/backtest_results/NFlX8/<YYYYMM>_trades.md
"""

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "user_data" / "backtest_results" / "NFlX8"
STRATEGY = "NostalgiaForInfinityX8_8020"


def load_trades(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        if not names:
            raise SystemExit(f"{zip_path} 内没有 json")
        data = json.loads(zf.read(names[0]))
    key = None
    for k in data.get("strategy", {}):
        if k == STRATEGY or k.startswith("NostalgiaForInfinityX8"):
            key = k
            break
    if key is None:
        raise SystemExit(f"未找到策略数据，现有: {list(data.get('strategy', {}))}")
    return data["strategy"][key].get("trades", []), data


def fmt_dt(ms):
    if ms is None:
        return "-"
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def stake_split(t: dict):
    """把一笔交易的保证金（stake_amount）拆成「初始投入」与「补仓投入」。

    回测包里 orders 没有 stake_amount 字段，只有 cost（名义成交额，含手续费、含杠杆），
    验证过：stake_amount == sum(entry cost) / (1 + fee) / leverage。
    因此这里按各次入场 order 的 cost 占比把 stake_amount 分摊，保证 初始+补仓 == stake_amount。
    返回 (初始投入, 补仓投入)。
    """
    orders = t.get("orders") or []
    entries = [o for o in orders if o.get("ft_is_entry")]
    total = float(t.get("stake_amount") or 0.0)
    if not entries or total <= 0:
        return total, 0.0
    costs = []
    for o in entries:
        c = o.get("cost")
        if not c:
            c = (o.get("amount") or 0) * (o.get("safe_price") or 0)
        costs.append(float(c or 0.0))
    s = sum(costs)
    if s <= 0:
        return total, 0.0
    init = total * costs[0] / s
    return init, total - init


def summarize(trades, raw):
    n = len(trades)
    long_n = sum(1 for t in trades if not t.get("is_short"))
    short_n = n - long_n
    profit = sum(t.get("profit_abs", 0) for t in trades)
    wins = sum(1 for t in trades if t.get("profit_abs", 0) > 0)
    draws = sum(1 for t in trades if t.get("profit_abs", 0) == 0)
    winrate = (wins / n * 100) if n else 0.0
    long_pl = sum(t.get("profit_abs", 0) for t in trades if not t.get("is_short"))
    short_pl = sum(t.get("profit_abs", 0) for t in trades if t.get("is_short"))
    reentries = 0
    for t in trades:
        orders = t.get("orders") or []
        entries = sum(1 for o in orders if o.get("ft_is_entry"))
        reentries += max(entries - 1, 0)
    tot_profit_pct = sum(t.get("profit_ratio", 0) * 100 for t in trades)
    # 账户级总收益率：以 1000 U 起始资金计算
    total_return_pct = profit / 1000 * 100
    stake_total = 0.0
    stake_init = 0.0
    stake_add = 0.0
    for t in trades:
        init, add = stake_split(t)
        stake_total += init + add
        stake_init += init
        stake_add += add
    stake_add_pct = (stake_add / stake_total * 100) if stake_total else 0.0
    return {
        "总投入(USDT)": round(stake_total, 2),
        "初始投入(USDT)": round(stake_init, 2),
        "补仓投入(USDT)": round(stake_add, 2),
        "补仓占比": f"{stake_add_pct:.2f}%",
        "交易数": n,
        "做多笔数": long_n,
        "做空笔数": short_n,
        "总利润(USDT)": round(profit, 2),
        "总收益率": f"{total_return_pct:.2f}%",
        "胜率": f"{winrate:.2f}%",
        "盈利笔数": wins,
        "平局笔数": draws,
        "多头盈亏(USDT)": round(long_pl, 2),
        "空头盈亏(USDT)": round(short_pl, 2),
        "总补仓次数": reentries,
        "累计收益率(逐笔相加)": f"{tot_profit_pct:.2f}%",
    }


def trades_markdown(trades):
    rows = [
        "| 序号 | 交易对 | 方向 | 买入时间 | 卖出时间 | 持仓(小时) | 补仓次数 | "
        "初始投入 | 补仓投入 | 买入价 | 卖出价 | 盈亏USDT | 收益率 | 入场信号 | 退出类型 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, t in enumerate(trades, 1):
        orders = t.get("orders") or []
        entries = sum(1 for o in orders if o.get("ft_is_entry"))
        reentry = max(entries - 1, 0)
        init_stake, add_stake = stake_split(t)
        open_ms, close_ms = t.get("open_timestamp"), t.get("close_timestamp")
        hours = ((close_ms - open_ms) / 3600000) if (open_ms and close_ms) else 0
        rows.append(
            "| {i} | {pair} | {side} | {ot} | {ct} | {hrs:.1f} | {re} | {init:.2f} | {add:.2f} | "
            "{op:.4f} | {cp:.4f} | {pl:+.2f} | {pr:+.2f}% | {tag} | {reason} |".format(
                i=i,
                pair=t.get("pair", "-"),
                side="做空" if t.get("is_short") else "做多",
                ot=fmt_dt(open_ms),
                ct=fmt_dt(close_ms),
                hrs=hours,
                re=reentry,
                init=init_stake,
                add=add_stake,
                op=t.get("open_rate", 0),
                cp=t.get("close_rate", 0),
                pl=t.get("profit_abs", 0),
                pr=t.get("profit_ratio", 0) * 100,
                tag=t.get("enter_tag") or "-",
                reason=t.get("exit_reason") or "-",
            )
        )
    return "\n".join(rows)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    ym = sys.argv[1]
    zip_path = OUT_DIR / f"binance_nflx8_4_full343_{ym}.zip"
    if not zip_path.exists():
        raise SystemExit(f"找不到结果包: {zip_path}")

    trades, raw = load_trades(zip_path)
    summary = summarize(trades, raw)

    print("===== 月度汇总 =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    md = trades_markdown(trades)
    print("\n===== 逐笔明细(markdown) =====")
    print(md)

    (OUT_DIR / f"{ym}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / f"{ym}_trades.md").write_text(md + "\n", encoding="utf-8")
    print(f"\n已保存: {OUT_DIR / (ym + '_summary.json')}")
    print(f"已保存: {OUT_DIR / (ym + '_trades.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
