import re
from collections import defaultdict

path = r"d:\量化分析软件\freqt\freqtrade\user_data\configs\binance\doc\NFlX8_4每月回测明细.md"

sig_pnl = defaultdict(float)      # 精确信号串 -> 总盈亏
sig_cnt = defaultdict(int)
tag_pnl = defaultdict(float)      # 拆分到单个 tag -> 总盈亏
tag_cnt = defaultdict(int)
exit_pnl = defaultdict(float)
exit_cnt = defaultdict(int)
exit_loss = defaultdict(float)

with open(path, encoding="utf-8") as f:
    for line in f:
        # 交易行：以 | 1 | 开头（序号为数字）
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        parts = [p.strip() for p in line.split("|")]
        # parts: ['', 序号, 交易对, 方向, 买, 卖, 持仓, 补仓次数, 初始, 补仓, 买价, 卖价, 盈亏, 收益率, 信号, 退出, '']
        if len(parts) < 16:
            continue
        pnl_s = parts[12].replace(",", "")
        sig_s = parts[14]
        exit_s = parts[15]
        try:
            pnl = float(pnl_s)
        except ValueError:
            continue
        # 精确信号串聚合
        sig_pnl[sig_s] += pnl
        sig_cnt[sig_s] += 1
        # 拆分到单个 tag
        tags = sig_s.split()
        for t in tags:
            tag_pnl[t] += pnl
            tag_cnt[t] += 1
        # 退出类型：取括号前的主体
        ex = exit_s.split("(")[0].strip()
        exit_pnl[ex] += pnl
        exit_cnt[ex] += 1
        if pnl < 0:
            exit_loss[ex] += pnl

print("=== 单个信号 tag 盈亏排序（净亏损的）===")
rows = [(t, tag_pnl[t], tag_cnt[t]) for t in tag_pnl if tag_pnl[t] < 0]
rows.sort(key=lambda x: x[1])
print(f"{'信号':<10}{'笔数':>6}{'净盈亏USDT':>14}")
for t, p, c in rows:
    print(f"{t:<10}{c:>6}{p:>14.2f}")

print("\n=== 精确信号串（含组合）净亏损的 ===")
rows2 = [(s, sig_pnl[s], sig_cnt[s]) for s in sig_pnl if sig_pnl[s] < 0]
rows2.sort(key=lambda x: x[1])
print(f"{'信号串':<16}{'笔数':>6}{'净盈亏USDT':>14}")
total_loss = 0.0
for s, p, c in rows2:
    total_loss += p
    print(f"{s:<16}{c:>6}{p:>14.2f}")
print(f"{'合计':<16}{'':>6}{total_loss:>14.2f}")

print("\n=== 退出类型盈亏 ===")
for ex in sorted(exit_pnl, key=lambda x: exit_pnl[x]):
    print(f"{ex:<40}{exit_cnt[ex]:>5}笔  净{exit_pnl[ex]:>10.2f}  亏损笔合计{exit_loss[ex]:>10.2f}")
