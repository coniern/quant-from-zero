"""
02 · 手算指标，和库算的对一遍

运行：python examples/02_metrics.py

目的只有一个：**证明你理解了公式。**
用一个手算得出的小例子，逐项和 src/metrics.py 的输出对比。
对不上就说明你（或我）理解错了 —— 这种"互相验证"是量化研究的基本功。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import (annual_return, annual_vol, equity_from_returns,
                         max_drawdown, sharpe, total_return, win_rate)

# ---- 一段可以口算的数据：4 天，收益 +10%、-10%、+10%、-10% ----
prices = [100, 110, 99, 108.9, 98.01]
r = [0.10, -0.10, 0.10, -0.10]

import pandas as pd
returns = pd.Series(r, dtype=float)

# ---------------- 手算 ----------------
total_hand = (1 + r[0]) * (1 + r[1]) * (1 + r[2]) * (1 + r[3]) - 1
annual_hand = (1 + total_hand) ** (252 / 4) - 1

import numpy as np
vol_hand = np.std(r, ddof=1) * np.sqrt(252)
mdd_hand = min([0,
                (110 - 110) / 110,          # 第 2 天是高点
                (99 - 110) / 110,           # 第 3 天离高点 -10%
                (98.01 - 110) / 110,        # 第 4 天离高点 -10.9%
                ])

print("=" * 64)
print(f"累计收益   手算 {total_hand:>9.4f}   代码 {total_return(returns):>9.4f}")
print(f"年化收益   手算 {annual_hand:>9.4f}   代码 {annual_return(returns):>9.4f}")
print(f"年化波动   手算 {vol_hand:>9.4f}   代码 {annual_vol(returns):>9.4f}")
print(f"最大回撤   手算 {mdd_hand:>9.4f}   代码 {max_drawdown(equity_from_returns(returns)):>9.4f}")
print(f"日胜率     手算 {0.5:>9.4f}   代码 {win_rate(returns):>9.4f}")
print(f"夏普比率   代码 {sharpe(returns):>9.4f}（手算：(均值-无风险/252)/标准差×√252）")
print("=" * 64)

# ---------------- 三个必须想通的问题 ----------------
print("""
看仔细：这 4 天里涨的日子和跌的日子一样多、幅度一样大，
但累计收益是 {:.2%}，不是 0！

这就是几何平均和算术平均的区别：
    算术平均：(10% - 10% + 10% - 10%) / 4 = 0%
    几何平均：(1.1 × 0.9 × 1.1 × 0.9) - 1 = -1.9%

先跌 50% 再涨 50%，你回了本吗？没有 —— (1-0.5)×(1+0.5) = 0.75，还亏 25%。
「亏损需要更大的涨幅才能补回来」，这是所有风控的数学起点。
""".format(total_hand))

# ---------------- 用真实数据再算一遍 ----------------
from src.data import load_daily

prices_real = load_daily("600519", "2020-01-01", "2024-12-31")["close"]
rets = prices_real.pct_change().dropna()

print(f"茅台 2020-2024（真实数据）")
print(f"  累计收益  {total_return(rets):>8.2%}")
print(f"  年化收益  {annual_return(rets):>8.2%}")
print(f"  年化波动  {annual_vol(rets):>8.2%}")
print(f"  最大回撤  {max_drawdown(equity_from_returns(rets)):>8.2%}")
print(f"  夏普比率  {sharpe(rets):>8.3f}")
