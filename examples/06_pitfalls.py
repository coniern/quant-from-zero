"""
06 · 回测的七个坑（本仓库最重要的一课）

运行：python examples/06_pitfalls.py

大部分"年化 80% 的策略"不是骗子，是**不小心作弊了**。
这个脚本用同一份数据、同一个策略，让你亲眼看见作弊和诚实差多少。
看完你会对任何"高收益回测"保持警惕 —— 这是做量化最重要的一种直觉。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import run_backtest
from src.data import load_daily
from src.metrics import annual_return, max_drawdown, sharpe
from src.strategies import ma_cross

prices = load_daily("600519", "2020-01-01", "2024-12-31")
close = prices["close"]
signal = ma_cross(close, fast=5, slow=20)


def brief(name: str, result: dict) -> None:
    r = result["returns"]
    print(f"{name:<28} 年化 {annual_return(r):>8.2%}   "
          f"夏普 {sharpe(r):>6.2f}   最大回撤 {max_drawdown(result['equity']):>8.2%}")


print("=" * 72)
print("同一策略，两种写法，结果天差地别")
print("=" * 72)

# ---------- 坑 1：未来函数（最致命）----------
honest = run_backtest(prices, signal, lag=1)
cheat = run_backtest(prices, signal, lag=0)
brief("✅ 诚实：信号延迟 1 天", honest)
brief("❌ 作弊：当天信号当天用", cheat)
print("   ↑ 只是少 shift 一天，年化就凭空多出一大截。这就是未来函数。\n")

# ---------- 坑 2：直接偷看未来 ----------
# 信号在 t 日的值 = t+1 日的涨跌方向（提前一天偷看答案）
future = close.pct_change().shift(-1)              # 未来一天的收益
oracle_signal = (future > 0).shift(1).astype(float)
oracle = run_backtest(prices, oracle_signal, lag=0)
brief("❌ 先知：提前知道涨跌再下单", oracle)
print("   ↑ 有了'先知'连手续费都打不垮它。任何回测里出现这种曲线，"
      "去查特征是不是泄露了未来数据。\n")

# ---------- 坑 3：忽略交易成本 ----------
no_cost = run_backtest(prices, signal, commission=0, slippage=0, stamp_tax=0)
brief("✅ 真实成本（万2.5+千0.5+滑点）", honest)
brief("❌ 零成本", no_cost)
print("   ↑ 双均线一年换手十几次，成本能吃掉几个点。高频策略尤其致命。\n")

# ---------- 坑 4：只挑最好的参数（过拟合）----------
print("=" * 72)
print("参数扫描：在样本内挑最优，等于偷看答案")
print("=" * 72)

rows = []
for fast in [3, 5, 10, 20]:
    for slow in [20, 30, 60]:
        if fast >= slow:
            continue
        sig = ma_cross(close, fast, slow)
        res = run_backtest(prices, sig)
        rows.append({
            "fast": fast, "slow": slow,
            "全样本年化": annual_return(res["returns"]),
            "全样本夏普": sharpe(res["returns"]),
        })
grid = pd.DataFrame(rows).sort_values("全样本夏普", ascending=False)
print(grid.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

best = grid.iloc[0]
print(f"\n样本内最优：MA{int(best['fast'])}/MA{int(best['slow'])}，"
      f"夏普 {best['全样本夏普']:.2f}")

# 用前 3 年选参数，后 2 年验证 —— 这才是诚实的做法
split = "2023-01-01"
train, test = prices.loc[:split], prices.loc[split:]

train_best, best_sharpe = None, -99
for fast in [3, 5, 10, 20]:
    for slow in [20, 30, 60]:
        if fast >= slow:
            continue
        r = run_backtest(train, ma_cross(train["close"], fast, slow))["returns"]
        s = sharpe(r)
        if s > best_sharpe:
            train_best, best_sharpe = (fast, slow), s

f, s_ = train_best
oos = run_backtest(test, ma_cross(test["close"], f, s_))["returns"]
print(f"只用 2020-2022 选参数 → 选出 MA{f}/MA{s_}（训练夏普 {best_sharpe:.2f}）")
print(f"拿到 2023-2024 的样本外表现 → 夏普 {sharpe(oos):.2f}，"
      f"年化 {annual_return(oos):.2%}")
print("   ↑ 样本内夏普和样本外夏普的落差，就是过拟合的程度。落差越大，越不可信。\n")

# ---------- 坑 5：幸存者偏差（用文字说明）----------
print("=" * 72)
print("还有三个坑，代码演示不了，但你必须知道")
print("=" * 72)
print("""
5. 幸存者偏差
   你现在能拉到的股票池，是「今天还活着」的股票。
   退市的、被 ST 的、暴雷的都不在里面。
   用这个池子回测，等于提前剔除了所有失败者 —— 收益必然虚高。
   → 本仓库 examples/04 的 20 只票就有这个问题，我在文档里标出来了。

6. 复权方式选错
   用不复权价格算收益，遇到分红送股那天会凭空多出 -30%。
   详见 docs/03-数据.md。

7. 忽略流动性
   回测假设"想买多少买多少"。实际小盘股一天成交几千万，
   你要买 5000 万就得买好几天，价格早跑了。
   → 资金量越大，策略容量越小。这是资金规模的天花板。
""")
