"""
01 · 30 行跑通人生第一个回测

运行：python examples/01_hello_backtest.py

这一步不要纠结原理，先让屏幕上出现一条净值曲线。
有了正反馈，后面的学习才撑得住。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import run_backtest
from src.data import load_daily
from src.metrics import format_summary, summary
from src.report import plot_equity
from src.strategies import ma_cross

# 1. 拿数据：贵州茅台，2020-2024
prices = load_daily("600519", "2020-01-01", "2024-12-31")
print(f"数据：{len(prices)} 个交易日，"
      f"{prices.index.min().date()} → {prices.index.max().date()}")

# 2. 出信号：5 日均线上穿 20 日均线就满仓，下穿就空仓
signal = ma_cross(prices["close"], fast=5, slow=20)
print(f"持仓天数占比：{signal.mean():.1%}")

# 3. 回测
result = run_backtest(prices, signal)

# 4. 看结果
print("\n" + format_summary(summary(result["returns"],
                                    result["benchmark_returns"])))
print(f"\n交易次数：{len(result['trades'])}")

path = plot_equity(result, title="茅台 5/20 双均线", name="01_equity.png")
print(f"\n图已保存：{path}")
