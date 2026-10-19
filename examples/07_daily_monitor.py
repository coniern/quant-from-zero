"""
风险与回撤监控：每天收盘后跑一次，看看策略今天怎么样。

这是「从回测到实盘」的桥梁：回测是研究过去，监控是照看现在。
实盘的第一阶段永远是「纸面交易」（paper trading）——
不发真钱，每天按收盘价模拟成交，和回测结果对比。
偏差超过阈值就停下来找原因，而不是加仓。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import run_backtest, rolling_sharpe, yearly_returns
from src.data import load_daily
from src.metrics import format_summary, summary
from src.strategies import ma_cross


def daily_check(symbol: str = "600519") -> None:
    """每天收盘后跑一次：重算信号 + 检查策略状态。"""
    prices = load_daily(symbol, "2023-01-01", "2024-12-31")
    signal = ma_cross(prices["close"], 5, 20)
    result = run_backtest(prices, signal)

    today = prices.index[-1]
    print(f"检查日期：{today.date()}")
    print(f"收盘价：{prices['close'].iloc[-1]:.2f}")
    print(f"当前信号：{'持仓' if signal.iloc[-1] > 0 else '空仓'}")
    print(f"净值：{result['equity'].iloc[-1]:.4f}")
    print(f"距一年高点：{result['equity'].iloc[-1] / result['equity'].iloc[-252:].max() - 1:.2%}")
    print(f"滚动一年夏普：{rolling_sharpe(result['returns']).iloc[-1]:.2f}")
    print(format_summary(summary(result["returns"], result["benchmark_returns"])))


if __name__ == "__main__":
    daily_check()
