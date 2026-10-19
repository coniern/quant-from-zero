"""
03 · 四种策略同台竞技

运行：python examples/03_compare_strategies.py

同样的数据、同样的成本、同样的评估标准 —— 唯一的变量是策略。
这是做策略研究的基本姿势：**控制变量**。

看完你会得到一个反直觉的结论：最简单的策略未必最差，
但所有择时策略都逃不过同一个困境（在结果里找）。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import run_backtest, trade_stats, yearly_returns
from src.data import load_daily
from src.metrics import format_summary, summary
from src.strategies import (STRATEGIES, breakout, ma_cross, mean_reversion,
                            momentum)
from src.config import apply_chinese_font

TICKERS = {
    "600519": "贵州茅台",
    "000300": "沪深300",
    "399006": "创业板指",
}


def run_all(symbol: str, kind: str = "stock") -> pd.DataFrame:
    prices = load_daily(symbol, "2020-01-01", "2024-12-31", kind=kind)
    strategies = {
        "双均线5/20": ma_cross(prices["close"], 5, 20),
        "双均线20/60": ma_cross(prices["close"], 20, 60),
        "通道突破20": breakout(prices["close"], 20),
        "动量20": momentum(prices["close"], 20),
        "均值回归20": mean_reversion(prices["close"], 20),
    }

    rows = {}
    for name, sig in strategies.items():
        res = run_backtest(prices, sig)
        s = summary(res["returns"], res["benchmark_returns"])
        t = trade_stats(res["trades"])
        rows[name] = {
            "年化收益": s["年化收益"],
            "最大回撤": s["最大回撤"],
            "夏普": s["夏普比率"],
            "交易次数": t["交易次数"],
            "买入持有": s["基准累计收益"],
        }
    return pd.DataFrame(rows).T, prices


apply_chinese_font()
for symbol, label in TICKERS.items():
    kind = "index" if symbol in ("000300", "399006") else "stock"
    table, prices = run_all(symbol, kind)
    print("\n" + "=" * 70)
    print(f"{label}（{symbol}）2020-2024，成本已计入")
    print("=" * 70)
    print(table.to_string(float_format=lambda x: f"{x:8.3f}"))

print("""
怎么读这张表：

1. 大多数策略跑不赢"买入持有" —— 这不是 bug，是常态。
   长期向上的资产，择时的收益是「少赚」；择时的价值在于「少亏」。

2. 对比「最大回撤」时要清醒：
   有的组合确实把回撤压浅了（看茅台的双均线 20/60），
   但创业板指上的双均线 5/20 回撤依然有 -40%。
   择时不是保险丝，它只在"跌得干脆"的行情里有用，
   阴跌不止的行情里它会被反复止损磨死。

3. 看交易次数：5/20 的双均线交易几十次，成本吃掉不少利润；
   20/60 交易少，单次利润更厚。参数短的策略对噪音更敏感。

4. 没有一个策略在三个标的上都最强。
   「参数/策略 × 标的 × 时间段」三维扫描后还能稳定盈利的，才值得信。
   单一组合的漂亮曲线，大概率是运气。

结论：这一页的意义不是"哪个策略最好"，
而是让你看到「换个标的、换个参数，结论就反转」——
这就是为什么量化研究必须做稳健性检验，而不是只看一条曲线。
""")
