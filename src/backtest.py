"""
回测引擎：60 行核心代码，每一行都讲清楚。

这是整个仓库的心脏。市面上的教程会让你去装 backtrader / vnpy / zipline，
然后花三天配环境、读文档、被版本冲突搞崩溃 —— 最后你还是不知道回测在算什么。

真相是：**一个日频、单标的、全仓进出的回测引擎，核心逻辑不到 30 行。**
你先把这 30 行看懂、改坏、修好，再去用框架，才知道框架在帮你做什么。

核心就三行：
    pos     = signal.shift(1)        # 昨天的信号今天才生效（防止未来函数）
    turnover = pos.diff().abs()      # 换了多少仓，用来算成本
    ret     = pos * 日收益 - 成本     # 策略每天赚多少
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (COMMISSION_RATE, SLIPPAGE_RATE, STAMP_TAX_RATE,
                     TRADING_DAYS)


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    commission: float = COMMISSION_RATE,
    slippage: float = SLIPPAGE_RATE,
    stamp_tax: float = STAMP_TAX_RATE,
    initial_capital: float = 1.0,
    lag: int = 1,
) -> dict:
    """跑一次回测。

    参数
    ----
    prices  : 至少包含 close 列，日期索引
    signals : 目标仓位，取值 [0, 1]（0=空仓，1=满仓）。
              索引必须和 prices 一致。**信号用当天收盘价算出，代表"明天开始持有"**。
    lag     : 信号延迟生效的天数，正常交易必须是 1。
              传 0 会引入未来函数 —— 只用来演示"作弊长什么样"，见 examples/06。

    返回
    ----
    dict: equity(净值), returns(日收益), position(实际仓位), trades(交易记录),
          turnover(换手), cost(成本), benchmark(买入持有净值)
    """
    close = prices["close"]
    ret = close.pct_change().fillna(0)

    # ★ 最关键的一行：信号右移一天。
    # 你今天收盘后才知道"均线金叉了"，最早也只能明天开盘买入。
    # 不 shift 就是把未来信息泄露给过去 —— 回测收益会好得离谱，实盘一分钱赚不到。
    pos = signals.reindex(close.index).ffill().shift(lag).fillna(0.0)

    # 换手率：仓位变化的绝对值。0.3→1.0 是买入 0.7，1.0→0 是卖出 1.0
    turnover = pos.diff().abs().fillna(pos.abs())

    # 成本 = 换手 ×（佣金 + 滑点）+ 卖出金额 × 印花税
    # 卖出怎么判断？仓位变小就是在卖。
    sell_amount = (-pos.diff()).clip(lower=0).fillna(0.0)
    cost = turnover * (commission + slippage) + sell_amount * stamp_tax

    strat_ret = pos * ret - cost
    equity = initial_capital * (1 + strat_ret).cumprod()
    benchmark = initial_capital * (1 + ret).cumprod()

    return {
        "equity": equity,
        "returns": strat_ret,
        "position": pos,
        "turnover": turnover,
        "cost": cost,
        "benchmark": benchmark,
        "benchmark_returns": ret,
        "trades": extract_trades(pos, close),
        "prices": prices,
    }


def extract_trades(pos: pd.Series, close: pd.Series) -> pd.DataFrame:
    """从仓位序列里还原出「一笔一笔的交易」。

    回测只看净值曲线是不够的 —— 你还要知道一年交易了几次、胜率多少、
    平均持有多久。交易次数太多，手续费会吃掉全部利润。
    """
    change = pos.diff().fillna(pos)
    rows = []
    open_date, open_price, direction = None, None, None

    for date, delta in change.items():
        if delta == 0:
            continue
        price = close.loc[date]
        if open_date is None:
            open_date, open_price, direction = date, price, np.sign(delta)
        else:
            pnl = (price / open_price - 1) * direction
            rows.append({
                "开仓日": open_date, "平仓日": date,
                "方向": "多" if direction > 0 else "空",
                "开仓价": round(open_price, 2), "平仓价": round(price, 2),
                "收益率": pnl, "持有天数": (date - open_date).days,
            })
            if abs(pos.loc[date]) > 1e-9:     # 反手：平掉旧仓，立刻开新仓
                open_date, open_price, direction = date, price, np.sign(delta)
            else:
                open_date, open_price, direction = None, None, None

    df = pd.DataFrame(rows)
    if len(df):
        df["收益率"] = df["收益率"].round(4)
    return df


def trade_stats(trades: pd.DataFrame) -> pd.Series:
    """交易层面的统计：次数、胜率、盈亏比、平均持有天数。"""
    if trades is None or len(trades) == 0:
        return pd.Series({"交易次数": 0})
    wins = trades[trades["收益率"] > 0]["收益率"]
    losses = trades[trades["收益率"] <= 0]["收益率"]
    return pd.Series({
        "交易次数": len(trades),
        "盈利次数": len(wins),
        "交易胜率": len(wins) / len(trades),
        "平均盈利": wins.mean() if len(wins) else 0.0,
        "平均亏损": losses.mean() if len(losses) else 0.0,
        "盈亏比": (wins.mean() / abs(losses.mean())) if len(losses) and losses.mean() != 0 else np.nan,
        "平均持有天数": trades["持有天数"].mean(),
    })


def yearly_returns(returns: pd.Series) -> pd.Series:
    """按年拆解收益。总收益好看但某一年巨亏，实盘你根本拿不住。"""
    return returns.groupby(returns.index.year).apply(
        lambda r: float((1 + r).prod() - 1)
    )


def rolling_sharpe(returns: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    """滚动一年夏普。用来判断「策略是持续有效，还是靠某一段行情撑起来的」。"""
    mean = returns.rolling(window).mean() * TRADING_DAYS
    vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return (mean / vol).replace([np.inf, -np.inf], np.nan)
