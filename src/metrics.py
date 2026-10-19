"""
收益与风险指标：全部手写，不调用任何现成库。

为什么不用 empyrical / quantstats 这类库？
因为"会调用"和"懂含义"是两回事。面试官问"你的夏普 1.8 是怎么算的、无风险利率取的多少、
为什么是 252 不是 365"，能当场写出公式的人才算懂。

每个函数上方都写了公式。看不懂公式就先看 docs/04-收益与风险.md。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RISK_FREE_RATE, TRADING_DAYS


def to_returns(close: pd.Series) -> pd.Series:
    """价格 → 日收益率。第一行是 NaN，要 dropna。"""
    return close.pct_change().dropna()


def equity_from_returns(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """日收益率 → 净值曲线（复利）。净值 = 初始资金 × Π(1+r_i)"""
    return initial * (1 + returns).cumprod()


# ---------------------------------------------------------------- 收益类

def total_return(returns: pd.Series) -> float:
    """区间累计收益率 = Π(1+r) - 1"""
    return float((1 + returns).prod() - 1)


def annual_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """年化收益率（CAGR，几何平均）

        (1 + 累计收益) ^ (周期数 / 天数) - 1

    注意：不是「累计收益 / 年数」，那是算术平均，会高估长期收益。
    """
    n = len(returns)
    if n == 0:
        return 0.0
    growth = float((1 + returns).prod())
    if growth <= 0:          # 本金亏光，年化无意义
        return -1.0
    return growth ** (periods / n) - 1


# ---------------------------------------------------------------- 风险类

def annual_vol(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """年化波动率 = 日收益标准差 × √252

    为什么乘 √252 而不是 252？
    因为波动率是「标准差」，方差随时间线性增长，标准差随 √时间 增长。
    这是新手最常写错的一行。
    """
    return float(returns.std(ddof=1) * np.sqrt(periods))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """回撤序列 = 净值 / 历史最高净值 - 1，永远 ≤ 0"""
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤（负数）。表示「从最高点跌下来最深亏了多少」。"""
    return float(drawdown_series(equity).min())


def max_drawdown_detail(equity: pd.Series) -> dict:
    """最大回撤的完整信息：幅度、峰值日、谷底日、修复天数（未修复则为 None）"""
    dd = drawdown_series(equity)
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()

    after = equity.loc[trough:]
    recovered = after[after >= equity.loc[peak]]
    recover_date = recovered.index[0] if len(recovered) else None

    return {
        "max_drawdown": float(dd.min()),
        "peak_date": peak,
        "trough_date": trough,
        "recover_date": recover_date,
        "days_to_recover": (recover_date - trough).days if recover_date is not None else None,
    }


def calmar(returns: pd.Series) -> float:
    """卡玛比率 = 年化收益 / |最大回撤|

    比夏普更贴近真实体验：它回答"为了这点收益，我要忍受多深的浮亏"。
    """
    mdd = max_drawdown(equity_from_returns(returns))
    if mdd == 0:
        return np.inf
    return annual_return(returns) / abs(mdd)


# ---------------------------------------------------------------- 风险调整收益

def sharpe(returns: pd.Series, rf: float = RISK_FREE_RATE,
           periods: int = TRADING_DAYS) -> float:
    """夏普比率 = 超额收益均值 / 收益标准差 × √252

        (mean(r) - rf/252) / std(r) * √252

    含义：每承担 1 单位波动，换来多少超额收益。
    经验值：<0.5 不值得看，0.5~1 一般，1~2 不错，>2 要怀疑是不是回测作弊了。
    """
    excess = returns - rf / periods
    sd = excess.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods))


def win_rate(returns: pd.Series) -> float:
    """胜率：上涨天数 / 总天数。注意高胜率 ≠ 赚钱（可能赢小亏大）。"""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def alpha_beta(returns: pd.Series, benchmark: pd.Series,
               rf: float = RISK_FREE_RATE) -> tuple[float, float]:
    """对基准做线性回归：r = α + β·r_bench + ε

    β（贝塔）：跟大盘同涨同跌的程度。β=1 跟大盘一样，β=0.5 只有半个大盘的波动。
    α（阿尔法）：扣掉大盘影响后的年化超额收益。**这才是"策略本身的能力"**。

    用最小二乘拟合，斜率是 β，截距日频 α 乘 252 成年化。
    """
    df = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(df) < 30:
        return 0.0, 0.0
    y, x = df.iloc[:, 0].values, df.iloc[:, 1].values
    beta, intercept = np.polyfit(x, y, 1)
    alpha = intercept * TRADING_DAYS      # 日频 α → 年化
    return float(alpha), float(beta)


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    """信息比率 IR = 超额收益均值 / 跟踪误差 × √252

    和夏普的区别：夏普比的是「无风险利率」，IR 比的是「基准」。
    做指数增强、相对收益策略必须看 IR。
    """
    active = (returns - benchmark).dropna()
    te = active.std(ddof=1)
    if te == 0:
        return 0.0
    return float(active.mean() / te * np.sqrt(TRADING_DAYS))


# ---------------------------------------------------------------- 汇总

def summary(returns: pd.Series, benchmark: pd.Series | None = None) -> pd.Series:
    """一次性算出全部指标，返回一张可以打印的表。"""
    equity = equity_from_returns(returns)
    dd = max_drawdown_detail(equity)
    m = {
        "累计收益": total_return(returns),
        "年化收益": annual_return(returns),
        "年化波动": annual_vol(returns),
        "最大回撤": dd["max_drawdown"],
        "回撤修复天数": dd["days_to_recover"],
        "夏普比率": sharpe(returns),
        "卡玛比率": calmar(returns),
        "日胜率": win_rate(returns),
    }
    if benchmark is not None:
        a, b = alpha_beta(returns, benchmark)
        m["年化Alpha"] = a
        m["Beta"] = b
        m["信息比率"] = information_ratio(returns, benchmark)
        m["基准累计收益"] = total_return(benchmark.dropna())
    return pd.Series(m)


def format_summary(s: pd.Series) -> str:
    """把指标表格式化成人类能读的样子（百分比 / 保留 2~3 位小数）。"""
    pct_keys = {"累计收益", "年化收益", "年化波动", "最大回撤",
                "年化Alpha", "日胜率", "基准累计收益"}
    lines = []
    for k, v in s.items():
        if pd.isna(v):
            lines.append(f"{k:>12}: —")
        elif k in pct_keys:
            lines.append(f"{k:>12}: {v:>8.2%}")
        elif k == "回撤修复天数":
            lines.append(f"{k:>12}: {int(v) if not pd.isna(v) else '—'} 天")
        else:
            lines.append(f"{k:>12}: {v:>8.3f}")
    return "\n".join(lines)
