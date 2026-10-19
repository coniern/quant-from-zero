"""
策略库：每个策略都是一个「从价格 → 目标仓位」的纯函数。

统一的约定，非常重要：
    输入  close (Series, 日期索引)
    输出  signal (Series, 同索引, 取值 0 或 1)

为什么强制 0/1 而不是 -1/0/1？
因为 A 股融券做空成本高、券源少，散户基本做不了空。
先学会「什么时候该在场」，再考虑做空 —— 这是更务实的顺序。

所有策略都**只用当天及之前的数据**，绝不偷看明天。
这是能回测的前提，也是新手写策略时最容易犯的错。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ma_cross(close: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    """双均线：快线上穿慢线持有，下穿空仓。

    这就是"金叉死叉"，最古典的择时方法。
    它有效的经济学解释是「趋势延续」：短期资金的动向领先于长期共识。

    缺点：震荡市里会反复被打脸（来回金叉死叉），交易成本极高。
    所以你会在 docs/05 里看到它在一段趋势行情里赚钱、在震荡行情里亏钱。
    """
    ma_fast = close.rolling(fast, min_periods=fast).mean()
    ma_slow = close.rolling(slow, min_periods=slow).mean()
    # 快线在上 → 1，否则 0。用 > 而不是 >=，避免持平当天的歧义
    return (ma_fast > ma_slow).astype(float)


def breakout(close: pd.Series, window: int = 20) -> pd.Series:
    """唐奇安通道突破：创 N 日新高就持有，跌破 N 日新低就离场。

    海龟交易法的简化版。和双均线比，它对"突破"更敏感、对"阴跌"反应更快，
    但在假突破多的市场里更难受。
    """
    high_n = close.rolling(window, min_periods=window).max()
    low_n = close.rolling(window, min_periods=window).min()

    signal = pd.Series(np.nan, index=close.index)
    signal[close >= high_n] = 1.0        # 突破上轨：进
    signal[close <= low_n] = 0.0         # 跌破下轨：出
    return signal.ffill().fillna(0.0)    # 中间状态维持原仓位


def momentum(close: pd.Series, window: int = 20, threshold: float = 0.0) -> pd.Series:
    """动量：过去 N 天涨幅 > 阈值就持有。

    「强者恒强」的横截面版本是多因子里的动量因子（见 src/factors.py），
    这里先用时间序列版本讲原理：涨得好的继续涨，跌得差的继续跌。
    """
    ret_n = close.pct_change(window)
    return (ret_n > threshold).astype(float)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI 相对强弱指标，0~100。>70 超买，<30 超卖。

    算法：先算每日涨跌幅，分别取上涨部分和下跌部分的 N 日均值，再求比值。
    注意这里用的是 Wilder 平滑（等价于 alpha=1/N 的 EMA），不是简单平均。
    """
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def mean_reversion(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """均值回归（布林带）：跌破下轨买入，回到中轨卖出。

    和动量策略是镜像关系：动量赌"延续"，均值回归赌"回归"。
    同一个市场里两者不可能同时长期有效 —— 这本身就是个很好的思考题。
    """
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std()
    lower = ma - n_std * sd
    upper = ma + n_std * sd

    signal = pd.Series(np.nan, index=close.index)
    signal[close < lower] = 1.0
    signal[close > upper] = 0.0
    return signal.ffill().fillna(0.0)


# 给参数扫描用：名字 → (函数, 参数网格)
STRATEGIES = {
    "双均线": (ma_cross, {"fast": [3, 5, 10, 20], "slow": [20, 30, 60]}),
    "通道突破": (breakout, {"window": [10, 20, 30, 60]}),
    "动量": (momentum, {"window": [10, 20, 60]}),
    "均值回归": (mean_reversion, {"window": [10, 20, 30]}),
}
