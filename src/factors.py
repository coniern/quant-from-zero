"""
横截面因子：从「一只票的时间序列」升级到「一堆票的比较」。

单标的择时（第 5 章）问的是"现在该不该在场"。
多因子选股（第 7 章）问的是"这一堆票里，哪 20 只更可能跑赢"。

因子研究的标准流程：
    构造因子 → 计算 IC → 分层回测 → 判断有效 → 合成打分 → 选股

这个文件实现前三步。**IC 是判断因子生死的唯一标准**，比看收益曲线靠谱得多。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 宽表工具

def to_wide(panel: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """长表 → 宽表：行=日期，列=股票。做横截面计算必须用这个形状。"""
    return panel.pivot_table(index=panel.index, columns="symbol", values=field)


# ---------------------------------------------------------------- 因子构造

def factor_momentum(close_wide: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """动量因子：过去 window 天收益率。

    逻辑：强者恒强。A 股 1 个月动量偏弱、3~12 个月动量较稳，
    所以 window 别只试 20 天。
    """
    return close_wide.pct_change(window)


def factor_reversal(close_wide: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """反转因子：过去 window 天收益率的相反数。

    A 股散户占比高、追涨杀跌明显，短期（1 周）反转效应比动量更强。
    注意它和动量因子的方向是反的 —— 同一个数，取负号就是另一个因子。
    """
    return -close_wide.pct_change(window)


def factor_volatility(close_wide: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """波动率因子：过去 window 天日收益标准差。取负号 = 低波动溢价。"""
    return -close_wide.pct_change().rolling(window, min_periods=window).std()


def factor_bias(close_wide: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """乖离率：(收盘价 - N日均线) / N日均线。衡量偏离程度。"""
    ma = close_wide.rolling(window, min_periods=window).mean()
    return (close_wide - ma) / ma


def factor_volume_ratio(volume_wide: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """量比：今日成交量 / N 日平均成交量。放量往往伴随趋势启动或见顶。"""
    return volume_wide / volume_wide.rolling(window, min_periods=window).mean()


# ---------------------------------------------------------------- 未来收益

def forward_return(close_wide: pd.DataFrame, periods: int = 5) -> pd.DataFrame:
    """未来 N 日收益率 —— 因子要预测的目标。

    ⚠️ 这个函数天然包含"未来信息"，它只能用来评估因子，绝不能进策略信号。
    很多人的回测作弊就是这样发生的：把 fwd_ret 当成特征喂给模型。
    """
    return close_wide.shift(-periods) / close_wide - 1


# ---------------------------------------------------------------- IC 检验

def _corr(a: pd.Series, b: pd.Series, method: str = "spearman") -> float:
    """相关系数。手写实现，不依赖 scipy。

    Spearman = 先把两组数各自转成排名，再算 Pearson。
    就这么简单 —— pandas 的 `method="spearman"` 内部其实要调 scipy，
    为了两个包能跑通整个教程，这里自己实现，顺便让你看清楚它是什么。
    """
    if method == "spearman":
        a, b = a.rank(), b.rank()
    a = a.to_numpy(dtype=float)
    b = b.to_numpy(dtype=float)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def ic_series(factor: pd.DataFrame, fwd: pd.DataFrame,
              method: str = "spearman") -> pd.Series:
    """逐日计算因子值与未来收益的秩相关系数（IC）。

    为什么用 Spearman 不用 Pearson？
    因为我们要的是"排序能力"而非"线性关系"。因子值 1.5 和 1.6 的差异不重要，
    重要的是"1.6 的那只票是否比 1.5 的涨得多"。Spearman 只看排名，对异常值也稳健。

    经验标准（日频、A股、5日收益）：
        |IC均值| > 0.03 且 ICIR > 0.3   → 值得继续研究
        |IC均值| > 0.05 且 ICIR > 0.5   → 相当不错
        IC 均值很漂亮但 ICIR 很低       → 靠某几个极端日撑起来的，别信
    """
    common_dates = factor.index.intersection(fwd.index)
    values = {}
    for date in common_dates:
        pair = pd.concat([factor.loc[date], fwd.loc[date]], axis=1).dropna()
        if len(pair) < 10:          # 样本太少，相关系数没有意义
            continue
        values[date] = _corr(pair.iloc[:, 0], pair.iloc[:, 1], method)
    return pd.Series(values).dropna()


def ic_summary(ic: pd.Series) -> pd.Series:
    """IC 的统计特征。ICIR = IC均值 / IC标准差，衡量因子的稳定性。"""
    if len(ic) == 0:
        return pd.Series({"IC均值": np.nan})
    return pd.Series({
        "IC均值": ic.mean(),
        "IC标准差": ic.std(),
        "ICIR": ic.mean() / ic.std() if ic.std() else np.nan,
        "IC>0占比": (ic > 0).mean(),
        "样本天数": len(ic),
    })


# ---------------------------------------------------------------- 分层回测

def layer_returns(factor: pd.DataFrame, fwd: pd.DataFrame,
                  n_layers: int = 5) -> pd.DataFrame:
    """分层回测：每天按因子值排序，等分成 N 组，看每组的未来收益。

    这是判断因子单调性最直观的方法。
    真正有效的因子，从第 1 层到第 N 层的收益应该接近单调递增（或递减）。
    如果只有第 1 层特别好、中间乱七八糟，那叫"尾部效应"，很难稳定赚到。

    返回：行=层号(1 最低 ~ N 最高)，列=每期各组平均未来收益
    """
    records = []
    for date in factor.index.intersection(fwd.index):
        f = factor.loc[date].dropna()
        r = fwd.loc[date].reindex(f.index)
        if len(f) < n_layers * 2:
            continue
        try:
            # rank 后按名次切层，比按值切更稳（避免因子值分布极度不均）
            layers = pd.qcut(f.rank(method="first"), n_layers,
                             labels=range(1, n_layers + 1))
        except ValueError:
            continue
        row = r.groupby(layers, observed=True).mean()
        row.name = date
        records.append(row)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_index()


def long_short(layer_ret: pd.DataFrame) -> pd.Series:
    """多空组合收益 = 最高层 - 最低层。

    多空收益才是因子的"纯 alpha"（扣掉了市场涨跌）。
    分层收益里最高层赚钱，可能只是因为它 beta 高 —— 多空一减就露馅了。
    """
    if layer_ret.empty:
        return pd.Series(dtype=float)
    return layer_ret.iloc[:, -1] - layer_ret.iloc[:, 0]


def zscore_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """横截面标准化：每天减去当日均值、除以当日标准差。

    不同因子的量纲天差地别（动量是百分比，量比是倍数），
    要合成打分必须先标准化。注意是**按行（每天）**标准化，不是按列。
    """
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def composite_score(factors: dict[str, pd.DataFrame],
                    weights: dict[str, float] | None = None) -> pd.DataFrame:
    """多因子合成打分：标准化后加权求和。权重不传就等权。"""
    if weights is None:
        weights = {k: 1.0 for k in factors}
    total = sum(weights.values())
    score = None
    for name, df in factors.items():
        z = zscore_cross_section(df)
        part = z * (weights[name] / total)
        score = part if score is None else score.add(part, fill_value=0)
    return score
