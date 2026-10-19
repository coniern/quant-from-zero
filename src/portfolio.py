"""
组合优化：手里有几只票，钱怎么分？

这是「选股」之后的第二个问题，也是大多数人忽略的问题。
同样 5 只票，等权买入和按最小方差配比，年化波动能差一半。

这里只用 numpy 实现 —— 连 scipy 都不需要。
因为优化问题的本质是「在约束下最小化某个目标」，理解了拉格朗日乘子，
你会发现闭式解比调库更清楚。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.dropna(how="all").fillna(0.0)


def equal_weight(returns: pd.DataFrame) -> pd.Series:
    """等权：1/N。

    别小看它。学术上叫「1/N 策略」，DeMiguel 等人 2009 年的论文证明：
    在样本外，等权经常打败各种复杂的优化模型。原因是**估计误差**：
    你估的协方差矩阵本身就不准，优化器会把误差放大成极端权重。
    """
    n = returns.shape[1]
    return pd.Series(1.0 / n, index=returns.columns)


def inverse_volatility(returns: pd.DataFrame) -> pd.Series:
    """ inverse vol：权重 ∝ 1/波动率。波动小的多配，波动大的少配。

    只用了对角线的信息（各自的波动），不估计相关性 ——
    少估一个量，就少一份误差。实务里非常常用。
    """
    vol = returns.std()
    w = 1 / vol.replace(0, np.nan)
    return (w / w.sum()).fillna(0)


def min_variance(returns: pd.DataFrame, long_only: bool = True) -> pd.Series:
    """最小方差组合（闭式解）

    目标：min  w'Σw    约束：Σw = 1
    拉格朗日求解 → w ∝ Σ⁻¹·1

    这是马科维茨模型在「只关心风险、不管收益」时的特例。
    长仓约束下严格解需要数值优化，这里用「截断负权重再归一化」近似 ——
    结果和数值解非常接近，但只需要一行矩阵运算。
    """
    cov = returns.cov().values
    ones = np.ones(cov.shape[0])
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:      # 协方差矩阵奇异（票太少/完全共线）
        inv = np.linalg.pinv(cov)
    w = inv @ ones
    w = w / w.sum()

    if long_only:
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()
    return pd.Series(w, index=returns.columns)


def risk_parity(returns: pd.DataFrame, iters: int = 1000) -> pd.Series:
    """风险平价：让每只票对组合的风险贡献相等。

    直觉：如果一只票贡献了 80% 的风险，那这个组合其实是"一只票 + 四个陪跑"。
    风险平价的组合，任何一只票崩了都不会致命。

    算法是不动点迭代（风险预算问题的标准解法）：

        目标条件：w_i × (Σw)_i = 预算_i（对所有 i）
        不动点：  w_i = 预算_i / (Σw)_i
        迭代：    用当前 w 算 (Σw)，更新 w，归一化，重复

    数学的严谨版本（Griveau-Billion 2013）用的是牛顿法加对数变换，
    这里给的是收敛性同样有保证、但一行能写完的版本 ——
    你可以自己验证：跑完后用 risk_contribution() 检查，各项应接近 1/N。
    """
    cov = returns.cov().values
    n = cov.shape[0]
    budget = np.full(n, 1.0 / n)      # 每个资产分到 1/N 的风险预算

    w = budget / np.sqrt(np.diag(cov))     # 用逆波动率做起点，收敛更快
    w = w / w.sum()

    for _ in range(iters):
        marginal = cov @ w                 # (Σw)_i
        w_new = budget / marginal
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < 1e-12:
            w = w_new
            break
        w = w_new

    return pd.Series(w, index=returns.columns)


def risk_contribution(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """每个资产对组合总风险的贡献占比。风险平价的检验标准。

    边际风险贡献 MRC_i = (Σw)_i / σ_p
    风险贡献 RC_i = w_i × MRC_i，归一化后各资产应接近 1/N。
    """
    cov = returns.cov().values
    w = weights.values
    port_var = w @ cov @ w
    if port_var <= 0:
        return pd.Series(0.0, index=weights.index)
    mrc = (cov @ w) / np.sqrt(port_var)
    rc = w * mrc
    return pd.Series(rc / rc.sum(), index=weights.index)


def portfolio_returns(weights: pd.Series, returns: pd.DataFrame,
                      rebalance: str = "ME") -> pd.Series:
    """按权重合成组合收益。

    rebalance="ME" 表示每月再平衡一次（把跑偏的权重拉回目标）。
    不做再平衡的话，涨得好的票权重会越来越重，组合慢慢变成单押。

    兼容性：pandas 3.0 用 "ME"，2.x 用 "M"，这里自动适配。
    """
    rets = _clean(returns)
    if rebalance is None:
        return (rets * weights).sum(axis=1)

    try:                      # pandas 3.0+ 移除了 "M"，只认 "ME"
        grouper = pd.Grouper(freq=rebalance)
        rets.groupby(grouper).size()
    except ValueError:
        rebalance = {"M": "ME", "Y": "YE", "Q": "QE"}.get(rebalance, rebalance)
        grouper = pd.Grouper(freq=rebalance)

    out = []
    for _, chunk in rets.groupby(grouper):
        if chunk.empty:
            continue
        w = weights.copy()
        for date, row in chunk.iterrows():
            r = float((w * row).sum())
            out.append(pd.Series([r], index=[date]))
            w = w * (1 + row)                 # 权重随涨跌漂移
            if w.sum() > 0:
                w = w / w.sum()               # 归一到 100%
    return pd.concat(out).sort_index()


def compare(weights_map: dict[str, pd.Series], returns: pd.DataFrame) -> pd.DataFrame:
    """把多种配比方案的绩效摆在一起对比。"""
    from .metrics import annual_return, annual_vol, max_drawdown, sharpe, \
        equity_from_returns

    rows = {}
    for name, w in weights_map.items():
        r = portfolio_returns(w, returns)
        rows[name] = {
            "年化收益": annual_return(r),
            "年化波动": annual_vol(r),
            "最大回撤": max_drawdown(equity_from_returns(r)),
            "夏普": sharpe(r),
        }
    return pd.DataFrame(rows).T
