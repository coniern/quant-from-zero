"""
出图：把数字变成一眼能看懂的东西。

量化研究里，一张净值曲线图能顶 500 字结论。
但图也有陷阱 —— 纵轴从 0.9 开始，能让 5% 的波动看起来像腰斩。
这个文件里的图统一从净值 1.0 的基准线画起，不做视觉夸张。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, apply_chinese_font
from .metrics import drawdown_series, equity_from_returns


def _save(fig, name: str) -> Path:
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_equity(result: dict, title: str = "策略净值", name: str = "equity.png") -> Path:
    """净值曲线 + 回撤填充，上下两栏共享时间轴。"""
    apply_chinese_font()
    equity = result["equity"]
    bench = result.get("benchmark")
    dd = drawdown_series(equity)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    ax1.plot(equity.index, equity.values, lw=1.6, color="#c0392b", label=title)
    if bench is not None:
        ax1.plot(bench.index, bench.values, lw=1.2, color="#7f8c8d",
                 ls="--", label="买入持有")
    ax1.axhline(1.0, color="#bdc3c7", lw=0.8)
    ax1.set_ylabel("净值")
    ax1.set_title(f"{title} · 净值与回撤", fontsize=13)
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(alpha=0.25)

    ax2.fill_between(dd.index, dd.values, 0, color="#2980b9", alpha=0.35)
    ax2.set_ylabel("回撤")
    ax2.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax2.grid(alpha=0.25)

    return _save(fig, name)


def plot_signals(prices: pd.DataFrame, signal: pd.Series,
                 fast: int = 5, slow: int = 20,
                 name: str = "signals.png") -> Path:
    """价格 + 均线 + 持仓区间。用来「看见」策略到底在干什么。"""
    apply_chinese_font()
    close = prices["close"]
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(close.index, close.values, lw=1.2, color="#2c3e50", label="收盘价")
    ax.plot(close.index, close.rolling(fast).mean(), lw=1.0,
            color="#e67e22", label=f"MA{fast}")
    ax.plot(close.index, close.rolling(slow).mean(), lw=1.0,
            color="#16a085", label=f"MA{slow}")

    # 有仓位的区间涂成浅红，一眼看出"什么时候在场"
    holding = signal.fillna(0) > 0
    ax.fill_between(close.index, close.min(), close.max(),
                    where=holding, color="#c0392b", alpha=0.10,
                    label="持仓期")
    ax.set_ylabel("价格（元）")
    ax.set_title("双均线信号可视化")
    ax.legend(loc="upper left", frameon=False, ncol=4, fontsize=9)
    ax.grid(alpha=0.25)

    return _save(fig, name)


def plot_ic(ic: pd.Series, title: str = "因子 IC", name: str = "ic.png") -> Path:
    """IC 柱状图 + 累计 IC。累计 IC 一路向上 = 因子持续有效。"""
    apply_chinese_font()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    colors = ["#c0392b" if v > 0 else "#27ae60" for v in ic.values]
    ax1.bar(ic.index, ic.values, color=colors, width=1.0, alpha=0.8)
    ax1.axhline(0, color="#2c3e50", lw=0.8)
    ax1.axhline(ic.mean(), color="#8e44ad", lw=1.0, ls="--",
                label=f"均值 {ic.mean():.4f}")
    ax1.set_ylabel("IC")
    ax1.set_title(f"{title} · 逐期 IC（均值 {ic.mean():.4f}，ICIR "
                  f"{ic.mean() / ic.std() if ic.std() else 0:.2f}）", fontsize=12)
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)

    ax2.plot(ic.index, ic.cumsum().values, lw=1.4, color="#2980b9")
    ax2.axhline(0, color="#bdc3c7", lw=0.8)
    ax2.set_ylabel("累计 IC")
    ax2.grid(alpha=0.25)

    return _save(fig, name)


def plot_layers(layer_ret: pd.DataFrame, title: str = "分层回测",
                name: str = "layers.png") -> Path:
    """各层平均收益柱状图。单调性一眼可见。"""
    apply_chinese_font()
    means = layer_ret.mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))

    colors = ["#27ae60" if v < 0 else "#c0392b" for v in means.values]
    bars = ax.bar([f"L{int(i)}" for i in means.index], means.values,
                  color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%.2f%%", fontsize=9)
    ax.axhline(0, color="#2c3e50", lw=0.8)
    ax.set_ylabel("平均每期收益（%）")
    ax.set_xlabel("因子分层（L1 最低 → 最高）")
    ax.set_title(f"{title} · 分层平均收益", fontsize=12)
    ax.grid(alpha=0.25, axis="y")

    return _save(fig, name)


def plot_yearly(returns: pd.Series, name: str = "yearly.png") -> Path:
    """年度收益柱状图。看策略是不是只在某一年爆发。"""
    apply_chinese_font()
    yearly = returns.groupby(returns.index.year).apply(lambda r: (1 + r).prod() - 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#c0392b" if v > 0 else "#27ae60" for v in yearly.values]
    bars = ax.bar(yearly.index.astype(str), yearly.values * 100,
                  color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=9)
    ax.axhline(0, color="#2c3e50", lw=0.8)
    ax.set_ylabel("年度收益（%）")
    ax.set_title("分年度收益", fontsize=12)
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, name)
