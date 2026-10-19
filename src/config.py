"""
全局配置。

为什么单独放一个文件？
因为"手续费率""无风险利率""交易日数"这类数字会在十几个地方用到。
散落在各处的魔法数字是 bug 的温床，集中一处改一次就全生效。
"""

from pathlib import Path

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "cache"
OUTPUT_DIR = ROOT / "output"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 交易成本（A 股真实规则） ----------
# 佣金：双边收取，万 2.5，不足 5 元按 5 元收（这里按比例近似，暂不算最低 5 元）
COMMISSION_RATE = 0.00025
# 印花税：只在卖出时收，千分之一（2023-08-28 起减半为万 5）
STAMP_TAX_RATE = 0.0005
# 滑点：买入成交价比收盘价高一点，卖出低一点。0.1% 是保守估计
SLIPPAGE_RATE = 0.001

# ---------- 收益计算 ----------
TRADING_DAYS = 252          # A 股一年约 242~244 个交易日，业界习惯用 252
RISK_FREE_RATE = 0.02       # 无风险利率，用一年期国债近似 2%

# ---------- 回测默认参数 ----------
INITIAL_CAPITAL = 100_000   # 初始资金 10 万


def apply_chinese_font() -> None:
    """让 matplotlib 能显示中文，并正常显示负号。

    不设这个，图上所有中文都会变成方块 —— 这是新手第一个必踩的坑。
    """
    import matplotlib
    matplotlib.use("Agg")           # 无界面后端，服务器/脚本里也能出图
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Arial Unicode MS",
        "Songti SC", "SimHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False   # 否则负号显示成方块
