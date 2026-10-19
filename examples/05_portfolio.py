"""
05 · 组合优化：钱怎么分，比选什么更重要

运行：python examples/05_portfolio.py

同样是 10 只票，四种分配方式，绩效差多少？
（1）等权            每只 10%
（2）逆波动率        波动小的多配
（3）最小方差        让组合波动最小
（4）风险平价        让每只票贡献的风险一样多

先猜哪个最好，再看结果 —— 对不上就说明你对"风险"的直觉有偏差。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data import load_panel
from src.metrics import annual_return, annual_vol, max_drawdown, sharpe, \
    equity_from_returns
from src.portfolio import (equal_weight, inverse_volatility, min_variance,
                           portfolio_returns, risk_contribution, risk_parity)

UNIVERSE = [
    "600519", "000858", "601318", "600036", "000001",
    "600030", "600887", "000651", "000333", "600900",
]

panel = load_panel(UNIVERSE, "2022-01-01", "2024-12-31")
close_wide = panel.pivot_table(index=panel.index, columns="symbol",
                               values="close")
rets = close_wide.pct_change().dropna()
print(f"{rets.shape[1]} 只票，{len(rets)} 个交易日\n")

# ---------------- 四种权重 ----------------
weights = {
    "等权": equal_weight(rets),
    "逆波动率": inverse_volatility(rets),
    "最小方差": min_variance(rets),
    "风险平价": risk_parity(rets),
}

print("=" * 84)
print("四种配比方案的绩效（2022-2024）")
print("=" * 84)

rows = {}
for name, w in weights.items():
    r = portfolio_returns(w, rets)
    rows[name] = {
        "年化收益": annual_return(r),
        "年化波动": annual_vol(r),
        "最大回撤": max_drawdown(equity_from_returns(r)),
        "夏普": sharpe(r),
    }
table = pd.DataFrame(rows).T
print(table.to_string(float_format=lambda x: f"{x:8.3f}"))

# ---------------- 权重明细 ----------------
print("\n权重明细（%）")
wdf = pd.DataFrame({k: (v * 100).round(1) for k, v in weights.items()}).T
print(wdf.to_string())

# ---------------- 风险贡献验证 ----------------
print("\n风险贡献占比（%）：风险平价的目标是每行都接近 10%")
rc = risk_contribution(weights["风险平价"], rets)
print((rc * 100).round(1).to_string())

print("""
怎么读这张表（对照上面跑出来的真实数字）：

1. 最小方差组合的「年化波动」一定最低 —— 这是它的数学目标，检验算法对不对就看这个。

2. 这次最小方差的年化收益反而最高（看权重就知道为什么：46% 押在长江电力上）。
   别过度解读 —— 2022-2024 恰好是防御股跑赢的行情，
   「最小方差」在这里实际上做了一次风格押注。换一个时间段结论就可能反转。

3. 风险平价的权重分布（长电 21.9%，其余 7~10%）比最小方差均衡得多，
   且跑完后每只票的风险贡献精确等于 10% —— 用 risk_contribution() 可自行验证。
   权重均衡意味着：任何一只票暴雷，对组合的伤害都被限制在 10% 附近。

4. 学术研究里等权经常打败复杂优化（DeMiguel 2009），但这次等权夏普垫底。
   两个结论都别当成真理：**优化模型依赖估计准确的协方差矩阵，
   估计误差会被优化器放大成极端权重 —— 谁赢谁输取决于运气和样本。**

5. 四种方案的回撤（-22% ~ -37%）都小于单押茅台（-51%）。
   分散化是唯一免费的午餐 —— 但它降低的是波动，不是亏损的概率。
""")
