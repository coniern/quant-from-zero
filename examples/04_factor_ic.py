"""
04 · 多因子选股：因子是活的还是死的，用 IC 说话

运行：python examples/04_factor_ic.py

这一课对应 docs/07。核心问题只有一个：
**动量因子到底能不能预测未来收益？**

流程：构造因子 → 逐日算 IC → 看 IC 统计 → 分层回测 → 得出结论。
流程走完，你会得到一个明确答案（可能和你想的不一样）。

⚠️ 已知缺陷：这 20 只票是"今天还活着"的票，有幸存者偏差。
结论只能用于教学演示，不能直接拿去实盘。详见 docs/03-数据.md 第七节。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import apply_chinese_font
from src.data import load_panel
from src.factors import (factor_bias, factor_momentum, factor_reversal,
                         factor_volatility, forward_return, ic_series,
                         ic_summary, layer_returns, long_short)
from src.report import plot_ic, plot_layers

# 20 只各行业代表性股票（有幸存者偏差，教学用）
UNIVERSE = [
    "600519", "000858", "601318", "600036", "000001",
    "600030", "601166", "000651", "000333", "600276",
    "002415", "300750", "002594", "601012", "600887",
    "000568", "601888", "600900", "601899", "600309",
]

apply_chinese_font()
START, END = "2022-01-01", "2024-12-31"

print("拉取 20 只股票数据（有本地缓存，第二次秒开）...")
panel = load_panel(UNIVERSE, START, END)
n_stocks = panel["symbol"].nunique()
print(f"完成：{n_stocks} 只票，{panel.index.nunique()} 个交易日\n")

close_wide = panel.pivot_table(index=panel.index, columns="symbol",
                               values="close")

# 未来 5 日收益 —— 因子要预测的目标
fwd5 = forward_return(close_wide, periods=5)

# ---------------- 逐个检验因子 ----------------
factors = {
    "动量20日": factor_momentum(close_wide, 20),
    "动量60日": factor_momentum(close_wide, 60),
    "反转5日": factor_reversal(close_wide, 5),
    "低波动20日": factor_volatility(close_wide, 20),
    "乖离率20日": factor_bias(close_wide, 20),
}

rows = {}
for name, f in factors.items():
    ic = ic_series(f, fwd5)
    stat = ic_summary(ic)
    # 分层回测 + 多空收益
    layers = layer_returns(f, fwd5, n_layers=5)
    ls = long_short(layers)
    rows[name] = {
        "IC均值": stat["IC均值"],
        "ICIR": stat["ICIR"],
        "IC>0占比": stat["IC>0占比"],
        "多空年化": (1 + ls).prod() ** (252 / len(ls)) - 1 if len(ls) else float("nan"),
    }
    plot_ic(ic, title=name, name=f"04_ic_{name}.png")

table = pd.DataFrame(rows).T
print("=" * 72)
print("因子检验结果（2022-2024，未来 5 日收益，Spearman IC）")
print("=" * 72)
print(table.to_string(float_format=lambda x: f"{x:8.4f}"))

# ---------------- 分层回测图 ----------------
best = table["ICIR"].abs().idxmax()
print(f"\nICIR 绝对值最大的因子：{best}，画出它的分层回测：")
layers = layer_returns(factors[best], fwd5, n_layers=5)
path = plot_layers(layers, title=best, name="04_layers.png")
print(f"图已保存：{path}")

print("""
怎么读这张表：

1. IC 均值：因子值和未来收益的相关性。日频 |IC|>0.03 算及格。
2. ICIR：IC 的稳定性（均值/标准差）。>0.3 才值得继续研究。
3. IC>0 占比：多少比例的时间因子方向是对的。60% 以上算稳定。
4. 多空年化：每天买最高组、空最低组的年化收益（已扣掉市场涨跌）。
5. 分层单调性（看图）：L1 到 L5 应该接近单调。

对照经验值，你大概率会得出一个让人失望的结论：
**这五个因子全死了。** |IC| 不到 0.01，ICIR 接近 0，多空收益不是负的
就是靠某几个极端日撑起来的。

这恰恰是最有价值的一课，原因有三个：

1. **20 只大白马本来就不该有横截面 alpha。**
   它们同涨同跌（beta 高度相关）、机构定价充分、消息早已反映。
   因子研究需要的是几百上千只、风格差异大的样本。

2. **幸存者偏差在这里是双重的。**
   池子本身是"活下来且活得很好"的票 —— 连"烂"的那一半都没有，
   什么因子都无法区分好坏，因为池子里只有"好"。

3. **失效是常态，有效是例外。**
   学术论文里 |IC| 0.05 的因子，发表十年后普遍衰减一半以上。
   你以后测因子，测出全灭是正常的，测出有效才要加倍怀疑
   （是数据泄露？过拟合？还是真的有效？）。

下一步动手：把 UNIVERSE 换成 100+ 只不同行业、不同市值的票，
或者把 periods 改成 20/60 日，看看结论变不变。
变，说明结果对样本敏感；不变，才有资格谈"因子有效"。
""")
