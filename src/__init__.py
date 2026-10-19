"""quant-from-zero：一个能跑通的 A 股量化入门项目。

模块地图
--------
config      全局参数（成本、无风险利率、路径）
data        数据获取与清洗（akshare + 本地缓存 + 合成兜底）
metrics     收益风险指标（全部手写）
backtest    回测引擎（60 行核心）
strategies  择时策略（双均线、突破、动量、均值回归）
factors     横截面因子（构造、IC、分层）
report      出图
"""

__version__ = "1.0.0"
