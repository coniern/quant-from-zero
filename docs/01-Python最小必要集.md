# 01 · Python 最小必要集

> 时间预算 6 小时。目标不是"学会 Python"，是"能用 pandas 处理价格数据"。

## 先说结论：你不需要学完 Python

很多人卡在这里：买一本 600 页的 Python 教程，学完前 5 章就放弃了。

**量化需要的 Python，其实只有三样东西：**

1. `pandas` 的两个数据结构：`Series`（一列）和 `DataFrame`（一张表）
2. 三个概念：**索引**、**对齐**、**向量化**
3. 十个左右的方法：`pct_change` `rolling` `shift` `groupby` `resample` `iloc` `loc` `merge` `fillna` `apply`

剩下的（装饰器、多线程、元类、设计模式）**现在一个都用不上**。需要的时候再学。

## 一、Series 和 DataFrame

`Series` = 带标签的一维数组。`DataFrame` = 带标签的二维表。

关键是"带标签"这三个字。普通 Python 列表只有位置（第 0 个、第 1 个），
pandas 的每一行都有一个**名字**（索引），这在金融数据里是日期：

```python
import pandas as pd

close = pd.Series(
    [1550.0, 1559.66, 1534.66],
    index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    name="close",
)
print(close)
# date
# 2024-01-02    1550.00
# 2024-01-03    1559.66
# 2024-01-04    1534.66
# Name: close, dtype: float64
```

**为什么索引是日期这么重要？** 因为金融数据天生是时间序列，而你后面所有的操作（算收益、滚动平均、对齐基准）都依赖"按日期对齐"。

## 二、概念一：对齐（最重要的一个）

这是 pandas 最容易被误解、也最有用的特性。

```python
a = pd.Series([1, 2, 3], index=["d1", "d2", "d3"])
b = pd.Series([10, 20, 30], index=["d2", "d3", "d4"])

print(a + b)
# d1     NaN      ← 只有 a 有
# d2    22.0
# d3    33.0
# d4     NaN      ← 只有 b 有
```

**pandas 按索引对齐相加，不是按位置。** 没有对应日期的位置自动变成 NaN。

对量化的意义：**你永远不会算错一行。** 如果两只票的停牌日不同，pandas 会老老实实给你 NaN，而不是悄悄错位一行 —— 错位一行在回测里就是灾难。

但也要小心：`NaN` 会顺着计算传染。所以永远记得 `dropna()` 或 `fillna()`。

## 三、概念二：向量化（为什么不能写 for 循环）

新手最容易写的代码：

```python
# ❌ 慢 100 倍，而且容易写错
rets = []
for i in range(1, len(close)):
    r = close.iloc[i] / close.iloc[i - 1] - 1
    rets.append(r)
```

正确的写法：

```python
# ✅ 一行，快 100 倍
rets = close.pct_change()
```

`pct_change()` 内部是 C 实现的批量运算。**数据量到 10 万行时，for 循环要跑 3 秒，向量化只要 0.03 秒。**

这不是"优雅"的问题，是"能不能跑"的问题：你要扫 1000 个参数组合时，差 100 倍就是 5 分钟 vs 8 小时。

## 四、十个必须掌握的方法

以下每个都用真实场景解释，不要背，理解用途即可。

### 1. `pct_change()` — 算收益率

```python
close.pct_change()          # 日收益率
close.pct_change(20)        # 20 日动量（这是因子！）
```

第一个值必然是 `NaN`（没有前一天），记得处理。

### 2. `rolling(n).mean()` — 滚动窗口

```python
close.rolling(20).mean()              # 20 日均线
close.rolling(20).std()               # 20 日波动率
close.rolling(20).max()               # 20 日最高（唐奇安上轨）
```

`rolling` 的本质是"开一个长度 n 的窗口，在时间轴上滑动"。
**所有技术指标都是 rolling 的某种组合** —— 均线是 mean，布林带是 mean±2std，RSI 是涨跌幅的均值比。

### 3. `shift(n)` — 平移（明天/昨天）

```python
close.shift(1)      # 昨天的收盘价
close.shift(-1)     # 明天的收盘价 ← 这是未来函数，慎用！
```

**`shift` 是回测里最重要的函数，没有之一。** 第 6 章会专门讲：忘了 `shift(1)`，回测收益能凭空多一倍。

### 4. `cumprod()` — 复利累积

```python
(1 + rets).cumprod()      # 净值曲线
```

净值 = 初始资金 × Π(1+r)。**注意是乘不是加**，这是复利。

### 5. `cummax()` — 历史最高（算回撤用）

```python
equity / equity.cummax() - 1      # 回撤序列
```

`cummax` 给出"到今天为止的最高点"，用它做分母就知道当前离高点跌了多少。

### 6. `loc` / `iloc` — 取值

```python
close.loc["2024-01-02"]        # 按标签（日期）取
close.iloc[0]                  # 按位置取（第 0 行）
df.loc["2024-01-01":"2024-06-30"]   # 按日期区间切片（含两端）
```

用错会报错或取错数据。记住：**`loc` 看名字，`iloc` 看序号。**

### 7. `groupby` — 分组统计

```python
rets.groupby(rets.index.year).apply(lambda r: (1 + r).prod() - 1)   # 年度收益
panel.groupby("symbol")["close"].last()                             # 每只票最新价
```

`groupby` 的思维是"拆分-计算-合并"。金融里最常见的用法是**按年/按月/按股票分组**。

### 8. `resample` — 时间重采样

```python
daily.close.resample("W").last()      # 日线 → 周线
daily.close.resample("ME").last()     # 日线 → 月线（pandas 2.2+ 用 ME，老版本用 M）
```

**降频用 `last`（取期末值），不要用 `mean`。** 周线的收盘价是周五的收盘价，不是一周平均。

### 9. `merge` — 拼表

```python
pd.merge(df_price, df_factor, on=["date", "symbol"], how="left")
```

相当于 SQL 的 JOIN。`how="left"` 最常用：保住主表所有行，缺的补 NaN。

### 10. `apply` — 兜底方案

```python
df.groupby("symbol").apply(lambda g: g["close"].max())
```

**`apply` 是最后手段，不是首选。** 它慢（本质是循环）。能用向量化表达就别用 `apply`。
但有些逻辑（比如复杂的自定义指标）确实只能用 `apply`，这时别犹豫。

## 五、新手最常踩的四个坑

**坑 1：链式赋值**

```python
df[df["close"] > 100]["signal"] = 1      # ❌ 不生效，还会警告
df.loc[df["close"] > 100, "signal"] = 1  # ✅
```

**坑 2：忘了索引对齐导致 NaN 一片**

两个序列算加减，先确认索引一致。用 `df.index.equals(other.index)` 检查。

**坑 3：用 `inplace=True`**

看起来省事，实际会让调试变难（你无法回退看中间结果）。**在新版 pandas 里大量 `inplace` 已被弃用，别养成习惯。**

**坑 4：以为 `df.copy()` 是多余的**

pandas 的操作有时返回视图、有时返回副本。加一句 `.copy()` 能避免 90% 的诡异 bug，代价是几乎为零。

## 六、动手练习（必须做，不做等于没学）

打开 Python，用 `src/data.py` 拉一段茅台的数据，然后：

1. 算出 2023 年每个月的收益率（提示：`resample`）
2. 算出 20 日均线，以及收盘价相对均线的偏离百分比
3. 找出 2023 年单日涨幅最大的 5 天
4. 算出 2023 年最大回撤，并找出回撤最深的那一天
5. 用 for 循环和向量化分别算一遍日收益，用 `%timeit` 比较耗时

## 自测题（不看笔记回答）

1. `close.pct_change()` 的第一个值是什么？为什么？
2. `shift(1)` 和 `shift(-1)` 分别代表什么？哪个在回测里是危险的？
3. 为什么年化波动率要乘 `√252` 而不是 `252`？
4. `rolling(20).mean()` 的第 1~19 个值是什么？
5. 两个日期不同的 Series 相加，pandas 会怎么处理？

答不上来就回去把对应的小节再看一遍。**这五题是后面所有内容的地基。**

---

下一章：[A 股的游戏规则](02-A股游戏规则.md) —— 这一章没有代码，但能让你少亏很多钱。
