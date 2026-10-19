"""
数据层：把"外部世界的数据"变成"干净的 DataFrame"。

一个量化项目 70% 的时间花在数据上，而 90% 的错误也来自数据。
这个文件只做三件事：
  1. 拉 A 股/指数日线（东方财富 + 腾讯双数据源，免费、不注册、不要 token）
  2. 缓存到本地 parquet，第二次运行秒开，断网也能用
  3. 三级兜底：真实快照 → 合成数据，保证教程在任何环境下都能跑

⚠️ 为什么不直接用 akshare？
akshare 很好用，但它依赖 30+ 个包、版本更新频繁、接口三天两头改名。
这个项目的目标是「让你看懂数据是怎么来的」，所以用最朴素的方式：
一个 HTTP 请求 + 一段 JSON 解析，60 行代码，你能逐行读懂、也能自己改接口。

⚠️ 为什么要两个数据源？
因为免费接口会限流、会改协议、会抽风。
**线上系统永远要有备份数据源** —— 这不是过度设计，是基本素养。
本文件里东财挂了自动切腾讯，这是真实工程里必须有的降级逻辑。

设计原则：**接口稳定，实现可替换**。
上层代码永远只调 load_daily()，不关心数据是下载的、读缓存的、快照的还是合成的。
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from .config import DATA_DIR, ROOT

COLUMNS = ["open", "high", "low", "close", "volume", "amount", "pct_chg"]

SAMPLE_DIR = ROOT / "data" / "sample"      # 随仓库分发的真实数据快照

_EM_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_EM_FIELDS = "f51,f52,f53,f54,f55,f56,f57,f59,f61"   # 日期,开,收,高,低,量,额,涨跌幅,换手

_TX_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"


# ---------------------------------------------------------------- 代码转换

def _em_secid(symbol: str, kind: str = "stock") -> str:
    """东财的 secid：'1.' 上交所，'0.' 深交所。"""
    if kind == "index":
        market = "1" if symbol.startswith("000") else "0"
    else:
        market = "1" if symbol[0] in "569" else "0"
    return f"{market}.{symbol}"


def _tx_code(symbol: str, kind: str = "stock") -> str:
    """腾讯的代码：'sh600519' / 'sz000001'。"""
    if kind == "index":
        prefix = "sh" if symbol.startswith("000") else "sz"
    else:
        prefix = "sh" if symbol[0] in "569" else "sz"
    return f"{prefix}{symbol}"


def _get_json(url: str, retries: int = 3, timeout: int = 20) -> dict:
    """带重试和退避的 GET。网络抖动是常态，不重试的爬虫都是半成品。"""
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
                "Referer": "https://quote.eastmoney.com/",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last = exc
            time.sleep(0.8 * (i + 1))       # 指数退避，别把服务器打疼
    raise RuntimeError(f"请求失败: {last}")


# ---------------------------------------------------------------- 数据源 1：东财

def fetch_eastmoney(symbol: str, start: str, end: str,
                    kind: str = "stock", adjust: str = "qfq") -> pd.DataFrame:
    """东方财富日线。字段最全（含成交额、涨跌幅），优先用它。"""
    fqt = {"": 0, "none": 0, "qfq": 1, "hfq": 2}[adjust]
    params = {
        "secid": _em_secid(symbol, kind),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": _EM_FIELDS,
        "klt": "101",          # 101=日线 102=周线 103=月线
        "fqt": str(fqt),       # 0=不复权 1=前复权 2=后复权
        "beg": start.replace("-", ""),
        "end": end.replace("-", ""),
        "lmt": "100000",
    }
    url = f"{_EM_URL}?{urllib.parse.urlencode(params)}"
    data = _get_json(url).get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        raise ValueError(f"{symbol} 无数据")

    rows = [line.split(",") for line in klines]
    df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low",
                                     "volume", "amount", "pct_chg", "turnover"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for c in COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[COLUMNS]


# ---------------------------------------------------------------- 数据源 2：腾讯

def fetch_tencent(symbol: str, start: str, end: str,
                  kind: str = "stock", adjust: str = "qfq") -> pd.DataFrame:
    """腾讯日线。单次最多返回 ~640 根，所以要分段拉。

    **这是工程上的常见模式：接口有单次上限，就按时间切片循环拉，再拼起来。**
    """
    code = _tx_code(symbol, kind)
    key = {"qfq": "qfqday", "hfq": "hfqday", "": "day", "none": "day"}[adjust]

    chunks = []
    s = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while s < end_ts:
        e = min(s + pd.DateOffset(years=2), end_ts)
        param = f"{code},day,{s:%Y-%m-%d},{e:%Y-%m-%d},640,{adjust or ''}"
        payload = _get_json(f"{_TX_URL}?param={param}")

        data = payload.get("data")
        node = data.get(code) if isinstance(data, dict) else None
        rows = (node or {}).get(key) or (node or {}).get("day") or []
        if rows:
            chunks.append(pd.DataFrame(
                [r[:6] for r in rows],
                columns=["date", "open", "close", "high", "low", "volume"],
            ))
        s = e
        time.sleep(0.25)
        if e >= end_ts:
            break

    if not chunks:
        raise ValueError(f"{symbol} 无数据")

    df = pd.concat(chunks).drop_duplicates("date")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = df["volume"] * 100            # 腾讯给的是「手」，统一成「股」
    df["amount"] = np.nan                        # 腾讯不提供，留空
    df["pct_chg"] = df["close"].pct_change() * 100
    df = df.loc[(df.index >= start) & (df.index <= end)]
    return df[COLUMNS]


def fetch_any(symbol: str, start: str, end: str,
              kind: str = "stock", adjust: str = "qfq") -> pd.DataFrame:
    """双源依次尝试。第一个失败就换第二个，两个都挂就抛异常。"""
    errors = []
    for name, fn in (("eastmoney", fetch_eastmoney), ("tencent", fetch_tencent)):
        try:
            df = fn(symbol, start, end, kind=kind, adjust=adjust)
            if len(df):
                return df
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__} {exc}")
    raise RuntimeError(" | ".join(errors))


# ---------------------------------------------------------------- 合成数据

def synthetic_prices(
    symbol: str = "SYN",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    annual_drift: float = 0.06,
    annual_vol: float = 0.28,
    seed: int | None = None,
) -> pd.DataFrame:
    """生成一段「像股票」的合成价格。

    用几何布朗运动（GBM）：每天的对数收益 ~ 正态分布。
    参数按 A 股个股的典型值给：年化波动 28%、年化漂移 6%。

    种子由 symbol 决定，所以同一只票每次生成的结果完全一样 ——
    这一点在调试策略时非常重要，否则你改一行代码结果就变，根本没法对比。
    """
    if seed is None:
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    idx = pd.bdate_range(start, end)      # 只取工作日，近似交易日
    n = len(idx)
    dt = 1 / 252
    mu = annual_drift - 0.5 * annual_vol**2
    log_ret = rng.normal(mu * dt, annual_vol * np.sqrt(dt), n)

    close = 10.0 * np.exp(np.cumsum(log_ret))
    intraday = np.abs(rng.normal(0, 0.008, n)) + 0.002
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)
    prev = np.concatenate([[close[0]], close[:-1]])

    df = pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": rng.integers(1e6, 5e7, n).astype(float),
            "amount": close * rng.integers(1e6, 5e7, n),
            "pct_chg": (close / prev - 1) * 100,
        },
        index=idx,
    )
    df.index.name = "date"
    return df


# ---------------------------------------------------------------- 统一入口

def _sample_file(symbol: str, adjust: str, kind: str):
    return SAMPLE_DIR / f"{kind}_{symbol}_{adjust or 'raw'}.csv"


def load_daily(
    symbol: str = "000001",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    adjust: str = "qfq",
    kind: str = "stock",
    use_cache: bool = True,
    offline: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    """加载单只标的的日线数据。这是全项目唯一的数据入口。

    数据查找顺序（很重要，决定了结果的可复现性）：
        本地缓存 → 仓库自带快照 → 联网下载 → 合成数据

    参数
    ----
    symbol  : 6 位代码，如 "000001"（平安银行）、"600519"（贵州茅台）
    adjust  : "qfq" 前复权 / "hfq" 后复权 / "" 不复权
              —— 新手最容易忽略、后果最严重的参数，详见 docs/03-数据.md
    kind    : "stock" 个股 / "index" 指数
    offline : 强制离线，跳过联网
    refresh : 强制联网刷新（忽略缓存和快照）

    返回
    ----
    DataFrame，索引是日期，列是 open/high/low/close/volume/amount/pct_chg
    """
    name = f"{kind}_{symbol}_{adjust or 'raw'}_{start}_{end}"
    cache_path = DATA_DIR / f"{name}.parquet"

    # 1) 精确缓存
    if use_cache and not refresh and cache_path.exists():
        return pd.read_parquet(cache_path)

    # 2) 仓库自带真实快照（覆盖请求区间就用它，保证结果可复现）
    sample_path = _sample_file(symbol, adjust, kind)
    if not refresh and sample_path.exists():
        snap = pd.read_csv(sample_path, parse_dates=["date"]).set_index("date")
        if snap.index.min() <= pd.Timestamp(start) and snap.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=10):
            sub = snap.loc[start:end]
            if len(sub) > 20:
                if use_cache:
                    sub.to_parquet(cache_path)
                return sub.copy()

    # 3) 联网
    if offline:
        df = synthetic_prices(symbol, start, end)
    else:
        try:
            df = fetch_any(symbol, start, end, kind=kind, adjust=adjust)
        except Exception as exc:
            print(f"[warn] {symbol} 下载失败（{type(exc).__name__}: {exc}），改用合成数据")
            df = synthetic_prices(symbol, start, end)

    if use_cache:
        try:
            df.to_parquet(cache_path)
        except Exception:
            pass          # 缓存失败不影响主流程
    return df


def load_panel(
    symbols: list[str],
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """加载多只标的，拼成「长表」（long format）。

    长表结构：每行是 (日期, 股票, 各列)。做横截面因子计算时用它。
    对应地，宽表（pivot）更适合做时间序列计算。两种形状的转换见 src/factors.py。

    注意这里的 sleep(0.2)：并发打接口容易被封 IP，慢慢拉是美德。
    """
    frames = []
    for sym in symbols:
        try:
            df = load_daily(sym, start, end, adjust=adjust)
        except Exception as exc:
            print(f"[warn] 跳过 {sym}: {exc}")
            continue
        frames.append(df.assign(symbol=sym))
        time.sleep(0.2)
    if not frames:
        raise RuntimeError("一只标的都没加载成功")
    return pd.concat(frames).sort_index()


def snapshot(symbols: list[str], start: str, end: str,
             adjust: str = "qfq") -> None:
    """把当前数据存成仓库自带的 CSV 快照（维护者用，普通学习者不需要跑）。"""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        for kind in ("stock", "index"):
            try:
                df = load_daily(sym, start, end, adjust=adjust, kind=kind,
                                refresh=True)
            except Exception:
                continue
            out = _sample_file(sym, adjust, kind)
            df.to_csv(out)
            print(f"saved {out.name} ({len(df)} 行)")


# A 股常见指数，方便直接引用
INDEX = {
    "上证指数": "000001",
    "沪深300": "000300",
    "中证500": "000905",
    "创业板指": "399006",
}


if __name__ == "__main__":
    # 自检：python -m src.data
    df = load_daily("600519", "2023-01-01", "2024-12-31")
    print(df.tail(3))
    print(f"行数 {len(df)}，区间 {df.index.min().date()} → {df.index.max().date()}")
