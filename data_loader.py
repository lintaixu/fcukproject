"""
台股資料下載器 (使用 yfinance, 需 .TW suffix).

若 yfinance 連線失敗（例如此處沙箱無外網），會 fallback 到合成資料模擬。
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd


# 台灣 50 ETF 成分股 (完整清單)
TW50_TICKERS = [
    "2330.TW",  # 台積電
    "2317.TW",  # 鴻海
    "2454.TW",  # 聯發科
    "2412.TW",  # 中華電
    "2308.TW",  # 台達電
    "2382.TW",  # 廣達
    "2881.TW",  # 富邦金
    "2882.TW",  # 國泰金
    "2891.TW",  # 中信金
    "2303.TW",  # 聯電
    "1301.TW",  # 台塑
    "1303.TW",  # 南亞
    "2002.TW",  # 中鋼
    "3711.TW",  # 日月光投控
    "2207.TW",  # 和泰車
    "2886.TW",  # 兆豐金
    "2884.TW",  # 玉山金
    "2892.TW",  # 第一金
    "2880.TW",  # 華南金
    "2885.TW",  # 元大金
    "2887.TW",  # 台新金
    "5880.TW",  # 合庫金
    "2890.TW",  # 永豐金
    "1326.TW",  # 台化
    "1101.TW",  # 台泥
    "1102.TW",  # 亞泥
    "2912.TW",  # 統一超
    "2301.TW",  # 光寶科
    "2357.TW",  # 華碩
    "2395.TW",  # 研華
    "3008.TW",  # 大立光
    "2327.TW",  # 國巨
    "3045.TW",  # 台灣大
    "4904.TW",  # 遠傳
    "2379.TW",  # 瑞昱
    "3034.TW",  # 聯詠
    "2345.TW",  # 智邦
    "2603.TW",  # 長榮
    "2609.TW",  # 陽明
    "2615.TW",  # 萬海
    "5871.TW",  # 中租-KY
    "2105.TW",  # 正新
    "1216.TW",  # 統一
    "2633.TW",  # 台灣高鐵
    "9910.TW",  # 豐泰
    "6505.TW",  # 台塑化
    "2474.TW",  # 可成
    "4938.TW",  # 和碩
    "2347.TW",  # 聯強
    "8046.TW",  # 南電
]


def fetch_yfinance(tickers, start: str, end: str, cache_dir: str = "./cache"):
    """
    用 yfinance 下載資料, 並 cache 在本地以避免重複下載。
    """
    import yfinance as yf
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    out = {}
    for tk in tickers:
        cache_file = os.path.join(cache_dir, f"{tk}_{start}_{end}.parquet")
        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
        else:
            print(f"下載 {tk} ...")
            df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=False)
            if df.empty:
                continue
            # 攤平 multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.columns = [c.lower() for c in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
            df.to_parquet(cache_file)
        out[tk] = df
    return out


def synthesize_taiwan_stocks(tickers, start: str, end: str, seed: int = 42):
    """
    當無法連線時 fallback: 用 GBM 合成「假台股」資料,
    讓整個 pipeline 仍可運行驗證。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)

    out = {}
    for i, tk in enumerate(tickers):
        # 不同股票用不同 drift / vol
        drift = rng.uniform(-0.0005, 0.001)
        vol = rng.uniform(0.012, 0.025)
        # GBM
        rets = rng.normal(drift, vol, n_days)
        close = 100 * np.exp(np.cumsum(rets))
        # 加一點 chart-like 結構: 周期性 + 趨勢
        close += np.sin(np.linspace(0, 8 * np.pi, n_days)) * 5

        # 推導 OHLV
        open_ = close * (1 + rng.normal(0, 0.005, n_days))
        high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        volume = rng.integers(1_000_000, 50_000_000, n_days)

        df = pd.DataFrame({
            'open': open_, 'high': high, 'low': low,
            'close': close, 'volume': volume
        }, index=dates)
        df.index.name = 'Date'
        out[tk] = df

    return out


def fetch_tw_stocks(tickers=None, start="2018-01-01", end="2024-12-31",
                    use_synthetic: bool = False):
    """
    主接口: 嘗試用 yfinance, 失敗則 fallback 到合成資料。
    """
    if tickers is None:
        tickers = TW50_TICKERS

    if use_synthetic:
        print("[INFO] 使用合成資料 (synthetic mode).")
        return synthesize_taiwan_stocks(tickers, start, end)

    try:
        data = fetch_yfinance(tickers, start, end)
        if not data:
            raise RuntimeError("yfinance 回傳空")
        return data
    except Exception as e:
        print(f"[WARN] yfinance 失敗 ({e}), 改用合成資料.")
        return synthesize_taiwan_stocks(tickers, start, end)


if __name__ == "__main__":
    # 測試: 用合成模式 (沙箱無外網)
    data = fetch_tw_stocks(
        tickers=TW50_TICKERS[:3],
        start="2020-01-01",
        end="2024-12-31",
        use_synthetic=True,
    )
    for tk, df in data.items():
        print(f"{tk}: {len(df)} 天, 範圍 {df.index[0].date()} ~ {df.index[-1].date()}")
        print(df.head(3))
        print()
