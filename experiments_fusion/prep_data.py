"""下載/讀取股票資料, 建構 X9 / X16 / 平面特徵表並存檔."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_dataset import (build_samples, compute_indicators_raw,
                            compute_candle_features)

CACHE_DIR = r"c:\Users\alex\Desktop\專題進度\chartgcn\cache"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
START, END = "2018-01-01", "2024-12-31"
STATS_END = "2022-12-31"     # z-score 統計量只用到此日 (train 期)
WINDOW, M_PIPS, N, G = 100, 40, 15, 5

TICKERS = [
    "2330.TW", "2317.TW", "2454.TW", "2412.TW", "2308.TW", "2382.TW",
    "2881.TW", "2882.TW", "2891.TW", "2303.TW", "1301.TW", "1303.TW",
    "2002.TW", "3711.TW", "2207.TW", "2886.TW", "2884.TW", "2892.TW",
    "2880.TW", "2885.TW", "2887.TW", "5880.TW", "2890.TW", "1326.TW",
    "1101.TW", "1102.TW", "2912.TW", "2301.TW", "2357.TW", "2395.TW",
]


def fetch(tickers):
    import yfinance as yf
    out = {}
    for tk in tickers:
        cache_file = os.path.join(CACHE_DIR, f"{tk}_{START}_{END}.parquet")
        if os.path.exists(cache_file):
            out[tk] = pd.read_parquet(cache_file)
            continue
        try:
            print(f"下載 {tk} ...", flush=True)
            df = yf.download(tk, start=START, end=END,
                             progress=False, auto_adjust=True)
            if df.empty:
                print(f"  {tk} 空白, 跳過", flush=True)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.columns = [c.lower() for c in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
            df.to_parquet(cache_file)
            out[tk] = df
        except Exception as e:
            print(f"  {tk} 失敗: {e}", flush=True)
    return out


def build_flat(stock_data):
    """每檔股票每個決策日的平面特徵 (16 維, train 統計量 z-score) 與報酬."""
    stats_end = pd.to_datetime(STATS_END)
    rows = {'dates': [], 'tickers': [], 'feat': [],
            'y_next': [], 'y_5d': [], 'ret5d': []}
    for tk, df in stock_data.items():
        T = len(df)
        if T < WINDOW + 6:
            continue
        ind = compute_indicators_raw(df, n=WINDOW)
        cnd = compute_candle_features(df)
        feats = np.concatenate([ind, cnd], axis=1)
        mask = df.index <= stats_end
        mean = feats[mask].mean(axis=0, keepdims=True)
        std = feats[mask].std(axis=0, keepdims=True) + 1e-9
        feats = ((feats - mean) / std).astype(np.float32)
        close = df['close'].values.astype(np.float64)
        for t in range(WINDOW - 1, T - 5):
            rows['dates'].append(str(df.index[t])[:10])
            rows['tickers'].append(tk)
            rows['feat'].append(feats[t])
            rows['y_next'].append(1 if close[t + 1] > close[t] else 0)
            rows['y_5d'].append(1 if close[t + 5] > close[t] else 0)
            rows['ret5d'].append((close[t + 5] / close[t] - 1) * 100)
    return {
        'dates': np.array(rows['dates']),
        'tickers': np.array(rows['tickers']),
        'feat': np.stack(rows['feat']),
        'y_next': np.array(rows['y_next'], dtype=np.int64),
        'y_5d': np.array(rows['y_5d'], dtype=np.int64),
        'ret5d': np.array(rows['ret5d'], dtype=np.float64),
    }


if __name__ == '__main__':
    data = fetch(TICKERS)
    print(f"取得 {len(data)} 檔股票", flush=True)
    for tk, df in list(data.items())[:3]:
        print(f"  {tk}: {len(df)} 天 {df.index[0].date()}~{df.index[-1].date()}")

    print("\n[FLAT] 平面特徵表...", flush=True)
    flat = build_flat(data)
    np.savez_compressed(os.path.join(OUT_DIR, 'flat.npz'), **flat)
    print(f"  flat: {flat['feat'].shape}", flush=True)

    print("\n[X9] 圖特徵 (9 指標)...", flush=True)
    d9 = build_samples(data, STATS_END, window=WINDOW, m_pips=M_PIPS,
                       N=N, g=G, with_candle=False, n_workers=10)
    np.savez_compressed(os.path.join(OUT_DIR, 'data_X9.npz'), **d9)

    print("\n[X16] 圖特徵 (9 指標 + 7 K棒)...", flush=True)
    d16 = build_samples(data, STATS_END, window=WINDOW, m_pips=M_PIPS,
                        N=N, g=G, with_candle=True, n_workers=10)
    np.savez_compressed(os.path.join(OUT_DIR, 'data_X16.npz'), **d16)

    print("\nPREP DONE", flush=True)
