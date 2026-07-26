"""
融合實驗用資料建構 — 修正 C-2/C-3/C-4/C-5 後的版本.

與 core/dataset.py 的差異:
  C-3 修正: 指標在「完整序列」上因果計算 (rolling), z-score 只用訓練期統計量,
            且 train/val/test 用同一組統計量
  C-4 修正: 測試期樣本的 window 可回溯訓練期資料 (完整序列滑窗, 依決策日分割)
  C-5 修正: meta date = 決策日 t (window 最後一天), label = t+1 (或 t+5) 相對 t 的漲跌
  C-2 修正: 時間序 val (非 random_split), 模型選擇只看 val

特徵:
  F=9  : 論文 9 技術指標
  F=16 : 9 指標 + main 分支 features_1day 的 7 個 K 棒特徵 (融合)

標籤 (同一批 X 共用):
  y_next: close[t+1] > close[t]
  y_5d  : close[t+5] > close[t]
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed

from pip_algorithm import extract_pips
from vg_graph import build_visibility_graph
from subgraph import build_3d_feature


# ────────────────────────────────────────────────
# 指標 (與 core/indicators.py 相同公式, 但不做 z-score — 正規化移到外層)
# ────────────────────────────────────────────────

def compute_indicators_raw(df: pd.DataFrame, n: int = 100) -> np.ndarray:
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)

    sma = close.rolling(n, min_periods=1).mean()
    weights = np.arange(1, n + 1)
    wma = close.rolling(n, min_periods=1).apply(
        lambda x: np.dot(x[-len(weights):] if len(x) >= n else x,
                         weights[-len(x):]) / weights[-len(x):].sum(),
        raw=True,
    )
    mom = close - close.shift(n - 1)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    macd = diff.ewm(span=n, adjust=False).mean()
    hh = high.rolling(n, min_periods=1).max()
    ll = low.rolling(n, min_periods=1).min()
    williams_r = (hh - close) / (hh - ll + 1e-9) * 100
    typical = (high + low + close) / 3
    sm = typical.rolling(n, min_periods=1).mean()
    md = typical.rolling(n, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (typical - sm) / (0.015 * md + 1e-9)
    k = (close - ll) / (hh - ll + 1e-9) * 100
    d = k.rolling(n, min_periods=1).mean()
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n, min_periods=1).mean()
    dw = (-delta).clip(lower=0).rolling(n, min_periods=1).mean()
    rs = up / (dw + 1e-9)
    rsi = 100 - 100 / (1 + rs)

    feats = np.stack([sma, wma, mom, macd, williams_r, cci, k, d, rsi], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# ────────────────────────────────────────────────
# main 分支 features_1day 的 7 個 K 棒特徵 (欄位改小寫, 還原價)
# ────────────────────────────────────────────────

def compute_candle_features(df: pd.DataFrame) -> np.ndarray:
    O, H, L, C, V = (df['open'], df['high'], df['low'],
                     df['close'], df['volume'])
    Cp = C.shift(1)
    upper = (H - np.maximum(O, C)) / C * 100
    lower = (np.minimum(O, C) - L) / C * 100
    body = (C - O) / C * 100
    gap = (O - Cp) / C * 100
    close_chg = (C - Cp) / C * 100
    v5ma = V.rolling(5, min_periods=1).mean()
    vol_ratio = (V - v5ma) / v5ma.replace(0, np.nan)
    trend = (C.shift(2) - C.shift(7)) / C.shift(7) * 100
    feats = np.stack([upper, lower, body, gap, close_chg,
                      vol_ratio, trend], axis=1)
    return np.nan_to_num(feats.astype(np.float64), nan=0.0,
                         posinf=0.0, neginf=0.0)


def _build_one_sample(args):
    series, feat_window, y_next, y_5d, m_pips, N, g, ticker, date = args
    try:
        pips, scores = extract_pips(series, m=m_pips)
        G = build_visibility_graph(series, pips)
        X = build_3d_feature(series, pips, scores, feat_window, G, N=N, g=g)
        return (X, y_next, y_5d, ticker, date)
    except Exception:
        return None


def build_samples(
    stock_data: dict,
    train_end: str,
    window: int = 100,
    m_pips: int = 40,
    N: int = 15,
    g: int = 5,
    with_candle: bool = False,
    n_workers: int = 10,
):
    """
    對每檔股票的「完整序列」滑窗建樣本 (stride=1).
    決策日 t = window 最後一天; z-score 統計量只取決策日 <= train_end 的資料.

    Returns: dict(X, y_next, y_5d, tickers, dates)
    """
    train_end_ts = pd.to_datetime(train_end)
    all_tasks = []

    for ticker, df in stock_data.items():
        T = len(df)
        if T < window + 6:
            continue
        ind = compute_indicators_raw(df, n=window)          # (T, 9)
        if with_candle:
            cnd = compute_candle_features(df)               # (T, 7)
            feats = np.concatenate([ind, cnd], axis=1)      # (T, 16)
        else:
            feats = ind

        # C-3 修正: z-score 只用訓練期列
        train_mask = df.index <= train_end_ts
        mean = feats[train_mask].mean(axis=0, keepdims=True)
        std = feats[train_mask].std(axis=0, keepdims=True) + 1e-9
        feats = ((feats - mean) / std).astype(np.float32)

        close = df['close'].values.astype(np.float64)

        # 決策日 t: 需要完整 window 與 t+5 的未來價格
        for t in range(window - 1, T - 5):
            series = close[t - window + 1: t + 1].copy()
            feat_window = feats[t - window + 1: t + 1].copy()
            y_next = 1 if close[t + 1] > close[t] else 0
            y_5d = 1 if close[t + 5] > close[t] else 0
            all_tasks.append((series, feat_window, y_next, y_5d,
                              m_pips, N, g, ticker, df.index[t]))

    total = len(all_tasks)
    print(f"  建構 {total} 個樣本 (workers={n_workers}, "
          f"F={'16' if with_candle else '9'})...", flush=True)

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_build_one_sample, task) for task in all_tasks]
        for future in as_completed(futures):
            r = future.result()
            done += 1
            if r is not None:
                results.append(r)
            if done % 2000 == 0:
                print(f"    {done}/{total} ({done/total:.0%})", flush=True)

    # 依 (date, ticker) 排序
    results.sort(key=lambda r: (r[4], r[3]))
    X = np.stack([r[0] for r in results])
    y_next = np.array([r[1] for r in results], dtype=np.int64)
    y_5d = np.array([r[2] for r in results], dtype=np.int64)
    tickers = np.array([r[3] for r in results])
    dates = np.array([str(r[4])[:10] for r in results])

    print(f"  完成: {len(y_next)} samples, X shape={X.shape}", flush=True)
    return {'X': X, 'y_next': y_next, 'y_5d': y_5d,
            'tickers': tickers, 'dates': dates}


class ArrayDataset(Dataset):
    """包一份 (X, y) 陣列 + date_to_indices, 相容 DateGroupedBatchSampler."""

    def __init__(self, X, y, dates, tickers=None):
        self.X = X
        self.y = y
        self.dates = dates
        self.tickers = tickers
        self.date_to_indices = {}
        for i, d in enumerate(dates):
            self.date_to_indices.setdefault(d, []).append(i)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx]).float(),
                torch.tensor(self.y[idx], dtype=torch.long))


def split_indices_by_date(dates, train_end, val_end, train_stride=1):
    """
    時間序切分 (C-2 修正: val 為訓練期尾段, 不用 random_split).
    train: date <= train_end (依日期做 stride 抽樣)
    val:   train_end < date <= val_end
    test:  date > val_end
    """
    train_idx, val_idx, test_idx = [], [], []
    uniq_dates = sorted(set(dates))
    train_dates = [d for d in uniq_dates if d <= train_end]
    kept_train_dates = set(train_dates[::train_stride])

    for i, d in enumerate(dates):
        if d <= train_end:
            if d in kept_train_dates:
                train_idx.append(i)
        elif d <= val_end:
            val_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, val_idx, test_idx
