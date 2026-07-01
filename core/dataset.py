"""
ChartGCNDataset — 嚴格對齊論文 (Li et al., KBS 2022).
加入多線程並行建構 (PIP+VG+Subgraph 為 CPU-bound, 用 ProcessPoolExecutor 加速).

變更:
  - 移除 adjacency matrix A (論文模型不含顯式 GCN 層)
  - 指標窗口 n 預設跟隨 rolling window
  - __getitem__ 回傳 (X, y) 而非 (X, A, y)
  - 多線程/多進程並行建構樣本
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
from indicators import compute_indicators


def _build_one_sample(args):
    """
    建構單一樣本 (設計為 top-level function 供 multiprocessing 呼叫).
    Returns: (X, label, meta_key) or None
    """
    series, feat_window, label, m_pips, N, g, meta_key = args
    try:
        pips, scores = extract_pips(series, m=m_pips)
        G = build_visibility_graph(series, pips)
        X = build_3d_feature(series, pips, scores, feat_window, G, N=N, g=g)
        return (X, label, meta_key)
    except Exception:
        return None


class ChartGCNDataset(Dataset):
    def __init__(
        self,
        stock_data: dict,           # {ticker: DataFrame}
        window: int = 100,
        m_pips: int = 40,
        N: int = 15,
        g: int = 5,
        indicator_n: int = None,    # None → 跟隨 window
        stride: int = 1,
        n_workers: int = None,      # None → auto (CPU cores - 1)
        verbose: bool = True,
    ):
        self.window = window
        self.m_pips = m_pips
        self.N = N
        self.g = g
        self.indicator_n = indicator_n if indicator_n is not None else window

        if n_workers is None:
            n_workers = max(1, os.cpu_count() // 2)

        self.X_list = []
        self.y_list = []
        self.meta = []

        # 先計算所有股票的指標和 task 清單
        all_tasks = []
        for ticker, df in stock_data.items():
            if len(df) < window + 1:
                continue
            feats = compute_indicators(df, n=self.indicator_n)
            close = df['close'].values

            for end in range(window, len(df) - 1, stride):
                start = end - window
                series = close[start:end].copy()
                feat_window = feats[start:end].copy()
                label = 1 if close[end] > close[end - 1] else 0
                meta_key = (ticker, df.index[end])

                all_tasks.append(
                    (series, feat_window, label, m_pips, N, g, meta_key)
                )

        total = len(all_tasks)
        if verbose:
            print(f"  建構 {total} 個樣本 (workers={n_workers})...")

        # 多進程並行建構
        done = 0
        if n_workers > 1 and total > 50:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_build_one_sample, t): i
                           for i, t in enumerate(all_tasks)}
                for future in as_completed(futures):
                    result = future.result()
                    done += 1
                    if result is not None:
                        X, label, meta_key = result
                        self.X_list.append(X)
                        self.y_list.append(label)
                        self.meta.append(meta_key)
                    if verbose and done % 500 == 0:
                        print(f"    {done}/{total} ({done/total:.0%})")
        else:
            # 單線程 fallback
            for task in all_tasks:
                result = _build_one_sample(task)
                done += 1
                if result is not None:
                    X, label, meta_key = result
                    self.X_list.append(X)
                    self.y_list.append(label)
                    self.meta.append(meta_key)
                if verbose and done % 500 == 0:
                    print(f"    {done}/{total} ({done/total:.0%})")

        # 依 meta (ticker, date) 排序以確保 date_to_indices 正確
        if self.X_list:
            sorted_indices = sorted(range(len(self.meta)),
                                    key=lambda i: (self.meta[i][1], self.meta[i][0]))
            self.X_list = [self.X_list[i] for i in sorted_indices]
            self.y_list = [self.y_list[i] for i in sorted_indices]
            self.meta = [self.meta[i] for i in sorted_indices]

        # 轉成 numpy
        self.X = np.stack(self.X_list, axis=0) if self.X_list else np.zeros((0, N, g, 9))
        self.y = np.array(self.y_list, dtype=np.int64) if self.y_list else np.zeros(0, dtype=np.int64)

        # 日期 → sample index 映射
        self.date_to_indices = {}
        for i, (ticker, date) in enumerate(self.meta):
            date_key = str(date)[:10]
            if date_key not in self.date_to_indices:
                self.date_to_indices[date_key] = []
            self.date_to_indices[date_key].append(i)

        if verbose:
            n_pos = (self.y == 1).sum()
            n_neg = (self.y == 0).sum()
            n_dates = len(self.date_to_indices)
            total_samples = len(self.y)
            ratio = n_pos / total_samples if total_samples > 0 else 0
            print(f"  完成: {total_samples} samples "
                  f"(漲={n_pos}, 跌={n_neg}, 漲比例={ratio:.2%})")
            if n_dates > 0:
                print(f"  Unique dates: {n_dates}, "
                      f"avg stocks/date: {total_samples/n_dates:.1f}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]).float(),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def split_by_date(stock_data: dict, train_end: str, test_end: str = None):
    """依日期切分 dict of DataFrames 成 (train_dict, test_dict)."""
    train, test = {}, {}
    train_end = pd.to_datetime(train_end)
    test_end = pd.to_datetime(test_end) if test_end else None

    for tk, df in stock_data.items():
        train_part = df[df.index <= train_end]
        if test_end:
            test_part = df[(df.index > train_end) & (df.index <= test_end)]
        else:
            test_part = df[df.index > train_end]
        if len(train_part) > 0:
            train[tk] = train_part
        if len(test_part) > 0:
            test[tk] = test_part
    return train, test


if __name__ == "__main__":
    from data_loader import fetch_tw_stocks, TW50_TICKERS

    data = fetch_tw_stocks(
        tickers=TW50_TICKERS[:3],
        start="2020-01-01", end="2024-12-31",
    )

    train_data, test_data = split_by_date(data, "2023-12-31")
    print(f"訓練股票: {list(train_data.keys())}")

    ds = ChartGCNDataset(
        train_data,
        window=100, m_pips=40, N=15, g=5,
        stride=5,
    )
    X, y = ds[0]
    print(f"\nSample X shape: {X.shape}, y: {y.item()}")
