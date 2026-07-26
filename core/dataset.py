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
from concurrent.futures import ProcessPoolExecutor

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
        norm_stats: dict = None,    # {ticker: (mean, std)} — 測試集應傳入
                                    # 訓練集的統計量, 避免前視偏誤 (C-3)
        min_date=None,              # 只保留決策日 > min_date 的樣本;
                                    # 搭配 warm-up 資料使用 (C-4)
        verbose: bool = True,
    ):
        self.window = window
        self.m_pips = m_pips
        self.N = N
        self.g = g
        self.indicator_n = indicator_n if indicator_n is not None else window
        self.norm_stats = {}        # 本 dataset 使用的統計量 (可傳給測試集)

        if n_workers is None:
            n_workers = max(1, os.cpu_count() // 2)
        if min_date is not None:
            min_date = pd.to_datetime(min_date)

        # ── Phase 1: 逐股計算指標, 只建「輕量描述子」(不複製資料切片) ──
        # 記憶體改善: 舊版把每個樣本的 series/feats 切片複製進 task 清單
        # (30 萬樣本 ≈ 2GB), 再 np.stack 又複製一次 → 大規模時 MemoryError。
        # 新版描述子排序後預先配置整塊 X, 逐筆就地填入, 峰值僅一份 X。
        close_by_tk = {}
        feats_by_tk = {}
        descriptors = []   # (ticker, start, end, label, decision_date)
        for ticker, df in stock_data.items():
            if len(df) < window + 1:
                continue
            stats = norm_stats.get(ticker) if norm_stats else None
            feats, used_stats = compute_indicators(
                df, n=self.indicator_n, stats=stats, return_stats=True)
            self.norm_stats[ticker] = used_stats
            close = df['close'].values
            close_by_tk[ticker] = close
            feats_by_tk[ticker] = feats

            for end in range(window, len(df) - 1, stride):
                start = end - window
                # 決策日 = window 最後一天 (end-1); label = 隔日相對決策日的漲跌
                decision_date = df.index[end - 1]
                if min_date is not None and decision_date <= min_date:
                    continue
                label = 1 if close[end] > close[end - 1] else 0
                descriptors.append((ticker, start, end, label, decision_date))

        # 依 (date, ticker) 排序 → 填入順序即最終順序, 免去事後重排複製
        descriptors.sort(key=lambda d: (d[4], d[0]))

        total = len(descriptors)
        F_dim = next(iter(feats_by_tk.values())).shape[1] if feats_by_tk else 9
        if verbose:
            print(f"  建構 {total} 個樣本 (workers={n_workers})...")

        X_all = np.zeros((total, N, g, F_dim), dtype=np.float32)
        y_all = np.zeros(total, dtype=np.int64)
        self.meta = []
        n_ok = 0
        done = 0

        def _iter_tasks(desc_chunk):
            for ticker, start, end, label, ddate in desc_chunk:
                yield (close_by_tk[ticker][start:end],
                       feats_by_tk[ticker][start:end],
                       label, m_pips, N, g, (ticker, ddate))

        if n_workers > 1 and total > 50:
            # 分塊送進 pool.map (保序), 記憶體以塊為上限
            CHUNK = 2000
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                for ci in range(0, total, CHUNK):
                    chunk = descriptors[ci:ci + CHUNK]
                    for result in pool.map(_build_one_sample,
                                           _iter_tasks(chunk)):
                        done += 1
                        if result is not None:
                            X, label, meta_key = result
                            X_all[n_ok] = X
                            y_all[n_ok] = label
                            self.meta.append(meta_key)
                            n_ok += 1
                        if verbose and done % 5000 == 0:
                            print(f"    {done}/{total} ({done/total:.0%})")
        else:
            for task in _iter_tasks(descriptors):
                result = _build_one_sample(task)
                done += 1
                if result is not None:
                    X, label, meta_key = result
                    X_all[n_ok] = X
                    y_all[n_ok] = label
                    self.meta.append(meta_key)
                    n_ok += 1
                if verbose and done % 5000 == 0:
                    print(f"    {done}/{total} ({done/total:.0%})")

        # 截尾 view (失敗樣本極少; 不另複製)
        self.X = X_all[:n_ok]
        self.y = y_all[:n_ok]

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


def split_by_date(stock_data: dict, train_end: str, test_end: str = None,
                  warmup_rows: int = 0):
    """
    依日期切分 dict of DataFrames 成 (train_dict, test_dict).

    warmup_rows: 測試段往前多帶的歷史列數 (供指標 rolling 與滑窗 warm-up,
                 修正 C-4)。搭配 ChartGCNDataset(min_date=train_end) 使用,
                 確保 warm-up 段不會產生測試樣本。
    """
    train, test = {}, {}
    train_end = pd.to_datetime(train_end)
    test_end = pd.to_datetime(test_end) if test_end else None

    for tk, df in stock_data.items():
        train_part = df[df.index <= train_end]
        pos = df.index.searchsorted(train_end, side='right')
        start_pos = max(0, pos - warmup_rows)
        if test_end:
            test_part = df.iloc[start_pos:][df.iloc[start_pos:].index <= test_end]
        else:
            test_part = df.iloc[start_pos:]
        if len(train_part) > 0:
            train[tk] = train_part
        if len(test_part) > pos - start_pos:
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
