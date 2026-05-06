"""
ChartGCNDataset — 把多檔台股資料切成 rolling windows,
每個 window 經過 PIP+VG+Subgraph → 一個 (X, label) 樣本.

注意: 為了避免每次 epoch 重新跑 PIP+VG (慢), 這裡會在初始化時
       一次性把所有 sample 都建好快取在記憶體中。
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pip_algorithm import extract_pips
from vg_graph import build_visibility_graph
from subgraph import build_3d_feature
from indicators import compute_indicators


class ChartGCNDataset(Dataset):
    def __init__(
        self,
        stock_data: dict,           # {ticker: DataFrame}
        window: int = 100,
        m_pips: int = 40,
        N: int = 15,
        g: int = 5,
        indicator_n: int = 14,
        stride: int = 1,
        verbose: bool = True,
    ):
        self.window = window
        self.m_pips = m_pips
        self.N = N
        self.g = g

        self.X_list = []        # list of (N, g, F) arrays
        self.A_list = []        # list of (N, g, g) adjacency matrices
        self.y_list = []        # list of int labels
        self.meta = []          # (ticker, end_date) for traceability

        for ticker, df in stock_data.items():
            if len(df) < window + 1:
                continue
            feats = compute_indicators(df, n=indicator_n)  # (T, F)
            close = df['close'].values

            samples = 0
            for end in range(window, len(df) - 1, stride):
                start = end - window
                series = close[start:end]                  # (window,)
                feat_window = feats[start:end]             # (window, F)
                label = 1 if close[end] > close[end - 1] else 0

                try:
                    pips, scores = extract_pips(series, m=m_pips)
                    G = build_visibility_graph(series, pips)
                    X, _A = build_3d_feature(
                        series, pips, scores, feat_window, G,
                        N=N, g=g,
                    )
                except Exception as e:
                    if verbose:
                        print(f"[skip] {ticker} @ {df.index[end]}: {e}")
                    continue

                self.X_list.append(X)
                self.A_list.append(_A)
                self.y_list.append(label)
                self.meta.append((ticker, df.index[end]))
                samples += 1

            if verbose:
                print(f"  {ticker}: {samples} samples")

        # 轉成 numpy / tensor
        self.X = np.stack(self.X_list, axis=0)             # (M, N, g, F)
        self.A = np.stack(self.A_list, axis=0)             # (M, N, g, g)
        self.y = np.array(self.y_list, dtype=np.int64)     # (M,)

        if verbose:
            n_pos = (self.y == 1).sum()
            n_neg = (self.y == 0).sum()
            print(f"\nTotal samples: {len(self.y)} "
                  f"(漲={n_pos}, 跌={n_neg}, 漲比例={n_pos/len(self.y):.2%})")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]).float(),
            torch.from_numpy(self.A[idx]).float(),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def split_by_date(stock_data: dict, train_end: str, test_end: str = None):
    """
    依日期切分 dict of DataFrames 成 (train_dict, test_dict).
    """
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
        use_synthetic=True,
    )

    train_data, test_data = split_by_date(data, "2023-12-31")
    print(f"訓練股票: {list(train_data.keys())}")

    ds = ChartGCNDataset(
        train_data,
        window=100, m_pips=40, N=15, g=5,
        stride=5,  # 加大 stride 加速測試
    )
    X, y = ds[0]
    print(f"\nSample X shape: {X.shape}, y: {y.item()}")
