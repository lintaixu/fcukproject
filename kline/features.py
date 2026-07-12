"""
K 線特徵計算。

提供三套特徵集（1 日 / 2 日 / 3 日）與除權息偵測。
`FEATURE_SETS` 為 (名稱, 函式) 清單，供上層迴圈套用比較。
"""

import numpy as np
import pandas as pd


def detect_exdiv_mask(df: pd.DataFrame, tol: float = 0.005) -> pd.Series:
    """以 Adj Close / Close 的跳動偵測除權息日，回傳布林遮罩。"""
    if 'Adj Close' not in df.columns or 'Close' not in df.columns:
        return pd.Series(False, index=df.index)
    ratio = df['Adj Close'] / df['Close']
    return ratio.pct_change().abs() > tol


def features_1day(df: pd.DataFrame) -> pd.DataFrame:
    """1 日特徵集（7 維）。"""
    O, H, L, C, V = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
    Cp = C.shift(1)
    upper     = (H - np.maximum(O, C)) / C * 100
    lower     = (np.minimum(O, C) - L) / C * 100
    body      = (C - O) / C * 100
    gap       = (O - Cp) / C * 100
    close_chg = (C - Cp) / C * 100
    v5ma      = V.rolling(5, min_periods=5).mean()
    vol_ratio = (V - v5ma) / v5ma.replace(0, np.nan)
    trend     = (C.shift(2) - C.shift(7)) / C.shift(7) * 100
    return pd.DataFrame({
        'upper': upper, 'lower': lower, 'body': body,
        'gap': gap, 'close_chg': close_chg,
        'vol_ratio': vol_ratio, 'trend': trend,
    }, index=df.index)


def features_2day(df: pd.DataFrame) -> pd.DataFrame:
    """2 日特徵集（10 維）：1 日特徵 + 前一日影線/實體。"""
    O, H, L, C, V = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
    O1, H1, L1, C1 = O.shift(1), H.shift(1), L.shift(1), C.shift(1)
    upper     = (H  - np.maximum(O,  C )) / C  * 100
    lower     = (np.minimum(O,  C ) - L ) / C  * 100
    body      = (C  - O ) / C  * 100
    p_upper   = (H1 - np.maximum(O1, C1)) / C1 * 100
    p_lower   = (np.minimum(O1, C1) - L1) / C1 * 100
    p_body    = (C1 - O1) / C1 * 100
    gap       = (O  - C1) / C  * 100
    close_chg = (C  - C1) / C  * 100
    v5ma      = V.rolling(5, min_periods=5).mean()
    vol_ratio = (V  - v5ma) / v5ma.replace(0, np.nan)
    trend     = (C.shift(2) - C.shift(7)) / C.shift(7) * 100
    return pd.DataFrame({
        'upper': upper, 'lower': lower, 'body': body,
        'p_upper': p_upper, 'p_lower': p_lower, 'p_body': p_body,
        'gap': gap, 'close_chg': close_chg,
        'vol_ratio': vol_ratio, 'trend': trend,
    }, index=df.index)


def features_3day(df: pd.DataFrame) -> pd.DataFrame:
    """3 日特徵集（12 維）：加入前兩日型態、累積報酬與區間位階。"""
    O, H, L, C, V = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
    O1, H1, L1, C1 = O.shift(1), H.shift(1), L.shift(1), C.shift(1)
    O2, H2, L2, C2 = O.shift(2), H.shift(2), L.shift(2), C.shift(2)
    upper        = (H  - np.maximum(O,  C )) / C  * 100
    lower        = (np.minimum(O,  C ) - L ) / C  * 100
    body         = (C  - O ) / C  * 100
    p1_body      = (C1 - O1) / C1 * 100
    p2_body      = (C2 - O2) / C2 * 100
    gap          = (O  - C1) / C  * 100
    close_chg    = (C  - C1) / C  * 100
    p1_gap       = (O1 - C2) / C1 * 100
    p1_close_chg = (C1 - C2) / C1 * 100
    cum_return   = (C  - C2) / C2 * 100
    v3ma         = V.rolling(3, min_periods=3).mean()
    vol_ratio_3  = (V - v3ma) / v3ma.replace(0, np.nan)
    h_max = pd.concat([H, H1, H2], axis=1).max(axis=1)
    l_min = pd.concat([L, L1, L2], axis=1).min(axis=1)
    denom = h_max - l_min
    range_pos = pd.Series(
        np.where(denom > 0, (C - l_min) / denom * 100, 50.0),
        index=df.index)
    return pd.DataFrame({
        'upper': upper, 'lower': lower, 'body': body,
        'p1_body': p1_body, 'p2_body': p2_body,
        'gap': gap, 'close_chg': close_chg,
        'p1_gap': p1_gap, 'p1_close_chg': p1_close_chg,
        'cum_return': cum_return,
        'vol_ratio_3': vol_ratio_3, 'range_pos': range_pos,
    }, index=df.index)


FEATURE_SETS = [
    ('1日-7特徵',  features_1day),
    ('2日-10特徵', features_2day),
    ('3日-12特徵', features_3day),
]


if __name__ == '__main__':
    n = 300
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({
        'Open': close + rng.normal(0, 0.5, n), 'High': close + 1.0,
        'Low': close - 1.0, 'Close': close,
        'Volume': rng.integers(1_000_000, 5_000_000, n), 'Adj Close': close,
    }, index=pd.date_range('2020-01-01', periods=n, freq='B'))
    for name, fn in FEATURE_SETS:
        f = fn(df)
        print(f"{name}: shape={f.shape}  欄位={list(f.columns)}")
    print(f"除權息偵測: {int(detect_exdiv_mask(df).sum())} 天 (合成資料應為 0)")
