import numpy as np
import pandas as pd

def compute_indicators(df: pd.DataFrame, n: int = 100) -> np.ndarray:
    o = df['open'].astype(float).values
    h = df['high'].astype(float).values
    l = df['low'].astype(float).values
    c = df['close'].astype(float).values
    v = df['volume'].astype(float).values
    T = len(c)
    hl = h - l
    hl = np.where(hl < 1e-9, 1e-9, hl)
    prev_c = np.empty(T); prev_c[0] = c[0]; prev_c[1:] = c[:-1]
    upper_shadow = (h - np.maximum(o, c)) / hl
    lower_shadow = (np.minimum(o, c) - l) / hl
    body = (c - o) / hl
    gap = (o - prev_c) / (np.abs(prev_c) + 1e-9)
    close_change = (c - prev_c) / (np.abs(prev_c) + 1e-9)
    vol_ma5 = pd.Series(v).rolling(5, min_periods=1).mean().values
    vol_ma5 = np.where(vol_ma5 < 1, 1, vol_ma5)
    vol_ratio_5d = v / vol_ma5
    c5 = np.empty(T); c5[:5] = c[:5]; c5[5:] = c[:-5]
    trend_5d = (c - c5) / (np.abs(c5) + 1e-9)
    prev_upper = np.empty(T); prev_upper[0] = 0; prev_upper[1:] = upper_shadow[:-1]
    prev_lower = np.empty(T); prev_lower[0] = 0; prev_lower[1:] = lower_shadow[:-1]
    prev_body = np.empty(T); prev_body[0] = 0; prev_body[1:] = body[:-1]
    prev2_body = np.empty(T); prev2_body[:2] = 0; prev2_body[2:] = body[:-2]
    c3 = np.empty(T); c3[:3] = c[:3]; c3[3:] = c[:-3]
    cumul_3d = (c - c3) / (np.abs(c3) + 1e-9)
    vol_ma3 = pd.Series(v).rolling(3, min_periods=1).mean().values
    vol_ma3 = np.where(vol_ma3 < 1, 1, vol_ma3)
    vol_ratio_3d = v / vol_ma3
    h20 = pd.Series(h).rolling(20, min_periods=1).max().values
    l20 = pd.Series(l).rolling(20, min_periods=1).min().values
    range20 = h20 - l20
    range20 = np.where(range20 < 1e-9, 1e-9, range20)
    range_pos = (c - l20) / range20
    feats = np.stack([upper_shadow, lower_shadow, body, gap, close_change,
        vol_ratio_5d, trend_5d, prev_upper, prev_lower, prev_body,
        prev2_body, cumul_3d, vol_ratio_3d, range_pos], axis=1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-9
    feats = (feats - mean) / std
    return feats.astype(np.float32)
