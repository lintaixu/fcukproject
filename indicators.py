"""
9 個技術指標 (對應論文 Table 1).

Inputs: pandas DataFrame 含 'open', 'high', 'low', 'close' columns.
Output: shape (T, 9) 的 numpy array, 已經 z-score 標準化.
"""
import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    """
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close']
        n:  時間窗口

    Returns:
        np.ndarray of shape (T, 9), z-score normalized
    """
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)

    # 1. Simple n-day MA
    sma = close.rolling(n).mean()

    # 2. Weighted n-day MA
    weights = np.arange(1, n + 1)
    wma = close.rolling(n).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )

    # 3. Momentum (n-day)
    mom = close - close.shift(n - 1)

    # 4. MACD (簡化版: 12-day EMA - 26-day EMA, 再做 9-day EMA)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    macd = diff.ewm(span=9, adjust=False).mean()

    # 5. Larry Williams %R
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    williams_r = (hh - close) / (hh - ll + 1e-9) * 100

    # 6. CCI
    typical = (high + low + close) / 3
    sm = typical.rolling(n).mean()
    md = typical.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (typical - sm) / (0.015 * md + 1e-9)

    # 7. Stochastic K%
    k = (close - ll) / (hh - ll + 1e-9) * 100

    # 8. Stochastic D% (10-day moving average of K%, 對應論文 Table 1)
    d = k.rolling(10).mean()

    # 9. RSI
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dw = (-delta).clip(lower=0).rolling(n).mean()
    rs = up / (dw + 1e-9)
    rsi = 100 - 100 / (1 + rs)

    feats = np.stack(
        [sma, wma, mom, macd, williams_r, cci, k, d, rsi], axis=1
    )

    # fill NaN with 0
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    # z-score per column (over the entire series)
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-9
    feats = (feats - mean) / std

    return feats.astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_days = 300
    close = 100 + np.cumsum(rng.normal(0, 1, n_days))
    df = pd.DataFrame({
        'open': close + rng.normal(0, 0.3, n_days),
        'high': close + np.abs(rng.normal(0, 0.5, n_days)),
        'low': close - np.abs(rng.normal(0, 0.5, n_days)),
        'close': close,
    })
    feats = compute_indicators(df, n=14)
    print(f"feats shape: {feats.shape}")
    print(f"feats stats: mean={feats.mean():.3f}, std={feats.std():.3f}")
    print(f"feats sample (last row): {feats[-1]}")
