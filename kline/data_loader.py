"""
原始股價下載與資料集建構。

- `download_raw`：多線程並行下載 TW50 全部股票。
- `download_one`：下載單一股票（供補充下載自訂代號使用）。
- `build_dataset`：套用特徵函式、加入未來報酬標籤、過濾除權息與異常值。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

from kline.config import HOLD_DAYS, TW50
from kline.features import detect_exdiv_mask


def download_one(ticker: str):
    """下載單一股票；資料不足或缺欄位時回傳 (ticker, None)。"""
    try:
        df = yf.download(
            ticker, start='2015-01-01', end='2025-12-31',
            auto_adjust=False, progress=False, actions=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        required = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}
        if df.empty or len(df) < 100 or not required.issubset(df.columns):
            return ticker, None
        return ticker, df
    except Exception:
        return ticker, None


# 舊名相容別名（原始程式使用 _download_one）
_download_one = download_one


def download_raw(log_fn: callable, progress_fn: callable = None) -> dict:
    """並行下載 TW50 全部股票，回傳 {ticker: DataFrame}。"""
    raw_data = {}
    tickers = list(TW50.keys())
    total = len(tickers)
    log_fn(f"  並行下載 {total} 檔股票（10 線程）...")

    done = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_one, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, df = future.result()
            done += 1
            if df is not None:
                raw_data[ticker] = df
                log_fn(f"  [{done:02d}/{total}] {ticker} {TW50[ticker]} → {len(df)} 筆")
            else:
                log_fn(f"  [{done:02d}/{total}] {ticker} → 跳過")
            if progress_fn:
                progress_fn(done, total)

    return raw_data


def build_dataset(raw_data: dict, feat_fn) -> pd.DataFrame:
    """把每檔股票套上特徵函式並合併，加上 future_ret / ticker 欄，過濾除權息與 NaN。"""
    parts = []
    for ticker, df in raw_data.items():
        try:
            exdiv = detect_exdiv_mask(df)
            feat  = feat_fn(df)
            fret  = (df['Close'].shift(-HOLD_DAYS) - df['Close']) / df['Close'] * 100
            combined = feat.copy()
            combined['future_ret'] = fret
            combined['exdiv']      = exdiv
            combined['ticker']     = ticker
            parts.append(combined)
        except Exception:
            continue
    full = pd.concat(parts, axis=0)
    full = full[~full['exdiv']].drop(columns='exdiv')
    full = full.replace([np.inf, -np.inf], np.nan).dropna()
    return full


if __name__ == '__main__':
    from kline.features import features_1day

    def _mk(seed):
        rng = np.random.default_rng(seed); m = 400
        close = 100 + np.cumsum(rng.normal(0, 1, m))
        return pd.DataFrame({
            'Open': close, 'High': close + 1, 'Low': close - 1, 'Close': close,
            'Volume': rng.integers(1_000_000, 5_000_000, m), 'Adj Close': close,
        }, index=pd.date_range('2016-01-01', periods=m, freq='B'))

    raw = {f'{2330 + i}.TW': _mk(i) for i in range(3)}
    df = build_dataset(raw, features_1day)
    print(f"build_dataset: {len(raw)} 檔合成股票 → shape={df.shape}")
    print(f"  欄位={list(df.columns)}")
    print(f"  ticker 分佈={df['ticker'].value_counts().to_dict()}")
    print("（download_raw / download_one 需連網，此自測略過）")
