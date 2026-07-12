"""
特徵驗證（不回測）。

在訓練期上訓練 AE，對看漲 / 看跌各訓練一個 One-Class SVM，
以混淆矩陣（Precision / Recall / Accuracy / F1）評估各特徵集的判別力。
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from kline.config import (
    TRAIN_START, TRAIN_END, LATENT_DIM, RISE_THRESH, FALL_THRESH, OCSVM_NU,
)
from kline.autoencoder import train_ae


def _calc_metrics(actual_pos: np.ndarray, pred_pos: np.ndarray,
                  label: str, log_fn) -> dict:
    N   = len(actual_pos)
    TP  = int(( pred_pos &  actual_pos).sum())
    TN  = int((~pred_pos & ~actual_pos).sum())
    FP  = int(( pred_pos & ~actual_pos).sum())
    FN  = int((~pred_pos &  actual_pos).sum())

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy  = (TP + TN) / N  if N > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    log_fn(f"\n  [{label}] 混淆矩陣：")
    log_fn(f"    TP={TP:,}  TN={TN:,}  FP={FP:,}  FN={FN:,}")
    log_fn(f"    Precision : {precision:.4f}  ({precision:.2%})")
    log_fn(f"    Recall    : {recall:.4f}  ({recall:.2%})")
    log_fn(f"    Accuracy  : {accuracy:.4f}  ({accuracy:.2%})")
    log_fn(f"    F1-Score  : {f1:.4f}  ({f1:.2%})")

    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
            'precision': precision, 'recall': recall,
            'accuracy': accuracy, 'f1': f1}


def validate_features(df, feat_cols: list, log_fn: callable) -> dict:
    train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()

    # 依日期切 80/20：前 80% 訓練 AE+OCSVM，後 20% 做 out-of-sample 驗證
    dates      = train_df.index.unique().sort_values()
    split_date = dates[int(len(dates) * 0.8)]
    fit_df     = train_df[train_df.index <  split_date]
    val_df     = train_df[train_df.index >= split_date]
    log_fn(f"  訓練段：{len(fit_df):,} 筆 (~{str(split_date)[:10]})  "
           f"驗證段：{len(val_df):,} 筆")

    X_fit = fit_df[feat_cols].values.astype(np.float32)
    y_fit = fit_df['future_ret'].values.astype(np.float32)
    X_val = val_df[feat_cols].values.astype(np.float32)
    y_val = val_df['future_ret'].values.astype(np.float32)

    scaler  = StandardScaler()
    X_fit_s = np.clip(scaler.fit_transform(X_fit), -5, 5)
    X_val_s = np.clip(scaler.transform(X_val), -5, 5)

    log_fn(f"  [AE] 訓練中 (input={len(feat_cols)}, latent={LATENT_DIM}, "
           f"樣本={len(X_fit_s):,})...")
    ae = train_ae(X_fit_s, len(feat_cols), log_fn)
    ae.eval()

    with torch.no_grad():
        Z     = ae.encode(torch.FloatTensor(X_fit_s)).numpy()
        Z_val = ae.encode(torch.FloatTensor(X_val_s)).numpy()

    bull_mask = y_fit > RISE_THRESH
    bear_mask = y_fit < FALL_THRESH
    log_fn(f"  看漲種子：{bull_mask.sum():,}  看跌種子：{bear_mask.sum():,}")

    metrics = {}

    # 在驗證段評估；正例門檻與訓練種子一致 (RISE_THRESH / FALL_THRESH)
    if bull_mask.sum() >= 5:
        oc_bull = OneClassSVM(kernel='rbf', nu=OCSVM_NU, gamma='scale')
        oc_bull.fit(Z[bull_mask])
        pred_pos = oc_bull.predict(Z_val) == 1
        actual   = y_val > RISE_THRESH
        metrics['bull'] = _calc_metrics(actual, pred_pos, '看漲', log_fn)
    else:
        metrics['bull'] = {}

    if bear_mask.sum() >= 5:
        oc_bear = OneClassSVM(kernel='rbf', nu=OCSVM_NU, gamma='scale')
        oc_bear.fit(Z[bear_mask])
        pred_pos = oc_bear.predict(Z_val) == 1
        actual   = y_val < FALL_THRESH
        metrics['bear'] = _calc_metrics(actual, pred_pos, '看跌', log_fn)
    else:
        metrics['bear'] = {}

    return metrics


if __name__ == '__main__':
    import pandas as pd
    from kline.data_loader import build_dataset
    from kline.features import features_1day

    def _mk(seed):
        rng = np.random.default_rng(seed); m = 2200
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, m)))
        return pd.DataFrame({
            'Open': close, 'High': close * 1.01, 'Low': close * 0.99, 'Close': close,
            'Volume': rng.integers(1_000_000, 5_000_000, m), 'Adj Close': close,
        }, index=pd.date_range('2016-01-01', periods=m, freq='B'))

    raw = {f'{2330 + i}.TW': _mk(i) for i in range(3)}
    df = build_dataset(raw, features_1day)
    cols = [c for c in df.columns if c not in ('future_ret', 'ticker')]
    metrics = validate_features(df, cols, print)
    print()
    for d, label in [('bull', '看漲'), ('bear', '看跌')]:
        r = metrics.get(d, {})
        if r:
            print(f"validate[{label}]: TP={r['TP']} FP={r['FP']} "
                  f"P={r['precision']:.2%} R={r['recall']:.2%} "
                  f"Acc={r['accuracy']:.2%} F1={r['f1']:.2%}")
