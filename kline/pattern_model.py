"""
型態搜尋 + 模型訓練。

`find_and_train` 完成一次完整的訓練流程：
  1. 以 Euclidean 距離找出「相似型態群」並計算重要性權重。
  2. 對看漲 / 看跌兩方向各自訓練 Autoencoder + One-Class SVM。
回傳 {'scaler', 'bull', 'bear'} 供回測使用。
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import OneClassSVM

from kline.config import (
    TRAIN_START, TRAIN_END, RISE_THRESH, FALL_THRESH, DIST_THRESH,
    MIN_FREQ, MIN_BULL_RET, MIN_BEAR_RET, LATENT_DIM, OCSVM_NU,
)
from kline.autoencoder import train_ae


def find_and_train(df: pd.DataFrame, feat_cols: list, log_fn: callable) -> dict:
    train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    X = train_df[feat_cols].values.astype(np.float32)
    y = train_df['future_ret'].values.astype(np.float32)

    scaler   = StandardScaler()
    X_scaled = np.clip(scaler.fit_transform(X), -5, 5)

    bull_seeds = np.where(y > RISE_THRESH)[0]
    bear_seeds = np.where(y < FALL_THRESH)[0]
    log_fn(f"  看漲種子：{len(bull_seeds):,}  看跌種子：{len(bear_seeds):,}")

    models = {'scaler': scaler}

    for direction, seeds in [('bull', bull_seeds), ('bear', bear_seeds)]:
        label  = '看漲' if direction == 'bull' else '看跌'
        if len(seeds) == 0:
            models[direction] = None; continue

        idx_importance = {}
        valid_count    = 0
        top_patterns   = []
        BATCH = 200

        for batch_start in range(0, len(seeds), BATCH):
            batch_seeds = seeds[batch_start: batch_start + BATCH]
            batch_vecs  = X_scaled[batch_seeds]
            dist_batch  = euclidean_distances(batch_vecs, X_scaled)

            for j, seed_idx in enumerate(batch_seeds):
                dists        = dist_batch[j].copy()
                dists[seed_idx] = np.inf
                similar_idx  = np.where(dists <= DIST_THRESH)[0]
                if len(similar_idx) < MIN_FREQ:
                    continue
                similar_rets = y[similar_idx]

                if direction == 'bull':
                    if not (similar_rets > MIN_BULL_RET).all():
                        continue
                else:
                    if not (similar_rets < MIN_BEAR_RET).all():
                        continue

                avg_ret    = float(similar_rets.mean())
                freq       = len(similar_idx)
                win_rate   = float((similar_rets > 0).mean()) if direction == 'bull' \
                             else float((similar_rets < 0).mean())
                importance = freq * abs(avg_ret)

                for idx in [seed_idx, *similar_idx.tolist()]:
                    if idx not in idx_importance or idx_importance[idx] < importance:
                        idx_importance[idx] = importance
                valid_count += 1
                top_patterns.append((freq, avg_ret, win_rate, importance))

        log_fn(f"  {label} 有效型態群：{valid_count}  有效樣本：{len(idx_importance):,}")

        top_patterns.sort(key=lambda x: x[3], reverse=True)
        if top_patterns:
            log_fn(f"  {'排名':>4}  {'頻率':>6}  {'平均報酬':>10}  {'勝率':>8}  {'重要性':>10}")
            for rank, (freq, avg, wr, imp) in enumerate(top_patterns[:10], 1):
                log_fn(f"  {rank:>4}  {freq:>6}  {avg:>+9.2f}%  {wr:>7.1%}  {imp:>10.2f}")

        if len(idx_importance) == 0:
            models[direction] = None; continue

        sorted_idx    = sorted(idx_importance.keys())
        X_pattern     = X_scaled[sorted_idx]
        log_fn(f"\n  [{label}] 訓練 AE (input={len(feat_cols)}, latent={LATENT_DIM}, "
               f"樣本={len(X_pattern):,})...")
        ae = train_ae(X_pattern, len(feat_cols), log_fn)
        ae.eval()

        with torch.no_grad():
            Z = ae.encode(torch.FloatTensor(X_pattern)).numpy()

        weights = np.array([idx_importance[i] for i in sorted_idx], dtype=np.float32)
        weights = weights / weights.sum() * len(weights)

        log_fn(f"  [{label}] 訓練 OCSVM（含重要性權重）...")
        ocsvm = OneClassSVM(kernel='rbf', nu=OCSVM_NU, gamma='scale')
        ocsvm.fit(Z, sample_weight=weights)
        models[direction] = {'ae': ae, 'ocsvm': ocsvm}
        log_fn(f"  [{label}] 完成")

    return models


if __name__ == '__main__':
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
    models = find_and_train(df, cols, print)
    print(f"\nfind_and_train → keys={list(models.keys())}")
    print(f"  scaler={'有' if 'scaler' in models else '無'}  "
          f"bull={'有模型' if models.get('bull') else 'None'}  "
          f"bear={'有模型' if models.get('bear') else 'None'}")
