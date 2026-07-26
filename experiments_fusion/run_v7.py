"""V7: 逐月 walk-forward 滾動訓練 (2024 共 12 次重訓) × 3 變體.

V7a: 擴張視窗, 無衰減      (V6b 的逐月版)
V7b: 擴張視窗 + 樣本衰減   (loss 權重 = 0.5^(樣本距今天數/180), 近期樣本重)
V7c: 滾動視窗 (只用最近 2 年), 無衰減

共同: GCN F=9, 5 日標籤, val = 預測月前 60 天 (選最佳 epoch), epochs=12
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_dataset import ArrayDataset
from fusion_train import (DateGroupedBatchSampler, predict_probs,
                          metrics_from_pred, macro_f1)
from model import ChartGCN

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 42
EPOCHS = 12
HALF_LIFE = 180.0   # 天
RES = {}


def log(m):
    print(m, flush=True)


def save():
    with open(os.path.join(OUT, 'results_v7.json'), 'w', encoding='utf-8') as f:
        json.dump(RES, f, ensure_ascii=False, indent=2, default=float)


def fit_weighted(tr_ds, w, val_ds, epochs=EPOCHS, device='cpu'):
    """帶樣本權重的訓練; 以 val macro-F1 選最佳權重, 回傳 model."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = ChartGCN(N=15, g=5, F_dim=9, dropout=0.3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    sampler = DateGroupedBatchSampler(tr_ds, shuffle=True)
    wt = torch.from_numpy(w.astype(np.float32))

    best_val, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for indices in sampler:
            X = torch.from_numpy(tr_ds.X[indices]).float().to(device)
            y = torch.from_numpy(tr_ds.y[indices]).long().to(device)
            wi = wt[indices].to(device)
            opt.zero_grad()
            losses = F.cross_entropy(model(X), y, reduction='none')
            loss = (losses * wi).sum() / wi.sum()
            loss.backward()
            opt.step()
        vp, vt = predict_probs(model, val_ds, device)
        vf1 = macro_f1(metrics_from_pred(vt, vp.argmax(1)))
        if vf1 > best_val:
            best_val = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_variant(tag, note, d9, dates_dt, rolling_days=None, decay=False):
    log(f"\n{'='*60}\n [{tag}] {note}\n{'='*60}")
    t0 = time.time()
    months = [f"2024-{m:02d}" for m in range(1, 13)]
    all_true, all_pred = [], []
    per_m = []

    for mo in months:
        S = np.datetime64(f"{mo}-01")
        val_start = S - np.timedelta64(60, 'D')
        tr_mask = dates_dt < val_start
        if rolling_days is not None:
            tr_mask &= dates_dt >= (S - np.timedelta64(rolling_days, 'D'))
        va_mask = (dates_dt >= val_start) & (dates_dt < S)
        te_mask = (d9['dates'] >= mo) & (d9['dates'] < (f"2025-01" if mo == "2024-12"
                                                        else f"2024-{int(mo[5:])+1:02d}"))
        mk = lambda m: ArrayDataset(d9['X'][m], d9['y_5d'][m],
                                    d9['dates'][m], d9['tickers'][m])
        tr_ds, va_ds, te_ds = mk(tr_mask), mk(va_mask), mk(te_mask)
        if decay:
            age = (S - dates_dt[tr_mask]).astype('timedelta64[D]').astype(float)
            w = 0.5 ** (age / HALF_LIFE)
        else:
            w = np.ones(tr_mask.sum())

        model = fit_weighted(tr_ds, w, va_ds)
        tp, tt = predict_probs(model, te_ds)
        pred = tp.argmax(1)
        acc = float((pred == tt).mean())
        all_true.append(tt)
        all_pred.append(pred)
        per_m.append({'month': mo, 'n': len(tt), 'n_train': len(tr_ds), 'acc': acc})
        log(f"  {mo}: n={len(tt)} train={len(tr_ds)} acc={acc:.4f}")

    true = np.concatenate(all_true)
    pred = np.concatenate(all_pred)
    agg = metrics_from_pred(true, pred)
    agg['f1_macro'] = macro_f1(agg)
    agg['elapsed_min'] = (time.time() - t0) / 60
    RES[tag] = {'note': note, 'per_month': per_m, 'aggregate': agg}
    save()
    log(f"  → 全年 acc={agg['acc']:.4f} f1_macro={agg['f1_macro']:.4f} "
        f"({agg['elapsed_min']:.1f} min)")


if __name__ == '__main__':
    d9 = dict(np.load(os.path.join(OUT, 'data_X9.npz'), allow_pickle=True))
    dates_dt = d9['dates'].astype('datetime64[D]')

    run_variant('V7a', '逐月滾動, 擴張視窗, 無衰減', d9, dates_dt)
    run_variant('V7b', f'逐月滾動 + 樣本衰減 (半衰期 {HALF_LIFE:.0f} 天)',
                d9, dates_dt, decay=True)
    run_variant('V7c', '逐月滾動, 滾動視窗 (最近 730 天)', d9, dates_dt,
                rolling_days=730)
    log("\nV7 DONE")
