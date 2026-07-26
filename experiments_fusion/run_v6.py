"""V6: 準確率未達六成後的替代方法探討.

V6a 死區標籤: 只用 |5日報酬| > 2% 的樣本訓練與評估 (方向訊號更乾淨)
V6b 滾動重訓: 2024 逐季 walk-forward (訓練集隨時間前進, 對抗分佈漂移)
V6c GBDT 死區版
另輸出各 split 標籤分佈供報告分析.
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_dataset import ArrayDataset
from fusion_train import fit, metrics_from_pred, macro_f1

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 42
RES = {}


def log(m):
    print(m, flush=True)


def save():
    with open(os.path.join(OUT, 'results_v6.json'), 'w', encoding='utf-8') as f:
        json.dump(RES, f, ensure_ascii=False, indent=2, default=float)


if __name__ == '__main__':
    d9 = dict(np.load(os.path.join(OUT, 'data_X9.npz'), allow_pickle=True))
    flat = dict(np.load(os.path.join(OUT, 'flat.npz'), allow_pickle=True))

    # 對齊 ret5d 到圖樣本
    key2ret = {f"{d}|{t}": r for d, t, r in
               zip(flat['dates'], flat['tickers'], flat['ret5d'])}
    ret5d_g = np.array([key2ret.get(f"{d}|{t}", 0.0)
                        for d, t in zip(d9['dates'], d9['tickers'])])

    # ── 標籤分佈統計 ──
    stats = {}
    for name, lo, hi in [('train', '', '2022-12-31'),
                         ('val', '2022-12-31', '2023-12-31'),
                         ('test', '2023-12-31', '9999')]:
        m = (d9['dates'] > lo) & (d9['dates'] <= hi)
        stats[name] = {
            'n': int(m.sum()),
            'up_next': float(d9['y_next'][m].mean()),
            'up_5d': float(d9['y_5d'][m].mean()),
            'big_move_rate': float((np.abs(ret5d_g[m]) > 2).mean()),
            'mean_ret5d': float(ret5d_g[m].mean()),
        }
    RES['label_stats'] = stats
    log(f"label_stats: {json.dumps(stats, indent=1)}")
    save()

    # ── V6a: 死區標籤 (|ret5d|>2%) GCN ──
    log(f"\n{'='*60}\n [V6a] 死區標籤 GCN: 只取 |5日報酬|>2%\n{'='*60}")
    t0 = time.time()
    dz = np.abs(ret5d_g) > 2.0
    tr = dz & (d9['dates'] <= '2022-12-31')
    va = dz & (d9['dates'] > '2022-12-31') & (d9['dates'] <= '2023-12-31')
    te = dz & (d9['dates'] > '2023-12-31')
    y = (ret5d_g > 0).astype(np.int64)
    mk = lambda m: ArrayDataset(d9['X'][m], y[m], d9['dates'][m], d9['tickers'][m])
    tr_ds, va_ds, te_ds = mk(tr), mk(va), mk(te)
    log(f"  train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)} "
        f"(test 覆蓋率={te.sum()/ (d9['dates'] > '2023-12-31').sum():.2%}, "
        f"test 漲比例={te_ds.y.mean():.2%})")
    _, tm, _ = fit(tr_ds, va_ds, te_ds, F_dim=9, epochs=20, seed=SEED, verbose=False)
    tm['coverage_of_test_days'] = float(te.sum() / (d9['dates'] > '2023-12-31').sum())
    tm['elapsed_min'] = (time.time() - t0) / 60
    RES['V6a'] = {'note': '死區標籤 GCN (F=9): 訓練與評估只用 |ret5d|>2% 樣本',
                  'metrics': tm}
    log(f"  → test acc={tm['acc']:.4f} f1_macro={tm['f1_macro']:.4f}")
    save()

    # ── V6b: 2024 逐季 walk-forward (GCN F=9, 5日標籤) ──
    log(f"\n{'='*60}\n [V6b] 逐季 walk-forward 重訓\n{'='*60}")
    t0 = time.time()
    quarters = [('2024-01-01', '2024-03-31'), ('2024-04-01', '2024-06-30'),
                ('2024-07-01', '2024-09-30'), ('2024-10-01', '2024-12-31')]
    all_true, all_pred = [], []
    per_q = []
    for qs, qe in quarters:
        va_start = str(np.datetime64(qs) - np.timedelta64(180, 'D'))[:10]
        tr = d9['dates'] < va_start
        va = (d9['dates'] >= va_start) & (d9['dates'] < qs)
        te = (d9['dates'] >= qs) & (d9['dates'] <= qe)
        mk5 = lambda m: ArrayDataset(d9['X'][m], d9['y_5d'][m],
                                     d9['dates'][m], d9['tickers'][m])
        _, tm, tp = fit(mk5(tr), mk5(va), mk5(te), F_dim=9, epochs=15,
                        seed=SEED, verbose=False)
        pred = tp.argmax(1)
        true = d9['y_5d'][te]
        all_true.append(true)
        all_pred.append(pred)
        per_q.append({'quarter': qs[:7], 'n': int(te.sum()),
                      'acc': tm['acc'], 'f1_macro': tm['f1_macro']})
        log(f"  {qs[:7]}: n={te.sum()} acc={tm['acc']:.4f} f1m={tm['f1_macro']:.4f}")
    true = np.concatenate(all_true)
    pred = np.concatenate(all_pred)
    agg = metrics_from_pred(true, pred)
    agg['f1_macro'] = macro_f1(agg)
    agg['elapsed_min'] = (time.time() - t0) / 60
    RES['V6b'] = {'note': '逐季 walk-forward: 每季用截至該季前的資料重訓 (5日標籤)',
                  'per_quarter': per_q, 'aggregate': agg}
    log(f"  → 全年彙總 acc={agg['acc']:.4f} f1_macro={agg['f1_macro']:.4f}")
    save()

    # ── V6c: GBDT 死區版 ──
    log(f"\n{'='*60}\n [V6c] GBDT 死區標籤\n{'='*60}")
    from sklearn.ensemble import HistGradientBoostingClassifier
    fdz = np.abs(flat['ret5d']) > 2.0
    ftr = fdz & (flat['dates'] <= '2022-12-31')
    fte = fdz & (flat['dates'] > '2023-12-31')
    yf = (flat['ret5d'] > 0).astype(np.int64)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         max_depth=6, random_state=SEED)
    clf.fit(flat['feat'][ftr], yf[ftr])
    pm = metrics_from_pred(yf[fte], clf.predict(flat['feat'][fte]))
    pm['f1_macro'] = macro_f1(pm)
    RES['V6c'] = {'note': 'GBDT 平面特徵 + 死區標籤', 'metrics': pm}
    log(f"  → test acc={pm['acc']:.4f} f1_macro={pm['f1_macro']:.4f}")
    save()
    log("\nV6 DONE")
