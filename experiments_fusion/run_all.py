"""融合實驗主程式 — V1~V5 依序執行, 結果寫 results.json."""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_dataset import ArrayDataset, split_indices_by_date
from fusion_train import fit, metrics_from_pred, macro_f1

OUT = os.path.dirname(os.path.abspath(__file__))
TRAIN_END, VAL_END = "2022-12-31", "2023-12-31"
EPOCHS, SEED = 20, 42
RESULTS = {}


def log(msg):
    print(msg, flush=True)


def save():
    with open(os.path.join(OUT, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=float)


def make_splits(d, label_key):
    tr, va, te = split_indices_by_date(d['dates'], TRAIN_END, VAL_END)
    y = d[label_key]
    mk = lambda idx: ArrayDataset(d['X'][idx], y[idx],
                                  d['dates'][idx], d['tickers'][idx])
    return mk(tr), mk(va), mk(te)


def run_gcn(name, d, label_key, F_dim, note):
    t0 = time.time()
    log(f"\n{'='*60}\n [{name}] {note}\n{'='*60}")
    train_ds, val_ds, test_ds = make_splits(d, label_key)
    log(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"(test 漲比例={test_ds.y.mean():.2%})")
    model, tm, tp = fit(train_ds, val_ds, test_ds,
                        F_dim=F_dim, epochs=EPOCHS, seed=SEED)
    tm['elapsed_min'] = (time.time() - t0) / 60
    tm['n_train'], tm['n_test'] = len(train_ds), len(test_ds)
    tm['test_base_rate'] = float(max(test_ds.y.mean(), 1 - test_ds.y.mean()))
    RESULTS[name] = {'note': note, 'metrics': tm}
    save()
    log(f"  → test acc={tm['acc']:.4f} f1_macro={tm['f1_macro']:.4f} "
        f"({tm['elapsed_min']:.1f} min)")
    return model, test_ds, tp


# ─────────────────────────────────────────────
# 系統 A: K線型態挖掘 + AE + OCSVM (main 分支邏輯移植)
# ─────────────────────────────────────────────

class AE(nn.Module):
    def __init__(self, input_dim, latent=4):
        super().__init__()
        h = max(16, input_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, 8), nn.ReLU(), nn.Linear(8, latent))
        self.decoder = nn.Sequential(
            nn.Linear(latent, 8), nn.ReLU(),
            nn.Linear(8, h), nn.ReLU(), nn.Linear(h, input_dim))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_ae(X, epochs=60):
    torch.manual_seed(SEED)
    Xt = torch.FloatTensor(X)
    ds = torch.utils.data.TensorDataset(Xt)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=min(256, len(X)), shuffle=True,
        drop_last=len(X) > 256)
    m = AE(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    m.train()
    for _ in range(epochs):
        for (xb,) in loader:
            opt.zero_grad()
            rec, _ = m(xb)
            loss = lossf(rec, xb)
            loss.backward()
            opt.step()
    m.eval()
    return m


def train_system_A(feat, ret5d, rise=5.0, fall=-5.0, dist_th=3.0,
                   min_freq=5, min_bull=2.0, min_bear=-2.0, nu=0.1):
    """main 分支 find_and_train 的移植版 (輸入已 z-score 的 K 棒 7 維特徵)."""
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.svm import OneClassSVM

    Xs = np.clip(feat, -5, 5)
    models = {}
    for direction, seeds in [('bull', np.where(ret5d > rise)[0]),
                             ('bear', np.where(ret5d < fall)[0])]:
        log(f"  [A/{direction}] 種子 {len(seeds)}")
        idx_imp = {}
        BATCH = 200
        for bs in range(0, len(seeds), BATCH):
            bseeds = seeds[bs:bs + BATCH]
            dist = euclidean_distances(Xs[bseeds], Xs)
            for j, si in enumerate(bseeds):
                dd = dist[j].copy()
                dd[si] = np.inf
                sim = np.where(dd <= dist_th)[0]
                if len(sim) < min_freq:
                    continue
                rets = ret5d[sim]
                if direction == 'bull':
                    if not (rets > min_bull).all():
                        continue
                else:
                    if not (rets < min_bear).all():
                        continue
                imp = len(sim) * abs(float(rets.mean()))
                for ii in [si, *sim.tolist()]:
                    if ii not in idx_imp or idx_imp[ii] < imp:
                        idx_imp[ii] = imp
        log(f"  [A/{direction}] 有效樣本 {len(idx_imp)}")
        if len(idx_imp) < 10:
            models[direction] = None
            continue
        sidx = sorted(idx_imp.keys())
        Xp = Xs[sidx]
        ae = train_ae(Xp)
        with torch.no_grad():
            Z = ae.encoder(torch.FloatTensor(Xp)).numpy()
        w = np.array([idx_imp[i] for i in sidx], dtype=np.float32)
        w = w / w.sum() * len(w)
        oc = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
        oc.fit(Z, sample_weight=w)
        models[direction] = {'ae': ae, 'ocsvm': oc}
    return models


def predict_system_A(models, feat):
    Xs = np.clip(feat, -5, 5)
    out = {}
    for direction in ('bull', 'bear'):
        if models.get(direction) is None:
            out[direction] = np.zeros(len(Xs), dtype=bool)
            continue
        ae, oc = models[direction]['ae'], models[direction]['ocsvm']
        with torch.no_grad():
            Z = ae.encoder(torch.FloatTensor(Xs)).numpy()
        out[direction] = oc.predict(Z) == 1
    return out


def selective_report(true, pred_mask_up, pred_mask_down):
    """在被選中的樣本上計算方向準確率與覆蓋率."""
    sel = pred_mask_up | pred_mask_down
    n = int(sel.sum())
    if n == 0:
        return {'coverage': 0.0, 'n_signals': 0, 'acc': None}
    correct = ((pred_mask_up & (true == 1)) |
               (pred_mask_down & (true == 0))).sum()
    return {'coverage': float(sel.mean()), 'n_signals': n,
            'acc': float(correct / n),
            'n_up': int(pred_mask_up.sum()),
            'n_down': int(pred_mask_down.sum())}


if __name__ == '__main__':
    t_start = time.time()
    d9 = dict(np.load(os.path.join(OUT, 'data_X9.npz'), allow_pickle=True))
    d16 = dict(np.load(os.path.join(OUT, 'data_X16.npz'), allow_pickle=True))
    flat = dict(np.load(os.path.join(OUT, 'flat.npz'), allow_pickle=True))

    # ── V1: 修正版基準線 (F=9, 隔日標籤) ──
    run_gcn('V1', d9, 'y_next', 9,
            '修正版 Chart-GCN 基準線: F=9 指標, 預測隔日漲跌')

    # ── V2: 5 日視野標籤 ──
    m2, te2, tp2 = run_gcn('V2', d9, 'y_5d', 9,
                           'F=9 指標, 標籤改為未來 5 日漲跌 (與系統 A 對齊)')

    # ── V3a/V3b: 特徵融合 ──
    run_gcn('V3a', d16, 'y_next', 16, '融合特徵 F=16 (9指標+7K棒), 隔日標籤')
    m3, te3, tp3 = run_gcn('V3b', d16, 'y_5d', 16,
                           '融合特徵 F=16, 5 日標籤')

    # ── V4: 系統 A 訊號閘門 ──
    log(f"\n{'='*60}\n [V4] 系統 A (AE+OCSVM) 閘門融合\n{'='*60}")
    t0 = time.time()
    ftr, fva, fte = split_indices_by_date(flat['dates'], TRAIN_END, VAL_END)
    candle_cols = slice(9, 16)   # flat feat 後 7 欄 = K 棒特徵
    A_models = train_system_A(flat['feat'][ftr][:, candle_cols],
                              flat['ret5d'][ftr])

    # 對齊 flat test 列與圖模型 test 樣本 (key = date|ticker)
    flat_test_keys = {f"{d}|{t}": i for i, (d, t) in
                      enumerate(zip(flat['dates'][fte], flat['tickers'][fte]))}
    A_pred_te = predict_system_A(A_models, flat['feat'][fte][:, candle_cols])

    def align_A(test_ds):
        rows = [flat_test_keys.get(f"{d}|{t}", -1)
                for d, t in zip(test_ds.dates, test_ds.tickers)]
        rows = np.array(rows)
        ok = rows >= 0
        bull = np.zeros(len(rows), dtype=bool)
        bear = np.zeros(len(rows), dtype=bool)
        bull[ok] = A_pred_te['bull'][rows[ok]]
        bear[ok] = A_pred_te['bear'][rows[ok]]
        return bull, bear

    for tag, (te_ds, tp) in [('V4_gate_V2', (te2, tp2)),
                             ('V4_gate_V3b', (te3, tp3))]:
        bull, bear = align_A(te_ds)
        b_pred = tp.argmax(1)
        up = (b_pred == 1) & bull
        down = (b_pred == 0) & bear
        rep = selective_report(te_ds.y, up, down)
        rep['A_bull_rate'] = float(bull.mean())
        rep['A_bear_rate'] = float(bear.mean())
        RESULTS[tag] = {'note': f'{tag}: B 預測與 A 型態閘門同向才出訊號 (5日方向)',
                        'metrics': rep}
        log(f"  {tag}: coverage={rep['coverage']:.2%} "
            f"n={rep['n_signals']} acc={rep['acc']}")
    RESULTS['V4_meta'] = {'elapsed_min': (time.time() - t0) / 60}
    save()

    # ── V5a: 信心門檻 (機率閘門) ──
    log(f"\n{'='*60}\n [V5a] 信心門檻選擇性預測\n{'='*60}")
    for tag, (te_ds, tp) in [('V5a_conf_V2', (te2, tp2)),
                             ('V5a_conf_V3b', (te3, tp3))]:
        curves = []
        for tau in [0.55, 0.60, 0.65, 0.70, 0.75]:
            up = tp[:, 1] >= tau
            down = tp[:, 0] >= tau
            rep = selective_report(te_ds.y, up, down)
            rep['tau'] = tau
            curves.append(rep)
            log(f"  {tag} tau={tau}: cov={rep['coverage']:.2%} acc={rep['acc']}")
        RESULTS[tag] = {'note': f'{tag}: softmax 機率 ≥ tau 才出訊號', 'curves': curves}
    save()

    # ── V5b: GBDT 平面特徵基準 ──
    log(f"\n{'='*60}\n [V5b] HistGradientBoosting (平面 16 維特徵, 5 日標籤)\n{'='*60}")
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xf, yf = flat['feat'], flat['y_5d']
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         max_depth=6, random_state=SEED,
                                         validation_fraction=None)
    clf.fit(Xf[ftr], yf[ftr])
    for split, idx in [('val', fva), ('test', fte)]:
        pm = metrics_from_pred(yf[idx], clf.predict(Xf[idx]))
        pm['f1_macro'] = macro_f1(pm)
        RESULTS.setdefault('V5b', {'note': 'GBDT 平面特徵 (無圖結構), 5 日標籤'})[split] = pm
        log(f"  {split}: acc={pm['acc']:.4f} f1_macro={pm['f1_macro']:.4f}")
    # GBDT + 信心門檻
    proba = clf.predict_proba(Xf[fte])
    curves = []
    for tau in [0.55, 0.60, 0.65, 0.70, 0.75]:
        rep = selective_report(yf[fte], proba[:, 1] >= tau, proba[:, 0] >= tau)
        rep['tau'] = tau
        curves.append(rep)
        log(f"  V5b conf tau={tau}: cov={rep['coverage']:.2%} acc={rep['acc']}")
    RESULTS['V5b_conf'] = {'note': 'GBDT + 信心門檻', 'curves': curves}

    RESULTS['_meta'] = {
        'train_end': TRAIN_END, 'val_end': VAL_END,
        'epochs': EPOCHS, 'seed': SEED,
        'total_elapsed_min': (time.time() - t_start) / 60,
    }
    save()
    log("\nALL DONE")
