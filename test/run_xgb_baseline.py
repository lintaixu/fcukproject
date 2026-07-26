"""
XGBoost 同條件對照基線 — 驗證 Chart GCN 管線是否增值.

協定: 測試樣本/標籤/指標與 Chart GCN 實驗完全相同 (直接讀 dscache 快取),
只換模型。兩個特徵變體:
  A. subgraph : Chart GCN 的 (N,g,9) 子圖特徵攤平 → 同輸入比模型
  B. plain    : 決策日當天的 9 個原始技術指標   → 驗證 PIP/VG/子圖加工是否增值

用法:
  python test/run_xgb_baseline.py --market tw
  python test/run_xgb_baseline.py --market cn
"""
import argparse
import json
import os as _os
import sys as _sys
from datetime import datetime

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)
from xgboost import XGBClassifier

_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _root)
_sys.path.insert(0, _os.path.join(_root, "core"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from data_loader import fetch_tw_stocks
from indicators import compute_indicators
from run_paper_repro import load_ds_cache, TICKER_SETS

MARKETS = {
    "tw": dict(tickers="tw50", start="2016-01-01", train_end="2023-12-31",
               end="2024-12-31", window=140, m=60, N=15, g=5,
               out_dir="experiments_paper", md_file="實驗記錄_論文對齊.md",
               ref=("Chart GCN rawfeat", 52.70, 50.33)),
    "cn": dict(tickers="sse50", start="2010-01-01", train_end="2017-12-31",
               end="2018-12-31", window=140, m=60, N=15, g=5,
               out_dir="chinatest", md_file="chinatest/實驗記錄_china.md",
               ref=("Chart GCN cn-sse50-raw", 52.24, 44.93)),
}

SEED = 42


def metrics_of(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "pre_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "pre_0": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }


def fit_xgb(Xtr, ytr, Xva, yva, Xte):
    model = XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=SEED, n_jobs=-1,
        eval_metric="logloss", early_stopping_rounds=50,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return model.predict(Xte), model.best_iteration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="tw", choices=["tw", "cn"])
    args = ap.parse_args()
    cfg = MARKETS[args.market]

    key = (f"{cfg['tickers']}0_{cfg['start']}_{cfg['train_end']}_{cfg['end']}"
           f"_w{cfg['window']}m{cfg['m']}N{cfg['N']}g{cfg['g']}s1_raw")
    cache = _os.path.join(_root, cfg["out_dir"], "dscache")
    train_ds = load_ds_cache(_os.path.join(cache, f"{key}_train.npz"))
    test_ds = load_ds_cache(_os.path.join(cache, f"{key}_test.npz"))
    print(f"[DATA] {key}: train {len(train_ds)}, test {len(test_ds)}")

    # 與 Chart GCN 相同的隨機 80/20 切分 (同 seed)
    rng = np.random.default_rng(SEED)
    n = len(train_ds)
    perm = rng.permutation(n)
    tr_idx, va_idx = perm[:int(n * 0.8)], perm[int(n * 0.8):]

    results = {}

    # ── 變體 A: 子圖特徵攤平 (同輸入比模型) ──
    Xf = train_ds.X.reshape(n, -1)
    Xte = test_ds.X.reshape(len(test_ds), -1)
    pred, bi = fit_xgb(Xf[tr_idx], train_ds.y[tr_idx],
                       Xf[va_idx], train_ds.y[va_idx], Xte)
    results["A_subgraph"] = metrics_of(test_ds.y, pred)
    results["A_subgraph"]["best_iter"] = int(bi)
    print(f"[A subgraph] acc={results['A_subgraph']['acc']:.4f}")

    # ── 變體 B: 決策日當天 9 個原始指標 ──
    data = fetch_tw_stocks(tickers=TICKER_SETS[cfg["tickers"]],
                           start=cfg["start"], end=cfg["end"])
    feat_row = {}
    for tk, df in data.items():
        feats = compute_indicators(df, n=cfg["window"], stats=(0.0, 1.0))
        for i, dt in enumerate(df.index):
            feat_row[(tk, str(dt)[:10])] = feats[i]

    def plain_features(ds):
        X = np.zeros((len(ds), 9), dtype=np.float32)
        for i, (tk, dt) in enumerate(ds.meta):
            X[i] = feat_row[(tk, str(dt)[:10])]
        return X

    Xp_tr = plain_features(train_ds)
    Xp_te = plain_features(test_ds)
    pred, bi = fit_xgb(Xp_tr[tr_idx], train_ds.y[tr_idx],
                       Xp_tr[va_idx], train_ds.y[va_idx], Xp_te)
    results["B_plain9"] = metrics_of(test_ds.y, pred)
    results["B_plain9"]["best_iter"] = int(bi)
    print(f"[B plain9]   acc={results['B_plain9']['acc']:.4f}")

    # ── 輸出 ──
    ref_name, ref_acc, ref_f1m = cfg["ref"]
    exp_id = "XGB-" + datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{args.market}"
    out_json = _os.path.join(_root, cfg["out_dir"], f"{exp_id}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"exp_id": exp_id, "key": key, "results": {
            k: {m: round(float(v), 4) for m, v in r.items()}
            for k, r in results.items()}}, f, indent=2, ensure_ascii=False)

    lines = ["", f"### {exp_id} XGBoost 同條件對照基線", "",
             f"- 測試資料/標籤與 Chart GCN 完全相同（快取 `{key}`）；"
             f"同 seed 隨機 80/20 切分；XGB: depth=6, lr=0.05, "
             f"early stopping on val logloss",
             f"- 對照組: {ref_name} = Acc {ref_acc:.2f}% / F1m {ref_f1m:.2f}%；"
             f"論文宣稱 69.26%", "",
             "| 模型 | 特徵 | Acc | F1_1 | F1_0 | F1 macro |",
             "|---|---|---|---|---|---|"]
    for tag, label in [("A_subgraph", "XGB / 子圖特徵攤平 (同輸入)"),
                       ("B_plain9", "XGB / 決策日 9 指標 (裸指標)")]:
        r = results[tag]
        f1m = (r["f1_1"] + r["f1_0"]) / 2
        lines.append(f"| {label} | {'N×g×9' if tag=='A_subgraph' else '9'} | "
                     f"{r['acc']*100:.2f}% | {r['f1_1']*100:.2f}% | "
                     f"{r['f1_0']*100:.2f}% | {f1m*100:.2f}% |")
    lines.append(f"| {ref_name} (對照) | N×g×9 | {ref_acc:.2f}% | — | — | "
                 f"{ref_f1m:.2f}% |")
    with open(_os.path.join(_root, cfg["md_file"]), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[SAVED] {out_json}")
    for tag, r in results.items():
        f1m = (r["f1_1"] + r["f1_0"]) / 2
        print(f"  {tag}: acc={r['acc']*100:.2f}% f1_macro={f1m*100:.2f}%")


if __name__ == "__main__":
    main()
