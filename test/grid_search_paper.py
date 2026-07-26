"""
論文網格搜尋 (對齊版): raw 特徵 + paper_exact 模型 + 隨機批次 128.

論文 4.3 範圍:
  window n ∈ {100,120,130,140}, m ∈ {30,40,60,80},
  N ∈ {10,15,20,25}, g ∈ {3,4,6,5}
兩階段搜尋 (Stage1 固定 N=15,g=5 搜 w,m; Stage2 固定最佳 w,m 搜 N,g),
以 validation macro-F1 選參 (無測試集洩漏), 最後用最佳參數 stride=1 完整訓練.

結果自動 append 到 實驗記錄_論文對齊.md.
"""
import argparse
import itertools
import json
import time
import os as _os
import sys as _sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import random_split

_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _root)
_sys.path.insert(0, _os.path.join(_root, "core"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from data_loader import fetch_tw_stocks
from dataset import ChartGCNDataset, split_by_date
from train import fit
from backtest import run_backtest, plot_backtest
from run_paper_repro import (ArrayDataset, save_ds_cache, load_ds_cache,
                             TICKER_SETS)

SEED = 42
SEARCH_STRIDE, SEARCH_EPOCHS = 5, 15

ap = argparse.ArgumentParser()
ap.add_argument("--tickers", default="tw50", choices=list(TICKER_SETS))
ap.add_argument("--start", default="2016-01-01")
ap.add_argument("--train-end", default="2023-12-31")
ap.add_argument("--end", default="2024-12-31")
ap.add_argument("--batch-mode", default="paper", choices=["paper", "date"])
ap.add_argument("--out-dir", default="experiments_paper")
ap.add_argument("--md-file", default="實驗記錄_論文對齊.md")
ARGS = ap.parse_args()

START, TRAIN_END, END = ARGS.start, ARGS.train_end, ARGS.end
OUT_DIR = _os.path.join(_root, ARGS.out_dir)
CACHE_DIR = _os.path.join(OUT_DIR, "dscache")
_os.makedirs(CACHE_DIR, exist_ok=True)


def build_or_load(data, window, m, N, g, stride):
    key = (f"{ARGS.tickers}0_{START}_{TRAIN_END}_{END}_w{window}m{m}N{N}g{g}"
           f"s{stride}_raw")
    tr_c = _os.path.join(CACHE_DIR, f"{key}_train.npz")
    te_c = _os.path.join(CACHE_DIR, f"{key}_test.npz")
    if _os.path.exists(tr_c) and _os.path.exists(te_c):
        return load_ds_cache(tr_c), load_ds_cache(te_c)

    raw_stats = {tk: (0.0, 1.0) for tk in data}
    train_data, test_data = split_by_date(data, TRAIN_END,
                                          warmup_rows=2 * window)
    train_full = ChartGCNDataset(
        train_data, window=window, m_pips=m, N=N, g=g,
        stride=stride, n_workers=1, norm_stats=raw_stats, verbose=False)
    test_ds = ChartGCNDataset(
        test_data, window=window, m_pips=m, N=N, g=g,
        stride=1, n_workers=1, norm_stats=raw_stats,
        min_date=TRAIN_END, verbose=False)
    save_ds_cache(tr_c, train_full)
    save_ds_cache(te_c, test_ds)
    return train_full, test_ds


def eval_combo(data, window, m, N, g):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_full, test_ds = build_or_load(data, window, m, N, g, SEARCH_STRIDE)
    n_total = len(train_full)
    n_train = int(n_total * 0.8)
    train_ds, val_ds = random_split(
        train_full, [n_train, n_total - n_train],
        generator=torch.Generator().manual_seed(SEED))
    _, metrics = fit(train_ds, val_ds, test_ds,
                     N=N, g=g, F_dim=9,
                     epochs=SEARCH_EPOCHS, lr=1e-3, weight_decay=5e-5,
                     batch_mode=ARGS.batch_mode, paper_exact=True,
                     verbose=False)
    return metrics


def main():
    t_start = time.time()
    data = fetch_tw_stocks(tickers=TICKER_SETS[ARGS.tickers],
                           start=START, end=END)
    print(f"[DATA] {ARGS.tickers}: {len(data)} stocks")

    results = []

    # Stage 1
    combos1 = list(itertools.product([100, 120, 130, 140], [30, 40, 60, 80]))
    for i, (w, m) in enumerate(combos1):
        t0 = time.time()
        mt = eval_combo(data, w, m, 15, 5)
        r = {"window": w, "m": m, "N": 15, "g": 5,
             "val_f1_macro": mt["val_f1_macro"],
             "test_acc": mt["acc"],
             "test_f1_macro": (mt["f1_1"] + mt["f1_0"]) / 2}
        results.append(r)
        print(f"[S1 {i+1}/{len(combos1)}] w={w} m={m} "
              f"val_f1m={r['val_f1_macro']:.4f} acc={r['test_acc']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    best1 = max(results, key=lambda x: x["val_f1_macro"])
    bw, bm = best1["window"], best1["m"]
    print(f"\n[S1 BEST] window={bw}, m={bm}, "
          f"val_f1_macro={best1['val_f1_macro']:.4f}\n")

    # Stage 2
    combos2 = [(n, g) for n, g in itertools.product([10, 15, 20, 25],
                                                    [3, 4, 5, 6])
               if not (n == 15 and g == 5)]
    for i, (n, g) in enumerate(combos2):
        t0 = time.time()
        mt = eval_combo(data, bw, bm, n, g)
        r = {"window": bw, "m": bm, "N": n, "g": g,
             "val_f1_macro": mt["val_f1_macro"],
             "test_acc": mt["acc"],
             "test_f1_macro": (mt["f1_1"] + mt["f1_0"]) / 2}
        results.append(r)
        print(f"[S2 {i+1}/{len(combos2)}] N={n} g={g} "
              f"val_f1m={r['val_f1_macro']:.4f} acc={r['test_acc']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    best = max(results, key=lambda x: x["val_f1_macro"])
    print(f"\n[BEST] {best}")

    with open(_os.path.join(OUT_DIR,
                            f"grid_search_{ARGS.tickers}_{ARGS.batch_mode}"
                            f"_results.json"),
              "w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best}, f,
                  indent=2, ensure_ascii=False)

    # append 到實驗記錄
    md_path = _os.path.join(_root, ARGS.md_file)
    rows = sorted(results, key=lambda x: -x["val_f1_macro"])
    lines = ["", f"### GRID-{datetime.now().strftime('%Y%m%d-%H%M')} "
                 f"論文網格搜尋（{ARGS.tickers}, raw 特徵、對齊版管線、"
                 f"batch={ARGS.batch_mode}、val macro-F1 選參）", "",
             f"- 範圍: 論文 4.3（兩階段）; stride={SEARCH_STRIDE}, "
             f"epochs={SEARCH_EPOCHS}; 總耗時 {(time.time()-t_start)/60:.0f} 分",
             f"- **最佳: window={best['window']}, m={best['m']}, "
             f"N={best['N']}, g={best['g']} "
             f"(val_f1_macro={best['val_f1_macro']:.4f})**", "",
             "| window | m | N | g | val F1 macro | test acc | test F1 macro |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = " **←最佳**" if r is best else ""
        lines.append(f"| {r['window']} | {r['m']} | {r['N']} | {r['g']} | "
                     f"{r['val_f1_macro']:.4f} | {r['test_acc']*100:.2f}% | "
                     f"{r['test_f1_macro']*100:.2f}%{mark} |")
    with open(md_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n[NEXT] 用最佳參數執行完整訓練:")
    print(f"  python test/run_paper_repro.py --raw-features "
          f"--tickers {ARGS.tickers} --start {START} "
          f"--train-end {TRAIN_END} --end {END} "
          f"--out-dir {ARGS.out_dir} --md-file {ARGS.md_file} "
          f"--window {best['window']} --m-pips {best['m']} "
          f"--N {best['N']} --g {best['g']} "
          f"--batch-mode {ARGS.batch_mode} --tag gridbest")


if __name__ == "__main__":
    main()
