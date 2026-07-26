"""
論文復現實驗執行器 (Li et al., KBS 2022 Chart GCN).

每次執行會自動:
  1. 訓練 + 測試 + 交易模擬
  2. 結果 JSON 存到 experiments_paper/<EXP_ID>.json
  3. 實驗摘要 append 到 實驗記錄_論文對齊.md (供人工審閱)
  4. 回測淨值圖存到 experiments_paper/<EXP_ID>_backtest.png

用法範例:
  python test/run_paper_repro.py --tag baseline                 # TW50 論文設定
  python test/run_paper_repro.py --tickers tw100 --tag tw100    # 100 檔
  python test/run_paper_repro.py --tickers electronics --tag el # 電子族群
  python test/run_paper_repro.py --batch-mode date --tag dateb  # 同日批次變體
"""
import argparse
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

from data_loader import (fetch_tw_stocks, TW50_TICKERS, TW100_TICKERS,
                         TW200_TICKERS, TW500_TICKERS, TW_ELECTRONICS, TW_FINANCE,
                         TW_ELEC_ALL, TW_SEMI, TW_ELEC_COMP, TW_FIN_ALL,
                         SSE50_TICKERS)
from dataset import ChartGCNDataset, split_by_date
from train import fit
from backtest import run_backtest, plot_backtest

import pandas as pd


class ArrayDataset(torch.utils.data.Dataset):
    """從快取陣列重建、介面相容 ChartGCNDataset 的輕量 Dataset."""

    def __init__(self, X, y, meta):
        self.X = X
        self.y = y
        self.meta = meta
        self.date_to_indices = {}
        for i, (tk, date) in enumerate(meta):
            self.date_to_indices.setdefault(str(date)[:10], []).append(i)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx]).float(),
                torch.tensor(self.y[idx], dtype=torch.long))


def save_ds_cache(path, ds):
    np.savez_compressed(
        path, X=ds.X, y=ds.y,
        tickers=np.array([m[0] for m in ds.meta]),
        dates=np.array([str(m[1]) for m in ds.meta]),
    )


def load_ds_cache(path):
    z = np.load(path, allow_pickle=False)
    meta = [(str(tk), pd.Timestamp(str(d)))
            for tk, d in zip(z["tickers"], z["dates"])]
    return ArrayDataset(z["X"], z["y"], meta)

# 論文 Table 3 結果 (供對照)
PAPER = {
    "SZ-50":   {"acc": 69.26, "pre_1": 65.76, "pre_0": 72.26, "f1_1": 66.40, "f1_0": 71.67},
    "CSI-300": {"acc": 68.62, "pre_1": 64.87, "pre_0": 71.91, "f1_1": 65.93, "f1_0": 70.91},
}

TICKER_SETS = {
    "tw50": TW50_TICKERS,
    "tw100": TW100_TICKERS,
    "tw200": TW200_TICKERS,
    "tw500": TW500_TICKERS,
    "elec_all": TW_ELEC_ALL,
    "semi": TW_SEMI,
    "elec_comp": TW_ELEC_COMP,
    "fin_all": TW_FIN_ALL,
    "electronics": TW_ELECTRONICS,
    "finance": TW_FINANCE,
    "sse50": SSE50_TICKERS,     # 論文原始資料 (上證50, chinatest 用)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="tw50", choices=list(TICKER_SETS))
    ap.add_argument("--n-stocks", type=int, default=0, help="0 = 全部")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--train-end", default="2023-12-31")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--window", type=int, default=140)
    ap.add_argument("--m-pips", type=int, default=60)
    ap.add_argument("--N", type=int, default=15)
    ap.add_argument("--g", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch-mode", default="paper", choices=["paper", "date"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--raw-features", action="store_true",
                    help="不做 z-score, 用原始指標值 (論文未提及正規化)")
    ap.add_argument("--no-attention", action="store_true",
                    help="論文消融變體 Chart GCN-2 (無 self-attention)")
    ap.add_argument("--out-dir", default="experiments_paper",
                    help="結果輸出目錄 (相對專案根)")
    ap.add_argument("--md-file", default="實驗記錄_論文對齊.md",
                    help="實驗記錄 markdown (相對專案根)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    exp_id = "EXP-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.tag:
        exp_id += f"_{args.tag}"
    out_dir = _os.path.join(_root, args.out_dir)
    _os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}\n {exp_id}\n{'='*70}")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 1. 資料 ──────────────────────────────────────────────
    tickers = TICKER_SETS[args.tickers]
    if args.n_stocks > 0:
        tickers = tickers[:args.n_stocks]
    t0 = time.time()
    data = fetch_tw_stocks(tickers=tickers, start=args.start, end=args.end)
    print(f"\n[DATA] {len(data)} 檔股票 ({time.time()-t0:.0f}s)")

    train_data, test_data = split_by_date(
        data, args.train_end, warmup_rows=2 * args.window)

    # ── 2. Dataset (訓練期統計量 → 測試集, 無前視) ────────────
    t0 = time.time()
    cache_dir = _os.path.join(out_dir, "dscache")
    _os.makedirs(cache_dir, exist_ok=True)
    key = (f"{args.tickers}{args.n_stocks}_{args.start}_{args.train_end}_"
           f"{args.end}_w{args.window}m{args.m_pips}N{args.N}g{args.g}"
           f"s{args.stride}{'_raw' if args.raw_features else ''}")
    tr_cache = _os.path.join(cache_dir, f"{key}_train.npz")
    te_cache = _os.path.join(cache_dir, f"{key}_test.npz")

    if _os.path.exists(tr_cache) and _os.path.exists(te_cache):
        print(f"[BUILD] 使用快取 {key}")
        train_full = load_ds_cache(tr_cache)
        test_ds = load_ds_cache(te_cache)
    else:
        raw_stats = ({tk: (0.0, 1.0) for tk in data}
                     if args.raw_features else None)
        print("[BUILD] train dataset ...")
        train_full = ChartGCNDataset(
            train_data, window=args.window, m_pips=args.m_pips,
            N=args.N, g=args.g, stride=args.stride, n_workers=args.workers,
            norm_stats=raw_stats,
        )
        print("[BUILD] test dataset (沿用訓練期 norm_stats) ...")
        test_ds = ChartGCNDataset(
            test_data, window=args.window, m_pips=args.m_pips,
            N=args.N, g=args.g, stride=1, n_workers=args.workers,
            norm_stats=(raw_stats if args.raw_features
                        else train_full.norm_stats),
            min_date=args.train_end,
        )
        save_ds_cache(tr_cache, train_full)
        save_ds_cache(te_cache, test_ds)
    build_time = time.time() - t0

    # 論文 4.3: 隨機 80/20 切 train/val
    n_total = len(train_full)
    n_train = int(n_total * 0.8)
    train_ds, val_ds = random_split(
        train_full, [n_train, n_total - n_train],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_pos = float((train_full.y == 1).mean()) * 100
    test_pos = float((test_ds.y == 1).mean()) * 100
    print(f"[STATS] train={n_total} (漲 {train_pos:.1f}%) | "
          f"test={len(test_ds)} (漲 {test_pos:.1f}%)")

    # ── 3. 訓練 ──────────────────────────────────────────────
    t0 = time.time()
    model, metrics = fit(
        train_ds, val_ds, test_ds,
        N=args.N, g=args.g, F_dim=9,
        epochs=args.epochs, lr=args.lr, weight_decay=5e-5,
        batch_mode=args.batch_mode, batch_size=128,
        paper_exact=True,
        use_attention=not args.no_attention,
    )
    train_time = time.time() - t0

    f1_macro = (metrics["f1_1"] + metrics["f1_0"]) / 2

    # ── 4. 對照論文 ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f" {'指標':<10} {'本實驗':>10} {'論文 SZ-50':>12} {'差距':>10}")
    print(f"{'─'*60}")
    for k, pk in [("acc", "acc"), ("pre_1", "pre_1"), ("pre_0", "pre_0"),
                  ("f1_1", "f1_1"), ("f1_0", "f1_0")]:
        ours = metrics[k] * 100
        ref = PAPER["SZ-50"][pk]
        print(f" {k:<10} {ours:>9.2f}% {ref:>11.2f}% {ours-ref:>+9.2f}%")

    # ── 5. 交易模擬 ──────────────────────────────────────────
    bt_summary = None
    if not args.no_backtest:
        bt = run_backtest(model, test_ds, test_data,
                          indicator_n=args.window,
                          batch_mode=args.batch_mode,
                          )
        png_path = _os.path.join(out_dir, f"{exp_id}_backtest.png")
        plot_backtest(bt, save_path=png_path)
        bt_summary = {
            "strategy_nv": bt["avg_final_nv"],
            "benchmark_nv": bt["benchmark_final"],
            "excess_pct": (bt["avg_final_nv"] - bt["benchmark_final"]) * 100,
        }

    # ── 6. 記錄 ──────────────────────────────────────────────
    record = {
        "exp_id": exp_id,
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "params": vars(args),
        "n_stocks": len(data),
        "n_train_samples": n_total,
        "n_test_samples": len(test_ds),
        "train_pos_pct": round(train_pos, 2),
        "test_pos_pct": round(test_pos, 2),
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "f1_macro": round(f1_macro, 4),
        "paper_ref_sz50": PAPER["SZ-50"],
        "acc_gap_vs_paper": round(metrics["acc"] * 100 - PAPER["SZ-50"]["acc"], 2),
        "backtest": bt_summary,
        "build_time_s": round(build_time, 1),
        "train_time_s": round(train_time, 1),
    }
    json_path = _os.path.join(out_dir, f"{exp_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    model_path = _os.path.join(out_dir, f"{exp_id}_model.pt")
    torch.save(model.state_dict(), model_path)

    # append 到實驗記錄 markdown
    md_path = _os.path.join(_root, args.md_file)
    test_year = args.end[:4]
    bt_txt = (f"策略 {bt_summary['strategy_nv']:.4f} vs 大盤 "
              f"{bt_summary['benchmark_nv']:.4f} "
              f"(超額 {bt_summary['excess_pct']:+.2f}%)"
              if bt_summary else "未執行")
    block = f"""
### {exp_id}

- **時間**: {record['datetime']}　**標的**: {args.tickers} ({len(data)} 檔)　**期間**: {args.start} ~ {args.end} (train_end={args.train_end})
- **參數**: window={args.window}, m={args.m_pips}, N={args.N}, g={args.g}, stride={args.stride}, epochs={args.epochs}, lr={args.lr}, batch={args.batch_mode}, seed={args.seed}
- **樣本**: train {n_total} (漲 {train_pos:.1f}%) / test {len(test_ds)} (漲 {test_pos:.1f}%)

| 指標 | 本實驗 | 論文 SZ-50 | 差距 |
|---|---|---|---|
| Accuracy | {metrics['acc']*100:.2f}% | 69.26% | {metrics['acc']*100-69.26:+.2f}% |
| Pre_1 / Pre_0 | {metrics['pre_1']*100:.2f}% / {metrics['pre_0']*100:.2f}% | 65.76% / 72.26% | — |
| F1_1 / F1_0 | {metrics['f1_1']*100:.2f}% / {metrics['f1_0']*100:.2f}% | 66.40% / 71.67% | — |
| F1 macro | {f1_macro*100:.2f}% | 69.04% | {f1_macro*100-69.04:+.2f}% |
| Val F1 macro (選模) | {metrics.get('val_f1_macro', 0)*100:.2f}% | — | — |

- **回測 ({test_year})**: {bt_txt}
- **耗時**: 建構 {build_time:.0f}s / 訓練 {train_time:.0f}s
- **檔案**: `experiments_paper/{exp_id}.json`
"""
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(block)

    print(f"\n[SAVED] {json_path}")
    print(f"[SAVED] {md_path} (已 append 實驗摘要)")
    print(f"\n{exp_id} 完成. Test Acc = {metrics['acc']*100:.2f}% "
          f"(論文 SZ-50: 69.26%)")


if __name__ == "__main__":
    main()
