"""
End-to-end pipeline — Chart GCN 台股實作.

Grid Search (論文 Section 4.3):
  - window (n):  {100, 120, 130, 140}
  - m_pips:      {30, 40, 60, 80}
  - N:           {10, 15, 20, 25}
  - g:           {3, 4, 5, 6}

兩階段搜尋 (加速):
  Stage 1: 固定 N=15, g=5, 搜尋 (window, m_pips) → 16 組
  Stage 2: 固定最佳 (window, m_pips), 搜尋 (N, g) → 16 組
  Final:   用最佳參數完整訓練 + 回測

用法:
    python main.py                          # 預設參數單次執行
    python main.py --grid-search            # 兩階段 grid search
    python main.py --grid-search --grid-full  # 完整 256 組 grid search
"""
import argparse
import itertools
import time
import torch
from torch.utils.data import random_split

from data_loader import fetch_tw_stocks, TW150_TICKERS
from dataset import ChartGCNDataset, split_by_date
from train import fit
from backtest import run_backtest, plot_backtest


def build_datasets(data, train_end, window, m_pips, N, g, stride,
                   seed=42, verbose=True):
    """建構 train/val/test datasets."""
    train_data, test_data = split_by_date(data, train_end)

    if verbose:
        print(f"\n[BUILD] window={window}, m_pips={m_pips}, N={N}, g={g}, "
              f"stride={stride}")

    train_full = ChartGCNDataset(
        train_data, window=window, m_pips=m_pips,
        N=N, g=g, stride=stride, verbose=verbose,
    )

    n_total = len(train_full)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    test_ds = ChartGCNDataset(
        test_data, window=window, m_pips=m_pips,
        N=N, g=g, stride=1, verbose=verbose,
    )

    return train_ds, val_ds, test_ds, test_data


def run_single(args, data):
    """單次執行 (無 grid search)."""
    torch.manual_seed(args.seed)

    train_ds, val_ds, test_ds, test_data = build_datasets(
        data, args.train_end,
        args.window, args.m_pips, args.N, args.g,
        args.stride, args.seed,
    )

    print(f"\n[TRAIN] epochs={args.epochs}, lr={args.lr}")
    model, metrics = fit(
        train_ds, val_ds, test_ds,
        N=args.N, g=args.g, F_dim=14,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
    )

    f1_macro = (metrics['f1_1'] + metrics['f1_0']) / 2
    print("\n=== 分類結果 ===")
    print(f"Test accuracy:  {metrics['acc']:.4f}")
    print(f"Test F1 (漲):   {metrics['f1_1']:.4f}")
    print(f"Test F1 (跌):   {metrics['f1_0']:.4f}")
    print(f"Test F1 macro:  {f1_macro:.4f}")

    print(f"\n[BACKTEST]")
    bt_results = run_backtest(model, test_ds, test_data,
                              indicator_n=args.window)
    plot_backtest(bt_results, save_path='backtest_result.png')

    return metrics


def grid_search(args, data):
    """
    論文 Section 4.3: Grid Search 尋找最佳超參數.
    排序/選擇標準: macro F1 = (f1_1 + f1_0) / 2
    """
    WINDOWS = [100, 120, 130, 140]
    M_PIPS = [30, 40, 60, 80]
    NS = [10, 15, 20, 25]
    GS = [3, 4, 5, 6]

    search_epochs = max(15, args.epochs // 2)
    search_stride = max(5, args.stride)

    all_results = []

    if args.grid_full:
        combos = list(itertools.product(WINDOWS, M_PIPS, NS, GS))
        combos = [(w, m, n, g) for w, m, n, g in combos
                  if n >= 5 and m > n]
        print(f"\n{'='*60}")
        print(f" 完整 Grid Search: {len(combos)} 組合")
        print(f" search_epochs={search_epochs}, stride={search_stride}")
        print(f"{'='*60}")

        for i, (w, m, n, g) in enumerate(combos):
            t0 = time.time()
            print(f"\n[{i+1}/{len(combos)}] window={w}, m_pips={m}, "
                  f"N={n}, g={g}")
            try:
                torch.manual_seed(args.seed)
                train_ds, val_ds, test_ds, _ = build_datasets(
                    data, args.train_end, w, m, n, g,
                    search_stride, args.seed, verbose=False,
                )
                _, metrics = fit(
                    train_ds, val_ds, test_ds,
                    N=n, g=g, F_dim=14,
                    epochs=search_epochs,
                    lr=args.lr,
                    dropout=args.dropout,
                    verbose=False,
                )
                elapsed = time.time() - t0
                f1_macro = (metrics['f1_1'] + metrics['f1_0']) / 2
                result = {
                    'window': w, 'm_pips': m, 'N': n, 'g': g,
                    'f1_macro': f1_macro,
                    'acc': metrics['acc'],
                    'f1_1': metrics['f1_1'], 'f1_0': metrics['f1_0'],
                    'time': elapsed,
                }
                all_results.append(result)
                print(f"  f1_macro={f1_macro:.4f}  acc={metrics['acc']:.4f}  "
                      f"f1_1={metrics['f1_1']:.4f}  ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  [FAIL] {e}")

    else:
        # ── 兩階段 Grid Search ──

        # Stage 1: 固定 N=15, g=5, 搜尋 (window, m_pips)
        stage1_combos = [(w, m) for w, m in itertools.product(WINDOWS, M_PIPS)
                         if m > 15]
        print(f"\n{'='*60}")
        print(f" Stage 1: 搜尋 (window, m_pips) — {len(stage1_combos)} 組合")
        print(f" 固定 N=15, g=5, epochs={search_epochs}, stride={search_stride}")
        print(f"{'='*60}")

        stage1_results = []
        for i, (w, m) in enumerate(stage1_combos):
            t0 = time.time()
            print(f"\n[S1 {i+1}/{len(stage1_combos)}] window={w}, m_pips={m}")
            try:
                torch.manual_seed(args.seed)
                train_ds, val_ds, test_ds, _ = build_datasets(
                    data, args.train_end, w, m, 15, 5,
                    search_stride, args.seed, verbose=False,
                )
                _, metrics = fit(
                    train_ds, val_ds, test_ds,
                    N=15, g=5, F_dim=14,
                    epochs=search_epochs,
                    lr=args.lr,
                    dropout=args.dropout,
                    verbose=False,
                )
                elapsed = time.time() - t0
                f1_macro = (metrics['f1_1'] + metrics['f1_0']) / 2
                result = {
                    'window': w, 'm_pips': m, 'N': 15, 'g': 5,
                    'f1_macro': f1_macro,
                    'acc': metrics['acc'],
                    'f1_1': metrics['f1_1'], 'f1_0': metrics['f1_0'],
                    'time': elapsed,
                }
                stage1_results.append(result)
                print(f"  f1_macro={f1_macro:.4f}  acc={metrics['acc']:.4f}  "
                      f"f1_1={metrics['f1_1']:.4f}  ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  [FAIL] {e}")

        if not stage1_results:
            print("[ERROR] Stage 1 全部失敗")
            return

        best_s1 = max(stage1_results, key=lambda x: x['f1_macro'])
        best_w, best_m = best_s1['window'], best_s1['m_pips']

        print(f"\n{'─'*60}")
        print(f" Stage 1 最佳: window={best_w}, m_pips={best_m}, "
              f"f1_macro={best_s1['f1_macro']:.4f}")
        print(f"{'─'*60}")

        # Stage 2: 固定最佳 (window, m_pips), 搜尋 (N, g)
        stage2_combos = [(n, g) for n, g in itertools.product(NS, GS)
                         if n >= 5 and best_m > n]
        print(f"\n{'='*60}")
        print(f" Stage 2: 搜尋 (N, g) — {len(stage2_combos)} 組合")
        print(f" 固定 window={best_w}, m_pips={best_m}")
        print(f"{'='*60}")

        stage2_results = []
        for i, (n, g) in enumerate(stage2_combos):
            t0 = time.time()
            print(f"\n[S2 {i+1}/{len(stage2_combos)}] N={n}, g={g}")
            try:
                torch.manual_seed(args.seed)
                train_ds, val_ds, test_ds, _ = build_datasets(
                    data, args.train_end, best_w, best_m, n, g,
                    search_stride, args.seed, verbose=False,
                )
                _, metrics = fit(
                    train_ds, val_ds, test_ds,
                    N=n, g=g, F_dim=14,
                    epochs=search_epochs,
                    lr=args.lr,
                    dropout=args.dropout,
                    verbose=False,
                )
                elapsed = time.time() - t0
                f1_macro = (metrics['f1_1'] + metrics['f1_0']) / 2
                result = {
                    'window': best_w, 'm_pips': best_m, 'N': n, 'g': g,
                    'f1_macro': f1_macro,
                    'acc': metrics['acc'],
                    'f1_1': metrics['f1_1'], 'f1_0': metrics['f1_0'],
                    'time': elapsed,
                }
                stage2_results.append(result)
                print(f"  f1_macro={f1_macro:.4f}  acc={metrics['acc']:.4f}  "
                      f"f1_1={metrics['f1_1']:.4f}  ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  [FAIL] {e}")

        all_results = stage1_results + stage2_results

    # ── 輸出結果總表 ──
    print(f"\n{'='*80}")
    print(f" Grid Search 結果總表 (排序: macro F1)")
    print(f"{'='*80}")
    print(f"  {'window':>6} {'m_pips':>6} {'N':>4} {'g':>4} | "
          f"{'f1_macro':>8} {'acc':>7} {'f1_1':>7} {'f1_0':>7} | {'time':>6}")
    print(f"  {'─'*70}")

    all_results.sort(key=lambda x: x['f1_macro'], reverse=True)
    for r in all_results:
        marker = " ★" if r == all_results[0] else ""
        print(f"  {r['window']:>6} {r['m_pips']:>6} {r['N']:>4} {r['g']:>4} | "
              f"{r['f1_macro']:>8.4f} {r['acc']:>7.4f} "
              f"{r['f1_1']:>7.4f} {r['f1_0']:>7.4f} | "
              f"{r['time']:>5.1f}s{marker}")

    best = all_results[0]
    print(f"\n{'='*60}")
    print(f" 最佳參數: window={best['window']}, m_pips={best['m_pips']}, "
          f"N={best['N']}, g={best['g']}")
    print(f" 最佳 f1_macro={best['f1_macro']:.4f}")
    print(f"{'='*60}")

    # ── 用最佳參數完整訓練 + 回測 ──
    print(f"\n[FINAL] 以最佳參數完整訓練 (epochs={args.epochs}, stride=1)")
    torch.manual_seed(args.seed)
    train_ds, val_ds, test_ds, test_data = build_datasets(
        data, args.train_end,
        best['window'], best['m_pips'], best['N'], best['g'],
        stride=1, seed=args.seed, verbose=True,
    )

    model, metrics = fit(
        train_ds, val_ds, test_ds,
        N=best['N'], g=best['g'], F_dim=14,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
    )

    f1_macro = (metrics['f1_1'] + metrics['f1_0']) / 2
    print("\n=== 最終測試結果 ===")
    print(f"Test accuracy:  {metrics['acc']:.4f}")
    print(f"Test F1 (漲):   {metrics['f1_1']:.4f}")
    print(f"Test F1 (跌):   {metrics['f1_0']:.4f}")
    print(f"Test F1 macro:  {f1_macro:.4f}")

    bt_results = run_backtest(model, test_ds, test_data,
                              indicator_n=best['window'])
    plot_backtest(bt_results, save_path='backtest_result.png')

    return best, metrics


def main():
    p = argparse.ArgumentParser(
        description="Chart GCN — 台股實作 (對齊論文 Li et al., KBS 2022)"
    )

    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--train-end", default="2023-12-31")

    p.add_argument("--window", type=int, default=100)
    p.add_argument("--m-pips", type=int, default=80)
    p.add_argument("--N", type=int, default=15)
    p.add_argument("--g", type=int, default=5)

    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--grid-search", action="store_true",
                   help="啟用 Grid Search (兩階段)")
    p.add_argument("--grid-full", action="store_true",
                   help="完整 Grid Search (所有組合)")

    args = p.parse_args()

    tickers = args.tickers or TW150_TICKERS
    print(f"[STEP 1] 下載台股資料: {len(tickers)} 檔")
    data = fetch_tw_stocks(tickers=tickers, start=args.start, end=args.end)
    print(f"  成功取得 {len(data)} 檔資料")

    if args.grid_search or args.grid_full:
        grid_search(args, data)
    else:
        run_single(args, data)


if __name__ == "__main__":
    main()
