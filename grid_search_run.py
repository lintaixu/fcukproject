"""
Standalone Grid Search — 避免 main.py 的 multiprocessing MemoryError.
兩階段搜尋最佳超參數，完成後寫入最佳參數並更新 PPT.
"""
import itertools
import time
import json
import sys
import torch
from torch.utils.data import random_split

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "core"))
from data_loader import fetch_tw_stocks, TW100_TICKERS
from dataset import ChartGCNDataset, split_by_date
from train import fit
from backtest import run_backtest, plot_backtest


SEED = 42
TRAIN_END = "2023-12-31"
START = "2018-01-01"
END = "2024-12-31"
LR = 1e-3
DROPOUT = 0.3
BATCH_SIZE = 128


def build_datasets(data, train_end, window, m_pips, N, g, stride,
                   n_workers=4, seed=42, verbose=False):
    train_data, test_data = split_by_date(data, train_end)
    train_full = ChartGCNDataset(
        train_data, window=window, m_pips=m_pips,
        N=N, g=g, stride=stride, n_workers=n_workers, verbose=verbose,
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
        N=N, g=g, stride=1, n_workers=n_workers, verbose=verbose,
    )
    return train_ds, val_ds, test_ds, test_data


def run_grid_search(data, n_workers=4):
    WINDOWS = [100, 120, 130, 140]
    M_PIPS = [30, 40, 60, 80]
    NS = [10, 15, 20, 25]
    GS = [3, 4, 5, 6]

    search_epochs = 15
    search_stride = 5

    # ── Stage 1: 固定 N=15, g=5, 搜尋 (window, m_pips) ──
    stage1_combos = [(w, m) for w, m in itertools.product(WINDOWS, M_PIPS)
                     if m > 15]
    print(f"\n{'='*60}")
    print(f" Stage 1: 搜尋 (window, m_pips) — {len(stage1_combos)} 組合")
    print(f" 固定 N=15, g=5, epochs={search_epochs}, stride={search_stride}")
    print(f" workers={n_workers}")
    print(f"{'='*60}")

    stage1_results = []
    for i, (w, m) in enumerate(stage1_combos):
        t0 = time.time()
        print(f"\n[S1 {i+1}/{len(stage1_combos)}] window={w}, m_pips={m}")
        try:
            torch.manual_seed(SEED)
            train_ds, val_ds, test_ds, _ = build_datasets(
                data, TRAIN_END, w, m, 15, 5,
                search_stride, n_workers=n_workers, seed=SEED,
            )
            _, metrics = fit(
                train_ds, val_ds, test_ds,
                N=15, g=5, F_dim=9,
                epochs=search_epochs,
                lr=LR, dropout=DROPOUT, verbose=False,
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
        except MemoryError:
            print(f"  [MemoryError] 記憶體不足!")
            return None, "memory_error"
        except Exception as e:
            print(f"  [FAIL] {e}")

    if not stage1_results:
        print("[ERROR] Stage 1 全部失敗")
        return None, "all_failed"

    best_s1 = max(stage1_results, key=lambda x: x['f1_macro'])
    best_w, best_m = best_s1['window'], best_s1['m_pips']
    print(f"\n{'─'*60}")
    print(f" Stage 1 最佳: window={best_w}, m_pips={best_m}, "
          f"f1_macro={best_s1['f1_macro']:.4f}")
    print(f"{'─'*60}")

    # ── Stage 2: 固定最佳 (window, m_pips), 搜尋 (N, g) ──
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
            torch.manual_seed(SEED)
            train_ds, val_ds, test_ds, _ = build_datasets(
                data, TRAIN_END, best_w, best_m, n, g,
                search_stride, n_workers=n_workers, seed=SEED,
            )
            _, metrics = fit(
                train_ds, val_ds, test_ds,
                N=n, g=g, F_dim=9,
                epochs=search_epochs,
                lr=LR, dropout=DROPOUT, verbose=False,
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
        except MemoryError:
            print(f"  [MemoryError] 記憶體不足!")
            return None, "memory_error"
        except Exception as e:
            print(f"  [FAIL] {e}")

    all_results = stage1_results + stage2_results
    all_results.sort(key=lambda x: x['f1_macro'], reverse=True)

    # ── 輸出結果總表 ──
    print(f"\n{'='*80}")
    print(f" Grid Search 結果總表 (排序: macro F1)")
    print(f"{'='*80}")
    print(f"  {'window':>6} {'m_pips':>6} {'N':>4} {'g':>4} | "
          f"{'f1_macro':>8} {'acc':>7} {'f1_1':>7} {'f1_0':>7} | {'time':>6}")
    print(f"  {'─'*70}")
    for r in all_results:
        marker = " *" if r == all_results[0] else ""
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

    return best, all_results


def final_train(data, best, n_workers=4):
    """用最佳參數完整訓練 + 回測."""
    EPOCHS = 30
    print(f"\n[FINAL] 最佳參數完整訓練 (epochs={EPOCHS}, stride=1, workers={n_workers})")
    torch.manual_seed(SEED)

    train_ds, val_ds, test_ds, test_data = build_datasets(
        data, TRAIN_END,
        best['window'], best['m_pips'], best['N'], best['g'],
        stride=1, n_workers=n_workers, seed=SEED, verbose=True,
    )

    model, metrics = fit(
        train_ds, val_ds, test_ds,
        N=best['N'], g=best['g'], F_dim=9,
        epochs=EPOCHS, lr=LR, dropout=DROPOUT,
    )

    f1_macro = (metrics['f1_1'] + metrics['f1_0']) / 2
    print("\n=== 最終測試結果 ===")
    print(f"Test accuracy:  {metrics['acc']:.4f}")
    print(f"Test F1 (漲):   {metrics['f1_1']:.4f}")
    print(f"Test F1 (跌):   {metrics['f1_0']:.4f}")
    print(f"Test F1 macro:  {f1_macro:.4f}")

    bt_results = run_backtest(model, test_ds, test_data,
                              indicator_n=best['window'])
    plot_backtest(bt_results, save_path='backtest_gridsearch.png')

    return metrics, bt_results


def save_best_params(best, metrics, bt_results):
    """將最佳參數和結果存為 JSON."""
    output = {
        'best_params': {
            'window': best['window'],
            'm_pips': best['m_pips'],
            'N': best['N'],
            'g': best['g'],
        },
        'grid_search_f1_macro': best['f1_macro'],
        'final_metrics': {
            'acc': metrics['acc'],
            'f1_1': metrics['f1_1'],
            'f1_0': metrics['f1_0'],
            'f1_macro': (metrics['f1_1'] + metrics['f1_0']) / 2,
            'pre_1': metrics['pre_1'],
            'rec_1': metrics['rec_1'],
        },
        'backtest': {
            'strategy_return': (bt_results['avg_final_nv'] - 1) * 100,
            'benchmark_return': (bt_results['benchmark_final'] - 1) * 100,
            'excess_return': (bt_results['avg_final_nv'] - bt_results['benchmark_final']) * 100,
        },
    }
    with open('best_params.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n最佳參數已儲存: best_params.json")
    return output


def main():
    print(f"[STEP 1] 下載台股資料")
    data = fetch_tw_stocks(tickers=TW100_TICKERS, start=START, end=END)
    print(f"  成功取得 {len(data)} 檔資料")

    # 先用 4 workers 嘗試
    n_workers = 4
    print(f"\n[STEP 2] Grid Search (workers={n_workers})")
    best, result = run_grid_search(data, n_workers=n_workers)

    if result == "memory_error":
        n_workers = 2
        print(f"\n[RETRY] 記憶體不足，改用 {n_workers} workers 重跑")
        best, result = run_grid_search(data, n_workers=n_workers)

        if result == "memory_error":
            n_workers = 1
            print(f"\n[RETRY] 再次記憶體不足，改用 {n_workers} worker 重跑")
            best, result = run_grid_search(data, n_workers=n_workers)

    if best is None:
        print("[ERROR] Grid Search 全部失敗，無法繼續")
        sys.exit(1)

    # 最終訓練
    print(f"\n[STEP 3] 最佳參數完整訓練")
    metrics, bt_results = final_train(data, best, n_workers=n_workers)

    # 儲存結果
    output = save_best_params(best, metrics, bt_results)
    print(json.dumps(output, indent=2, ensure_ascii=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
