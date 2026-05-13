"""
End-to-end pipeline: 從台股資料 → Chart GCN → 訓練 → 測試評估.

執行方式:
    python main.py                    # 預設用 yfinance 抓真實台股
    python main.py --synthetic        # 沙箱無外網時用合成資料
    python main.py --tickers 2330.TW 2317.TW
"""
import argparse
import torch
from torch.utils.data import random_split

from data_loader import fetch_tw_stocks, TW50_TICKERS
from dataset import ChartGCNDataset, split_by_date
from train import fit
from backtest import run_backtest, plot_backtest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=None,
                   help="台股代號 (含 .TW), 預設用 TW50_TICKERS")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--train-end", default="2023-12-31",
                   help="切分訓練/測試的日期 (含)")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--m-pips", type=int, default=5)
    p.add_argument("--N", type=int, default=15)
    p.add_argument("--g", type=int, default=5)
    p.add_argument("--stride", type=int, default=1,
                   help="rolling window 的步長, 越大訓練樣本越少但越快")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # 1. 下載資料
    tickers = args.tickers or TW50_TICKERS
    print(f"[STEP 1] 下載台股資料: {len(tickers)} 檔")
    data = fetch_tw_stocks(
        tickers=tickers,
        start=args.start, end=args.end,
    )
    print(f"  成功取得 {len(data)} 檔資料")

    # 2. 切分 train / test (按時間)
    print(f"\n[STEP 2] 時間切分 (train_end={args.train_end})")
    train_data, test_data = split_by_date(data, args.train_end)
    for tk in train_data:
        print(f"  {tk}: train={len(train_data[tk])}, test={len(test_data.get(tk, []))}")

    # 3. 建 Dataset (內含 PIP+VG+Subgraph 所有預處理)
    print(f"\n[STEP 3] 建構訓練集 (window={args.window}, stride={args.stride})")
    train_full = ChartGCNDataset(
        train_data, window=args.window, m_pips=args.m_pips,
        N=args.N, g=args.g, stride=args.stride,
    )

    # 隨機 80/20 切 train/val
    n_total = len(train_full)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"\n[STEP 4] 建構測試集")
    test_ds = ChartGCNDataset(
        test_data, window=args.window, m_pips=args.m_pips,
        N=args.N, g=args.g, stride=1,   # 測試集每天都要評估
    )

    # 4. 訓練
    print(f"\n[STEP 5] 訓練 Chart GCN (epochs={args.epochs})")
    model, metrics = fit(
        train_ds, val_ds, test_ds,
        N=args.N, g=args.g, F_dim=9,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print("\n=== 分類結果 ===")
    print(f"Test accuracy: {metrics['acc']:.4f}")
    print(f"Test F1 (漲): {metrics['f1_1']:.4f}")
    print(f"Test F1 (跌): {metrics['f1_0']:.4f}")

    # 5. 交易模擬 (論文 Section 6.3)
    print(f"\n[STEP 6] 交易模擬")
    bt_results = run_backtest(model, test_ds, test_data)
    plot_backtest(bt_results, save_path='backtest_result.png')


if __name__ == "__main__":
    main()
