"""用全部 50 檔台股執行訓練 + 回測（測試資料量是否改善結果）."""
import torch
from data_loader import fetch_tw_stocks, TW50_TICKERS
from dataset import ChartGCNDataset, split_by_date
from torch.utils.data import random_split
from train import fit
from backtest import run_backtest, plot_backtest

WINDOW, M_PIPS, N, G = 140, 60, 15, 5
SEED = 42
EPOCHS = 30
STRIDE_TRAIN = 3
STRIDE_TEST = 1

print("[STEP 1] Downloading data — ALL 50 TW stocks...")
data = fetch_tw_stocks(tickers=TW50_TICKERS, start="2018-01-01", end="2024-12-31")
print(f"  Got {len(data)} stocks")

train_data, test_data = split_by_date(data, "2023-12-31")
print(f"  Train stocks: {len(train_data)}, Test stocks: {len(test_data)}")

print(f"\n[BUILD] window={WINDOW}, m_pips={M_PIPS}, N={N}, g={G}")
print(f"Building train dataset (stride={STRIDE_TRAIN})...")
train_full = ChartGCNDataset(
    train_data, window=WINDOW, m_pips=M_PIPS,
    N=N, g=G, stride=STRIDE_TRAIN, n_workers=1,
)

torch.manual_seed(SEED)
n_total = len(train_full)
n_train = int(n_total * 0.8)
n_val = n_total - n_train
train_ds, val_ds = random_split(
    train_full, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED),
)

print(f"\nBuilding test dataset (stride={STRIDE_TEST})...")
test_ds = ChartGCNDataset(
    test_data, window=WINDOW, m_pips=M_PIPS,
    N=N, g=G, stride=STRIDE_TEST, n_workers=1,
)

print(f"\n[TRAIN] epochs={EPOCHS}")
model, metrics = fit(
    train_ds, val_ds, test_ds,
    N=N, g=G, F_dim=9,
    epochs=EPOCHS, lr=1e-3, dropout=0.3,
)

acc = metrics["acc"]
f1_1 = metrics["f1_1"]
f1_0 = metrics["f1_0"]
pre_1 = metrics["pre_1"]
rec_1 = metrics["rec_1"]
print(f"\n=== Final Test Results (50 stocks) ===")
print(f"Test accuracy:    {acc:.4f}")
print(f"Test F1 (up):     {f1_1:.4f}")
print(f"Test F1 (down):   {f1_0:.4f}")
print(f"Test Pre (up):    {pre_1:.4f}")
print(f"Test Rec (up):    {rec_1:.4f}")

print(f"\n[BACKTEST]")
bt_results = run_backtest(model, test_ds, test_data, indicator_n=WINDOW)
plot_backtest(bt_results, save_path="backtest_50stocks.png")
print("\nDone!")
