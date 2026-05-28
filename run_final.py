"""用 Grid Search 最佳參數執行最終訓練 + 回測."""
import torch
from data_loader import fetch_tw_stocks, TW50_TICKERS
from dataset import ChartGCNDataset, split_by_date
from torch.utils.data import random_split
from train import fit
from backtest import run_backtest, plot_backtest

WINDOW, M_PIPS, N, G = 140, 60, 15, 5
SEED = 42
EPOCHS = 30

print("[STEP 1] Downloading data...")
tickers = TW50_TICKERS[:15]
data = fetch_tw_stocks(tickers=tickers, start="2018-01-01", end="2024-12-31")
print(f"  Got {len(data)} stocks")

train_data, test_data = split_by_date(data, "2023-12-31")

print(f"\n[BUILD] Best params: window={WINDOW}, m_pips={M_PIPS}, N={N}, g={G}")
print("Building train dataset (single-thread)...")
train_full = ChartGCNDataset(
    train_data, window=WINDOW, m_pips=M_PIPS,
    N=N, g=G, stride=1, n_workers=1,
)

torch.manual_seed(SEED)
n_total = len(train_full)
n_train = int(n_total * 0.8)
n_val = n_total - n_train
train_ds, val_ds = random_split(
    train_full, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED),
)

print("Building test dataset (single-thread)...")
test_ds = ChartGCNDataset(
    test_data, window=WINDOW, m_pips=M_PIPS,
    N=N, g=G, stride=1, n_workers=1,
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
print(f"\n=== Final Test Results ===")
print(f"Test accuracy: {acc:.4f}")
print(f"Test F1 (up):  {f1_1:.4f}")
print(f"Test F1 (down): {f1_0:.4f}")

print("\n[BACKTEST]")
bt_results = run_backtest(model, test_ds, test_data, indicator_n=WINDOW)
plot_backtest(bt_results, save_path="backtest_result.png")
print("\nDone!")
