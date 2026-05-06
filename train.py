"""
Training & Evaluation script for Chart GCN.

重點修改:
  - DateGroupedBatchSampler: 確保每個 batch 只包含同一天的所有股票,
    讓 Self-Attention 計算「同一天不同股票之間的關聯」(對齊論文設計).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from model import ChartGCN


# ──────────────────────────────────────────────
#  DateGroupedBatchSampler (對齊論文 Self-Attention)
# ──────────────────────────────────────────────

class DateGroupedBatchSampler(Sampler):
    """
    每個 batch 包含同一交易日的所有股票 sample.

    論文中 Self-Attention 的設計意圖:
      同一天的 S 檔股票互相 attend, 捕捉跨股票的市場結構.
      若 batch 隨機混入不同日期的 sample, attention 計算的是無意義的跨時間關聯.

    用法:
      sampler = DateGroupedBatchSampler(dataset, shuffle=True)
      loader  = DataLoader(dataset, batch_sampler=sampler)
      # 注意: 使用 batch_sampler 時不能設 batch_size / shuffle / drop_last

    每個 iteration yield 的是一組 indices (同日所有股票).
    batch size = 該日有效股票數 (約 50), 無法自由控制.
    """

    def __init__(self, dataset, shuffle=True):
        """
        Args:
            dataset: ChartGCNDataset (或任何有 date_to_indices 屬性的 Dataset)
                     如果傳入 Subset, 會自動追溯底層 dataset 並重新映射 indices.
            shuffle: 是否隨機打亂日期順序 (每個 epoch 不同).
        """
        self.shuffle = shuffle

        # 處理 Subset 的情況 (train/val split 後)
        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            # dataset 是 Subset → 取底層 dataset 的 date_to_indices
            base_ds = dataset.dataset
            # 建立 base_idx → subset_pos 的映射
            # Subset.__getitem__(pos) → base_ds[self.indices[pos]]
            # batch_sampler 需 yield 的是 pos (0..len-1), 不是 base_idx
            base_to_pos = {int(base_idx): pos
                           for pos, base_idx in enumerate(dataset.indices)}

            self.date_groups = []
            for date_key, base_indices in base_ds.date_to_indices.items():
                # 只保留屬於此 Subset 的, 且轉換成 positional index
                filtered = [base_to_pos[i] for i in base_indices
                            if i in base_to_pos]
                if filtered:
                    self.date_groups.append(filtered)
        else:
            # 正常 ChartGCNDataset
            self.date_groups = list(dataset.date_to_indices.values())

    def __iter__(self):
        order = list(range(len(self.date_groups)))
        if self.shuffle:
            np.random.shuffle(order)
        for idx in order:
            yield self.date_groups[idx]

    def __len__(self):
        return len(self.date_groups)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    for X, A, y in loader:
        X, A, y = X.to(device), A.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X, A)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        n_correct += (logits.argmax(1) == y).sum().item()
        n_total += X.size(0)
    return total_loss / n_total, n_correct / n_total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    for X, A, y in loader:
        X, A, y = X.to(device), A.to(device), y.to(device)
        logits = model(X, A)
        all_pred.append(logits.argmax(1).cpu().numpy())
        all_true.append(y.cpu().numpy())
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)

    return {
        'acc': accuracy_score(true, pred),
        'pre_1': precision_score(true, pred, pos_label=1, zero_division=0),
        'pre_0': precision_score(true, pred, pos_label=0, zero_division=0),
        'rec_1': recall_score(true, pred, pos_label=1, zero_division=0),
        'rec_0': recall_score(true, pred, pos_label=0, zero_division=0),
        'f1_1': f1_score(true, pred, pos_label=1, zero_division=0),
        'f1_0': f1_score(true, pred, pos_label=0, zero_division=0),
    }


def fit(
    train_ds, val_ds, test_ds,
    *,
    N=15, g=5, F_dim=9,
    epochs=30,
    batch_size=64,
    lr=1e-3,
    weight_decay=5e-5,
    device=None,
    verbose=True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[INFO] device = {device}")

    # --- 使用 DateGroupedBatchSampler (對齊論文 Self-Attention 設計) ---
    # 每個 batch = 同一天的所有股票, 讓 attention 捕捉跨股票關聯
    # 注意: batch_sampler 與 batch_size/shuffle/drop_last 互斥
    train_sampler = DateGroupedBatchSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler)

    # val / test 也用同日 batch, 確保 attention 語意一致
    val_sampler = DateGroupedBatchSampler(val_ds, shuffle=False)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler)

    test_sampler = DateGroupedBatchSampler(test_ds, shuffle=False)
    test_loader = DataLoader(test_ds, batch_sampler=test_sampler)

    if verbose:
        print(f"[INFO] DateGroupedBatchSampler: "
              f"train={len(train_sampler)} dates, "
              f"val={len(val_sampler)} dates, "
              f"test={len(test_sampler)} dates")

    model = ChartGCN(N=N, g=g, F_dim=F_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_metrics = evaluate(model, val_loader, device)
        val_acc = val_metrics['acc']

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} | "
                  f"val_acc={val_acc:.4f} val_f1_1={val_metrics['f1_1']:.4f}")

    # 載入最佳 weights 跑 test
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)

    if verbose:
        print("\n=== Test Metrics ===")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

    return model, test_metrics
