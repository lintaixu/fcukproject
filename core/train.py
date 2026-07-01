"""
Training & Evaluation — 嚴格對齊論文 (Li et al., KBS 2022).

變更:
  - 模型只接收 X, 不接收 A (移除 GCN 層後不需要 adjacency)
  - DateGroupedBatchSampler 保留 (對齊論文 Self-Attention 設計)
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from model import ChartGCN


# ──────────────────────────────────────────────
#  DateGroupedBatchSampler (對齊論文 Self-Attention)
# ──────────────────────────────────────────────

class DateGroupedBatchSampler(Sampler):
    """
    每個 batch 包含同一交易日的所有股票 sample.

    論文 Self-Attention (Eq.7-9):
      同一天的 S 檔股票互相 attend, 捕捉跨股票的市場結構.
    """

    def __init__(self, dataset, shuffle=True):
        self.shuffle = shuffle

        # 處理 Subset 的情況 (train/val split 後)
        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            base_ds = dataset.dataset
            base_to_pos = {int(base_idx): pos
                           for pos, base_idx in enumerate(dataset.indices)}
            self.date_groups = []
            for date_key, base_indices in base_ds.date_to_indices.items():
                filtered = [base_to_pos[i] for i in base_indices
                            if i in base_to_pos]
                if filtered:
                    self.date_groups.append(filtered)
        else:
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
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
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
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
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
    dropout=0.3,
    device=None,
    verbose=True,
):
    # 預設 CPU — 模型僅 26K 參數，GPU 優勢極小，且避免螢幕黑屏
    if device is None:
        device = "cpu"
    if verbose:
        print(f"[INFO] device = {device}")

    # DateGroupedBatchSampler
    train_sampler = DateGroupedBatchSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler)

    val_sampler = DateGroupedBatchSampler(val_ds, shuffle=False)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler)

    test_sampler = DateGroupedBatchSampler(test_ds, shuffle=False)
    test_loader = DataLoader(test_ds, batch_sampler=test_sampler)

    if verbose:
        print(f"[INFO] DateGroupedBatchSampler: "
              f"train={len(train_sampler)} dates, "
              f"val={len(val_sampler)} dates, "
              f"test={len(test_sampler)} dates")

    model = ChartGCN(N=N, g=g, F_dim=F_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_score = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_metrics = evaluate(model, val_loader, device)
        val_f1_macro = (val_metrics['f1_1'] + val_metrics['f1_0']) / 2

        if val_f1_macro > best_val_score:
            best_val_score = val_f1_macro
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} | "
                  f"val_acc={val_metrics['acc']:.4f} "
                  f"val_f1_1={val_metrics['f1_1']:.4f} "
                  f"val_f1_macro={val_f1_macro:.4f}")

    # 載入最佳 weights 跑 test
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)

    if verbose:
        print("\n=== Test Metrics ===")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

    return model, test_metrics
