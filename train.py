"""
Training & Evaluation script for Chart GCN.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from model import ChartGCN


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
    device=None,
    verbose=True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[INFO] device = {device}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

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
