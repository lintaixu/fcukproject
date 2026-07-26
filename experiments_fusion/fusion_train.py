"""訓練/評估 — 沿用 core/train.py 邏輯, 改為時間序 val 與自帶 sampler."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from model import ChartGCN


class DateGroupedBatchSampler(Sampler):
    def __init__(self, dataset, shuffle=True):
        self.shuffle = shuffle
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
    total_loss, n_correct, n_total = 0.0, 0, 0
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
def predict_probs(model, dataset, device='cpu'):
    """回傳 (probs (n,2), true (n,)) — 依 date group 批次推論."""
    model.eval()
    sampler = DateGroupedBatchSampler(dataset, shuffle=False)
    probs = np.zeros((len(dataset), 2), dtype=np.float32)
    for indices in sampler:
        X = torch.from_numpy(dataset.X[indices]).float().to(device)
        p = torch.softmax(model(X), dim=1).cpu().numpy()
        for i, idx in enumerate(indices):
            probs[idx] = p[i]
    return probs, dataset.y


def metrics_from_pred(true, pred):
    return {
        'acc': accuracy_score(true, pred),
        'f1_1': f1_score(true, pred, pos_label=1, zero_division=0),
        'f1_0': f1_score(true, pred, pos_label=0, zero_division=0),
        'pre_1': precision_score(true, pred, pos_label=1, zero_division=0),
        'pre_0': precision_score(true, pred, pos_label=0, zero_division=0),
        'rec_1': recall_score(true, pred, pos_label=1, zero_division=0),
        'rec_0': recall_score(true, pred, pos_label=0, zero_division=0),
    }


def macro_f1(m):
    return (m['f1_1'] + m['f1_0']) / 2


def fit(train_ds, val_ds, test_ds, *, N=15, g=5, F_dim=9,
        epochs=20, lr=1e-3, weight_decay=5e-5, dropout=0.3,
        seed=42, device='cpu', verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(
        train_ds, batch_sampler=DateGroupedBatchSampler(train_ds, shuffle=True))

    model = ChartGCN(N=N, g=g, F_dim=F_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, optimizer,
                                 criterion, device)
        vp, vt = predict_probs(model, val_ds, device)
        vm = metrics_from_pred(vt, vp.argmax(1))
        vf1 = macro_f1(vm)
        if vf1 > best_val:
            best_val = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  Ep{epoch:02d} loss={tl:.4f} tr_acc={ta:.4f} "
                  f"val_acc={vm['acc']:.4f} val_f1m={vf1:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    tp, tt = predict_probs(model, test_ds, device)
    tm = metrics_from_pred(tt, tp.argmax(1))
    tm['f1_macro'] = macro_f1(tm)
    tm['best_val_f1_macro'] = best_val
    return model, tm, tp
