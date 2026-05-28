"""
Comprehensive debug: find why Chart GCN doesn't learn.

Tests:
  1. Feature statistics — are features degenerate?
  2. Feature-label correlation — do features carry signal?
  3. Logistic Regression baseline — is signal extractable?
  4. Model WITHOUT self-attention — is attention the problem?
  5. Model with class-weighted loss
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_fn
from torch.utils.data import DataLoader, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data_loader import fetch_tw_stocks, TW50_TICKERS
from dataset import ChartGCNDataset, split_by_date

WINDOW, M_PIPS, N, G = 140, 60, 15, 5
SEED = 42

# ── 1. Load data (fewer stocks + larger stride for speed) ──
print("=" * 60)
print("  Chart GCN Debug — 5 stocks, stride=10")
print("=" * 60)

tickers = TW50_TICKERS[:5]
data = fetch_tw_stocks(tickers=tickers, start="2018-01-01", end="2024-12-31")
train_data, test_data = split_by_date(data, "2023-12-31")

train_full = ChartGCNDataset(
    train_data, window=WINDOW, m_pips=M_PIPS,
    N=N, g=G, stride=10, n_workers=1, verbose=True,
)

torch.manual_seed(SEED)
n_total = len(train_full)
n_train = int(n_total * 0.8)
n_val = n_total - n_train
train_ds, val_ds = random_split(
    train_full, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED),
)

test_ds = ChartGCNDataset(
    test_data, window=WINDOW, m_pips=M_PIPS,
    N=N, g=G, stride=5, n_workers=1, verbose=True,
)

X_all = train_full.X
y_all = train_full.y

# ── 2. Feature Statistics ──
print("\n" + "=" * 60)
print("  [Test 1] Feature Statistics")
print("=" * 60)
print(f"X shape: {X_all.shape}")
print(f"X range: [{X_all.min():.4f}, {X_all.max():.4f}]")
print(f"X mean:  {X_all.mean():.4f}, std: {X_all.std():.4f}")
print(f"Fraction zero: {(X_all == 0).mean():.4f}")
print(f"Fraction NaN:  {np.isnan(X_all).mean():.4f}")
print(f"y: up={y_all.mean():.4f}, down={1-y_all.mean():.4f}")

feat_names = ["SMA", "WMA", "Mom", "MACD", "WillR", "CCI", "K%", "D%", "RSI"]
X_flat = X_all.reshape(-1, 9)
print("\nPer-feature stats (across all nodes in all samples):")
for i in range(9):
    f = X_flat[:, i]
    print(f"  {feat_names[i]:>5s}: mean={f.mean():+.4f} std={f.std():.4f} "
          f"[{f.min():.2f}, {f.max():.2f}] zero%={100*(f==0).mean():.1f}%")

# Check within-sample variance
sample_vars = []
for s in range(min(200, len(X_all))):
    x = X_all[s]  # (N, g, F)
    sample_vars.append(x.var())
print(f"\nWithin-sample variance: mean={np.mean(sample_vars):.6f}, "
      f"std={np.std(sample_vars):.6f}")

# ── 3. Feature-Label Correlation ──
print("\n" + "=" * 60)
print("  [Test 2] Feature-Label Correlation")
print("=" * 60)
X_sample_flat = X_all.reshape(len(y_all), -1)

# Mean feature diff between up/down
up_mask = y_all == 1
down_mask = y_all == 0
mean_up = X_sample_flat[up_mask].mean(axis=0)
mean_down = X_sample_flat[down_mask].mean(axis=0)
diff = mean_up - mean_down
abs_diff = np.abs(diff)
top_idx = np.argsort(abs_diff)[-10:]
print(f"Top 10 feature dimensions with largest mean diff (up vs down):")
for idx in top_idx[::-1]:
    subgraph = idx // (G * 9)
    node = (idx % (G * 9)) // 9
    feat = idx % 9
    print(f"  dim={idx} (sub={subgraph},node={node},{feat_names[feat]}): "
          f"diff={diff[idx]:+.6f}")
print(f"Max abs diff: {abs_diff.max():.6f}")

# ── 4. Logistic Regression Baseline ──
print("\n" + "=" * 60)
print("  [Test 3] Logistic Regression Baseline")
print("=" * 60)
X_train_lr = X_all[list(train_ds.indices)].reshape(len(train_ds), -1)
y_train_lr = y_all[list(train_ds.indices)]
X_val_lr = X_all[list(val_ds.indices)].reshape(len(val_ds), -1)
y_val_lr = y_all[list(val_ds.indices)]
X_test_lr = test_ds.X.reshape(len(test_ds), -1)
y_test_lr = test_ds.y

lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
lr.fit(X_train_lr, y_train_lr)
val_pred = lr.predict(X_val_lr)
test_pred = lr.predict(X_test_lr)
print(f"LR (balanced) Val  acc={accuracy_score(y_val_lr, val_pred):.4f} "
      f"f1_1={f1_score(y_val_lr, val_pred, pos_label=1, zero_division=0):.4f}")
print(f"LR (balanced) Test acc={accuracy_score(y_test_lr, test_pred):.4f} "
      f"f1_1={f1_score(y_test_lr, test_pred, pos_label=1, zero_division=0):.4f}")
print(f"LR val prediction dist: up={sum(val_pred)}, down={len(val_pred)-sum(val_pred)}")

# Also test unbalanced
lr2 = LogisticRegression(max_iter=1000, random_state=SEED)
lr2.fit(X_train_lr, y_train_lr)
val_pred2 = lr2.predict(X_val_lr)
print(f"LR (default)  Val  acc={accuracy_score(y_val_lr, val_pred2):.4f} "
      f"f1_1={f1_score(y_val_lr, val_pred2, pos_label=1, zero_division=0):.4f}")
print(f"LR val prediction dist: up={sum(val_pred2)}, down={len(val_pred2)-sum(val_pred2)}")


# ── 5. Model WITHOUT Self-Attention ──
print("\n" + "=" * 60)
print("  [Test 4] CNN Model WITHOUT Self-Attention")
print("=" * 60)

class ChartCNN_NoAttn(nn.Module):
    def __init__(self, N=15, g=5, F_dim=9, k1=32, k2=32, k3=16,
                 F1=84, F2=32, n_classes=2, dropout=0.3):
        super().__init__()
        self.N, self.g, self.F_dim, self.k1 = N, g, F_dim, k1
        self.conv1 = nn.Conv2d(1, k1, kernel_size=(g, F_dim))
        self.conv2 = nn.Conv2d(k1, k2, kernel_size=(5, 1))
        self.conv3 = nn.Conv2d(k2, k3, kernel_size=(1, 1))
        self.fc1 = nn.Linear(k3 * (N - 4), F1)
        self.fc2 = nn.Linear(F1, F2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(F2, n_classes)
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
        for m in [self.fc1, self.fc2, self.classifier]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, X):
        B = X.shape[0]
        H = X.reshape(B * self.N, 1, self.g, self.F_dim)
        H = F_fn.relu(self.conv1(H))
        H = H.reshape(B, self.N, self.k1)
        H = H.permute(0, 2, 1).unsqueeze(-1)
        H = F_fn.relu(self.conv2(H))
        H = F_fn.relu(self.conv3(H))
        H = H.reshape(B, -1)
        H = F_fn.relu(self.fc1(H))
        H = self.dropout(H)
        H = F_fn.relu(self.fc2(H))
        return self.classifier(H)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

def train_and_eval(model, name, epochs=30, use_class_weight=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    if use_class_weight:
        n_pos = (y_all == 1).sum()
        n_neg = (y_all == 0).sum()
        w = torch.tensor([n_pos / len(y_all), n_neg / len(y_all)], dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        tl, nc, nt = 0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * xb.size(0)
            nc += (logits.argmax(1) == yb).sum().item()
            nt += xb.size(0)

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            vp, vt = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    vp.extend(model(xb).argmax(1).numpy())
                    vt.extend(yb.numpy())
            va = accuracy_score(vt, vp)
            vf = f1_score(vt, vp, pos_label=1, zero_division=0)
            n_up = sum(vp)
            print(f"  Ep {epoch:2d} | loss={tl/nt:.4f} acc={nc/nt:.4f} | "
                  f"val_acc={va:.4f} f1_1={vf:.4f} pred_up={n_up}")
            if va > best_val_acc:
                best_val_acc = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            tp.extend(model(xb).argmax(1).numpy())
            tt.extend(yb.numpy())
    ta = accuracy_score(tt, tp)
    tf1 = f1_score(tt, tp, pos_label=1, zero_division=0)
    tf0 = f1_score(tt, tp, pos_label=0, zero_division=0)
    n_up = sum(tp)
    print(f"  {name} BEST: test_acc={ta:.4f} f1_1={tf1:.4f} f1_0={tf0:.4f} "
          f"pred_up={n_up}/{len(tp)}")

torch.manual_seed(SEED)
model_no_attn = ChartCNN_NoAttn(N=N, g=G, F_dim=9, dropout=0.3)
print("Training WITHOUT attention, regular batches:")
train_and_eval(model_no_attn, "NoAttn")

# ── 6. No Attention + Class Weight ──
print("\n" + "=" * 60)
print("  [Test 5] CNN + Class Weight")
print("=" * 60)
torch.manual_seed(SEED)
model_cw = ChartCNN_NoAttn(N=N, g=G, F_dim=9, dropout=0.3)
print("Training WITHOUT attention + class weight:")
train_and_eval(model_cw, "NoAttn+CW", use_class_weight=True)

# ── 7. Try different LR ──
print("\n" + "=" * 60)
print("  [Test 6] CNN, lr=5e-4")
print("=" * 60)
torch.manual_seed(SEED)
model_lr = ChartCNN_NoAttn(N=N, g=G, F_dim=9, dropout=0.1)

optimizer = torch.optim.Adam(model_lr.parameters(), lr=5e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
best_val_acc = 0
for epoch in range(1, 51):
    model_lr.train()
    tl, nc, nt = 0, 0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model_lr(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        tl += loss.item() * xb.size(0)
        nc += (logits.argmax(1) == yb).sum().item()
        nt += xb.size(0)
    if epoch % 10 == 0 or epoch == 1:
        model_lr.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                vp.extend(model_lr(xb).argmax(1).numpy())
                vt.extend(yb.numpy())
        va = accuracy_score(vt, vp)
        vf = f1_score(vt, vp, pos_label=1, zero_division=0)
        print(f"  Ep {epoch:2d} | loss={tl/nt:.4f} acc={nc/nt:.4f} | "
              f"val_acc={va:.4f} f1_1={vf:.4f}")

print("\n" + "=" * 60)
print("  DEBUG COMPLETE")
print("=" * 60)
