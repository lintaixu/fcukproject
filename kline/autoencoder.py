"""
Autoencoder 模型與訓練。

將標準化後的特徵壓縮到 LATENT_DIM 維潛在空間，供 One-Class SVM 使用。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kline.config import EPOCHS, LATENT_DIM


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 4):
        super().__init__()
        h = max(16, input_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, 8), nn.ReLU(),
            nn.Linear(8, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, h), nn.ReLU(),
            nn.Linear(h, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


def train_ae(X: np.ndarray, input_dim: int, log_fn: callable) -> Autoencoder:
    """以 MSE + CosineAnnealing 訓練 Autoencoder，回傳訓練完成的模型。"""
    X_t    = torch.FloatTensor(X)
    loader = DataLoader(TensorDataset(X_t),
                        batch_size=min(256, len(X)),
                        shuffle=True,
                        drop_last=len(X) > 256)
    model   = Autoencoder(input_dim, LATENT_DIM)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(1, EPOCHS + 1):
        ep_loss = 0.0
        for (xb,) in loader:
            opt.zero_grad()
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        sched.step()
        if ep % 10 == 0:
            log_fn(f"      Epoch {ep:3d}/{EPOCHS}  loss={ep_loss/len(loader):.5f}")
    return model


if __name__ == '__main__':
    X = np.random.RandomState(0).randn(500, 7).astype(np.float32)
    ae = train_ae(X, input_dim=7, log_fn=print)
    ae.eval()
    with torch.no_grad():
        z = ae.encode(torch.FloatTensor(X)).numpy()
    n_params = sum(p.numel() for p in ae.parameters())
    print(f"輸入 {X.shape} → 潛在 {z.shape} (latent_dim={LATENT_DIM})")
    print(f"模型參數量: {n_params:,}")
