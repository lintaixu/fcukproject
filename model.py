"""
Chart GCN model — 嚴格對齊論文 (Li et al., KBS 2022).

論文架構 (Section 3.2.3, Fig. 4):
  Input:  X (B, N, g, F)
  Conv1:  kernel = (g, F)  → (B, k1, N, 1)   每個子圖壓縮成 k1 維特徵
  Conv2:  kernel = (5, 1)  → (B, k2, N-4, 1) 跨子圖組合高階圖表型態
  Conv3:  kernel = (1, 1)  → (B, k3, N-4, 1) channel mixing
  FC1 + FC2 (LeNet 風格)   → (B, F2)
  Self-Attention (Eq.7-9)  → (B, kv)
  Classifier               → (B, 2)

注意: 論文的 "spatial GCN" 來自 [22] (Niepert et al., ICML 2016),
      是透過子圖選取+正規化+CNN 的 pipeline, 不含顯式 GCN message-passing 層.
      原實作多加了兩層 GraphConvLayer, 已依論文移除.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChartGCN(nn.Module):
    def __init__(
        self,
        N: int = 15,
        g: int = 5,
        F_dim: int = 9,
        k1: int = 32,
        k2: int = 32,
        k3: int = 16,
        F1: int = 84,
        F2: int = 32,
        ka: int = 16,
        kv: int = 16,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.N = N
        self.g = g
        self.F_dim = F_dim
        self.k1 = k1

        # Conv1: kernel (g, F) — 論文: "a convolution kernel of G × F"
        self.conv1 = nn.Conv2d(1, k1, kernel_size=(g, F_dim))
        self.bn1 = nn.BatchNorm2d(k1)

        # Conv2: kernel (5, 1) — 論文: "a kernel of 5 × 1"
        self.conv2 = nn.Conv2d(k1, k2, kernel_size=(5, 1))
        self.bn2 = nn.BatchNorm2d(k2)

        # Conv3: kernel (1, 1)
        self.conv3 = nn.Conv2d(k2, k3, kernel_size=(1, 1))
        self.bn3 = nn.BatchNorm2d(k3)

        # FC (LeNet 風格, 論文 [42])
        flat_dim = k3 * (N - 4)
        self.fc1 = nn.Linear(flat_dim, F1)
        self.fc2 = nn.Linear(F1, F2)
        self.dropout = nn.Dropout(dropout)

        # Self-Attention (Eq.7-9) — 跨 batch 樣本 (同日不同股票) 互相 attend
        self.kv = kv
        self.Wq = nn.Linear(F2, ka)
        self.Wk = nn.Linear(F2, ka)
        self.Wv = nn.Linear(F2, kv)
        self.scale = ka ** 0.5
        self.res_proj = nn.Linear(F2, kv)

        # Classifier — 輸入為 kv 維 (attention 輸出 + residual)
        self.classifier = nn.Linear(kv, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
        for m in [self.fc1, self.fc2, self.classifier, self.res_proj]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
        for m in [self.Wq, self.Wk, self.Wv]:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, X):
        """
        Args:
            X: (B, N, g, F)
        Returns:
            logits: (B, n_classes)
        """
        B = X.shape[0]

        # --- Conv1: 逐子圖卷積 ---
        # (B, N, g, F) → (B*N, 1, g, F)
        H = X.reshape(B * self.N, 1, self.g, self.F_dim)
        H = F.relu(self.bn1(self.conv1(H)))  # (B*N, k1, 1, 1)

        # → (B, N, k1) → (B, k1, N, 1)
        H = H.reshape(B, self.N, self.k1)
        H = H.permute(0, 2, 1).unsqueeze(-1)

        # --- Conv2 + Conv3 ---
        H = F.relu(self.bn2(self.conv2(H)))  # (B, k2, N-4, 1)
        H = F.relu(self.bn3(self.conv3(H)))  # (B, k3, N-4, 1)

        # --- FC ---
        H = H.reshape(B, -1)
        H = F.relu(self.fc1(H))
        H = self.dropout(H)
        l = F.relu(self.fc2(H))         # (B, F2)

        # --- Self-Attention (Eq.7-9) + Residual ---
        Q = self.Wq(l)                  # (B, ka)
        K = self.Wk(l)
        V = self.Wv(l)                  # (B, kv)
        scores = (Q @ K.T) / self.scale
        attn = F.softmax(scores, dim=-1)
        Va = attn @ V + self.res_proj(l)  # attention + residual

        logits = self.classifier(Va)
        return logits


if __name__ == "__main__":
    model = ChartGCN(N=15, g=5, F_dim=9)
    x = torch.randn(8, 15, 5, 9)
    out = model(x)
    print(f"Output shape: {out.shape}")   # (8, 2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數量: {n_params:,}")
