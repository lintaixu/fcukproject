"""
Chart GCN model.

Pipeline:
  Input:  X of shape (B, N, g, F)   -- B 是 batch size
  Conv1:  kernel = (g, F)            -> (B, k1, N, 1)  抓單一子圖內的局部圖表特徵
  Squeeze + Conv2: kernel = (k1, 5)  -> (B, k2, N-4)   跨子圖組成更高階型態
  Conv3:  kernel = (k2, 1)           -> (B, k3, N-4)
  Flatten + FC1 + FC2                -> (B, F2)
  Self-Attention (within batch)      -> (B, kv)
  Output: (B, 2)  二分類漲跌

注意: 論文的 self-attention 設計是跨 S 個樣本 (不同股票同一天). 這裡實作為
       對 batch 內樣本做 self-attention, 等價於在訓練時把同日不同股票的樣本
       打包成同一 batch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChartGCN(nn.Module):
    def __init__(
        self,
        N: int = 15,        # 子圖數
        g: int = 5,         # 子圖節點數
        F_dim: int = 9,     # 技術指標數
        k1: int = 32,
        k2: int = 32,
        k3: int = 16,
        F1: int = 84,
        F2: int = 32,
        ka: int = 16,
        kv: int = 16,
        n_classes: int = 2,
    ):
        super().__init__()
        self.N = N
        self.g = g
        self.F_dim = F_dim
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        # Conv1: 把每個子圖 (g x F) 卷積成 1 個 channel
        # input  : (B, 1, N, g*F)  ← 我們會把 (g, F) flatten 成 g*F 一維
        # 但更標準做法是用 2D conv:
        # input  : (B, 1, N*g, F) — 不適合
        # 我們改用: (B, 1, N, g*F),  kernel=(1, g*F),  stride=(1, g*F)
        # 等價於對每個子圖內的 g*F 個元素做一次卷積 → 輸出 (B, k1, N, 1)
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=k1,
            kernel_size=(1, g * F_dim),
            stride=(1, g * F_dim),
        )

        # Conv2: 跨子圖卷積  (B, k1, N, 1) → (B, k2, N-4, 1)
        self.conv2 = nn.Conv2d(
            in_channels=k1, out_channels=k2,
            kernel_size=(5, 1),
        )

        # Conv3: (B, k2, N-4, 1) → (B, k3, N-4, 1)
        self.conv3 = nn.Conv2d(
            in_channels=k2, out_channels=k3,
            kernel_size=(1, 1),
        )

        # Flatten + FC
        flat_dim = k3 * (N - 4)
        self.fc1 = nn.Linear(flat_dim, F1)
        self.fc2 = nn.Linear(F1, F2)

        # Self-Attention
        self.Wq = nn.Linear(F2, ka)
        self.Wk = nn.Linear(F2, ka)
        self.Wv = nn.Linear(F2, kv)
        self.scale = ka ** 0.5

        # Final classifier
        self.classifier = nn.Linear(kv, n_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, X):
        """
        Args:
            X: tensor of shape (B, N, g, F)
        Returns:
            logits: (B, n_classes)
        """
        B, N, g, F_dim = X.shape
        assert N == self.N and g == self.g and F_dim == self.F_dim

        # reshape 成 (B, 1, N, g*F)
        x = X.reshape(B, 1, N, g * F_dim)

        # Conv1 → (B, k1, N, 1)
        x = F.relu(self.conv1(x))

        # Conv2 → (B, k2, N-4, 1)
        x = F.relu(self.conv2(x))

        # Conv3 → (B, k3, N-4, 1)
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.reshape(B, -1)
        x = self.dropout(F.relu(self.fc1(x)))
        l = F.relu(self.fc2(x))      # (B, F2)

        # Self-Attention (within batch)
        Q = self.Wq(l)               # (B, ka)
        K = self.Wk(l)
        V = self.Wv(l)               # (B, kv)
        scores = (Q @ K.T) / self.scale
        attn = F.softmax(scores, dim=-1)
        Va = attn @ V                # (B, kv)

        logits = self.classifier(Va)
        return logits


if __name__ == "__main__":
    model = ChartGCN(N=15, g=5, F_dim=9)
    x = torch.randn(8, 15, 5, 9)
    out = model(x)
    print(f"Output shape: {out.shape}")  # (8, 2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數量: {n_params:,}")
