"""
Chart GCN model (對齊論文).

Pipeline:
  Input:  X (B, N, g, F),  A (B, N, g, g)
  GCN:    兩層圖卷積, 利用鄰接矩陣 A 做 spatial graph convolution
  Conv1:  kernel = (g, F)  → (B, k1, N, 1)   每個子圖壓縮成 k1 維特徵
  Conv2:  kernel = (5, 1)  → (B, k2, N-4, 1) 跨子圖組合高階圖表型態
  Conv3:  kernel = (1, 1)  → (B, k3, N-4, 1) channel mixing
  FC1 + FC2                → (B, F2)
  Self-Attention           → (B, kv)    ← 論文原始設計, 無 residual
  Output: (B, 2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    Spatial graph convolution: H = σ(D^{-1} A_hat X W + b)
    使用 row-normalized adjacency, 對應 DGCNN [22] 的做法。
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')

    def forward(self, X, A):
        """
        Args:
            X: (B, N, g, F_in)
            A: (B, N, g, g)
        Returns:
            H: (B, N, g, F_out)
        """
        I = torch.eye(A.size(-1), device=A.device)
        A_hat = A + I
        D = A_hat.sum(dim=-1, keepdim=True).clamp(min=1)
        A_norm = A_hat / D
        H = A_norm @ X
        H = self.linear(H)
        return H


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
    ):
        super().__init__()
        self.N = N
        self.g = g
        self.F_dim = F_dim
        self.k1 = k1

        # GCN layers
        self.gcn1 = GraphConvLayer(F_dim, F_dim)
        self.gcn2 = GraphConvLayer(F_dim, F_dim)

        # Conv1: kernel (g, F)
        self.conv1 = nn.Conv2d(1, k1, kernel_size=(g, F_dim))

        # Conv2: kernel (5, 1) 跨子圖卷積
        self.conv2 = nn.Conv2d(k1, k2, kernel_size=(5, 1))

        # Conv3: kernel (1, 1)
        self.conv3 = nn.Conv2d(k2, k3, kernel_size=(1, 1))

        # FC
        flat_dim = k3 * (N - 4)
        self.fc1 = nn.Linear(flat_dim, F1)
        self.fc2 = nn.Linear(F1, F2)

        # Self-Attention (Eq.7-9) — 論文原始設計, kv 為獨立超參數
        self.kv = kv
        self.Wq = nn.Linear(F2, ka)
        self.Wk = nn.Linear(F2, ka)
        self.Wv = nn.Linear(F2, kv)
        self.scale = ka ** 0.5

        # Classifier — 輸入為 kv 維 (attention 輸出), 非 F2
        self.classifier = nn.Linear(kv, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
        for m in [self.fc1, self.fc2, self.classifier]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
        for m in [self.Wq, self.Wk, self.Wv]:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, X, A):
        """
        Args:
            X: (B, N, g, F)
            A: (B, N, g, g)
        Returns:
            logits: (B, n_classes)
        """
        B = X.shape[0]

        # --- GCN ---
        H = F.relu(self.gcn1(X, A))
        H = F.relu(self.gcn2(H, A))

        # --- Conv1: 逐子圖卷積 ---
        H = H.reshape(B * self.N, 1, self.g, self.F_dim)
        H = F.relu(self.conv1(H))

        H = H.reshape(B, self.N, self.k1)
        H = H.permute(0, 2, 1).unsqueeze(-1)

        # --- Conv2 + Conv3 ---
        H = F.relu(self.conv2(H))
        H = F.relu(self.conv3(H))

        # --- FC ---
        H = H.reshape(B, -1)
        H = F.relu(self.fc1(H))
        l = F.relu(self.fc2(H))           # (B, F2)

        # --- Self-Attention (Eq.7-9) — 論文原始設計, 無 residual ---
        Q = self.Wq(l)                    # (B, ka)
        K = self.Wk(l)
        V = self.Wv(l)                    # (B, kv)
        scores = (Q @ K.T) / self.scale
        attn = F.softmax(scores, dim=-1)
        Va = attn @ V                     # (B, kv)

        logits = self.classifier(Va)
        return logits


if __name__ == "__main__":
    model = ChartGCN(N=15, g=5, F_dim=9)
    x = torch.randn(8, 15, 5, 9)
    a = torch.randint(0, 2, (8, 15, 5, 5)).float()
    out = model(x, a)
    print(f"Output shape: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數量: {n_params:,}")
