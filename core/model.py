"""
Chart GCN model — 嚴格對齊論文 (Li et al., KBS 2022).

論文架構 (Section 3.2.3, Fig. 4), paper_exact=True (預設):
  Input:  X (B, N, g, F)
  Conv1:  kernel = (g, F),  k1 個  → 每個子圖壓成 k1 維   (論文: "kernel of G × F")
  Conv2:  kernel = (5, 1),  k2 個  → 跨子圖組合高階圖表型態 (論文: "kernel of 5 × 1")
  Conv3:  kernel = (N-4, 1), k3 個 → 全高度卷積 (仿 LeNet C5 層)
  FC1(F1=84) + FC2(F2=32)          → l (B, F2)
  Self-Attention (Eq.7-9): Va = softmax(Q·K^T)·V   ← 無縮放、無殘差 (論文原式)
  Classifier                        → (B, 2)

  通道數依論文 "following LeNet [42]" 對應 LeNet-5:
    C1=6, C3=16, C5=120, F6=84  →  k1=6, k2=16, k3=120, F1=84
  (論文只明定 F1=84, F2=32; k1/k2/k3/ka/kv 未明定, 以上為 LeNet 對應之假設)

  論文未提及 BatchNorm / Dropout / 殘差 / attention 縮放 → paper_exact 模式一律不用。
  ReLU 用於所有卷積與全連接層 (論文 Section 3.2.3)。

paper_exact=False 保留舊版改進型 (BatchNorm + Dropout + Residual Attention +
scaled dot-product), 供 experiments_fusion/ 等既有實驗沿用。
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
        k1: int = None,
        k2: int = None,
        k3: int = None,
        F1: int = 84,
        F2: int = 32,
        ka: int = None,
        kv: int = None,
        n_classes: int = 2,
        dropout: float = 0.3,
        paper_exact: bool = True,
        use_attention: bool = True,   # False = 論文消融變體 Chart GCN-2
    ):
        super().__init__()
        self.N = N
        self.g = g
        self.F_dim = F_dim
        self.paper_exact = paper_exact
        self.use_attention = use_attention

        if paper_exact:
            # LeNet-5 對應 (論文 "following LeNet")
            k1 = 6 if k1 is None else k1
            k2 = 16 if k2 is None else k2
            k3 = 120 if k3 is None else k3
            ka = 32 if ka is None else ka
            kv = 32 if kv is None else kv
        else:
            k1 = 32 if k1 is None else k1
            k2 = 32 if k2 is None else k2
            k3 = 16 if k3 is None else k3
            ka = 16 if ka is None else ka
            kv = 16 if kv is None else kv

        self.k1 = k1

        # Conv1: kernel (g, F) — 論文: "a convolution kernel of G × F"
        self.conv1 = nn.Conv2d(1, k1, kernel_size=(g, F_dim))
        # Conv2: kernel (5, 1) — 論文: "a kernel of 5 × 1"
        self.conv2 = nn.Conv2d(k1, k2, kernel_size=(5, 1))

        if paper_exact:
            # Conv3: 全高度卷積 (N-4, 1) — 仿 LeNet C5 (输出 1×1×k3)
            self.conv3 = nn.Conv2d(k2, k3, kernel_size=(N - 4, 1))
            flat_dim = k3
        else:
            self.bn1 = nn.BatchNorm2d(k1)
            self.bn2 = nn.BatchNorm2d(k2)
            self.conv3 = nn.Conv2d(k2, k3, kernel_size=(1, 1))
            self.bn3 = nn.BatchNorm2d(k3)
            flat_dim = k3 * (N - 4)

        # FC (LeNet 風格, 論文 [42]): F1=84, F2=32
        self.fc1 = nn.Linear(flat_dim, F1)
        self.fc2 = nn.Linear(F1, F2)
        self.dropout = nn.Dropout(dropout)

        # Self-Attention (Eq.7-9)
        self.kv = kv
        self.Wq = nn.Linear(F2, ka)
        self.Wk = nn.Linear(F2, ka)
        self.Wv = nn.Linear(F2, kv)
        if not paper_exact:
            self.scale = ka ** 0.5
            self.res_proj = nn.Linear(F2, kv)

        # Classifier (無 attention 時直接吃 l ∈ R^F2)
        self.classifier = nn.Linear(kv if use_attention else F2, n_classes)

        self._init_weights()

    def _init_weights(self):
        mods = [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2,
                self.classifier]
        if not self.paper_exact:
            mods.append(self.res_proj)
        for m in mods:
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
        H = X.reshape(B * self.N, 1, self.g, self.F_dim)
        if self.paper_exact:
            H = F.relu(self.conv1(H))            # (B*N, k1, 1, 1)
        else:
            H = F.relu(self.bn1(self.conv1(H)))

        # → (B, N, k1) → (B, k1, N, 1)
        H = H.reshape(B, self.N, self.k1)
        H = H.permute(0, 2, 1).unsqueeze(-1)

        # --- Conv2 + Conv3 ---
        if self.paper_exact:
            H = F.relu(self.conv2(H))            # (B, k2, N-4, 1)
            H = F.relu(self.conv3(H))            # (B, k3, 1, 1)
        else:
            H = F.relu(self.bn2(self.conv2(H)))
            H = F.relu(self.bn3(self.conv3(H)))

        # --- FC ---
        H = H.reshape(B, -1)
        H = F.relu(self.fc1(H))
        if not self.paper_exact:
            H = self.dropout(H)
        l = F.relu(self.fc2(H))                  # (B, F2)

        # --- Self-Attention (Eq.7-9); use_attention=False = Chart GCN-2 ---
        if not self.use_attention:
            return self.classifier(l)

        Q = self.Wq(l)
        K = self.Wk(l)
        V = self.Wv(l)                           # (B, kv)
        if self.paper_exact:
            attn = F.softmax(Q @ K.T, dim=-1)    # Eq.(8)(9): 無縮放
            Va = attn @ V                        # Va = α · V(l), 無殘差
        else:
            attn = F.softmax((Q @ K.T) / self.scale, dim=-1)
            Va = attn @ V + self.res_proj(l)

        logits = self.classifier(Va)
        return logits


if __name__ == "__main__":
    for exact in [True, False]:
        model = ChartGCN(N=15, g=5, F_dim=9, paper_exact=exact)
        x = torch.randn(8, 15, 5, 9)
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"paper_exact={exact}: out={tuple(out.shape)}, params={n_params:,}")
