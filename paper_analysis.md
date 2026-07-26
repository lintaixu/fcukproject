# Chart GCN 論文 — 完整技術清單

> 論文：Chart GCN: Learning chart information with a graph convolutional network for stock movement prediction
> 期刊：Knowledge-Based Systems 248 (2022) 108842

---

## 一、資料前處理層

### 1. Perceptually Important Points (PIP) 演算法
**論文 Section 3.1 | `pip_algorithm.py`**

| 項目 | 內容 |
|------|------|
| 目的 | 從長度 n 的價格序列中萃取 m 個最具代表性的關鍵轉折點 |
| 演算法 | 迭代式貪心選取 — 初始只有起點和終點，每輪加入「與兩側已選點距離最大」的候選點 |
| 距離公式 | **Euclidean Distance**：`d(k) = d_left + d_right`，其中 `d_left = √((k-left)² + (P_k - P_left)²)`，`d_right = √((right-k)² + (P_right - P_k)²)` |
| Chart Importance Score | 每個 PIP 加入時的最大距離值 d(k)，代表該點的圖表重要性 |
| 超參數 | m（PIP 數量），論文搜索範圍 {30, 40, 60, 80} |

### 2. 9 個技術指標 (Technical Indicators)
**論文 Section 3.4, Table 1 | `indicators.py`**

| # | 指標名稱 | 公式 |
|---|----------|------|
| 1 | **SMA** (Simple Moving Average) | `SMA_n = (1/n) Σ C_t-i` |
| 2 | **WMA** (Weighted Moving Average) | `WMA_n = Σ(i × C_t-i) / Σi`，權重 1,2,...,n |
| 3 | **Momentum** | `MOM = C_t - C_t-(n-1)` |
| 4 | **MACD** | `MACD = EMA_9(EMA_12 - EMA_26)`，其中 `EMA_k = α×C + (1-α)×EMA_prev`，`α = 2/(k+1)` |
| 5 | **Larry Williams %R** | `%R = (H_n - C) / (H_n - L_n) × 100` |
| 6 | **CCI** (Commodity Channel Index) | `CCI = (TP - SMA_TP) / (0.015 × MD)`，`TP = (H+L+C)/3` |
| 7 | **Stochastic K%** | `K% = (C - L_n) / (H_n - L_n) × 100` |
| 8 | **Stochastic D%** | `D% = SMA_10(K%)` (K% 的 10 日移動平均) |
| 9 | **RSI** (Relative Strength Index) | `RSI = 100 - 100/(1+RS)`，`RS = avg_gain / avg_loss` |

---

## 二、圖結構建構層

### 3. Visibility Graph (VG)
**論文 Section 3.2, Eq.(3) | `vg_graph.py`**

| 項目 | 內容 |
|------|------|
| 目的 | 將 PIP 時間序列轉換為圖結構，捕捉「哪些關鍵點之間可以互相看見」 |
| 可見性條件 | 兩點 (t_a, y_a) 和 (t_b, y_b) 之間，若所有中間點 (t_c, y_c) 滿足：**y_c < y_a + (y_b - y_a) × (t_c - t_a) / (t_b - t_a)**，則建邊 |
| 物理意義 | 相當於從每個 PIP 點向外「看」，若視線不被其他 PIP 擋住則連線 |
| 圖性質 | 無向圖、連通圖，節點屬性含 time 和 price |

### 4. Chart Importance Score
**論文 Section 3.3, Eq.(4)**

| 項目 | 內容 |
|------|------|
| 目的 | 量化每個 PIP 點在圖表中的重要程度，用於排序選取核心節點 |
| 公式 | PIP 加入時的 Euclidean distance 值 |
| 與 Betweenness Centrality 的差異 | 論文提出的 importance score 比 betweenness centrality 更好（消融實驗 Table 4 驗證） |

### 5. Subgraph Generation (子圖生成)
**論文 Section 3.3 | `subgraph.py`**

| 步驟 | 演算法 | 說明 |
|------|--------|------|
| (a) 選取核心節點 | **Top-N Selection** | 按 (importance score → degree → neighbor mean score) 三級排序，取前 N 個 |
| (b) 展開子圖 | **BFS (Breadth-First Search)** | 從每個核心節點出發 BFS 展開到 g 個節點 |
| (c) 子圖正規化 | **Normalize & Padding** | 子圖內節點按重要性排序，不足 g 個補 dummy node (-1) |
| (d) 鄰接矩陣 | **Adjacency Matrix** | 子圖內的邊轉成 g×g 矩陣 A，dummy 行/列為 0 |
| (e) 附加特徵 | **Feature Attachment** | 每個節點映射回原序列 index，取出 9 個技術指標 → (g, F) |
| 輸出 | **3D Feature Tensor** | X: (N, g, F)，A: (N, g, g) |
| 超參數 | N ∈ {10,15,20,25}，g ∈ {3,4,5,6} |

---

## 三、模型架構層

### 6. Spatial Graph Convolution (GCN)
**論文 Section 3.2.3（2026-07-24 依原文 PDF 更正）**

| 項目 | 內容 |
|------|------|
| ⚠ 更正 | 論文**沒有** `H = σ(D⁻¹ Â X W + b)` 這類 GCN 傳播式(全文式 (1)~(11) 無此式)|
| 論文的 "spatial GCN" | = 「正規化子圖(PATCHY-SAN 式排序 [22])+ LeNet 式卷積」整體,即 Eq.(6) `l = f_LeNet(X)` |
| 鄰接矩陣用途 | 僅在子圖正規化階段使用(排序/補 dummy),不進入前向傳播 —— 與 `core/model.py` 現行實作一致 |

### 7. Conv1 — 子圖內卷積
**論文 Section 4.2 | `model.py`**

| 項目 | 內容 |
|------|------|
| Kernel size | (g, F_dim) — 完整覆蓋一個子圖 |
| 輸入 → 輸出 | (B×N, 1, g, F) → (B×N, k1, 1, 1) |
| 意義 | 將每個子圖壓縮成 k1 維的「圖表模式特徵向量」 |
| 激活 | ReLU |
| k1 | ⚠ 論文未給數值(「typical configuration」);paper_exact 實作採 LeNet 對應 k1=6(舊改進版為 32)|

### 8. Conv2 — 跨子圖卷積
**論文 Section 4.2 | `model.py`**

| 項目 | 內容 |
|------|------|
| Kernel size | (5, 1) — 跨 5 個相鄰子圖 |
| 輸入 → 輸出 | (B, k1, N, 1) → (B, k2, N-4, 1) |
| 意義 | 組合多個子圖的模式，捕捉「高階圖表型態」 |
| 激活 | ReLU |
| k2 | ⚠ 論文未給數值;paper_exact 採 LeNet 對應 k2=16(舊改進版為 32)|

### 9. Conv3 — Channel Mixing
**論文 Section 4.2 | `model.py`**

| 項目 | 內容 |
|------|------|
| Kernel size | paper_exact: (N-4, 1) 全高度(仿 LeNet C5);舊改進版: (1, 1) |
| 輸入 → 輸出 | paper_exact: (B, k2, N-4, 1) → (B, k3, 1, 1) |
| 意義 | 跨 channel/全高度混合特徵(論文只說「use k3 kernels」,kernel 形狀未明定)|
| 激活 | ReLU |
| k3 | ⚠ 論文未給數值;paper_exact 採 LeNet 對應 k3=120(舊改進版為 16)|

### 10. Fully Connected Layers
**論文 Section 4.2 | `model.py`**

| 項目 | 內容 |
|------|------|
| FC1 | Linear(k3 × (N-4), F1) + ReLU，F1=84 |
| FC2 | Linear(F1, F2) + ReLU，F2=32 |
| 輸出 | l ∈ ℝ^(B, F2) — 每個樣本的特徵向量 |

### 11. Self-Attention Mechanism
**論文 Section 4.3, Eq.(7)-(9) | `model.py`**

| 公式 | 表達式 | 說明 |
|------|--------|------|
| **Eq.(7)** | `Q = l·W_Q, K = l·W_K, V = l·W_V` | 三個投影,W_Q/W_K ∈ R^(F2×ka), W_V ∈ R^(F2×kv) |
| **Eq.(8)** | `S = Q·Kᵀ` | ⚠ 原文**無 /√k_a 縮放**(2026-07-24 依 PDF 更正)|
| **Eq.(9)** | `α = softmax(S)`,之後 `V_a = α·V` | 注意力權重與加權聚合 |
| 超參數 | k_a, k_v | ⚠ 論文**未給數值**(只說 ka 供 Q/K 共用、kv 可不同);16/16 是我們的假設 |
| 作用域 | 「between **S samples**」 | ⚠ 論文**未定義 S 是誰**;「同一天的 S 檔股票」是我們(合理且無洩漏)的解讀,「連續 128 筆 batch」解讀會造成未來洩漏(見 實驗記錄_論文對齊.md P1) |

### 12. Classifier
**論文 Section 4.3 | `model.py`**

| 項目 | 內容 |
|------|------|
| 公式 | `logits = W_c × V_a + b_c` |
| 輸入 | V_a ∈ ℝ^(B, k_v) |
| 輸出 | logits ∈ ℝ^(B, 2)，二分類（漲/跌） |

---

## 四、訓練與優化

### 13. 損失函數 — Cross-Entropy Loss
**論文 Section 5 | `train.py`**

| 項目 | 內容 |
|------|------|
| 公式 | `L = -Σ y_i × log(p_i)` |
| y_i | one-hot 標籤 (漲=1, 跌=0) |
| p_i | softmax(logits) |

### 14. 優化器 — Adam
**論文提到 weight_decay=0.00005 | `train.py`**

| 超參數 | 值 |
|--------|---|
| Learning rate | 1e-3 |
| Weight decay | 5e-5 |

### 15. 標籤定義
**論文 Section 3.4**

| 項目 | 內容 |
|------|------|
| 公式 | `label = 1 if C_t > C_{t-1} else 0` |
| 意義 | 今日收盤價高於昨日 → 漲 (1)，否則跌 (0) |

---

## 五、評估指標

### 16. Classification Metrics
**論文 Section 5.2, Table 3**

| 指標 | 公式 |
|------|------|
| **Accuracy** | `(TP+TN) / (TP+TN+FP+FN)` |
| **Precision** | `TP / (TP+FP)` |
| **Recall** | `TP / (TP+FN)` |
| **F1-Score** | `2 × Pre × Rec / (Pre + Rec)` |

---

## 六、相似度框架 (Similarity Framework)

### 17. 基於 VG 的圖表相似度
**論文 Section 3.2, 5.1**

| 項目 | 內容 |
|------|------|
| 核心思想 | 兩段價格走勢若「長得像」，它們的 VG 圖結構也會相似 |
| 優勢 | 對 6 種圖表變異具魯棒性（見下） |

### 18. 對比方法 — DTW (Dynamic Time Warping)
**論文 Section 5.1, Fig.5**

| 項目 | 內容 |
|------|------|
| 公式 | `DTW(X,Y) = min Σ d(x_i, y_j)` subject to alignment path constraints |
| 用途 | 作為相似度比較的 baseline |

### 19. 對比方法 — Pearson Correlation
**論文 Section 5.1, Fig.5**

| 項目 | 內容 |
|------|------|
| 公式 | `r = Σ(x_i - x̄)(y_i - ȳ) / √(Σ(x_i - x̄)² × Σ(y_i - ȳ)²)` |
| 用途 | 作為相似度比較的 baseline |

### 20. 6 種圖表變異 (Chart Variations)
**論文 Section 5.1, Fig.5**

| # | 變異類型 | 說明 |
|---|----------|------|
| 1 | 振幅縮放 (Amplitude Scaling) | 價格乘以倍率 |
| 2 | 振幅平移 (Amplitude Shifting) | 價格加常數 |
| 3 | 時間縮放 (Time Scaling) | 拉伸/壓縮時間軸 |
| 4 | 時間平移 (Time Shifting) | 平移時間軸 |
| 5 | 旋轉 (Rotation) | 整體趨勢旋轉 |
| 6 | 高斯雜訊 (Gaussian Noise) | 加入隨機雜訊 |

---

## 七、交易模擬

### 21. Trading Strategy
**論文 Section 6.3 | `backtest.py`**

| 訊號 | 條件 | 動作 |
|------|------|------|
| **Long** | 模型預測「漲」且目前空手 | 全額買入 |
| **Short** | 模型預測「跌」**或** MACD < 0，且目前持倉 | 全數賣出 |
| MACD 風險過濾 | `MACD = EMA_12 - EMA_26`，若 < 0 則強制賣出 | 避免波動風險 |

### 22. Benchmark — Equal-Weight Buy-and-Hold
**論文 Section 6.3**

| 項目 | 內容 |
|------|------|
| 公式 | `NV_t = (1/S) Σ (Price_t / Price_0)` |
| 意義 | 第 0 天等權買入所有 S 檔股票，持有不動 |

### 23. 淨值計算

| 項目 | 內容 |
|------|------|
| 公式 | `NV = cash + shares × price` |
| 起始 | NV_0 = 1.0 |

---

## 八、消融實驗變體

### 24. 消融實驗設計
**論文 Section 6.1, Table 4**

| 變體 | 修改內容 | 目的 |
|------|----------|------|
| Chart GCN-1 | 用 **Betweenness Centrality** 替代 Chart Importance Score | 驗證 importance score 貢獻 |
| Chart GCN-2 | **移除 Self-Attention** | 驗證跨股票 attention 貢獻 |
| Chart GCN-3 | **兩者都移除** | 驗證兩組件交互效果 |
| Chart GCN-4 | 用**原始價格**替代 9 個技術指標 | 驗證技術指標的價值 |

### 25. Betweenness Centrality
**論文 Section 6.1（作為對照）**

| 項目 | 內容 |
|------|------|
| 公式 | `BC(v) = Σ_{s≠v≠t} σ_st(v) / σ_st` |
| σ_st | s 到 t 的最短路徑數 |
| σ_st(v) | 經過 v 的最短路徑數 |

---

## 九、替換實驗 (Kernel Replacement)

### 26. 替換 GCN 為其他模型
**論文 Section 6.2, Table 5**

| 替換模型 | 說明 |
|----------|------|
| **CNN** | 標準卷積神經網路 |
| **LSTM** | 長短期記憶網路 |
| **TCN** (Temporal Convolutional Network) | 時序卷積 |
| **GRU** | 門控循環單元 |

### 27. 61 種布林值圖表模式
**論文 Section 6.2**

| 項目 | 內容 |
|------|------|
| 目的 | 擴展技術指標，加入 61 個布林值圖表模式判定 |
| 用途 | 驗證 VG 相似度框架的通用性 |

---

## 十、Baseline 模型對比

### 28. Baseline 模型列表
**論文 Section 5.2, Table 3**

| 模型 | 類型 | 說明 |
|------|------|------|
| **CNN** | 卷積神經網路 | 標準 CNN 用於時序分類 |
| **LSTM** | 循環神經網路 | 長短期記憶，擅長序列建模 |
| **TCN** | 時序卷積 | 因果膨脹卷積，平行計算 |
| **GRU** | 循環神經網路 | LSTM 的簡化版本 |
| **DARNN** | 注意力 RNN | Dual-stage Attention-based RNN |
| **CA-FSCN** | 卷積 + 注意力 | Chart Analysis with Fully-connected Subgraph CNN |

---

## 十一、超參數搜索

### 29. Grid Search
**論文 Section 5.2**

| 超參數 | 搜索範圍 | 最佳值 |
|--------|---------|--------|
| window (觀察窗口) | {100, 120, 130, 140} | 依資料集而定 |
| m (PIP 數量) | {30, 40, 60, 80} | 依資料集而定 |
| N (核心節點數) | {10, 15, 20, 25} | 依資料集而定 |
| g (子圖大小) | {3, 4, 5, 6} | 依資料集而定 |

---

## 十二、完整 Pipeline 流程圖

```
原始股價序列 (window=100)
    │
    ├─① PIP 演算法 → 提取 m=40 個關鍵點 + importance score
    │
    ├─② Visibility Graph → m 個節點的無向圖
    │
    ├─③ Top-N Selection → 選出 N=15 個核心節點
    │
    ├─④ BFS Subgraph → 每核心展開 g=5 子圖 + 鄰接矩陣
    │
    ├─⑤ 9 個技術指標 → 附加到子圖節點
    │
    ├─⑥ 輸出: X(N,g,F) + A(N,g,g) → 一個訓練樣本
    │
    ├─⑦ GCN (2層) → 圖結構感知的特徵提取
    │
    ├─⑧ Conv1→Conv2→Conv3 → 子圖內/跨子圖卷積
    │
    ├─⑨ FC1→FC2 → 壓縮到 F2 維
    │
    ├─⑩ Self-Attention (同日 S 檔股票) → 跨股票關聯
    │
    └─⑪ Classifier → 二分類 (漲/跌)
```

---

## 十三、論文技術統計

| 類別 | 數量 | 內容 |
|------|------|------|
| 資料前處理演算法 | 3 | PIP、VG、Subgraph Generation |
| 技術指標 | 9 | SMA, WMA, Momentum, MACD, %R, CCI, K%, D%, RSI |
| 深度學習層 | 8 | GCN×2, Conv×3, FC×2, Self-Attention |
| 核心公式 | 9 | Eq.(3)~(9), GCN公式, CrossEntropy |
| 評估指標 | 4 | Accuracy, Precision, Recall, F1 |
| 相似度方法 | 3 | VG-based, DTW, Pearson |
| 圖表變異測試 | 6 | 振幅縮放/平移、時間縮放/平移、旋轉、雜訊 |
| Baseline 模型 | 6 | CNN, LSTM, TCN, GRU, DARNN, CA-FSCN |
| 消融變體 | 4 | GCN-1~4 |
| 超參數 | 4 | window, m, N, g |
