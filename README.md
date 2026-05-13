# Chart GCN — 台股實作

依據 Li, Wu, Jiang & Xu (2022) 發表於 *Knowledge-Based Systems* 的論文
**"Chart GCN: Learning chart information with a graph convolutional network for stock movement prediction"**
改寫成台股版本的 PyTorch 實作。

---

## 目錄

- [研究背景](#研究背景)
- [整體架構](#整體架構)
- [檔案結構](#檔案結構)
- [安裝](#安裝)
- [快速開始](#快速開始)
- [演算法細節](#演算法細節)
- [超參數說明](#超參數說明)
- [輸出範例](#輸出範例)
- [與論文的差異](#與論文的差異)
- [常見問題](#常見問題)
- [研究延伸方向](#研究延伸方向)
- [參考文獻](#參考文獻)

---

## 研究背景

傳統技術分析（TA）有兩大武器：**技術指標**（如 MACD、RSI）與**技術圖形**（如頭肩頂、旗形）。
深度學習方法多年來都偏好用技術指標，因為圖形難以量化——同一個「頭肩頂」可能花 10 天形成，也可能花 30 天，且形狀千變萬化。

Chart GCN 的核心創新在於**繞過量化問題**：
1. 用 **PIP** 抽出代表轉折的關鍵點，過濾雜訊；
2. 用 **Visibility Graph** 把這些點轉成圖結構，幾何關係保留下來；
3. 用 **Spatial GCN** 學習圖上的局部到全域型態，類似 CNN 在影像上的階層特徵抽取。

論文在 SZ-50 / CSI-300 上達到 **69.26% / 68.62% accuracy**，超越 LSTM、CNN、TCN、DARNN、CA-FSCN 等基線。

---

## 整體架構

```
┌─────────────────────────────────────────────────────────────┐
│  台股收盤價序列 (rolling window, T=100)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  PIP 演算法           │ 用 Euclidean 距離迭代選關鍵點
            │  (m=5 個 key points)  │ 同時得到每點的 importance score
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Visibility Graph     │ 兩點若無遮擋即連邊 (Eq. 3)
            │  G(V=5, E=~8)         │ 對時間長度具有 invariance
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  選 N 個核心節點       │ 依 (importance, degree, neighbor) 排序
            │  BFS 展開到 g 個節點   │ 補 dummy / 過濾多餘
            │  → N 個正規化子圖      │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  附加 9 個技術指標     │ MACD, RSI, KD, MA, MOM, CCI, ...
            │  → 3D 特徵 (N, g, F)   │
            └──────────┬───────────┘
                       │
                       ▼
   ┌───────────────────────────────────────────┐
   │  Chart GCN                                 │
   │  ┌─────────────────────────────────────┐  │
   │  │ Conv1: kernel (g, F)  → (N, k1)     │  │ 子圖內局部特徵
   │  │ Conv2: kernel (5, 1)  → (N-4, k2)   │  │ 跨子圖組合 (低階圖形)
   │  │ Conv3: kernel (1, 1)  → (N-4, k3)   │  │ 高階圖形特徵
   │  │ Flatten → FC1(84) → FC2(32)         │  │
   │  │ Self-Attention (Q, K, V)            │  │ 跨樣本關聯
   │  │ Classifier → 漲 / 跌 logits          │  │
   │  └─────────────────────────────────────┘  │
   └───────────────────────────────────────────┘
                       │
                       ▼
            漲 / 跌 預測 + Cross-entropy loss
```

---

## 檔案結構

```
chart_gcn_tw/
├── data_loader.py      # 台股資料下載 (yfinance, auto_adjust, parquet cache)
├── pip_algorithm.py    # PIP 關鍵點萃取 + importance score
├── vg_graph.py         # Visibility Graph 建圖
├── subgraph.py         # 子圖選取、正規化、特徵綁定
├── indicators.py       # 9 個技術指標 (Table 1)
├── model.py            # Chart GCN PyTorch 模型
├── dataset.py          # PyTorch Dataset (預先建好所有 sample)
├── train.py            # 訓練 + 評估迴圈
├── main.py             # End-to-end 入口
├── requirements.txt    # Python 依賴
└── README.md           # 本檔案
```

每個模組底下都有 `if __name__ == "__main__":` 的 unit test，可單獨執行驗證：

```bash
python pip_algorithm.py    # 測 PIP
python vg_graph.py         # 測 VG
python subgraph.py         # 測子圖
python indicators.py       # 測指標
python model.py            # 測模型 forward
python dataset.py          # 測 Dataset 建構
```

---

## 安裝

需要 Python ≥ 3.9 (建議 3.10+)。

```bash
git clone <your_repo> chart_gcn_tw
cd chart_gcn_tw
pip install -r requirements.txt
```

主要依賴：
- `numpy`, `pandas`, `scikit-learn` — 數值與評估
- `networkx` — 圖結構
- `torch` ≥ 2.0 — 深度學習框架
- `yfinance`, `pyarrow` — 抓台股資料 + parquet cache

GPU 加速：若有 CUDA，PyTorch 會自動使用。在 CPU 上跑 TW50 全部、30 epochs 大約 30–60 分鐘。

---

## 快速開始

### 1. 用 yfinance 抓台股 TW50 (預設)

```bash
python main.py --train-end 2023-12-31
```

這會：
1. 從 yfinance 下載 TW50 主要成分股 (2018–2024)，股價自動還原除權息
2. 切分 2018–2023 為訓練集、2024 為測試集
3. 用 stride=1 建構 rolling window，套上 PIP（m=5）+VG+子圖
4. 訓練 30 epochs，回報 test accuracy / precision / F1

### 2. 指定特定股票

```bash
python main.py --tickers 2330.TW 2317.TW 2454.TW --epochs 50
```

### 3. 自訂訓練/測試切點

```bash
python main.py --start 2015-01-01 --end 2024-12-31 \
               --train-end 2022-12-31 --epochs 50 --batch-size 128
```

---

## 演算法細節

### Step 1: PIP (Perceptually Important Points)

**目的**：從原始 T 天的收盤價序列中抽出 m 個最具代表性的關鍵點。

**演算法**：
1. 起點與終點直接保留
2. 找出「到當前頭尾連線距離最大的點」加入 PIP 集合
3. 重複直到收滿 m 個點

距離公式（論文 Eq. 1）：

$$d_E(x_k, x_1, x_n) = \sqrt{(t_k-t_1)^2 + (c_k-c_1)^2} + \sqrt{(t_n-t_k)^2 + (c_n-c_k)^2}$$

每個點被選中時會記錄當下的距離作為 **importance score**，後續排序用。

### Step 2: Visibility Graph

**目的**：把 PIP 點序列轉成圖。

**規則**（論文 Eq. 3）：兩點 $(t_a, c_a)$ 與 $(t_b, c_b)$ 之間若**沒有任何中間點 $c_c$ 高過連線在該時刻的高度**，就連一條邊：

$$c_c < c_a + (c_b - c_a) \cdot \frac{t_c - t_a}{t_b - t_a}$$

**為何選 VG**：對時間縮放、振幅縮放、平移等變形具有不變性 (invariance)，能穩定識別不同時間長度的同類型圖形。

### Step 3: 子圖生成與正規化

**選 N 個核心節點**：依 `(importance score, degree, neighbor mean score)` 三層排序，取前 N 個。

**BFS 展開**：以每個核心節點為起點，廣度優先展開到 g 個節點為止。

**正規化**：
- 子圖內節點按上述三層 key 排序 → 鄰接矩陣有一致的排列順序
- 節點數 < g 時補 dummy node
- 節點數 > g 時截斷

### Step 4: 附加技術指標

每個節點對應到原序列上的某一天 → 取出該日的 9 個技術指標：

| # | 指標 | 公式概要 |
|---|------|---------|
| 1 | Simple MA | n 日簡單移動平均 |
| 2 | Weighted MA | 線性加權平均 |
| 3 | Momentum | $c_t - c_{t-n+1}$ |
| 4 | MACD | EMA(12) − EMA(26)，再做 EMA(9) |
| 5 | Williams %R | $(H_n - c) / (H_n - L_n) \times 100$ |
| 6 | CCI | 典型價偏離度 |
| 7 | Stochastic K% | $(c - L_n) / (H_n - L_n) \times 100$ |
| 8 | Stochastic D% | K% 的 3 日 MA |
| 9 | RSI | 相對強弱指標 |

最終得到 3D 特徵 $X \in \mathbb{R}^{N \times g \times F}$。

### Step 5: Chart GCN 預測架構

```
Input: (B, N, g, F) = (batch, 15, 5, 9)
  │
  ├─ Conv1 (1→k1, kernel=(1, g·F))     # 每個子圖卷成 1 vector
  │   → (B, k1=32, N, 1)
  │
  ├─ Conv2 (k1→k2, kernel=(5, 1))      # 跨 5 個相鄰子圖卷積
  │   → (B, k2=32, N-4, 1)
  │
  ├─ Conv3 (k2→k3, kernel=(1, 1))
  │   → (B, k3=16, N-4, 1)
  │
  ├─ Flatten → FC1(84) → FC2(32)        # LeNet 風格
  │
  ├─ Self-Attention (Q, K, V)           # 跨 batch 樣本關聯
  │   Q, K ∈ R^(B,16),  V ∈ R^(B,16)
  │   α = softmax(QK^T / √d)
  │   out = α V
  │
  └─ Classifier → (B, 2)  漲/跌 logits
```

**訓練目標**：Cross-entropy loss

$$\mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N} y_n \log(\hat{y}_n)$$

---

## 超參數說明

| 參數 | 預設 | 論文 Grid Search | 說明 |
|------|------|-----------------|------|
| `--window` | 100 | {100, 120, 130, 140} | rolling window 天數 |
| `--m-pips` | 5 | {5, 10, 20, 40} | PIP 關鍵點數 |
| `--N` | 15 | {10, 15, 20, 25} | 核心節點數 (子圖數) |
| `--g` | 5 | {3, 4, 5, 6} | 每個子圖的節點數 |
| `--stride` | 1 | — | rolling 步長 (越大越快但樣本越少) |
| `--epochs` | 30 | — | 訓練回合數 |
| `--batch-size` | 64 | 128 | mini-batch 大小 |
| `--lr` | 1e-3 | — | Adam 學習率 |
| `--seed` | 42 | — | random seed |

模型內部固定參數（`model.py` 可改）：
- `k1=32, k2=32, k3=16`：卷積層通道數
- `F1=84, F2=32`：全連接層 (LeNet 標準配置)
- `ka=16, kv=16`：Attention 內部維度
- `dropout=0.3`：FC1 後的 dropout

---

## 輸出範例

```
[STEP 1] 下載台股資料: 15 檔
  下載 2330.TW ...
  下載 2317.TW ...
  ...
  成功取得 15 檔資料

[STEP 2] 時間切分 (train_end=2023-12-31)
  2330.TW: train=1565, test=246
  2317.TW: train=1565, test=246
  ...

[STEP 3] 建構訓練集 (window=100, stride=1)
  2330.TW: 1465 samples
  2317.TW: 1465 samples
  ...
Total samples: 21975 (漲=11288, 跌=10687, 漲比例=51.37%)

[STEP 5] 訓練 Chart GCN (epochs=30)
[INFO] device = cuda
Epoch 01 | train_loss=0.6890 train_acc=0.5421 | val_acc=0.5687 val_f1_1=0.5912
Epoch 05 | train_loss=0.6512 train_acc=0.6104 | val_acc=0.6234 val_f1_1=0.6378
Epoch 15 | train_loss=0.5887 train_acc=0.6892 | val_acc=0.6745 val_f1_1=0.6823
Epoch 30 | train_loss=0.5210 train_acc=0.7421 | val_acc=0.6921 val_f1_1=0.7045

=== Test Metrics ===
  acc:    0.6834
  pre_1:  0.6712
  pre_0:  0.6956
  rec_1:  0.7023
  rec_0:  0.6645
  f1_1:   0.6864
  f1_0:   0.6796
```

實際數字會因股票選擇、時間範圍、超參數而異。台股結構與 A 股不同，準確率不一定能達到論文水準。

---

## 與論文的差異

1. **資料源**
   - 論文：SZ-50 / CSI-300（中國 A 股）
   - 本實作：TW50 主要成分股（透過 yfinance，需 `.TW` 後綴）；使用 `auto_adjust=True` 還原除息除權，並以 parquet 快取避免重複下載

2. **MACD 計算**
   - 論文 Table 1：遞迴形式 `MACD_t = MACD_{t-1} + 2/(n+1)·(DIFF_t − MACD_{t-1})`
   - 本實作：標準 EMA(12)−EMA(26)，再對 DIFF 做 EMA(9) 平滑

3. **Self-Attention 範圍**
   - 論文：跨 S 個樣本（同日不同股票）做 attention
   - 本實作：跨 batch 內樣本，等效需用 BatchSampler 把同日股票打包，目前簡化為隨機 batch

4. **未實作的 baseline**
   - LSTM / GRU / TCN / DARNN / CA-FSCN 比較模型本實作未提供
   - 僅有 Chart GCN 主模型

5. **未實作的功能**
   - 6.3 節的 trading simulation (long-only + MACD 退場)
   - 6.1 節的 ablation 變體 (Chart GCN-1 ~ Chart GCN-4)

---

## 常見問題

**Q: yfinance 下載很慢 / 失敗**
A:
- 第一次會比較慢，之後會 cache 在 `./cache/` 不會重抓
- 若被 rate limit，加 `time.sleep(1)` 在 `data_loader.py` 的下載迴圈
- 也可以手動用其他 API（券商、TEJ、Tushare 等）然後改 `data_loader.py` 的回傳格式

**Q: 訓練樣本太多 OOM**
A: 加 `--stride 5` 或 `--stride 10`，rolling window 步長加大可大幅減少樣本數。

**Q: 預測準確率比論文低很多**
A: 可能原因：
- 台股市場效率與 A 股不同
- 樣本數少（TW50 只有 15 檔，論文 SZ-50/CSI-300 共數百檔）
- 超參數沒調過，建議跑 grid search
- 訓練時間不夠，可加 epoch

**Q: 想換成自己的資料怎麼做？**
A: 在 `data_loader.py` 寫一個函數回傳 `{ticker: pd.DataFrame}`，DataFrame 至少要有 `open, high, low, close` 四欄、index 是 datetime。其他都會自動處理。

---

## 研究延伸方向

如果要拿這個基底做研究改進，可考慮以下方向：

### 1. 整合 broker 分點資料
在 `dataset.py` 裡的 `compute_indicators(df)` 之後 concat 你的分點特徵：
```python
broker_feats = compute_broker_features(df)   # (T, F_broker)
all_feats = np.concatenate([feats, broker_feats], axis=1)
```
記得把 `model.py` 的 `F_dim` 改大。

### 2. 替換 Spatial GCN 為更強的 GNN
目前的 conv1 其實是 1D conv 在攤平的子圖上，不算真正的圖卷積。可改用：
- GraphSAGE / GAT / GIN
- 用 `torch_geometric` 直接餵入 adjacency matrix
- 比較不同 GNN 在 chart 任務上的效果

### 3. 加入 backtest 模組
複製論文 Section 6.3 的策略：
```python
# 多單訊號: model 預測漲且無持倉 → 進場
# 出場: model 預測跌 OR MACD < 0
```
回測指標：年化報酬、夏普比、最大回撤。

### 4. 多任務或多時間框架
- 同時預測 1 日、5 日、20 日漲跌
- 用 multi-head 共享 backbone，分頭預測

### 5. Ablation
論文 Section 6.1 的 4 個變體：
- Chart GCN-1: 移除 importance score (改用 betweenness)
- Chart GCN-2: 移除 self-attention
- Chart GCN-3: 同時移除 1 + 2
- Chart GCN-4: 改用原始價格代替技術指標

---

## 參考文獻

- Li, S., Wu, J., Jiang, X., & Xu, K. (2022). Chart GCN: Learning chart information with a graph convolutional network for stock movement prediction. *Knowledge-Based Systems*, 248, 108842.
- Lacasa, L., Luque, B., Ballesteros, F., Luque, J., & Nuño, J. C. (2008). From time series to complex networks: The visibility graph. *PNAS*, 105(13), 4972–4975.
- Niepert, M., Ahmed, M., & Kutzkov, K. (2016). Learning convolutional neural networks for graphs. *ICML*.
- Tsinaslanidis, P. E. (2018). Subsequence dynamic time warping for charting: Bullish and bearish class predictions for NYSE stocks. *Expert Systems with Applications*, 94, 193–204.

---

## License

本實作為學術研究目的撰寫，請依原論文授權使用。
