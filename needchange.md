# Chart GCN - 與論文差異及待修改項目

## 一、方法論層面差異

### 1. Self-Attention Residual Connection（嚴重度：高）

| | 論文 | 目前程式碼 |
|---|---|---|
| 公式 | `logits = classifier(Va)` | `logits = classifier(l + Va)` |
| 位置 | — | `model.py:145` |

**說明：**
論文 Self-Attention 的輸出 Va 直接送入 classifier 做分類。程式碼額外加了 residual connection（`l + Va`），因為不加的話 attention 會在隨機 batch 中把所有樣本特徵平均化，導致模型完全無法訓練（所有樣本輸出相同 logits）。

**影響：** 改變了 attention 的語意。論文中 attention 完全決定最終特徵；程式碼中 attention 只是「輔助」，主要依賴 FC 層的原始輸出 `l`。

**修改方向：** 若能實作同日 batch（見第 3 點），可嘗試移除 residual，回歸論文原始設計。

---

### 2. Self-Attention kv 維度綁定（嚴重度：中）

| | 論文 | 目前程式碼 |
|---|---|---|
| kv | 獨立超參數，可與 F2 不同 | 強制 `kv = F2 = 32` |
| 位置 | — | `model.py:88` |

**說明：**
論文中 kv 是自由超參數。程式碼因為 residual connection 需要 `l`（F2 維）和 `Va`（kv 維）相加，所以強制 kv = F2。

**修改方向：** 若移除 residual，可恢復 kv 為獨立參數；或改用投影層 `nn.Linear(kv, F2)` 對齊維度。

---

### 3. Self-Attention 作用域 — 未按日期分組 batch（嚴重度：高）

| | 論文 | 目前程式碼 |
|---|---|---|
| 對象 | 同一天的 S 檔股票互相 attend | batch 內隨機樣本（不保證同日） |
| 位置 | — | `model.py:137-143`、`train.py:71` |

**說明：**
論文的 Self-Attention 設計目的是捕捉「同一天不同股票之間的關聯」。例如某天金融股全部出現下跌圖表，attention 可以將這個訊號傳遞給其他股票。

目前程式碼的 DataLoader 使用 `shuffle=True` 隨機打包，一個 batch 內可能包含不同年份、不同股票的樣本，attention 計算的是無意義的跨時間關聯。

**修改方向：**
需要實作 `DateGroupedBatchSampler`，確保每個 batch 只包含同一天的所有股票。具體修改：
- `dataset.py`：記錄每個樣本的日期，建立日期 → index 的映射
- `train.py`：自訂 BatchSampler，按日期分組
- 每個 batch size = 該日有效股票數（約 50），無法自由調整 batch_size

**預估影響：** 論文消融實驗顯示，Self-Attention 貢獻約 5% 準確率（SZ-50: 64.59% → 69.26%）。

---

### 4. Conv3 Kernel Size（嚴重度：中）

| | 論文 | 目前程式碼 |
|---|---|---|
| 描述 | 「用 k3 個 kernel generate k2 x k3 matrix」 | `kernel_size=(1, 1)` 只做 channel mixing |
| 位置 | — | `model.py:78` |

**說明：**
論文描述 Conv3 會「進一步卷積更高階的圖表」，暗示可能有空間維度的卷積。程式碼使用 `(1,1)` kernel 只做 channel transformation，無法跨子圖組合特徵。

**修改方向：** 嘗試不同 kernel size（如 `(3,1)` 或 `(N-4,1)`），比較效果。論文描述模糊，需要實驗驗證。

---

## 二、工程實作差異（論文未提到，程式碼自行加入）

### 5. 技術指標 z-score 標準化（嚴重度：低）

| | 論文 | 目前程式碼 |
|---|---|---|
| 描述 | 未提到對技術指標做標準化 | 對 9 個指標做 z-score |
| 位置 | — | `indicators.py:73-76` |

**說明：**
論文提到其相似度框架「不需要標準化前處理」，但這指的是圖表相似度部分。技術指標的 z-score 是常見的工程實踐，有助於模型訓練穩定性。

**修改方向：** 可保留，或做消融實驗比較有無標準化的差異。

---

### 6. 權重初始化（嚴重度：低）

| | 論文 | 目前程式碼 |
|---|---|---|
| 描述 | 未提到 | Conv 用 Kaiming，Attention 用 Xavier |
| 位置 | — | `model.py:96-105` |

**說明：** 論文未指定初始化方式。程式碼使用 Kaiming（適用 ReLU）和 Xavier（適用 Attention）是業界標準做法。

**修改方向：** 可保留。

---

### 7. Optimizer 選擇（嚴重度：低）

| | 論文 | 目前程式碼 |
|---|---|---|
| 描述 | 未指定（只提到 weight_decay=0.00005） | Adam（lr=1e-3, weight_decay=5e-5） |
| 位置 | — | `train.py:76` |

**修改方向：** 可嘗試 SGD with momentum 比較效果。

---

### 8. Best Model 保存（嚴重度：低）

| | 論文 | 目前程式碼 |
|---|---|---|
| 描述 | 未提到 early stopping 或 best model 選取 | 追蹤最佳 val_acc，載入最佳權重做測試 |
| 位置 | — | `train.py:79-100` |

**修改方向：** 可保留，這是標準實踐。

---

## 三、資料集差異

### 9. 市場與資料源（嚴重度：中）

| | 論文 | 目前程式碼 |
|---|---|---|
| 市場 | 中國 A 股（SZ-50 / CSI-300） | 台灣 50 |
| 資料源 | Tushare | yfinance |
| 訓練期間 | 2010-2017 | 2010-2023 |
| 測試期間 | 2018 整年 | 2024 |
| 訓練樣本 | 85,767 / 543,262 | 55,339（stride=3） |
| stride | 1 | 3（記憶體限制） |

**說明：** 台股與 A 股有不同的交易規則（台股可放空、漲跌幅不同）。資料量差距約 10 倍（尤其 CSI-300）。stride=3 是因為 50 檔 x stride=1 會超出記憶體。

**修改方向：**
- 增加記憶體或改為 lazy loading Dataset，以支援 stride=1
- 若需完全對齊論文，改用 Tushare 取 A 股資料

---

### 10. 超參數搜索（嚴重度：中）

| | 論文 | 目前程式碼 |
|---|---|---|
| 方法 | Grid search | 固定單一值 |
| window | {100, 120, 130, 140} | 100 |
| PIPs | {30, 40, 60, 80} | 40 |
| N | {10, 15, 20, 25} | 15 |
| G | {3, 4, 5, 6} | 5 |

**修改方向：** 實作 grid search 迴圈，或使用 Optuna 做自動超參數搜索。

---

## 四、缺少的實驗分析

### 11. 相似度魯棒性比較（論文 Section 5.1, Fig.5）

比較 PIP+VG 方法 vs DTW vs Pearson 在 6 種圖表變異下的魯棒性。

**需要實作：**
- 6 種變異函數（振幅縮放/平移、時間縮放/平移、旋轉、高斯雜訊）
- 三種相似度方法的計算
- 視覺化比較圖

---

### 12. Baseline 模型對比（論文 Section 5.2, Table 3）

與 CNN、LSTM、TCN、GRU、DARNN、CA-FSCN 比較。

**需要實作：**
- 各 baseline 模型的定義
- 統一的訓練與評估流程
- 結果比較表

---

### 13. 消融實驗（論文 Section 6.1, Table 4）

| 變體 | 描述 |
|------|------|
| Chart GCN-1 | 無 chart importance（用 betweenness centrality 替代） |
| Chart GCN-2 | 無 self-attention |
| Chart GCN-3 | 兩者都無 |
| Chart GCN-4 | 用原始價格替代技術指標 |

**需要實作：**
- 4 個模型變體
- 分別訓練並比較結果

---

### 14. 替換 Kernel 實驗（論文 Section 6.2, Table 5）

將 GCN 替換成 CNN/LSTM/TCN 等 baseline，驗證相似度框架的通用性。加入 61 種布林值圖表比較。

---

## 五、優先修改順序建議

| 優先度 | 項目 | 預估影響 |
|--------|------|---------|
| 1 | 同日 batch Self-Attention（第 3 點） | +5% acc |
| 2 | 移除 residual，恢復論文原始設計（第 1、2 點） | 需搭配第 3 點 |
| 3 | Grid search 超參數（第 10 點） | +2~5% acc |
| 4 | stride=1 + lazy loading（第 9 點） | 更多訓練樣本 |
| 5 | Conv3 kernel 實驗（第 4 點） | 未知 |
| 6 | 消融實驗（第 13 點） | 驗證各組件貢獻 |
| 7 | Baseline 對比（第 12 點） | 驗證方法優越性 |
| 8 | 相似度魯棒性（第 11 點） | 驗證相似度框架 |
