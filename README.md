# K 線型態分析系統

以 Autoencoder + One-Class SVM 進行 K 線型態偵測與回測，搭配 Tkinter GUI 操作介面。

## 方法流程

1. **資料取得** — 透過 yfinance 下載台灣前 50 大股票歷史 OHLCV（2016-2025），自動偵測並排除除權息日
2. **特徵計算** — 三套 K 線特徵集：
   - 1 日（7 特徵）：上影線、下影線、實體線、開盤缺口、收盤漲跌、5 日均量比、前五日趨勢
   - 2 日（10 特徵）：加上前日上/下影線、前日實體線
   - 3 日（12 特徵）：加上前兩日實體線、前日開/收盤型態、3 日累積漲跌、3 日均量比、區間位階
3. **型態篩選** — 歐幾里得距離找相似 K 線群，過濾條件：出現頻率 ≥ 5 且每根相似 K 線皆符合報酬門檻
4. **重要性權重** — 重要性 = 出現頻率 × |平均報酬|
5. **Autoencoder 壓縮** — PyTorch AE 將特徵壓至 4 維潛在空間
6. **One-Class SVM** — 以重要性作為 sample_weight 訓練 OC-SVM，區分看漲/看跌型態
7. **回測** — 2025 年測試期，扣除交易成本（手續費 + 證交稅 = 0.42%/筆）

## GUI 功能

| 按鈕 | 說明 |
|------|------|
| ① 下載全部資料 | 下載台灣 50 成分股歷史資料 |
| ② 特徵驗證（不回測） | 訓練期依日期切 80/20，後 20% 做 out-of-sample 驗證，計算 Precision / Recall / Accuracy / F1，顯示混淆矩陣 |
| ③ 分析 + 回測 | 三特徵集完整訓練 + 指定股票回測（非重疊持倉），顯示累積報酬曲線 |

## 快速開始

```bash
pip install -r requirements.txt
python main.py                    # 建議入口
# 或 python kline_pattern_search.py   # 舊入口（向後相容）
```

## 模組結構

程式已模組化為 `kline/` 套件，運算邏輯與 GUI 介面分離：

```
kline/
├── config.py         固定參數與 TW50 清單
├── features.py       K 線特徵計算（1/2/3 日）＋除權息偵測
├── data_loader.py    股價下載（多線程）與資料集建構
├── autoencoder.py    Autoencoder 模型與訓練
├── pattern_model.py  型態搜尋 + AE + OCSVM 訓練（find_and_train）
├── backtest.py       回測（非重疊持倉）
├── validate.py       特徵驗證（out-of-sample 80/20）
└── gui.py            Tkinter 介面（只負責介面，不含演算法）
main.py               程式進入點
kline_pattern_search.py  向後相容層（re-export 舊 API，舊入口仍可用）
```

- **分層**：`config`（設定）→ 運算層（`features`/`data_loader`/`autoencoder`/`pattern_model`/`backtest`/`validate`，可單獨 `import` 測試）→ `gui`（介面層）。
- **各模組自測**：`python -m kline.<模組名>` 可獨立執行，以合成資料驗證該模組（不需連網）。

## 方法學修正

以下修正已納入對應模組，避免評估失真：

- **回測非重疊持倉**（`backtest.py`）：進場後持有 `HOLD_DAYS` 日內忽略新訊號，避免同一段行情重複計算導致累積報酬高估。
- **out-of-sample 驗證**（`validate.py`）：訓練期依日期切 80/20，後 20% 作驗證段，評估門檻與訓練種子一致。
- **量比分母**（`features.py`）：`vol_ratio` 分母改為均量（v5ma/v3ma），符合常規量比定義。
- **補充下載檢查**（`gui.py`）：非清單股票改用 `download_one`，補上必要欄位檢查（含 Adj Close）。

## 固定參數

| 參數 | 值 | 說明 |
|------|----|------|
| HOLD_DAYS | 5 | 持有天數 |
| RISE_THRESH | 5% | 大漲門檻 |
| FALL_THRESH | -5% | 大跌門檻 |
| DIST_THRESH | 3.0 | 歐幾里得距離門檻 |
| MIN_FREQ | 5 | 最少出現次數 |
| LATENT_DIM | 4 | AE 潛在維度 |
| EPOCHS | 60 | AE 訓練輪數 |
| OCSVM_NU | 0.1 | OC-SVM nu 參數 |
| 訓練期 | 2016-2024 | |
| 測試期 | 2025 | |

## 分支說明

| 分支 | 內容 |
|------|------|
| `main` | AE + OC-SVM K 線型態分析系統（本頁） |
| `Chart-GCN` | Chart GCN（圖卷積網路）論文復現 |
