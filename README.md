# Chart GCN — 台股實作

依據 Li, Wu, Jiang & Xu (2022) 發表於 *Knowledge-Based Systems* 的論文
**"Chart GCN: Learning chart information with a graph convolutional network for stock movement prediction"**
改寫成台股版本的 PyTorch 實作。

> **⚠ 專案現狀(2026-07-25)**:論文復現已完結。主要結論:
> 1. 無洩漏協定下,台股(TW50)與論文自身資料(上證 50)的 Test Acc 皆停在 **~50% 天花板**,論文宣稱的 69.26% 無法重現。
> 2. 論文的 batch 維 self-attention 在序向批次評估下會**洩漏未來資訊**;標準協定已改為 **date 批次(同日股票一批)+ raw 特徵**。
> 3. 同條件 XGBoost 對照在乾淨協定下與 Chart-GCN 同水準。
>
> 完整證據鏈與細節請讀 **`實驗記錄_論文對齊.md`**(審查報告+全部實驗記錄)。

## 權威文件指引

| 文件 | 內容 |
|---|---|
| `實驗記錄_論文對齊.md` | 全流程審查、確認問題清單、論文對齊清單(可調參數/拼湊部分)、全部實驗記錄 |
| `paper_analysis.md` | 論文技術逐條清單(已依原文 PDF 核對更正) |
| `台積電_資料流程.md` | 以台積電真實數值示範每一步輸入/輸出(教學用) |
| `融合實驗報告.md` | main(AE+OCSVM)×Chart-GCN 融合實驗 V1~V7c 完整記錄 |
| `週進度計畫.md` | 7/7~8/31 每週會議路線圖 |
| `chinatest/` | 上證 50 復現(否證論文 69.26% 的一手證據) |

---

## 檔案結構

```
chartgcn/
├── core/                     # 核心模組
│   ├── data_loader.py        #   台股資料下載 (yfinance, auto_adjust, parquet cache)
│   ├── pip_algorithm.py      #   PIP 關鍵點萃取 + importance score
│   ├── vg_graph.py           #   Visibility Graph 建圖
│   ├── subgraph.py           #   子圖選取、正規化、特徵綁定
│   ├── indicators.py         #   9 個技術指標 (論文 Table 1)
│   ├── model.py              #   Chart GCN 模型 (paper_exact=True 論文版 / False 舊改進版)
│   ├── dataset.py            #   Dataset 建構 (滑窗、標籤、npz 快取)
│   └── train.py              #   訓練迴圈 (batch_mode: 'paper' / 'date')
├── test/
│   ├── run_paper_repro.py    # ★ 標準實驗執行器 (自動 append 實驗記錄)
│   ├── grid_search_paper.py  # ★ 論文範圍網格搜尋
│   ├── run_xgb_baseline.py   # ★ XGBoost 同條件對照
│   └── main.py 等五支         # ⚠ 舊執行器,見下方警告
├── backtest.py               # 交易模擬 (論文 Section 6.3)
├── plot_pip.py               # PIP/VG/子圖視覺化
├── experiments_paper/        # 台股實驗產物 (JSON/model.pt/回測圖/dscache)
├── chinatest/                # 上證 50 復現實驗
└── experiments_fusion/       # 融合實驗 (未版控,重現方式見 融合實驗報告.md)
```

> **⚠ 舊執行器警告**:`test/main.py`、`run_50stocks.py`、`run_100stocks.py`、`run_final.py`、`grid_search_run.py`
> 為歷史腳本,**未跟上新標準協定**(未傳 batch_mode → 走含洩漏的 paper 批次;部分未傳 norm_stats/warmup)。
> 跑出的數字不可與現行記錄比較,僅供追溯舊 PPT 數據來源。新實驗一律用 ★ 標記的三支。

---

## 安裝

需要 Python ≥ 3.9(建議 3.10+)。

```bash
pip install -r requirements.txt
```

CPU 上跑 TW50、快取命中時完整訓練一次約 5 分鐘;首次建 dataset 快取約 10~30 分鐘。

---

## 快速開始(現行標準協定)

```bash
# 標準實驗:raw 特徵 + date 批次(無洩漏), 自動記錄到 實驗記錄_論文對齊.md
python test/run_paper_repro.py --raw-features --batch-mode date --tag my-exp

# 用網格最佳參數
python test/run_paper_repro.py --raw-features --batch-mode date \
    --window 140 --m-pips 80 --N 10 --g 4 --seed 42 --tag my-exp

# 網格搜尋(論文 4.3 範圍, 兩階段)
python test/grid_search_paper.py --batch-mode date

# XGBoost 同條件對照
python test/run_xgb_baseline.py --market tw

# 上證 50(論文自身資料)
python test/run_paper_repro.py --raw-features --batch-mode date \
    --tickers sse50 --start 2010-01-01 --train-end 2017-12-31 --end 2018-12-31 \
    --out-dir chinatest --md-file chinatest/實驗記錄_china.md --tag cn-exp
```

各 core 模組可單獨自測:`python core/pip_algorithm.py` 等(內含 `__main__` 測試)。

---

## 超參數

| 參數 | 論文網格 | date 批次最佳 (tw50) | 說明 |
|---|---|---|---|
| `--window` | {100, 120, 130, 140} | **140** | 滑窗長度 = 指標窗口 n |
| `--m-pips` | {30, 40, 60, 80} | **80** | PIP 關鍵點數 |
| `--N` | {10, 15, 20, 25} | **10** | 核心節點數(子圖數) |
| `--g` | {3, 4, 5, 6} | **4** | 每子圖節點數 |
| `--batch-mode` | 論文未定義 | **date** | paper=序向128筆(有洩漏)/ date=同日一批(乾淨) |

模型內部:`F1=84, F2=32` 為論文明定;`k1=6, k2=16, k3=120, ka=kv=16` 論文未給,
為 LeNet-5 對應假設(`paper_exact=False` 舊改進版為 32/32/16 + BN + 殘差)。
完整的「論文明定 vs 我們拼湊」清單見 實驗記錄_論文對齊.md 第六節。

## 主要結果(2024 台股測試集, 11,950 筆)

| 設定 | Acc | F1 macro |
|---|---|---|
| Chart-GCN, paper 批次(含洩漏, 舊協定) | 52.70% | 50.33% |
| Chart-GCN, date 批次最佳參數 ×3 seeds | 53.7% ± 0.6 | **38.3% ± 0.3**(塌縮全押跌) |
| XGBoost(裸 9 指標) | 53.87% | 39.44%(塌縮) |
| 全押「跌」的數學下限 | 54.2% | 35.1% |

回測(零成本、close-to-close)三個 seed 皆輸等權買入持有約 10~13%。

---

## 常見問題

**Q: yfinance 下載很慢 / 失敗**
A: 首次較慢,之後 cache 在 `./cache/`(請從 repo 根目錄執行,cache 路徑是相對 cwd)。注意 `end` 為右開區間、單檔失敗會被靜默跳過(已知問題,見 實驗記錄_論文對齊.md 第三節)。

**Q: 記憶體不足(30 萬+ 樣本)**
A: dataset 建構器已用描述子排序+預配置;避免多個訓練並行(Windows 分頁檔限制),或加 `--stride 5`。

**Q: 為什麼我的準確率只有 ~50%?**
A: 這就是結論——無洩漏協定下台股與上證的資訊天花板即 ~50%,不是你跑錯。細節見 實驗記錄_論文對齊.md。

**Q: 想換成自己的資料?**
A: 在 `core/data_loader.py` 寫函數回傳 `{ticker: pd.DataFrame}`(至少 open/high/low/close 四欄、datetime index)。

---

## 研究延伸方向

復現主線已完結,後續方向見 `週進度計畫.md`(路線圖)與 `融合實驗報告.md` 第七章(A-1~A-4 / B-1~B-4 精進清單,含 PIP 尺度正規化 B-1、橫斷面標籤 B-2)。模型層可嘗試:torch_geometric 真圖卷積(GAT/GIN)、多時間框架標籤(1/5/20 日)、論文消融變體(Chart GCN-1~4)。

## 參考文獻

- Li, S., Wu, J., Jiang, X., & Xu, K. (2022). Chart GCN: Learning chart information with a graph convolutional network for stock movement prediction. *Knowledge-Based Systems*, 248, 108842.
- Lacasa, L., Luque, B., Ballesteros, F., Luque, J., & Nuño, J. C. (2008). From time series to complex networks: The visibility graph. *PNAS*, 105(13), 4972–4975.
- Niepert, M., Ahmed, M., & Kutzkov, K. (2016). Learning convolutional neural networks for graphs. *ICML*.
- Tsinaslanidis, P. E. (2018). Subsequence dynamic time warping for charting: Bullish and bearish class predictions for NYSE stocks. *Expert Systems with Applications*, 94, 193–204.

## License

本實作為學術研究目的撰寫,請依原論文授權使用。
