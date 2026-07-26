# chinatest — 用論文原始資料復現 Chart GCN

目的：分辨「台股復現差距」是來自**資料差異**還是**方法本身**。
本資料夾用論文使用的資料設定重跑同一條對齊後管線：

| 項目 | 論文 | 本實驗 |
|---|---|---|
| 標的 | SZ-50（上證 50）成分股 | 同（2018 年前後成分股快照，見 `core/data_loader.py` 的 `SSE50_TICKERS`） |
| 期間 | 2010-01 ~ 2017-12 訓練/驗證（隨機 80/20）、2018 全年測試 | 相同 |
| 資料源 | Tushare | yfinance（.SS 代碼，auto_adjust 還原權息） |
| 方法 | Chart GCN 全流程 | `core/` 對齊版管線（見 `實驗記錄_論文對齊.md` 第一節審查表） |
| 論文結果 | Acc 69.26%（SZ-50） | 見 `實驗記錄_china.md` |

執行方式（由專案根目錄）：

```
python test/run_paper_repro.py --tickers sse50 ^
    --start 2010-01-01 --train-end 2017-12-31 --end 2018-12-31 ^
    --raw-features --out-dir chinatest --md-file chinatest/實驗記錄_china.md ^
    --tag cn-sse50-raw
```

結果 JSON / 模型 / 回測圖 / 資料快取都存在本資料夾；
實驗摘要自動 append 到 `實驗記錄_china.md`。

注意事項：
- 上證 50 成分股每半年調整，論文未載明快照日期；本清單為 2018 年前後代表性成分
  （含 601229、601360、601881 等 2016–2018 上市者，歷史較短、樣本較少，屬正常）。
- yfinance 的 A 股資料 2010 年起完整（驗證過 2188 個交易日），但個別代碼若下市/改碼會自動略過。
