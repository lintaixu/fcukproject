"""
K線型態分析系統 — 模組化套件。

流程：Euclidean Distance 找型態 → 重要性權重 → Autoencoder 壓縮 → One-Class SVM 判斷

模組：
    config        固定參數與 TW50 清單
    features      K 線特徵計算（1/2/3 日）
    data_loader   股價下載與資料集建構
    autoencoder   Autoencoder 模型與訓練
    pattern_model 型態搜尋 + AE + OCSVM 訓練
    backtest      回測
    validate      特徵驗證
    gui           Tkinter 介面
"""

__version__ = '1.0.0'
