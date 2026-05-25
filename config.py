"""
K線型態回測系統 - 配置模塊
"""

# 數據配置
DATA_CONFIG = {
    'cache_dir': 'data_cache',
    'default_interval': '1d',  # 日線
    'max_workers': 4,
}

# K線型態配置
PATTERN_CONFIG = {
    'min_body_ratio': 0.3,  # 最小K線實體比例
    'shadow_threshold': 0.1,  # 影線閾值
    'volume_threshold': 1.2,  # 成交量放大倍數
}

# 回測配置
BACKTEST_CONFIG = {
    'initial_capital': 100000,
    'commission': 0.001,  # 0.1% 手續費
    'slippage': 0.0005,  # 0.05% 滑點
    'stop_loss_ratio': 0.05,  # 5% 止損
    'take_profit_ratio': 0.10,  # 10% 止盈
}

# 統計配置
ANALYSIS_CONFIG = {
    'min_samples': 10,  # 最少樣本數
    'sharpe_risk_free_rate': 0.02,  # 無風險利率
}

# 可視化配置
PLOT_CONFIG = {
    'style': 'seaborn-v0_8-darkgrid',
    'figure_size': (14, 8),
    'dpi': 100,
}

# 股票代碼 - 台灣前50大公司（台灣50成分股）
STOCK_SYMBOLS = [
    '2330.TW',  # 台積電
    '2317.TW',  # 鴻海
    '2454.TW',  # 聯發科
    '2412.TW',  # 中華電
    '2308.TW',  # 台達電
    '2882.TW',  # 國泰金
    '2881.TW',  # 富邦金
    '2886.TW',  # 兆豐金
    '2891.TW',  # 中信金
    '2892.TW',  # 第一金
    '3008.TW',  # 尖端醫
    '2303.TW',  # 聯電
    '2002.TW',  # 鑫永銅
    '2301.TW',  # 光磊
    '1301.TW',  # 台塑
    '1303.TW',  # 南亞
    '2884.TW',  # 玉山金
    '5880.TW',  # 合庫金
    '2357.TW',  # 華碩
    '2382.TW',  # 廣達
    '2207.TW',  # 和泰車
    '2801.TW',  # 彩晶
    '2885.TW',  # 元大金
    '2395.TW',  # 宏達電
    '2912.TW',  # 統一超
    '3711.TW',  # 日月光
    '2327.TW',  # 光磊
    '6505.TW',  # 聯杰
    '2379.TW',  # 瑞昱
    '2408.TW',  # 南科
    '2474.TW',  # 佳能
    '3034.TW',  # 聯詠
    '6669.TW',  # 緯穎
    '2409.TW',  # 友達
    '2324.TW',  # 仁寶
    '2345.TW',  # 光磊
    '2347.TW',  # 芬蘭傳
    '3045.TW',  # 台灣大
    '2354.TW',  # 鴻準
    '2356.TW',  # 光聯
    '2377.TW',  # 微星
    '1216.TW',  # 統一
    '1326.TW',  # 天仁
    '2609.TW',  # 陽明
    '2603.TW',  # 長榮
    '2615.TW',  # 萬海
    '5871.TW',  # 中租
    '3231.TW',  # 緩達
    '4904.TW',  # 遠傳
    '4938.TW',  # 和碩
]

# 分析週期
ANALYSIS_PERIOD = {
    'start_date': '2022-01-01',
    'end_date': '2025-01-01',
}
