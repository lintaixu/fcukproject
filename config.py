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

# 股票代碼
STOCK_SYMBOLS = [
    '2330.TW',  # 台積電
    '1303.TW',  # 南亞
    '2454.TW',  # 聯發科
    '3711.TW',  # 日月光
    '2317.TW',  # 鴻海
    '1101.TW',  # 台泥
    '1301.TW',  # 台塑
    '6505.TW',  # 聯杰
]

# 分析週期
ANALYSIS_PERIOD = {
    'start_date': '2022-01-01',
    'end_date': '2025-01-01',
}
