"""
數據獲取模塊 - 使用yfinance獲取歷史數據
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """從yfinance獲取並管理股票數據"""
    
    def __init__(self, cache_dir='data_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def fetch_data(self, symbol, start_date, end_date, interval='1d'):
        """
        獲取股票歷史數據
        
        Args:
            symbol: 股票代碼 (e.g., '2330.TW')
            start_date: 開始日期 (str or datetime)
            end_date: 結束日期 (str or datetime)
            interval: 時間間隔 ('1d', '1h', '5m' 等)
            
        Returns:
            pd.DataFrame: OHLCV 數據
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed. Using mock data for demo.")
            return self._generate_mock_data(symbol, start_date, end_date)
        
        cache_file = self.cache_dir / f"{symbol}_{interval}_{start_date}_{end_date}.json"
        
        # 檢查快取
        if cache_file.exists():
            logger.info(f"載入快取: {cache_file}")
            try:
                df = pd.read_json(cache_file, orient='table')
                df.index = pd.to_datetime(df.index)
                return df
            except Exception as e:
                logger.warning(f"快取讀取失敗: {e}")
        
        # 從yfinance獲取數據
        logger.info(f"從yfinance獲取 {symbol} ({start_date} to {end_date})")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                logger.warning(f"未取得 {symbol} 的數據")
                return self._generate_mock_data(symbol, start_date, end_date)
            
            # 保存快取
            df.to_json(cache_file, orient='table')
            logger.info(f"數據已快取: {cache_file}")
            
            return df
        except Exception as e:
            logger.error(f"yfinance獲取失敗: {e}")
            return self._generate_mock_data(symbol, start_date, end_date)
    
    def _generate_mock_data(self, symbol, start_date, end_date, rows=252):
        """
        生成模擬K線數據用於測試
        """
        logger.info(f"生成模擬數據: {symbol}")
        
        dates = pd.date_range(start=start_date, end=end_date, periods=rows)
        
        np.random.seed(42 + hash(symbol) % 100)
        prices = np.cumsum(np.random.randn(rows) * 2) + 100
        
        data = {
            'Open': prices + np.random.randn(rows) * 0.5,
            'High': prices + abs(np.random.randn(rows) * 1.5),
            'Low': prices - abs(np.random.randn(rows) * 1.5),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, rows),
            'Dividends': 0,
            'Stock Splits': 0,
        }
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'Date'
        
        # 確保 High >= max(O,C) 且 Low <= min(O,C)
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        
        return df
    
    def validate_data(self, df):
        """
        驗證OHLCV數據完整性
        """
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if not all(col in df.columns for col in required_cols):
            logger.error(f"缺少必需欄位. 需要: {required_cols}")
            return False
        
        # 檢查 High >= max(O,C) 和 Low <= min(O,C)
        invalid_high = (df['High'] < df[['Open', 'Close']].max(axis=1)).sum()
        invalid_low = (df['Low'] > df[['Open', 'Close']].min(axis=1)).sum()
        
        if invalid_high > 0 or invalid_low > 0:
            logger.warning(f"發現 {invalid_high} 行High異常, {invalid_low} 行Low異常")
        
        # 檢查缺失值
        missing = df[required_cols].isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"發現缺失值:\n{missing}")
            return False
        
        logger.info(f"✓ 數據驗證通過 ({len(df)} 根K線)")
        return True
    
    def batch_fetch(self, symbols, start_date, end_date, interval='1d'):
        """
        批量獲取多個股票數據
        
        Returns:
            dict: {symbol: DataFrame}
        """
        data_dict = {}
        for symbol in symbols:
            try:
                df = self.fetch_data(symbol, start_date, end_date, interval)
                if self.validate_data(df):
                    data_dict[symbol] = df
            except Exception as e:
                logger.error(f"獲取 {symbol} 失敗: {e}")
        
        return data_dict


if __name__ == '__main__':
    fetcher = DataFetcher()
    
    # 測試單個股票
    df = fetcher.fetch_data('2330.TW', '2023-01-01', '2024-01-01')
    print(f"數據形狀: {df.shape}")
    print(df.head())
    
    # 批量獲取
    symbols = ['2330.TW', '1303.TW', '2454.TW']
    data_dict = fetcher.batch_fetch(symbols, '2023-01-01', '2024-01-01')
    print(f"\n獲取 {len(data_dict)} 個股票")
