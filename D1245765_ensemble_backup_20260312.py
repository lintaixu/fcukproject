"""
K線型態預測系統 - 使用 Autoencoder + One-Class SVM + 基因演算法
資料探勘期末作業
學號: D1245765

【核心流程】
Step 1: 特徵工程
    - 提取 10 種型態特徵 + 基礎量價指標
    - 注意事項：務必進行標準化（Scaling），避免股價絕對值影響 AE 訓練

Step 2: AE 預訓練
    - 在全市場或績優股歷史資料上訓練，提取 Latent Space
    - 注意事項：監控重構誤差（Reconstruction Error），確保特徵沒丟失

Step 3: OC-SVM 建模
    - 使用編碼後的特徵（Latent Code）訓練 OC-SVM
    - 注意事項：調整 nu 參數（預期的異常比例），這會直接影響交易頻率

Step 4: 回測驗證
    - 驗證通過 OC-SVM 過濾後的看漲型態，勝率是否顯著提升
    - 注意事項：警惕「存活者偏差」，確保回測包含已退市或下市的股票
"""

# ==================== 隨機種子設定（必須在最開始） ====================
import os
import random

# 設定為 None 可測試穩定度，設定為數字(如42)可固定結果
RANDOM_SEED = 327  # 改為 None 可測試穩定度

# 必須在導入其他套件前設定環境變數
if RANDOM_SEED is not None:
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(RANDOM_SEED)

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import euclidean
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 在導入 TensorFlow 後立即設定種子
if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    # 設定 TensorFlow 為單執行緒（確保確定性）
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

def set_random_seed(seed=None):
    """重新設定隨機種子（在訓練前調用）"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== DataProcessor 類別 ====================
class DataProcessor:
    """負責資料抓取、除權息清洗、特徵工程"""
    
    def __init__(self):
        self.feature_columns = [
            '上影線(%)', '下影線(%)', '實體線(%)',
            '前一日上影線(%)', '前一日下影線(%)', '前一日實體線(%)',
            '開盤型態(%)', '收盤型態(%)', '成交量突變率', '前五日趨勢'
        ]
        # 台股市場代表性樣本：涵蓋台灣 50 + 中型 100 主要成分股
        # 包含電子、金融、傳產、塑化、航運、食品等各產業龍頭
        self.market_stocks = [
            # 大型權值股（台灣 50 主要成分）
            '2330', '2317', '2454', '2412', '2308', '2882', '2881', '2886', '2891', '2892',
            '3008', '2303', '2002', '2301', '1301', '1303', '2884', '5880', '2357', '2382',
            '2207', '2801', '2885', '2395', '2912', '3711', '2327', '6505', '2379', '2408',
            # 中型股精選（產業代表）
            '2474', '3034', '6669', '2409', '2324', '2345', '2347', '3045', '2354', '2356',
            '2377', '1216', '1326', '2609', '2603', '2615', '5871', '3231', '4904', '4938',
            '6415', '9910', '2888', '2880', '2887', '5876', '2542', '2492', '2106', '2014',
            # 傳產、塑化、航運、食品
            '1102', '2313', '2105', '2201', '2609', '1101', '1476', '2618', '2633', '2337',
            '2610', '9904', '2324', '2204', '2206', '1590', '1802', '2352', '2383', '1605'
        ]
        # 預設使用前 50 檔（可透過參數調整）
        self.default_stocks = self.market_stocks[:50]
    
    def fetch_multiple_stocks(self, stock_codes, start_date, end_date):
        """
        抓取多檔股票資料
        
        Args:
            stock_codes: 股票代碼列表
            start_date: 開始日期
            end_date: 結束日期
            
        Returns:
            dict: {stock_code: DataFrame}
        """
        all_data = {}
        total = len(stock_codes)
        
        for idx, stock_code in enumerate(stock_codes, 1):
            try:
                print(f"[{idx}/{total}] 正在抓取 {stock_code} 資料...")
                data = self.fetch_data(stock_code, start_date, end_date)
                if not data.empty:
                    df = self.calculate_features(data)
                    if not df.empty:
                        all_data[stock_code] = df
                        print(f"✓ {stock_code} 資料處理完成，共 {len(df)} 筆")
            except Exception as e:
                print(f"✗ {stock_code} 處理失敗: {str(e)}")
                continue
        
        print(f"\n總結：成功 {len(all_data)}/{total} 檔股票\n")
        return all_data
    
    def fetch_data(self, stock_code, start_date, end_date):
        """
        抓取股票資料並清除除權息日期
        
        Args:
            stock_code: 股票代碼 (例如 '2330' 或 '2330.TW' 或 'AAPL')
            start_date: 開始日期
            end_date: 結束日期
            
        Returns:
            DataFrame: 清洗後的股價資料
        """
        try:
            # 智能處理股票代碼：純數字自動加上 .TW（台股）
            stock_code = stock_code.strip()
            if stock_code.isdigit():
                stock_code = f"{stock_code}.TW"
                print(f"自動識別為台股代碼: {stock_code}")
            
            ticker = yf.Ticker(stock_code)
            
            # 抓取股價資料
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"無法取得 {stock_code} 的資料")
            
            # 取得除權息資訊
            actions = ticker.actions
            
            if not actions.empty:
                # 找出有配息或股票分割的日期
                dividend_dates = actions[actions['Dividends'] > 0].index
                split_dates = actions[actions['Stock Splits'] > 0].index
                
                # 合併所有除權息日期
                ex_dates = dividend_dates.union(split_dates)
                
                # 從資料中移除這些日期
                data = data[~data.index.isin(ex_dates)]
                
                print(f"已移除 {len(ex_dates)} 個除權息日期")
            
            return data
            
        except Exception as e:
            raise Exception(f"資料抓取失敗: {str(e)}")
    
    def calculate_features(self, data):
        """
        計算 10 個 K 線特徵
        
        Args:
            data: 原始股價資料
            
        Returns:
            DataFrame: 包含特徵的資料
            
        注意事項：
            - 所有特徵均以百分比形式計算，避免股價絕對值影響
            - 特徵會在後續的 ModelEngine 中進行 StandardScaler 標準化
            - 標準化對於 Autoencoder 訓練至關重要
        """
        df = data.copy()
        
        # 避免除零問題，將 Close 為 0 的值替換為很小的數
        df['Close'] = df['Close'].replace(0, 1e-10)
        df['Volume'] = df['Volume'].replace(0, 1e-10)
        
        # 1. 上影線(%) - 相對於收盤價的百分比
        df['上影線(%)'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close'] * 100
        
        # 2. 下影線(%) - 相對於收盤價的百分比
        df['下影線(%)'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close'] * 100
        
        # 3. 實體線(%) - 相對於收盤價的百分比
        df['實體線(%)'] = (df['Close'] - df['Open']) / df['Close'] * 100
        
        # 4. 前一日上影線(%)
        df['前一日上影線(%)'] = df['上影線(%)'].shift(1)
        
        # 5. 前一日下影線(%)
        df['前一日下影線(%)'] = df['下影線(%)'].shift(1)
        
        # 6. 前一日實體線(%)
        df['前一日實體線(%)'] = df['實體線(%)'].shift(1)
        
        # 7. 開盤型態(%) - 跳空程度
        df['開盤型態(%)'] = (df['Open'] - df['Close'].shift(1)) / df['Close'] * 100
        
        # 8. 收盤型態(%)
        df['收盤型態(%)'] = (df['Close'] - df['Close'].shift(1)) / df['Close'] * 100
        
        # 9. 成交量突變率
        volume_ma5 = df['Volume'].rolling(window=5).mean()
        volume_ma5 = volume_ma5.replace(0, 1e-10)
        df['成交量突變率'] = (df['Volume'] - volume_ma5) / df['Volume']
        
        # 10. 前五日趨勢 - 修正公式: (Close_T-2 - Close_T-7) / Close_T-7
        close_t7 = df['Close'].shift(7).replace(0, 1e-10)
        df['前五日趨勢'] = (df['Close'].shift(2) - close_t7) / close_t7
        
        # 計算三天後的漲跌幅作為標籤
        df['三天後漲跌(%)'] = ((df['Close'].shift(-3) - df['Close']) / df['Close']) * 100
        
        # 定義標籤: 1=大漲, -1=大跌, 0=盤整
        df['Label'] = 0
        df.loc[df['三天後漲跌(%)'] > 5, 'Label'] = 1  # 大漲
        df.loc[df['三天後漲跌(%)'] < -5, 'Label'] = -1  # 大跌
        
        # 移除 NaN 和無窮大值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        return df
    
    def split_train_test(self, df, test_year=2025):
        """
        切分訓練集與測試集
        
        Args:
            df: 完整資料
            test_year: 測試集的年份
            
        Returns:
            train_df, test_df
        """
        train_df = df[df.index.year < test_year].copy()
        test_df = df[df.index.year >= test_year].copy()
        
        return train_df, test_df


# ==================== PatternSelector 類別 ====================
class PatternSelector:
    """負責型態篩選、頻率統計與獲利排序"""
    
    def __init__(self, feature_columns, min_frequency=3, min_avg_return=2.0):
        """
        Args:
            feature_columns: 特徵欄位
            min_frequency: 最小出現次數
            min_avg_return: 最小平均報酬率(%)
        """
        self.feature_columns = feature_columns
        self.min_frequency = min_frequency
        self.min_avg_return = min_avg_return
        self.selected_patterns = {'up': [], 'down': []}
    
    def find_frequent_patterns(self, all_stock_data, similarity_threshold=0.5):
        """
        在多檔股票中找出出現頻率高且預測準確的型態
        
        Args:
            all_stock_data: dict {stock_code: train_df}
            similarity_threshold: 相似度閾值
            
        Returns:
            dict: {'up': [patterns], 'down': [patterns]}
        """
        print("\n開始進行型態篩選...")
        
        # 收集所有大漲和大跌型態
        all_up_patterns = []
        all_down_patterns = []
        
        for stock_code, train_df in all_stock_data.items():
            up_samples = train_df[train_df['Label'] == 1]
            down_samples = train_df[train_df['Label'] == -1]
            
            for _, row in up_samples.iterrows():
                pattern = {
                    'features': row[self.feature_columns].values,
                    'return': row['三天後漲跌(%)'],
                    'stock': stock_code,
                    'date': row.name
                }
                all_up_patterns.append(pattern)
            
            for _, row in down_samples.iterrows():
                pattern = {
                    'features': row[self.feature_columns].values,
                    'return': row['三天後漲跌(%)'],
                    'stock': stock_code,
                    'date': row.name
                }
                all_down_patterns.append(pattern)
        
        print(f"收集到 {len(all_up_patterns)} 個大漲型態, {len(all_down_patterns)} 個大跌型態")
        
        # 篩選上漲型態
        selected_up = self._select_patterns(all_up_patterns, pattern_type='up')
        
        # 篩選下跌型態
        selected_down = self._select_patterns(all_down_patterns, pattern_type='down')
        
        self.selected_patterns = {'up': selected_up, 'down': selected_down}
        
        print(f"\n✓ 篩選完成: {len(selected_up)} 個上漲型態, {len(selected_down)} 個下跌型態")
        
        return self.selected_patterns
    
    def _select_patterns(self, patterns, pattern_type='up'):
        """
        二次篩選：找出頻率高且獲利好的型態
        """
        if len(patterns) == 0:
            return []
        
        # 使用聚類方法找出相似型態群組
        from sklearn.cluster import DBSCAN
        
        features = np.array([p['features'] for p in patterns])
        
        # 標準化特徵
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # DBSCAN 聚類
        clustering = DBSCAN(eps=0.8, min_samples=self.min_frequency).fit(features_scaled)
        labels = clustering.labels_
        
        # 分析每個群組
        selected = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # 噪音點
                continue
            
            # 取得該群組的所有型態
            cluster_indices = np.where(labels == label)[0]
            cluster_patterns = [patterns[i] for i in cluster_indices]
            
            # 計算平均報酬
            avg_return = np.mean([p['return'] for p in cluster_patterns])
            
            # 檢查是否符合條件
            frequency = len(cluster_patterns)
            
            if pattern_type == 'up':
                condition = avg_return >= self.min_avg_return
            else:  # down
                condition = avg_return <= -self.min_avg_return
            
            if frequency >= self.min_frequency and condition:
                # 計算群組中心（代表型態）
                center_features = np.mean([p['features'] for p in cluster_patterns], axis=0)
                
                selected.append({
                    'features': center_features,
                    'frequency': frequency,
                    'avg_return': avg_return,
                    'stocks': list(set([p['stock'] for p in cluster_patterns])),
                    'pattern_type': pattern_type
                })
        
        # 依獲利情形排序
        if pattern_type == 'up':
            selected = sorted(selected, key=lambda x: x['avg_return'], reverse=True)
        else:
            selected = sorted(selected, key=lambda x: x['avg_return'])
        
        return selected
    
    def get_pattern_summary(self):
        """
        取得型態摘要報告
        """
        summary = {
            'up_patterns': len(self.selected_patterns['up']),
            'down_patterns': len(self.selected_patterns['down']),
            'top_up': self.selected_patterns['up'][:5] if self.selected_patterns['up'] else [],
            'top_down': self.selected_patterns['down'][:5] if self.selected_patterns['down'] else []
        }
        return summary


# ==================== ModelEngine 類別 ====================
class ModelEngine:
    """包含 Autoencoder, OneClassSVM, GA 的核心演算法"""
    
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        self.scaler_up = StandardScaler()
        self.scaler_down = StandardScaler()
        # 兩個Autoencoder：一個學上漲、一個學下跌
        self.autoencoder_up = None
        self.autoencoder_down = None
        self.encoder_up = None
        self.encoder_down = None
        self.svm_up = None
        self.svm_down = None
        self.train_features_up = None
        self.train_features_down = None
        self.history_up = None
        self.history_down = None
        
    # ========== 加分項目 1：特徵權重 ==========
    def get_feature_weights(self, train_df):
        """
        使用 Random Forest 計算特徵重要性作為權重
        
        Args:
            train_df: 訓練資料
            
        Returns:
            weights: 特徵權重陣列
        """
        # 準備訓練數據
        X = train_df[self.feature_columns].values
        y = train_df['Label'].values
        
        # 只使用有標籤的樣本（排除0）
        mask = y != 0
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        if len(X_filtered) < 10:
            # 數據太少，返回均勻權重
            return np.ones(len(self.feature_columns))
        
        # 訓練 Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X_filtered, y_filtered)
        
        # 取得特徵重要性
        weights = rf.feature_importances_
        
        # 正規化權重（確保不為0）
        weights = weights / (weights.sum() + 1e-10)
        weights = np.maximum(weights, 0.01)  # 避免權重過小
        
        print(f"特徵權重: {dict(zip(self.feature_columns, weights.round(3)))}")
        
        return weights
    
    # ========== 方法一：相似度比對（支援加權）==========
    def euclidean_predict(self, train_df, test_df, weights=None):
        """
        使用加權歐式距離找出最相似的歷史型態
        
        Args:
            train_df: 訓練資料
            test_df: 測試資料
            weights: 特徵權重陣列（若為None則使用均勻權重）
            
        Returns:
            predictions: 預測結果 (1=漲, -1=跌)
        """
        # 設定權重
        if weights is None:
            weights = np.ones(len(self.feature_columns))
        
        # 取得大漲和大跌的樣本
        up_samples = train_df[train_df['Label'] == 1][self.feature_columns].values
        down_samples = train_df[train_df['Label'] == -1][self.feature_columns].values
        
        predictions = []
        
        for _, row in test_df.iterrows():
            test_vec = row[self.feature_columns].values
            
            # 計算與大漲樣本的最小加權距離
            min_up_dist = np.inf
            if len(up_samples) > 0:
                # 加權歐式距離
                weighted_dists = [np.sqrt(np.sum(weights * (test_vec - up_vec)**2)) for up_vec in up_samples]
                min_up_dist = min(weighted_dists)
            
            # 計算與大跌樣本的最小加權距離
            min_down_dist = np.inf
            if len(down_samples) > 0:
                weighted_dists = [np.sqrt(np.sum(weights * (test_vec - down_vec)**2)) for down_vec in down_samples]
                min_down_dist = min(weighted_dists)
            
            # 距離較小者為預測結果
            if min_up_dist < min_down_dist:
                predictions.append(1)
            else:
                predictions.append(-1)
        
        return np.array(predictions)
    
    # ========== 方法二：Autoencoder ==========
    def build_autoencoder(self, input_dim=10):
        """
        建立 Autoencoder 模型
        架構: Input(10) -> Dense(8) -> Dense(4) -> Dense(8) -> Output(10)
        
        Args:
            input_dim: 輸入維度
            
        Returns:
            autoencoder, encoder
        """
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(8, activation='relu')(input_layer)
        encoded = layers.Dense(4, activation='relu', name='latent_code')(encoded)
        
        # Decoder
        decoded = layers.Dense(8, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # 完整的 Autoencoder
        autoencoder = keras.Model(input_layer, decoded)
        
        # Encoder (用於提取 Latent Code)
        encoder = keras.Model(input_layer, encoded)
        
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder, encoder
    
    def train_autoencoder(self, train_df, epochs=50, batch_size=32):
        """
        訓練兩個 Autoencoder：一個學習上漲型態、一個學習下跌型態
        
        Args:
            train_df: 訓練資料
            epochs: 訓練輪數
            batch_size: 批次大小
            
        注意事項：
            - 監控重構誤差（MSE），確保特徵沒有丟失
            - 重構誤差越低，表示 Latent Space 保留了更多原始資訊
            - 典型的好結果：MSE < 0.01（取決於特徵的標準化範圍）
            - 若 MSE 過高，考慮增加 Latent Code 維度或調整網路架構
        """
        # 分別取出大漲和大跌的樣本（r > +5% 和 r < -5%）
        up_samples = train_df[train_df['Label'] == 1][self.feature_columns].values
        down_samples = train_df[train_df['Label'] == -1][self.feature_columns].values
        
        print(f"訓練上漲型態 Autoencoder：{len(up_samples)} 個樣本")
        print(f"訓練下跌型態 Autoencoder：{len(down_samples)} 個樣本")
        
        # 訓練上漲型態的 Autoencoder
        if len(up_samples) > 0:
            X_up_scaled = self.scaler_up.fit_transform(up_samples)
            self.autoencoder_up, self.encoder_up = self.build_autoencoder(input_dim=len(self.feature_columns))
            self.history_up = self.autoencoder_up.fit(
                X_up_scaled, X_up_scaled,
                epochs=epochs,
                batch_size=min(batch_size, len(up_samples)),
                validation_split=0.2 if len(up_samples) > 10 else 0,
                verbose=0
            )
            final_loss = self.history_up.history['loss'][-1]
            print(f"✓ 上漲型態 Autoencoder 訓練完成 (最終重構誤差 MSE: {final_loss:.6f})")
        
        # 訓練下跌型態的 Autoencoder
        if len(down_samples) > 0:
            X_down_scaled = self.scaler_down.fit_transform(down_samples)
            self.autoencoder_down, self.encoder_down = self.build_autoencoder(input_dim=len(self.feature_columns))
            self.history_down = self.autoencoder_down.fit(
                X_down_scaled, X_down_scaled,
                epochs=epochs,
                batch_size=min(batch_size, len(down_samples)),
                validation_split=0.2 if len(down_samples) > 10 else 0,
                verbose=0
            )
            final_loss = self.history_down.history['loss'][-1]
            print(f"✓ 下跌型態 Autoencoder 訓練完成 (最終重構誤差 MSE: {final_loss:.6f})")
    
    def get_latent_code(self, df, model_type='up'):
        """
        取得 Latent Code (4維壓縮特徵)
        
        Args:
            df: 資料
            model_type: 'up' 或 'down'，指定使用哪個模型
            
        Returns:
            latent_code: 4維特徵向量
        """
        X = df[self.feature_columns].values
        
        if model_type == 'up' and self.encoder_up is not None:
            X_scaled = self.scaler_up.transform(X)
            latent_code = self.encoder_up.predict(X_scaled, verbose=0)
        elif model_type == 'down' and self.encoder_down is not None:
            X_scaled = self.scaler_down.transform(X)
            latent_code = self.encoder_down.predict(X_scaled, verbose=0)
        else:
            # 備用：使用上漲模型
            X_scaled = self.scaler_up.transform(X)
            latent_code = self.encoder_up.predict(X_scaled, verbose=0)
        
        return latent_code
    
    def get_reconstruction_error(self, df, model_type='up'):
        """
        計算重建誤差（MSE）
        
        Args:
            df: 資料
            model_type: 'up' 或 'down'
            
        Returns:
            reconstruction_errors: 每個樣本的重建誤差
        """
        X = df[self.feature_columns].values
        
        if model_type == 'up' and self.autoencoder_up is not None:
            X_scaled = self.scaler_up.transform(X)
            X_reconstructed = self.autoencoder_up.predict(X_scaled, verbose=0)
        elif model_type == 'down' and self.autoencoder_down is not None:
            X_scaled = self.scaler_down.transform(X)
            X_reconstructed = self.autoencoder_down.predict(X_scaled, verbose=0)
        else:
            return np.zeros(len(df))
        
        # 計算每個樣本的MSE
        errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        return errors
    
    def autoencoder_predict(self, train_df, test_df, confidence_threshold=0.05):
        """
        方法二：使用 Autoencoder 的重建誤差進行預測（加入信心度閾值機制）
        邏輯：重建誤差小 = 與該型態相符
        
        Args:
            train_df: 訓練資料（未使用，保留接口一致性）
            test_df: 測試資料
            confidence_threshold: 信心度閾值（相對誤差差距）
            
        Returns:
            predictions: 預測結果 (1=漲, -1=跌, 0=觀望)
        """
        # 計算測試集在兩個模型上的重建誤差
        error_up = self.get_reconstruction_error(test_df, model_type='up')
        error_down = self.get_reconstruction_error(test_df, model_type='down')
        
        predictions = []
        
        for i in range(len(test_df)):
            # 計算兩個誤差的平均值
            avg_error = (error_up[i] + error_down[i]) / 2
            
            # 如果平均誤差太高，表示市場混亂，不做預測
            if avg_error > 0.5:  # 絕對誤差閾值
                predictions.append(0)  # 觀望
                continue
            
            # 計算相對差距
            if avg_error > 0:
                relative_diff = abs(error_up[i] - error_down[i]) / avg_error
            else:
                relative_diff = 0
            
            # 如果兩個誤差差距太小，表示信心度不足，觀望
            if relative_diff < confidence_threshold:
                predictions.append(0)  # 觀望
            elif error_up[i] < error_down[i]:
                predictions.append(1)  # 上漲型態的誤差較小且信心度足夠 -> 預測上漲
            else:
                predictions.append(-1)  # 下跌型態的誤差較小且信心度足夠 -> 預測下跌
        
        return np.array(predictions)
    
    # ========== 方法三：One-Class SVM ==========
    def train_one_class_svm(self, train_df):
        """
        訓練兩個 One-Class SVM (一個學習上漲、一個學習下跌)
        使用對應 Autoencoder 的 Latent Code
        
        Args:
            train_df: 訓練資料
            
        注意事項：
            - nu 參數控制預期的異常比例 (0.1 = 10%)
            - nu 越小，邊界越緊，交易訊號越少但品質越高
            - nu 越大，邊界越鬆，交易訊號越多但可能包含更多雜訊
            - 建議範圍：0.05 ~ 0.2
        """
        # 分別取出大漲和大跌的樣本
        up_samples = train_df[train_df['Label'] == 1]
        down_samples = train_df[train_df['Label'] == -1]
        
        # 取得對應的 Latent Code (使用 AE 編碼後的 4 維特徵)
        latent_up = self.get_latent_code(up_samples, model_type='up') if len(up_samples) > 0 else np.array([])
        latent_down = self.get_latent_code(down_samples, model_type='down') if len(down_samples) > 0 else np.array([])
        
        # 訓練 SVM (nu=0.1 表示預期 10% 的樣本為異常值)
        self.svm_up = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        self.svm_down = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        
        if len(latent_up) > 0:
            self.svm_up.fit(latent_up)
        if len(latent_down) > 0:
            self.svm_down.fit(latent_down)
        
        print(f"One-Class SVM 訓練完成 (上漲樣本: {len(latent_up)}, 下跌樣本: {len(latent_down)})")
        print(f"nu=0.1 設定: 預期約 10% 樣本為邊界外，將影響交易頻率")
    
    def svm_predict(self, test_df):
        """
        使用 One-Class SVM 預測
        分別用兩個模型的latent code計算決策分數
        
        Args:
            test_df: 測試資料
            
        Returns:
            predictions: 預測結果
        """
        # 分別取得兩個模型的latent code
        latent_code_up = self.get_latent_code(test_df, model_type='up')
        latent_code_down = self.get_latent_code(test_df, model_type='down')
        
        predictions = []
        for i in range(len(test_df)):
            lc_up = latent_code_up[i].reshape(1, -1)
            lc_down = latent_code_down[i].reshape(1, -1)
            
            # 檢查是否落在對應模型邊界內
            up_score = self.svm_up.decision_function(lc_up)[0] if self.svm_up else -np.inf
            down_score = self.svm_down.decision_function(lc_down)[0] if self.svm_down else -np.inf
            
            # 分數較高者為預測結果
            if up_score > down_score:
                predictions.append(1)
            else:
                predictions.append(-1)
        
        return np.array(predictions)
    
    # ========== 方法四：基因演算法 (GA) ==========
    def genetic_algorithm(self, train_df, population_size=50, generations=30):
        """
        使用基因演算法搜尋最佳 K 線特徵組合
        
        Args:
            train_df: 訓練資料
            population_size: 種群大小
            generations: 世代數
            
        Returns:
            best_pattern: 最佳型態向量
        """
        print("開始執行基因演算法...")
        
        # 初始化種群
        population = self._initialize_population(train_df, population_size)
        
        best_fitness = -np.inf
        best_pattern = None
        
        for gen in range(generations):
            # 計算適應度
            fitness_scores = [self._fitness_function(pattern, train_df) for pattern in population]
            
            # 記錄最佳解
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_pattern = population[max_idx].copy()
            
            # 選擇
            selected = self._selection(population, fitness_scores)
            
            # 交叉與突變
            population = self._crossover_and_mutation(selected)
            
            if (gen + 1) % 10 == 0:
                print(f"Generation {gen+1}/{generations}, Best Fitness: {best_fitness:.4f}")
        
        self.best_ga_pattern = best_pattern
        return best_pattern
    
    def _initialize_population(self, train_df, size):
        """初始化種群"""
        # 從訓練集中隨機選擇大漲或大跌的樣本作為初始種群
        extreme_samples = train_df[(train_df['Label'] == 1) | (train_df['Label'] == -1)]
        
        if len(extreme_samples) < size:
            # 如果樣本不足，進行複製
            indices = np.random.choice(len(extreme_samples), size, replace=True)
        else:
            indices = np.random.choice(len(extreme_samples), size, replace=False)
        
        population = []
        for idx in indices:
            pattern = extreme_samples.iloc[idx][self.feature_columns].values
            population.append(pattern)
        
        return population
    
    def _fitness_function(self, pattern, train_df):
        """
        適應度函數：考慮出現次數、平均獲利、最大回撤
        
        Args:
            pattern: 特徵向量
            train_df: 訓練資料
            
        Returns:
            fitness: 適應度分數
        """
        # 計算與訓練集所有樣本的距離
        X_train = train_df[self.feature_columns].values
        distances = np.array([euclidean(pattern, x) for x in X_train])
        
        # 找出相似的樣本 (距離閾值)
        threshold = np.percentile(distances, 10)  # 前 10% 最相似的
        similar_indices = distances < threshold
        
        if similar_indices.sum() == 0:
            return -1000  # 懲罰沒有相似樣本的情況
        
        similar_samples = train_df[similar_indices]
        
        # 1. 出現次數
        occurrence = len(similar_samples)
        if occurrence < 5:
            return -1000  # 不符合條件
        
        # 2. 平均獲利
        avg_profit = similar_samples['三天後漲跌(%)'].mean()
        if avg_profit < 3:
            return -500  # 不符合條件
        
        # 3. 最大回撤 (簡化計算：使用三天後漲跌的最小值)
        max_drawdown = abs(similar_samples['三天後漲跌(%)'].min())
        if max_drawdown > 3:
            max_drawdown_penalty = -max_drawdown * 10
        else:
            max_drawdown_penalty = 0
        
        # 綜合適應度
        fitness = occurrence * 0.5 + avg_profit * 10 + max_drawdown_penalty
        
        return fitness
    
    def _selection(self, population, fitness_scores):
        """選擇操作：輪盤賭選擇"""
        fitness_scores = np.array(fitness_scores)
        
        # 處理負數適應度
        min_fitness = fitness_scores.min()
        if min_fitness < 0:
            fitness_scores = fitness_scores - min_fitness + 1
        
        # 計算選擇概率
        total_fitness = fitness_scores.sum()
        if total_fitness == 0:
            probabilities = np.ones(len(population)) / len(population)
        else:
            probabilities = fitness_scores / total_fitness
        
        # 選擇
        selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
        selected = [population[i] for i in selected_indices]
        
        return selected
    
    def _crossover_and_mutation(self, selected):
        """交叉與突變操作"""
        new_population = []
        
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i+1] if i+1 < len(selected) else selected[0]
            
            # 單點交叉
            crossover_point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            
            # 突變 (10% 機率)
            if np.random.rand() < 0.1:
                mutation_point = np.random.randint(len(child1))
                child1[mutation_point] += np.random.randn() * 0.1
            
            if np.random.rand() < 0.1:
                mutation_point = np.random.randint(len(child2))
                child2[mutation_point] += np.random.randn() * 0.1
            
            new_population.append(child1)
            new_population.append(child2)
        
        return new_population[:len(selected)]
    
    def ga_predict(self, test_df, similarity_threshold=5.0):
        """
        使用 GA 找到的最佳型態進行預測（修正邏輯：不匹配=觀望，而非做空）
        
        Args:
            test_df: 測試資料
            similarity_threshold: 相似度閾值（歐式距離）
            
        Returns:
            predictions: 預測結果 (1=漲, 0=觀望)
        """
        if not hasattr(self, 'best_ga_pattern'):
            return np.zeros(len(test_df))
        
        X_test = test_df[self.feature_columns].values
        predictions = []
        
        for x in X_test:
            dist = euclidean(self.best_ga_pattern, x)
            
            # 修正邏輯：只有符合最佳型態時才做多
            # 不符合時保持觀望（0），而非做空（-1）
            # 理由：「不是上漲型態」不等於「是下跌型態」
            if dist < similarity_threshold:  # 相似度高，預測上漲
                predictions.append(1)
            else:
                predictions.append(0)  # 不符合型態，觀望（關鍵修正）
        
        return np.array(predictions)


# ==================== Backtester 類別 ====================
class Backtester:
    """
    回測系統：計算報酬率、勝率、最大回撤（支援做多/做空）
    
    【重要警示】存活者偏差 (Survivorship Bias)：
    - 當前回測僅包含目前仍在市場交易的股票
    - 未包含已下市、退市、或被併購的股票
    - 這可能導致回測績效被高估
    - 建議：實際應用時應考慮此偏差，或使用包含已下市股票的歷史資料庫
    """
    
    def __init__(self):
        self.strategy_returns = []  # 儲存策略報酬
        self.benchmark_returns = []  # 儲存基準報酬
    
    def backtest(self, test_df, predictions):
        """
        執行回測（支援做多與做空）
        
        交易策略：
        - predictions == 1: 做多，當日收盤價買入，3天後收盤價賣出
        - predictions == -1: 做空，當日收盤價賣出，3天後收盤價買回
        
        報酬率計算（含交易成本）：
        - 做多: (出場價 - 進場價) / 進場價 - 交易成本
        - 做空: (進場價 - 出場價) / 進場價 - 交易成本
        
        台灣交易成本：
        - 手續費（買進）：0.1425% * 折扣（預設6折 = 0.06%）
        - 手續費（賣出）：0.1425% * 折扣（預設6折 = 0.06%）
        - 證券交易稅（賣出）：0.3%
        - 總成本 = 0.06% + 0.06% + 0.3% = 0.42%
        
        Args:
            test_df: 測試資料
            predictions: 預測結果 (1=做多, -1=做空, 0=不操作)
            
        Returns:
            results: 回測結果字典
        """
        # 台灣證券交易成本設定
        commission_rate = 0.001425  # 手續費公定費率 0.1425%
        commission_discount = 0.6   # 手續費折扣（6折）
        transaction_tax = 0.003     # 證券交易稅 0.3%（賣出時收取）
        
        # 計算實際交易成本
        buy_commission = commission_rate * commission_discount   # 買進手續費
        sell_commission = commission_rate * commission_discount  # 賣出手續費
        total_cost = buy_commission + sell_commission + transaction_tax  # 總交易成本（做多/做空皆適用）
        # 篩選出有交易訊號的樣本（做多或做空）
        trade_signals = (predictions == 1) | (predictions == -1)
        
        if trade_signals.sum() == 0:
            return {
                'annual_return': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'trades': 0,
                'equity_curve': np.array([1.0])
            }
        
        # 準備資料
        test_df = test_df.copy()
        test_df['Prediction'] = predictions
        
        returns = []
        equity_curve = [1.0]  # 初始資金為 1
        
        # 遍歷所有有交易訊號的日期
        for i in range(len(test_df)):
            prediction = predictions[i]
            
            # 跳過無交易訊號
            if prediction == 0:
                continue
            
            # 檢查是否有足夠的未來資料（需要3天後的資料）
            if i + 3 >= len(test_df):
                continue  # 無法取得3天後的資料，跳過此筆交易
            
            # 使用 iloc 安全地取得價格
            entry_price = test_df.iloc[i]['Close']  # 進場價：當日收盤價
            exit_price = test_df.iloc[i + 3]['Close']  # 出場價：3天後收盤價
            
            # 根據預測方向計算報酬率（扣除交易成本）
            if prediction == 1:
                # 做多：買入後賣出
                ret = (exit_price - entry_price) / entry_price - total_cost
            elif prediction == -1:
                # 做空：賣出後買回
                ret = (entry_price - exit_price) / entry_price - total_cost
            else:
                continue
            
            returns.append(ret)
            
            # 更新權益曲線
            equity_curve.append(equity_curve[-1] * (1 + ret))
        
        # 如果沒有任何有效交易
        if len(returns) == 0:
            return {
                'annual_return': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'trades': 0,
                'equity_curve': np.array([1.0])
            }
        
        # 計算績效指標
        returns = np.array(returns)
        equity_curve = np.array(equity_curve)
        
        # 儲存策略報酬（用於統計檢定）
        self.strategy_returns = returns.copy()
        
        # 計算基準報酬（隨機買入）
        self._calculate_benchmark_returns(test_df)
        
        # 1. 總報酬率（轉換為百分比）
        total_return = (equity_curve[-1] - 1) * 100
        
        # 2. 勝率
        win_rate = (returns > 0).sum() / len(returns) * 100
        
        # 3. 最大回撤
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min()) * 100
        
        return {
            'annual_return': total_return,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'trades': len(returns),
            'equity_curve': equity_curve
        }
    
    def _calculate_benchmark_returns(self, test_df):
        """
        計算基準報酬（隨機買入策略）
        含交易成本，以公平比較
        
        Args:
            test_df: 測試資料
        """
        # 台灣證券交易成本設定（與 backtest 相同）
        commission_rate = 0.001425  # 手續費公定費率 0.1425%
        commission_discount = 0.6   # 手續費折扣（6折）
        transaction_tax = 0.003     # 證券交易稅 0.3%（賣出時收取）
        
        # 計算實際交易成本
        buy_commission = commission_rate * commission_discount
        sell_commission = commission_rate * commission_discount
        total_cost = buy_commission + sell_commission + transaction_tax
        
        benchmark_returns = []
        
        # 隨機選擇與策略相同數量的交易
        n_trades = len(self.strategy_returns)
        
        if n_trades == 0 or len(test_df) < 4:
            self.benchmark_returns = np.array([])
            return
        
        # 確保有足夠的數據點
        valid_indices = list(range(len(test_df) - 3))
        
        if len(valid_indices) < n_trades:
            # 允許重複抽樣
            selected_indices = np.random.choice(valid_indices, n_trades, replace=True)
        else:
            selected_indices = np.random.choice(valid_indices, n_trades, replace=False)
        
        for i in selected_indices:
            entry_price = test_df.iloc[i]['Close']
            exit_price = test_df.iloc[i + 3]['Close']
            ret = (exit_price - entry_price) / entry_price - total_cost  # 扣除交易成本
            benchmark_returns.append(ret)
        
        self.benchmark_returns = np.array(benchmark_returns)
    
    # ========== 加分項目 2：統計檢定 ==========
    def statistical_test(self):
        """
        使用 T-test 檢驗策略是否顯著優於隨機買入
        
        Returns:
            dict: 包含 t_statistic, p_value, is_significant, conclusion
        """
        if len(self.strategy_returns) == 0 or len(self.benchmark_returns) == 0:
            return {
                't_statistic': 0,
                'p_value': 1.0,
                'is_significant': False,
                'conclusion': '數據不足，無法進行統計檢定'
            }
        
        # 執行獨立樣本 T 檢定（單尾檢定：策略 > 基準）
        t_stat, p_value = stats.ttest_ind(
            self.strategy_returns, 
            self.benchmark_returns, 
            alternative='greater'
        )
        
        # 判斷顯著性（alpha = 0.05）
        is_significant = p_value < 0.05
        
        # 生成結論
        if is_significant:
            conclusion = f'✓ 策略顯著優於隨機買入 (p={p_value:.4f} < 0.05)'
        else:
            conclusion = f'✗ 策略未顯著優於隨機買入 (p={p_value:.4f} >= 0.05)'
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'conclusion': conclusion,
            'strategy_mean': self.strategy_returns.mean() * 100,
            'benchmark_mean': self.benchmark_returns.mean() * 100
        }


# ==================== Tkinter GUI 應用程式 ====================
class App:
    """主應用程式 GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("K線型態預測系統 - AE + One-Class SVM")
        self.root.geometry("1200x800")
        self.root.minsize(900, 650)
        self.root.configure(bg='#1e1e1e')
        
        self.data_processor = DataProcessor()
        self.model_engine = None
        self.backtester = Backtester()
        
        self.setup_ui()
    
    def setup_ui(self):
        """建立UI介面"""
        # 標題
        title_label = tk.Label(
            self.root, 
            text="K線型態預測系統 - AE + One-Class SVM",
            font=('Arial', 18, 'bold'),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        subtitle_label = tk.Label(
            self.root,
            text="Step 1: 特徵工程 → Step 2: AE 預訓練 → Step 3: OC-SVM 建模 → Step 4: 回測驗證",
            font=('Arial', 10),
            bg='#1e1e1e',
            fg='#888888'
        )
        title_label.pack(pady=(10, 0))
        subtitle_label.pack(pady=(0, 10))
        
        # 輸入區域
        input_frame = tk.Frame(self.root, bg='#2d2d2d')
        input_frame.pack(pady=8, padx=20, fill='x')
        # 設定欄權重，讓空白列吸收多餘空間，文字/輸入框固定寬度
        for col in range(7):
            input_frame.columnconfigure(col, weight=0)
        input_frame.columnconfigure(2, weight=1)  # 提示標籤後的間距列吸收多餘空間
        
        # 股票代碼
        tk.Label(input_frame, text="股票代碼:", bg='#2d2d2d', fg='#ffffff').grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.stock_entry = tk.Entry(input_frame, width=10)
        self.stock_entry.insert(0, "2330")
        self.stock_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        # 提示標籤
        tk.Label(input_frame, text="(台股可直接輸入代碼)", bg='#2d2d2d', fg='#888888', font=('Arial', 8)).grid(row=0, column=2, padx=5, pady=5, sticky='w')
        
        # 訓練開始日期
        tk.Label(input_frame, text="訓練開始:", bg='#2d2d2d', fg='#ffffff').grid(row=0, column=3, padx=5, pady=5, sticky='e')
        self.train_start_entry = tk.Entry(input_frame, width=12)
        self.train_start_entry.insert(0, "2016-01-01")
        self.train_start_entry.grid(row=0, column=4, padx=5, pady=5, sticky='w')
        
        # 訓練結束日期
        tk.Label(input_frame, text="訓練結束:", bg='#2d2d2d', fg='#ffffff').grid(row=0, column=5, padx=5, pady=5, sticky='e')
        self.train_end_entry = tk.Entry(input_frame, width=12)
        self.train_end_entry.insert(0, "2024-12-31")
        self.train_end_entry.grid(row=0, column=6, padx=5, pady=5, sticky='w')
        
        # 測試年份
        tk.Label(input_frame, text="測試年份:", bg='#2d2d2d', fg='#ffffff').grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.test_year_entry = tk.Entry(input_frame, width=8)
        self.test_year_entry.insert(0, "2025")
        self.test_year_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        # 市場股票數量（AE 預訓練用）
        tk.Label(input_frame, text="市場股票數:", bg='#2d2d2d', fg='#ffffff').grid(row=1, column=3, padx=5, pady=5, sticky='e')
        self.market_stocks_var = tk.StringVar(value="50")
        market_stocks_combo = ttk.Combobox(
            input_frame, 
            textvariable=self.market_stocks_var,
            values=["20", "30", "50", "80", "全部(~80)"],
            width=12,
            state='readonly'
        )
        market_stocks_combo.grid(row=1, column=4, padx=5, pady=5, sticky='w')
        tk.Label(input_frame, text="(用於 AE 預訓練)", bg='#2d2d2d', fg='#888888', font=('Arial', 8)).grid(row=1, column=5, padx=5, pady=5, sticky='w')
        
        # 執行按鈕
        self.run_button = tk.Button(
            input_frame,
            text="🚀 開始訓練與回測",
            command=self.run_analysis,
            bg='#0d7377',
            fg='white',
            font=('Arial', 11, 'bold'),
            relief='raised',
            cursor='hand2'
        )
        self.run_button.grid(row=1, column=6, columnspan=1, padx=10, pady=5, sticky='w')
        
        # 進度訊息
        self.status_label = tk.Label(
            self.root,
            text="請輸入參數後點擊開始",
            bg='#1e1e1e',
            fg='#00ff00',
            font=('Arial', 10)
        )
        self.status_label.pack(pady=5)
        
        # 主要內容區域 - 使用 Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, padx=20, fill='both', expand=True)
        
        # 分頁1: 回測結果
        self.result_frame = tk.Frame(self.notebook, bg='#2d2d2d')
        self.notebook.add(self.result_frame, text='📊 回測結果')
        
        # 上方：比較表格
        table_container = tk.Frame(self.result_frame, bg='#2d2d2d')
        table_container.pack(pady=10, padx=10, fill='x')
        
        tk.Label(
            table_container, 
            text="AE + OC-SVM 策略績效", 
            bg='#2d2d2d', 
            fg='#00ff00', 
            font=('Arial', 14, 'bold')
        ).pack(pady=5)
        
        # 使用 Treeview 創建表格
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview", 
                       background="#1e1e1e", 
                       foreground="white", 
                       fieldbackground="#1e1e1e",
                       rowheight=35,
                       font=('Arial', 10))
        style.configure("Treeview.Heading", 
                       background="#0d7377", 
                       foreground="white", 
                       font=('Arial', 10, 'bold'),
                       padding=5)
        style.map('Treeview', background=[('selected', '#0d7377')])
        
        # 創建表格（只有 OC-SVM 方法）
        columns = ('指標', '數值')
        self.result_table = ttk.Treeview(table_container, columns=columns, show='headings', height=5)
        
        # 設定欄位標題和寬度（stretch=True 讓欄位随視窗自動拉伸）
        self.result_table.heading('指標', text='評估指標')
        self.result_table.heading('數值', text='AE + OC-SVM 策略結果')
        
        self.result_table.column('指標', width=250, anchor='w', minwidth=150, stretch=True)
        self.result_table.column('數值', width=350, anchor='center', minwidth=200, stretch=True)
        
        # 加溻動條
        scrollbar = ttk.Scrollbar(table_container, orient='vertical', command=self.result_table.yview)
        self.result_table.configure(yscrollcommand=scrollbar.set)
        self.result_table.pack(side='left', pady=10, padx=(10, 0), fill='x', expand=True)
        scrollbar.pack(side='left', pady=10, padx=(0, 10), fill='y')
        
        # 下方：績效圖表
        self.chart_container = tk.Frame(self.result_frame, bg='#2d2d2d')
        self.chart_container.pack(pady=10, padx=10, fill='both', expand=True)
        
        # 分頁2: K線圖
        self.chart_frame = tk.Frame(self.notebook, bg='#2d2d2d')
        self.notebook.add(self.chart_frame, text='📈 K線圖與訊號')
        
        # 分頁3: Autoencoder Loss
        self.loss_frame = tk.Frame(self.notebook, bg='#2d2d2d')
        self.notebook.add(self.loss_frame, text='🧠 Autoencoder Loss')
    
    def run_analysis(self):
        """執行完整分析流程"""
        # 在新執行緒中執行，避免凍結UI
        thread = threading.Thread(target=self._run_analysis_thread, daemon=True)
        thread.start()
    
    def _run_analysis_thread(self):
        """
        實際執行分析的執行緒
        
        核心流程：
        Step 1: 特徵工程 - 提取 10 種型態特徵 + 基礎量價指標，並進行標準化
        Step 2: AE 預訓練 - 在全市場或績優股歷史資料上訓練，提取 Latent Space
        Step 3: OC-SVM 建模 - 使用編碼後的特徵訓練 OC-SVM
        Step 4: 回測驗證 - 驗證通過 OC-SVM 過濾後的看漲型態，勝率是否顯著提升
        """
        try:
            self.update_status("⏳ 開始執行 AE + OC-SVM 混合策略...")
            
            # 重新設定隨機種子（確保每次訓練都從相同狀態開始）
            if RANDOM_SEED is not None:
                set_random_seed(RANDOM_SEED)
            
            # 讀取參數
            stock_code = self.stock_entry.get()
            train_start = self.train_start_entry.get()
            train_end = self.train_end_entry.get()
            test_year = int(self.test_year_entry.get())
            
            # 讀取市場股票數量
            market_stocks_str = self.market_stocks_var.get()
            if "全部" in market_stocks_str:
                num_market_stocks = len(self.data_processor.market_stocks)
            else:
                num_market_stocks = int(market_stocks_str)
            
            # ========================================================================
            # Step 1: 特徵工程
            # 核心任務: 提取 10 種型態特徵 + 基礎量價指標
            # 注意事項: 務必進行標準化（Scaling），避免股價絕對值影響 AE 訓練
            # ========================================================================
            self.update_status("🔧 [Step 1/4] 特徵工程：提取 10 種型態特徵...")
            
            # 1-1. 抓取市場股票資料（用於 AE 預訓練）
            stock_list = self.data_processor.market_stocks[:num_market_stocks]
            self.update_status(f"📊 準備抓取 {len(stock_list)} 檔市場股票資料用於 AE 預訓練...")
            print(f"\n{'='*60}")
            print(f"開始抓取市場股票資料：共 {len(stock_list)} 檔")
            print(f"股票列表: {', '.join(stock_list[:10])}{'...' if len(stock_list) > 10 else ''}")
            print(f"{'='*60}\n")
            
            all_stock_data = self.data_processor.fetch_multiple_stocks(
                stock_list, 
                train_start, 
                train_end
            )
            
            print(f"\n{'='*60}")
            print(f"✅ 成功抓取 {len(all_stock_data)} 檔股票資料（嘗試 {len(stock_list)} 檔）")
            print(f"成功的股票: {', '.join(list(all_stock_data.keys())[:10])}{'...' if len(all_stock_data) > 10 else ''}")
            print(f"{'='*60}\n")
            
            if len(all_stock_data) == 0:
                raise Exception("無法取得任何股票資料")
            
            # 1-2. 型態篩選與排序（前置作業）
            self.update_status("🔍 進行型態篩選與獲利排序...")
            pattern_selector = PatternSelector(
                self.data_processor.feature_columns,
                min_frequency=3,  # 至少出現3次
                min_avg_return=2.0  # 平均報酬至少2%
            )
            
            # 只使用訓練集資料（避免未來資料洩漏）
            train_stock_data = {}
            for stock, df in all_stock_data.items():
                train_part = df[df.index.year < test_year]
                if len(train_part) > 0:
                    train_stock_data[stock] = train_part
            
            selected_patterns = pattern_selector.find_frequent_patterns(train_stock_data)
            pattern_summary = pattern_selector.get_pattern_summary()
            
            self.update_status(
                f"✅ [Step 1 完成] 找到 {pattern_summary['up_patterns']} 個上漲型態, "
                f"{pattern_summary['down_patterns']} 個下跌型態"
            )
            
            # 1-3. 抓取目標股票資料並計算特徵
            self.update_status(f"📥 抓取目標股票 {stock_code} 資料並計算特徵...")
            data = self.data_processor.fetch_data(
                stock_code, 
                train_start,
                f"{test_year}-12-31"
            )
            
            df = self.data_processor.calculate_features(data)
            train_df, test_df = self.data_processor.split_train_test(df, test_year)
            
            self.update_status(
                f"✅ 特徵工程完成：訓練集 {len(train_df)} 筆, 測試集 {len(test_df)} 筆 "
                f"(特徵已標準化處理)"
            )
            
            # ========================================================================
            # Step 2: AE 預訓練
            # 核心任務: 在全市場或績優股歷史資料上訓練，提取 Latent Space
            # 注意事項: 監控重構誤差（Reconstruction Error），確保特徵沒丟失
            # ========================================================================
            self.update_status("🧠 [Step 2/4] AE 預訓練：在全市場資料上訓練 Autoencoder...")
            
            # 2-1. 初始化模型引擎
            self.model_engine = ModelEngine(self.data_processor.feature_columns)
            
            # 2-2. 合併全市場資料進行 AE 預訓練
            # 使用市場股票的歷史資料來訓練 AE，讓它學習更泛化的型態特徵
            self.update_status("🔄 使用全市場資料訓練 Autoencoder（提取 Latent Space）...")
            
            # 合併所有股票的訓練資料
            all_train_data = pd.concat([df for df in train_stock_data.values()], ignore_index=True)
            self.update_status(f"📊 全市場訓練樣本總數: {len(all_train_data)} 筆")
            
            # 訓練 Autoencoder（上漲型態 & 下跌型態各一個）
            self.model_engine.train_autoencoder(all_train_data, epochs=50)
            
            # 2-3. 監控重構誤差
            if self.model_engine.history_up is not None:
                final_loss_up = self.model_engine.history_up.history['loss'][-1]
                self.update_status(f"✅ 上漲型態 AE 重構誤差 (MSE): {final_loss_up:.6f}")
            
            if self.model_engine.history_down is not None:
                final_loss_down = self.model_engine.history_down.history['loss'][-1]
                self.update_status(f"✅ 下跌型態 AE 重構誤差 (MSE): {final_loss_down:.6f}")
            
            self.update_status("✅ [Step 2 完成] Autoencoder 預訓練完成，Latent Space 提取成功")
            
            # ========================================================================
            # Step 3: OC-SVM 建模
            # 核心任務: 使用編碼後的特徵（Latent Code）訓練 OC-SVM
            # 注意事項: 調整 nu 參數（預期的異常比例），這會直接影響交易頻率
            # ========================================================================
            self.update_status("🤖 [Step 3/4] OC-SVM 建模：使用 Latent Code 訓練...")
            
            self.update_status("🔄 使用編碼後的 Latent Code 訓練 OC-SVM (nu=0.1)...")
            self.model_engine.train_one_class_svm(all_train_data)
            
            self.update_status(
                "✅ [Step 3 完成] OC-SVM 訓練完成 "
                "(nu=0.1 表示預期 10% 異常，將影響交易頻率)"
            )
            
            # ========================================================================
            # Step 4: 回測驗證
            # 核心任務: 驗證通過 OC-SVM 過濾後的看漲型態，勝率是否顯著提升
            # 注意事項: 警惕「存活者偏差」，確保回測包含已退市或下市的股票
            # ========================================================================
            self.update_status("📊 [Step 4/4] 回測驗證：評估 OC-SVM 過濾後的型態勝率...")
            
            # 4-1. OC-SVM 預測
            self.update_status("🔮 使用 OC-SVM 過濾看漲型態...")
            pred_svm = self.model_engine.svm_predict(test_df)
            
            # 4-2. 回測
            self.update_status("📈 執行回測分析...")
            result_svm = self.backtester.backtest(test_df, pred_svm)
            
            # 4-3. 統計檢定（驗證顯著性）
            self.update_status("📊 執行統計檢定（T-test）驗證策略顯著性...")
            stat_test = self.backtester.statistical_test()
            
            # 4-4. 警告存活者偏差
            self.update_status(
                "⚠️ 注意：當前回測基於現存股票，未包含已下市股票 "
                "(存活者偏差可能高估績效)"
            )
            
            # 4-5. 顯示結果（排回主執行緒執行）
            self.root.after(0, lambda r=result_svm, s=stat_test, p=pattern_summary, n=num_market_stocks:
                self.display_results(r, s, p, n))
            
            # 4-6. 繪製圖表（排回主執行緒執行）
            self.root.after(50, lambda: self.plot_kline_chart(test_df, pred_svm))
            self.root.after(100, lambda: self.plot_autoencoder_loss())
            
            self.update_status(
                f"✅ [Step 4 完成] 回測驗證完成！"
                f"OC-SVM 勝率: {result_svm['win_rate']:.2f}%, "
                f"年化報酬: {result_svm['annual_return']:.2f}%"
            )
            
            self.update_status("🎉 ===== 全部流程完成！ =====")
            
        except Exception as e:
            err_msg = str(e)
            self.update_status(f"❌ 錯誤: {err_msg}")
            self.root.after(0, lambda m=err_msg: messagebox.showerror("錯誤", m))
    
    def display_results(self, result_svm, stat_test, pattern_summary, num_market_stocks):
        """顯示 AE + OC-SVM 回測結果"""
        # 清空表格
        for item in self.result_table.get_children():
            self.result_table.delete(item)
        
        # 插入數據到表格（只有 OC-SVM）
        self.result_table.insert('', 'end', values=(
            '一年期總報酬率 (%)',
            f'{result_svm["annual_return"]:.2f}%'
        ), tags=('return',))
        
        self.result_table.insert('', 'end', values=(
            '勝率 (%)',
            f'{result_svm["win_rate"]:.2f}%'
        ), tags=('winrate',))
        
        self.result_table.insert('', 'end', values=(
            '最大回撤 (%)',
            f'{result_svm["max_drawdown"]:.2f}%'
        ), tags=('drawdown',))
        
        self.result_table.insert('', 'end', values=(
            '交易次數',
            f'{result_svm["trades"]}'
        ), tags=('trades',))
        
        sig_label = '✅ 顯著' if stat_test['is_significant'] else '⚠️ 不顯著'
        self.result_table.insert('', 'end', values=(
            f'統計檢定 (P-value={stat_test["p_value"]:.4f})',
            f'{sig_label} — {stat_test["conclusion"]}'
        ), tags=('stat',))
        
        # 繪製圖表
        self.plot_result_chart(result_svm, stat_test, pattern_summary, num_market_stocks)
    
    def _get_fig_size(self, container, fallback_w=12, fallback_h=8, dpi=96):
        """根據容器的實際像素寬高換算 matplotlib figure 的英寸大小"""
        container.update_idletasks()
        w_px = container.winfo_width()
        h_px = container.winfo_height()
        # 若視窗還未完成佈局，winfo 可能回傳 1，使用 fallback
        w_in = max(w_px / dpi, fallback_w) if w_px > 1 else fallback_w
        h_in = max(h_px / dpi, fallback_h) if h_px > 1 else fallback_h
        return w_in, h_in

    def plot_result_chart(self, result_svm, stat_test, pattern_summary, num_market_stocks):
        """繪製 AE + OC-SVM 結果圖表：績效指標 + 統計檢定 + 型態篩選摘要"""
        for widget in self.chart_container.winfo_children():
            widget.destroy()
        plt.close('all')

        w, h = self._get_fig_size(self.chart_container, fallback_w=13, fallback_h=8)
        fig = plt.figure(figsize=(w, h), facecolor='#2d2d2d')
        gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35,
                              top=0.95, bottom=0.05, left=0.06, right=0.97)

        # 圖1: 一年期總報酬率
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#1e1e1e')
        val = result_svm['annual_return']
        bar_color = '#4caf50' if val >= 0 else '#f44336'
        ax1.bar(['OC-SVM'], [val], color=bar_color, alpha=0.85, edgecolor='white', linewidth=1.5)
        ax1.set_title('一年期總報酬率 (%)', color='white', fontsize=11, fontweight='bold')
        ax1.set_ylabel('報酬率 (%)', color='white', fontsize=9)
        ax1.tick_params(colors='white', labelsize=8)
        ax1.grid(True, alpha=0.2, color='gray', axis='y')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax1.text(0, val, f'{val:.2f}%', ha='center',
                va='bottom' if val >= 0 else 'top',
                color='white', fontweight='bold', fontsize=12)

        # 圖2: 勝率
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('#1e1e1e')
        wr = result_svm['win_rate']
        ax2.bar(['OC-SVM'], [wr], color='#2196f3', alpha=0.85, edgecolor='white', linewidth=1.5)
        ax2.set_title('勝率 (%)', color='white', fontsize=11, fontweight='bold')
        ax2.set_ylabel('勝率 (%)', color='white', fontsize=9)
        ax2.tick_params(colors='white', labelsize=8)
        ax2.grid(True, alpha=0.2, color='gray', axis='y')
        ax2.axhline(y=50, color='yellow', linestyle='--', linewidth=1, alpha=0.7, label='50%基準')
        ax2.set_ylim(0, 100)
        ax2.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white', fontsize=8)
        ax2.text(0, wr, f'{wr:.2f}%', ha='center', va='bottom',
                color='white', fontweight='bold', fontsize=12)

        # 圖3: 最大回撤
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_facecolor('#1e1e1e')
        dd = result_svm['max_drawdown']
        ax3.bar(['OC-SVM'], [dd], color='#e91e63', alpha=0.85, edgecolor='white', linewidth=1.5)
        ax3.set_title('最大回撤 (%)', color='white', fontsize=11, fontweight='bold')
        ax3.set_ylabel('回撤 (%)', color='white', fontsize=9)
        ax3.tick_params(colors='white', labelsize=8)
        ax3.grid(True, alpha=0.2, color='gray', axis='y')
        ax3.text(0, dd, f'{dd:.2f}%', ha='center', va='bottom',
                color='white', fontweight='bold', fontsize=12)

        # 圖4: 統計檢定（中間整行）
        ax4 = fig.add_subplot(gs[1, :])
        ax4.set_facecolor('#1e1e1e')
        ax4.axis('off')
        ax4.text(0.5, 0.97, '統計檢定 (T-test)：AE + OC-SVM 策略 vs 隨機買入',
                ha='center', va='top', color='#00ff00', fontsize=12, fontweight='bold',
                transform=ax4.transAxes)
        info_texts = [
            f"策略平均報酬: {stat_test['strategy_mean']:.3f}%",
            f"隨機平均報酬: {stat_test['benchmark_mean']:.3f}%",
            f"T-statistic: {stat_test['t_statistic']:.4f}",
            f"P-value: {stat_test['p_value']:.4f}",
            f"結論: {stat_test['conclusion']}"
        ]
        for i, text in enumerate(info_texts):
            c = '#00ff00' if (i == len(info_texts)-1 and stat_test['is_significant']) else \
                '#ff9800' if (i == len(info_texts)-1) else 'white'
            ax4.text(0.5, 0.75 - i * 0.14, text, ha='center', va='top', color=c,
                    fontsize=10, fontweight='bold' if i == len(info_texts)-1 else 'normal',
                    transform=ax4.transAxes)
        ax4.text(0.5, 0.02, '註：P-value < 0.05 表示策略顯著優於隨機買入（95%信心水準）',
                ha='center', va='bottom', color='#888888', fontsize=8, style='italic',
                transform=ax4.transAxes)

        # 圖5: 型態篩選摘要（最下方整行）
        ax5 = fig.add_subplot(gs[2, :])
        ax5.set_facecolor('#1e1e1e')
        ax5.axis('off')
        ax5.text(0.5, 0.97, f'【Step 1 特徵工程】型態篩選結果 ({num_market_stocks} 檔市場股票)',
                ha='center', va='top', color='#ff9800', fontsize=12, fontweight='bold',
                transform=ax5.transAxes)
        pattern_texts = [
            f"✓ 上漲型態: {pattern_summary['up_patterns']} 個（出現次數≥3, 平均報酬≥2%）",
            f"✓ 下跌型態: {pattern_summary['down_patterns']} 個（出現次數≥3, 平均報酬≤-2%）",
        ]
        for i, text in enumerate(pattern_texts):
            ax5.text(0.5, 0.68 - i * 0.28, text, ha='center', va='top',
                    color='#4caf50', fontsize=10, fontweight='bold', transform=ax5.transAxes)
        if pattern_summary['top_up']:
            top_up = pattern_summary['top_up'][0]
            ax5.text(0.5, 0.12,
                    f"TOP 上漲型態：頻率={top_up['frequency']}, "
                    f"平均報酬={top_up['avg_return']:.2f}%, 出現於 {len(top_up['stocks'])} 檔股票",
                    ha='center', va='top', color='#888888', fontsize=8, style='italic',
                    transform=ax5.transAxes)

        canvas = FigureCanvasTkAgg(fig, master=self.chart_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def plot_kline_chart(self, test_df, predictions):
        """繪製 K 線圖並標記買賣點"""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        plt.close('all')

        w, h = self._get_fig_size(self.chart_frame, fallback_w=12, fallback_h=5)
        fig, ax = plt.subplots(figsize=(w, h), facecolor='#2d2d2d')
        fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.12)
        ax.set_facecolor('#1e1e1e')

        dates = test_df.index
        closes = test_df['Close'].values
        ax.plot(dates, closes, color='#00bcd4', linewidth=1.5, label='收盤價')

        buy_signals = predictions == 1
        buy_dates = test_df[buy_signals].index
        buy_prices = test_df[buy_signals]['Close'].values
        ax.scatter(buy_dates, buy_prices, color='#4caf50', marker='^', s=80, label='買入', zorder=5)

        for i, date in enumerate(buy_dates):
            try:
                sell_date_idx = test_df.index.get_loc(date) + 3
                if sell_date_idx < len(test_df):
                    sell_date = test_df.index[sell_date_idx]
                    sell_price = test_df.iloc[sell_date_idx]['Close']
                    ax.scatter(sell_date, sell_price, color='#f44336', marker='v', s=80, zorder=5)
            except:
                pass

        ax.set_title('K線圖與買賣訊號 (OC-SVM 預測)', color='white', fontsize=13)
        ax.set_xlabel('日期', color='white', fontsize=9)
        ax.set_ylabel('價格', color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        ax.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white', fontsize=9)
        ax.grid(True, alpha=0.3, color='gray')

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def plot_autoencoder_loss(self):
        """繪製兩個 Autoencoder 的 Loss 收斂圖"""
        if self.model_engine is None:
            return

        for widget in self.loss_frame.winfo_children():
            widget.destroy()
        plt.close('all')

        w, h = self._get_fig_size(self.loss_frame, fallback_w=12, fallback_h=5)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w, h), facecolor='#2d2d2d')
        fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.12, wspace=0.3)

        if self.model_engine.history_up is not None:
            ax1.set_facecolor('#1e1e1e')
            history_up = self.model_engine.history_up.history
            ax1.plot(history_up['loss'], color='#4caf50', linewidth=2, label='Training Loss')
            if 'val_loss' in history_up:
                ax1.plot(history_up['val_loss'], color='#81c784', linewidth=2, label='Validation Loss')
            ax1.set_title('上漲型態 Autoencoder Loss', color='white', fontsize=11, fontweight='bold')
            ax1.set_xlabel('Epoch', color='white', fontsize=9)
            ax1.set_ylabel('Loss (MSE)', color='white', fontsize=9)
            ax1.tick_params(colors='white', labelsize=8)
            ax1.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white', fontsize=9)
            ax1.grid(True, alpha=0.3, color='gray')

        if self.model_engine.history_down is not None:
            ax2.set_facecolor('#1e1e1e')
            history_down = self.model_engine.history_down.history
            ax2.plot(history_down['loss'], color='#f44336', linewidth=2, label='Training Loss')
            if 'val_loss' in history_down:
                ax2.plot(history_down['val_loss'], color='#e57373', linewidth=2, label='Validation Loss')
            ax2.set_title('下跌型態 Autoencoder Loss', color='white', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Epoch', color='white', fontsize=9)
            ax2.set_ylabel('Loss (MSE)', color='white', fontsize=9)
            ax2.tick_params(colors='white', labelsize=8)
            ax2.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white', fontsize=9)
            ax2.grid(True, alpha=0.3, color='gray')

        canvas = FigureCanvasTkAgg(fig, master=self.loss_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def update_status(self, message):
        """更新狀態訊息（線程安全）"""
        print(message)
        self.root.after(0, lambda msg=message: self.status_label.config(text=msg))


# ==================== 主程式進入點 ====================
if __name__ == "__main__":
    # 設定隨機種子
    set_random_seed(RANDOM_SEED)
    
    root = tk.Tk()
    app = App(root)
    root.mainloop()
