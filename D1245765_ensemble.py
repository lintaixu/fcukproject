"""
K線型態預測系統 - 使用 Autoencoder + One-Class SVM + 基因演算法
資料探勘期末作業
學號: D1245765

【核心流程】
Step 1: 特徵工程
    - 提取 10 種型態特徵 + 基礎量價指標
    - 注意事項：務必進行標準化（Scaling），避免股價絕對值影響 AE 訓練

Step 2: AE 預訓練
    - 在同產業股票歷史資料上訓練，提取 Latent Space
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

# 關閉 TensorFlow oneDNN 訊息，避免終端顯示大量非錯誤提示
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

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
    
    # 滑動視窗天數（單日特徵 → N日K線組合特徵）
    WINDOW_SIZE = 3

    def __init__(self):
        # 基礎 10 種單日特徵 + 第 11 種相對成交量特徵
        self._base_feature_columns = [
            '上影線(%)', '下影線(%)', '實體線(%)',
            '前一日上影線(%)', '前一日下影線(%)', '前一日實體線(%)',
            '開盤型態(%)', '收盤型態(%)', '成交量突變率', '前五日趨勢',
            '相對成交量'  # 第 11 特徵：當日成交量 / 5日均量
        ]
        # 滑動視窗後展開的最終特徵欄位（統計聚合：11特徵 × 2統計量 = 22維）
        # 會在 _apply_sliding_window 中動態生成
        self.feature_columns = []  # 暫時為空，待計算特徵時再設定
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
        
        # 產業分類對照表（TWSE/TPEx 產業代碼 → 產業名稱）
        self.industry_name_map = {
            1: '水泥工業', 2: '食品工業', 3: '塑膠工業', 4: '紡織纖維',
            5: '電機機械', 6: '電器電纜', 8: '玻璃陶瓷', 9: '造紙工業',
            10: '鋼鐵工業', 11: '橡膠工業', 12: '汽車工業', 14: '建材營造',
            15: '航運業', 16: '觀光餐旅', 17: '金融保險', 18: '貿易百貨',
            20: '其他電子業', 21: '化學工業', 22: '電腦及週邊設備', 23: '光電業',
            24: '半導體業', 25: '通信網路業', 26: '電子通路業', 27: '資訊服務業',
            28: '電子零組件業', 29: '其他', 30: '文化創意業', 31: '生技醫療業',
            32: '油電燃氣業', 33: '農業科技業', 35: '數位雲端', 36: '運動休閒',
            37: '居家生活', 38: '綠能環保', 91: '存託憑證'
        }

    def get_same_industry_stocks(self, stock_code, csv_path=None, max_stocks=50):
        """
        從 taiwan_stock_categories.csv 中找出與目標股票相同產業的股票列表，
        並依近期平均成交量（流動性）篩選前 max_stocks 檔。
        
        Args:
            stock_code: 目標股票代碼（如 '2330' 或 '2330.TW'）
            csv_path: CSV 檔案路徑（預設自動尋找）
            max_stocks: 最多返回幾檔（依流動性排序）
            
        Returns:
            (stock_list, industry_name): 同產業股票代碼列表 & 產業名稱
        """
        # 自動尋找 CSV 路徑
        if csv_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, 'taiwan_stock_categories.csv')
        
        if not os.path.exists(csv_path):
            print(f"⚠️  找不到 {csv_path}，改用預設股票清單")
            return self.market_stocks, '全市場（CSV未找到）'
        
        try:
            df = pd.read_csv(csv_path)
            clean_code = str(stock_code).replace('.TW', '').strip()
            df['股票代號'] = df['股票代號'].astype(str)
            
            target_row = df[df['股票代號'] == clean_code]
            if target_row.empty:
                print(f"⚠️  CSV 中找不到 {clean_code}，改用預設股票清單")
                return self.market_stocks, '全市場（股票代號未找到）'
            
            industry_code = target_row.iloc[0]['產業分類']
            industry_name = self.industry_name_map.get(int(industry_code), f'產業代碼{industry_code}')
            
            # 取得同產業所有股票（排除目標股票本身）
            same_industry = df[
                (df['產業分類'] == industry_code) &
                (df['股票代號'] != clean_code)
            ]['股票代號'].tolist()
            
            if len(same_industry) == 0:
                print(f"⚠️  {industry_name} 中無其他股票，改用預設股票清單")
                return self.market_stocks, industry_name
            
            print(f"✅ 目標股票：{clean_code}  產業：{industry_name}（代碼 {industry_code}）")
            print(f"   同產業共 {len(same_industry)} 檔，開始依流動性排序（抓取近 20 日均量）...")
            
            # ── 流動性篩選：依近 20 日平均成交量排序，取前 max_stocks 檔 ──
            # 建立代碼→市場別映射（決定使用 .TW 或 .TWO）
            market_map = dict(zip(df['股票代號'].astype(str), df['市場別']))
            liquidity = []
            sample_codes = same_industry[:min(len(same_industry), max_stocks * 3)]  # 抓 3 倍候選數量
            for code in sample_codes:
                try:
                    suffix = '.TWO' if market_map.get(code) == '上櫃' else '.TW'
                    ticker = yf.Ticker(f"{code}{suffix}")
                    hist = ticker.history(period='1mo')
                    if not hist.empty and 'Volume' in hist.columns:
                        avg_vol = hist['Volume'].mean()
                        avg_close = hist['Close'].mean() if 'Close' in hist.columns else 1
                        # 成交值 = 成交量 × 平均股價（更能反映流動性）
                        avg_value = avg_vol * avg_close
                        liquidity.append((code, avg_value))
                except Exception:
                    pass
            
            if liquidity:
                # 依成交值由大到小排序，取前 max_stocks 檔
                liquidity.sort(key=lambda x: x[1], reverse=True)
                filtered = [code for code, _ in liquidity[:max_stocks]]
                print(f"   流動性篩選後保留 {len(filtered)} 檔（成交值前段）")
                print(f"   前5檔：{filtered[:5]}")
            else:
                # 無法取得流動性資料時，直接截取前 max_stocks 檔
                filtered = same_industry[:max_stocks]
                print(f"   無法取得流動性資料，直接取前 {len(filtered)} 檔")
            
            return filtered, industry_name
            
        except Exception as e:
            print(f"⚠️  讀取 CSV 失敗 ({e})，改用預設股票清單")
            return self.market_stocks, '全市場（CSV讀取失敗）'

    def _resolve_tw_ticker(self, code):
        """
        對純數字台股代碼，優先查 CSV 確認市場別。
        上市 (.TW)；上櫃 (.TWO)。
        若 CSV 不含此代碼，則先試 .TW、再試 .TWO。
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'taiwan_stock_categories.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df['股票代號'] = df['股票代號'].astype(str)
                row = df[df['股票代號'] == code]
                if not row.empty:
                    market = row.iloc[0]['市場別']
                    return f"{code}.TWO" if market == '上櫃' else f"{code}.TW"
            except Exception:
                pass
        # Fallback: try .TW first; if empty, use .TWO
        try:
            hist = yf.Ticker(f"{code}.TW").history(period='5d')
            if not hist.empty:
                return f"{code}.TW"
        except Exception:
            pass
        return f"{code}.TWO"

    def fetch_multiple_stocks(self, stock_codes, start_date, end_date):
        """
        並行抓取多檔股票資料（ThreadPoolExecutor）
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        all_data = {}
        total = len(stock_codes)
        
        def _fetch_one(stock_code):
            try:
                data = self.fetch_data(stock_code, start_date, end_date)
                if not data.empty:
                    df = self.calculate_features(data)
                    if not df.empty:
                        return stock_code, df
            except Exception as e:
                print(f"✗ {stock_code} 處理失敗: {str(e)}")
            return stock_code, None
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_fetch_one, code): code for code in stock_codes}
            done = 0
            for future in as_completed(futures):
                done += 1
                code, df = future.result()
                if df is not None:
                    all_data[code] = df
                    print(f"[{done}/{total}] ✓ {code} 完成，共 {len(df)} 筆")
                else:
                    print(f"[{done}/{total}] ✗ {code} 跟鹈失敗")
        
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
            # 智能處理股票代碼：純數字自動加上台股後綴
            stock_code = stock_code.strip()
            if stock_code.isdigit():
                stock_code = self._resolve_tw_ticker(stock_code)
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
        計算 11 個 K 線特徵，並透過滑動視窗（Sliding Window）
        將單日特徵轉為 N 日 K 線組合特徵，捕捉時間序列慣性。

        基礎特徵（11 種）：
            1~3.  上/下影線/實體線(%)
            4~6.  前一日上/下影線/實體線(%)
            7.    開盤型態(%)（跳空程度）
            8.    收盤型態(%)（日內漲跌）
            9.    成交量突變率
            10.   前五日趨勢
            11.   相對成交量（當日量 / 5日均量，反映量能異常）

        最終特徵維度：WINDOW_SIZE × 11
        """
        df = data.copy()
        
        df['Close'] = df['Close'].replace(0, 1e-10)
        df['Volume'] = df['Volume'].replace(0, 1e-10)
        
        # 1. 上影線(%)
        df['上影線(%)'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close'] * 100
        # 2. 下影線(%)
        df['下影線(%)'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close'] * 100
        # 3. 實體線(%)
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
        volume_ma5 = df['Volume'].rolling(window=5).mean().replace(0, 1e-10)
        df['成交量突變率'] = (df['Volume'] - volume_ma5) / df['Volume']
        # 10. 前五日趨勢
        close_t7 = df['Close'].shift(7).replace(0, 1e-10)
        df['前五日趨勢'] = (df['Close'].shift(2) - close_t7) / close_t7
        # 11. 相對成交量（第 11 特徵：當日量 / 5日均量）
        df['相對成交量'] = df['Volume'] / volume_ma5
        
        # 標籤：三天後漲跌
        df['三天後漲跌(%)'] = ((df['Close'].shift(-3) - df['Close']) / df['Close']) * 100
        df['Label'] = 0
        df.loc[df['三天後漲跌(%)'] > 3, 'Label'] = 1
        df.loc[df['三天後漲跌(%)'] < -3, 'Label'] = -1
        
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # ── 滑動視窗展開（WINDOW_SIZE 日 × 11 特徵）──
        df = self._apply_sliding_window(df)
        
        return df

    def _apply_sliding_window(self, df):
        """
        將 WINDOW_SIZE 天的特徵直接攤平（Flatten），保留嚴格的時間序列順序。
        
        方法：將 N 天（例如 3 天）的特徵攤平成向量 [T_{-2}, T_{-1}, T_0]
        - 3 天 × 11 個基礎特徵 = 33 維向量
        - 完整保留時序信息，避免使用統計聚合造成的信息損失
        """
        W = self.WINDOW_SIZE  # 3
        base_cols = self._base_feature_columns  # 11 個基礎特徵
        
        # 建立 3 天的滑動視窗資料（d0=今天, d1=昨天, d2=前天）
        flatten_data = {}
        for d in range(W - 1, -1, -1):  # 從前天到今天，保持順序 [T-2, T-1, T0]
            for col in base_cols:
                # 命名格式：特徵名_T-2, 特徵名_T-1, 特徵名_T0
                flatten_data[f'{col}_T-{d}'] = df[col].shift(d)
        
        flatten_df = pd.DataFrame(flatten_data, index=df.index)
        
        # 更新 feature_columns（產生 3 × 11 = 33 個特徵，嚴格保持時序）
        self.feature_columns = []
        for d in range(W - 1, -1, -1):
            for col in base_cols:
                self.feature_columns.append(f'{col}_T-{d}')
        
        # 保留標籤欄位與回測需要的OHLCV欄位
        keep_cols = ['Open', 'High', 'Low', 'Close', 'Volume', '三天後漲跌(%)', 'Label']
        result = pd.concat([df[keep_cols], flatten_df], axis=1)
        result = result.replace([np.inf, -np.inf], np.nan).dropna()
        
        return result

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
        # 儲存屬於 DBSCAN 群組（非噪音）的樣本索引 (stock, date)
        self._clustered_up_keys = set()
        self._clustered_down_keys = set()
    
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
            up_samples   = train_df[train_df['Label'] == 1]
            down_samples = train_df[train_df['Label'] == -1]
            
            # 向量化：用 numpy 取得特徵矩陣，避免 iterrows()
            if len(up_samples) > 0:
                feats = up_samples[self.feature_columns].values
                rets  = up_samples['三天後漲跌(%)'].values
                dates = up_samples.index.tolist()
                for j in range(len(up_samples)):
                    all_up_patterns.append({'features': feats[j], 'return': rets[j],
                                            'stock': stock_code, 'date': dates[j]})
            if len(down_samples) > 0:
                feats = down_samples[self.feature_columns].values
                rets  = down_samples['三天後漲跌(%)'].values
                dates = down_samples.index.tolist()
                for j in range(len(down_samples)):
                    all_down_patterns.append({'features': feats[j], 'return': rets[j],
                                              'stock': stock_code, 'date': dates[j]})
        
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
        from sklearn.decomposition import PCA
        
        features = np.array([p['features'] for p in patterns])
        
        # 標準化特徵
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # ── 加入 PCA 降維以解決維度詛咒問題 ──────────────────────
        # 將 33 維特徵降至 5 維，讓 DBSCAN 在低維空間中更有效
        n_components = min(5, features_scaled.shape[0], features_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        features_pca = pca.fit_transform(features_scaled)
        
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"  PCA 降維: {features_scaled.shape[1]}維 → {n_components}維 "
              f"(保留 {explained_var*100:.1f}% 變異量)")
        
        # DBSCAN 聚類（在降維後的空間中運行，eps 可以設定較小值）
        clustering = DBSCAN(eps=1.2, min_samples=self.min_frequency).fit(features_pca)
        labels = clustering.labels_
        
        # 記錄屬於群組的樣本（非噪音），供後續篩選訓練資料
        clustered_keys = set()
        for i, lbl in enumerate(labels):
            if lbl != -1:
                clustered_keys.add((patterns[i]['stock'], patterns[i]['date']))
        if pattern_type == 'up':
            self._clustered_up_keys = clustered_keys
        else:
            self._clustered_down_keys = clustered_keys
        
        n_noise = int((labels == -1).sum())
        n_clustered = len(labels) - n_noise
        print(f"  DBSCAN ({pattern_type}): {n_clustered} 樣本屬於群組, {n_noise} 噪音點")
        
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
            'top_down': self.selected_patterns['down'][:5] if self.selected_patterns['down'] else [],
            'clustered_up_count': len(self._clustered_up_keys),
            'clustered_down_count': len(self._clustered_down_keys),
        }
        return summary
    
    def filter_training_data(self, train_stock_data):
        """
        用 DBSCAN 聚類結果過濾訓練資料，只保留屬於型態群組的高品質樣本。
        將 Step 1 的篩選結果傳遞給 Step 2 / Step 3。
        
        Args:
            train_stock_data: dict {stock_code: train_df}
        Returns:
            DataFrame: 只含屬於聚類群組的 Label=1/-1 樣本
        """
        all_keys = self._clustered_up_keys | self._clustered_down_keys
        filtered_dfs = []
        
        for stock, df in train_stock_data.items():
            # 用 index intersection 取代嵌套 for loop
            stock_dates = {date for (s, date) in all_keys if s == stock}
            if stock_dates:
                matched_idx = df.index.intersection(list(stock_dates))
                if len(matched_idx) > 0:
                    filtered_dfs.append(df.loc[matched_idx])
        
        if filtered_dfs:
            result = pd.concat(filtered_dfs, ignore_index=True)
            print(f"✓ 型態篩選後訓練樣本: {len(result)} 筆 "
                  f"(上漲 {len(self._clustered_up_keys)}, 下跌 {len(self._clustered_down_keys)})")
            return result
        
        return pd.DataFrame()


# ==================== ModelEngine 類別 ====================
class ModelEngine:
    """包含 Autoencoder, OneClassSVM, GA 的核心演算法"""
    
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns
        # 分別的特徵標準化器
        self.scaler_up = StandardScaler()    # 看漲型態專用標準化器
        self.scaler_down = StandardScaler()  # 看跌型態專用標準化器
        
        # 分別的 Autoencoder（看漲和看跌型態各自專用）
        self.autoencoder_up = None    # 看漲型態專用AE
        self.encoder_up = None
        self.history_up = None
        
        self.autoencoder_down = None  # 看跌型態專用AE  
        self.encoder_down = None
        self.history_down = None
        
        # 在各自 Latent Space 上的 OC-SVM
        self.svm_up = None
        self.svm_down = None
        
        # Latent Code 標準化器（各自在自己的潛在空間）
        self.latent_scaler_up = StandardScaler()   # 看漲型態Latent標準化器
        self.latent_scaler_down = StandardScaler() # 看跌型態Latent標準化器
    
    # ========== Autoencoder 建立與訓練 ==========
    def build_autoencoder(self, input_dim=22):
        """
        建立 Autoencoder 模型
        架構: Input -> Dense(16) -> Dense(4, latent) -> Dense(16) -> Output
        Latent 維度固定 4：壓縮至 4 維以保留時間序列慣性特徵
        
        Args:
            input_dim: 輸入維度（統計聚合版 = 11特徵 × 2統計量 = 22）
            
        Returns:
            autoencoder, encoder
        """
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(16, activation='relu')(input_layer)
        encoded = layers.Dense(4, activation='relu', name='latent_code')(encoded)
        
        # Decoder
        decoded = layers.Dense(16, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # 完整的 Autoencoder
        autoencoder = keras.Model(input_layer, decoded)
        
        # Encoder (用於提取 Latent Code)
        encoder = keras.Model(input_layer, encoded)
        
        return autoencoder, encoder
    
    def train_autoencoder(self, train_df, epochs=50, batch_size=32):
        """
        分別訓練看漲和看跌的 Autoencoder
        
        新架構：
        1. 為看漲型態訓練專用的 AE（學習看漲特徵表示）
        2. 為看跌型態訓練專用的 AE（學習看跌特徵表示）
        3. 在各自的 Latent Space 上訓練對應的 OC-SVM
        
        Args:
            train_df: 訓練資料（含 Label 欄位）
            epochs: 訓練輪數
            batch_size: 批次大小
        """
        # 分別收集看漲和看跌樣本
        up_samples = train_df[train_df['Label'] == 1][self.feature_columns].values
        down_samples = train_df[train_df['Label'] == -1][self.feature_columns].values
        
        print(f"分別訓練 AE：看漲 {len(up_samples)} 樣本, 看跌 {len(down_samples)} 樣本")
        
        # ========== 訓練看漲型態 Autoencoder ==========
        if len(up_samples) >= 10:
            print("🔺 訓練看漲型態 Autoencoder...")
            X_up_scaled = self.scaler_up.fit_transform(up_samples)
            
            # 建立看漲專用 AE
            self.autoencoder_up, self.encoder_up = self.build_autoencoder(len(self.feature_columns))
            self.autoencoder_up.compile(optimizer='adam', loss='mse')
            
            # 訓練看漲 AE
            history_up = self.autoencoder_up.fit(
                X_up_scaled, X_up_scaled,
                epochs=epochs,
                batch_size=min(batch_size, len(X_up_scaled)),
                verbose=0
            )
            
            self.history_up = history_up
            train_loss_up = history_up.history['loss'][-1]
            print(f"  ✓ 看漲 AE 訓練完成 (損失: {train_loss_up:.6f})")
        else:
            print("  ⚠️ 看漲樣本不足，跳過訓練")
        
        # ========== 訓練看跌型態 Autoencoder ==========
        if len(down_samples) >= 10:
            print("🔻 訓練看跌型態 Autoencoder...")
            X_down_scaled = self.scaler_down.fit_transform(down_samples)
            
            # 建立看跌專用 AE
            self.autoencoder_down, self.encoder_down = self.build_autoencoder(len(self.feature_columns))
            self.autoencoder_down.compile(optimizer='adam', loss='mse')
            
            # 訓練看跌 AE
            history_down = self.autoencoder_down.fit(
                X_down_scaled, X_down_scaled,
                epochs=epochs,
                batch_size=min(batch_size, len(X_down_scaled)),
                verbose=0
            )
            
            self.history_down = history_down
            train_loss_down = history_down.history['loss'][-1]
            print(f"  ✓ 看跌 AE 訓練完成 (損失: {train_loss_down:.6f})")
        else:
            print("  ⚠️ 看跌樣本不足，跳過訓練")
        
        print("✅ 分別訓練完成，各型態擁有專用的特徵表示空間")
    
    def _train_autoencoder_legacy(self, train_df, epochs=50, batch_size=32):
        """
        舊版：訓練兩個獨立的 Autoencoder（已廢棄，保留供參考）
        問題：兩個獨立的 latent space 導致 OC-SVM 分數不可比
        """
        # 分別取出大漲和大跌的樣本
        up_samples   = train_df[train_df['Label'] == 1][self.feature_columns].values
        down_samples = train_df[train_df['Label'] == -1][self.feature_columns].values
        
        print(f"[舊版] 對比損失 AE 訓練：上漲樣本 {len(up_samples)}, 下跌樣本 {len(down_samples)}")
        print("  警告：此方法已廢棄，請使用統一 AE 架構")
        
        # 後續代碼省略...
        pass
    
    def get_latent_code(self, df, pattern_type='auto'):
        """
        取得 Latent Code (4維壓縮特徵)
        根據型態類型使用對應的編碼器
        
        Args:
            df: 資料
            pattern_type: 'up'(看漲), 'down'(看跌), 'auto'(自動判斷)
            
        Returns:
            latent_code: 4維特徵向量
        """
        X = df[self.feature_columns].values
        
        # 自動判斷型態類型
        if pattern_type == 'auto':
            if 'Label' in df.columns:
                up_count = (df['Label'] == 1).sum()
                down_count = (df['Label'] == -1).sum()
                pattern_type = 'up' if up_count >= down_count else 'down'
            else:
                # 無標籤時預設使用看漲編碼器
                pattern_type = 'up'
        
        # 選擇對應的編碼器和標準化器
        if pattern_type == 'up':
            encoder = self.encoder_up
            scaler = self.scaler_up
            model_name = "看漲"
        elif pattern_type == 'down':
            encoder = self.encoder_down
            scaler = self.scaler_down
            model_name = "看跌"
        else:
            raise ValueError(f"不支援的型態類型: {pattern_type}")
        
        if encoder is None:
            print(f"警告：{model_name}模型尚未訓練，返回零向量")
            return np.zeros((len(df), 4))
        
        # 使用對應的標準化器轉換
        X_scaled = scaler.transform(X)
        latent_code = encoder.predict(X_scaled, verbose=0)
        
        return latent_code
    
    def get_reconstruction_error(self, df):
        """
        計算重建誤差（MSE）
        使用統一的 autoencoder
        
        Args:
            df: 資料
        
        Returns:
            reconstruction_error: 重建誤差陣列
        """
        X = df[self.feature_columns].values
        
        if self.autoencoder is None:
            print("警告：模型尚未訓練，返回零向量")
            return np.zeros(len(df))
        
        # 使用統一的 scaler 轉換
        X_scaled = self.scaler.transform(X)
        reconstructed = self.autoencoder.predict(X_scaled, verbose=0)
        
        # 計算 MSE（每個樣本的平均平方誤差）
        error = np.mean((X_scaled - reconstructed) ** 2, axis=1)
        
        return latent_code
    
    # ========== One-Class SVM 訓練與預測 ==========
    def train_one_class_svm(self, train_df):
        """
        訓練兩個 One-Class SVM (一個學習上漲、一個學習下跌)
        使用各自專用 Autoencoder 的 Latent Code
        
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
        
        # 使用各自專用編碼器取得 Latent Code
        latent_up = self.get_latent_code(up_samples, pattern_type='up') if len(up_samples) > 0 else np.array([])
        latent_down = self.get_latent_code(down_samples, pattern_type='down') if len(down_samples) > 0 else np.array([])
        
        # 訓練 SVM (nu=0.1 表示預期 10% 的樣本為異常值)
        self.svm_up = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        self.svm_down = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
        
        if len(latent_up) > 0:
            latent_up_scaled = self.latent_scaler_up.fit_transform(latent_up)
            self.svm_up.fit(latent_up_scaled)
            print(f"  🔺 看漲 OC-SVM 訓練完成 (樣本數: {len(latent_up)})")
        else:
            print("  ⚠️ 看漲樣本不足，跳過 OC-SVM 訓練")
            
        if len(latent_down) > 0:
            latent_down_scaled = self.latent_scaler_down.fit_transform(latent_down)
            self.svm_down.fit(latent_down_scaled)
            print(f"  🔻 看跌 OC-SVM 訓練完成 (樣本數: {len(latent_down)})")
        else:
            print("  ⚠️ 看跌樣本不足，跳過 OC-SVM 訓練")
        
        print(f"✅ 分別 OC-SVM 訓練完成 (nu=0.1，預期 10% 異常)")
        print("  優勢：各型態在專用 Latent Space 中學習，特徵表示更精確")
    
    def svm_predict(self, test_df, theta=0.1):
        """
        使用分別的 One-Class SVM 預測（含信心門檻 θ）
        
        新決策邏輯：
            1. 在看漲專用 Latent Space 中計算看漲分數
            2. 在看跌專用 Latent Space 中計算看跌分數
            3. 正規化後比較分數差異，應用信心門檻
        
        Args:
            test_df: 測試資料
            theta:   信心門檻（預設 0.1），差距不足視為雜訊
            
        Returns:
            predictions: 預測結果 (1=上漲, -1=下跌, 0=觀望)
        """
        # 分別在兩個專用 Latent Space 中計算 latent code
        latent_up = self.get_latent_code(test_df, pattern_type='up')
        latent_down = self.get_latent_code(test_df, pattern_type='down')
        
        predictions = []
        for i in range(len(test_df)):
            # 在看漲專用空間中評估
            lc_up = latent_up[i].reshape(1, -1)
            lc_up_scaled = self.latent_scaler_up.transform(lc_up)
            up_score = self.svm_up.decision_function(lc_up_scaled)[0] if self.svm_up else -np.inf
            
            # 在看跌專用空間中評估
            lc_down = latent_down[i].reshape(1, -1)
            lc_down_scaled = self.latent_scaler_down.transform(lc_down)
            down_score = self.svm_down.decision_function(lc_down_scaled)[0] if self.svm_down else -np.inf
            
            # 正規化分數（避免不同空間的分數直接比較問題）
            up_normalized = 1 / (1 + np.exp(-up_score))      # Sigmoid 正規化
            down_normalized = 1 / (1 + np.exp(-down_score))  # Sigmoid 正規化
            
            diff = up_normalized - down_normalized
            
            # 應用信心門檻
            if abs(diff) < theta:
                predictions.append(0)   # 差距不足 θ → 視為雜訊，不進場
            elif diff > 0:
                predictions.append(1)   # 看漲訊號更強
            else:
                predictions.append(-1)  # 看跌訊號更強
        
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
        slippage = 0.001            # 滑價損耗 0.1%（市場衝擊成本）
        
        # 計算實際交易成本（總計約 0.52%）
        buy_commission = commission_rate * commission_discount   # 買進手續費
        sell_commission = commission_rate * commission_discount  # 賣出手續費
        total_cost = buy_commission + sell_commission + transaction_tax + slippage  # ≈ 0.52%
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
        trailing_stop_pct = 0.02  # 2% 追蹤止損
        
        # 遍歷所有有交易訊號的日期
        for i in range(len(test_df)):
            prediction = predictions[i]
            
            # 跳過無交易訊號
            if prediction == 0:
                continue
            
            # 檢查是否有足夠的未來資料（需要至少4筆：隔天進場 + 3天持有）
            if i + 4 >= len(test_df):
                continue
            
            # ── 修正未來函數：進場價改為隔天開盤價 ──────────────────────
            entry_price = test_df.iloc[i + 1]['Open']  # 進場價：訊號日的隔天開盤價
            
            # ── 追蹤止損：在進場後的 1~3 天（共 3 個交易日）逐日檢查是否觸及止損 ──────────
            # i+1 是進場日，i+2 到 i+4 是持倉期（3天）
            exit_price = test_df.iloc[min(i + 4, len(test_df) - 1)]['Close']  # 預設第3天收盤出場
            peak_price = entry_price  # 追蹤高峰（做多）/ 低點（做空）
            
            if prediction == 1:
                # 做多：監控最高價，若收盤從高峰回落 ≥ 2% 則提前出場
                for day in range(1, 4):  # day=1,2,3 對應 i+2, i+3, i+4
                    if i + 1 + day >= len(test_df):
                        break
                    day_close = test_df.iloc[i + 1 + day]['Close']
                    peak_price = max(peak_price, day_close)   # 更新追蹤高峰
                    drawdown_from_peak = (peak_price - day_close) / peak_price
                    if drawdown_from_peak >= trailing_stop_pct:
                        exit_price = day_close  # 止損出場
                        break
                    exit_price = day_close  # 持續更新至最後一天
                ret = (exit_price - entry_price) / entry_price - total_cost
            
            elif prediction == -1:
                # 做空：監控最低價，若收盤從低點反彈 ≥ 2% 則提前回補
                trough_price = entry_price
                for day in range(1, 4):  # day=1,2,3 對應 i+2, i+3, i+4
                    if i + 1 + day >= len(test_df):
                        break
                    day_close = test_df.iloc[i + 1 + day]['Close']
                    trough_price = min(trough_price, day_close)  # 更新追蹤低點
                    rebound_from_trough = (day_close - trough_price) / trough_price
                    if rebound_from_trough >= trailing_stop_pct:
                        exit_price = day_close  # 止損回補
                        break
                    exit_price = day_close
                ret = (entry_price - exit_price) / entry_price - total_cost
            
            else:
                continue
            
            returns.append(ret)
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
        # 台灣證券交易成本設定（與 backtest 相同，含滑價）
        commission_rate = 0.001425  # 手續費公定費率 0.1425%
        commission_discount = 0.6   # 手續費折扣（6折）
        transaction_tax = 0.003     # 證券交易稅 0.3%（賣出時收取）
        slippage = 0.001            # 滑價損耗 0.1%
        
        # 計算實際交易成本
        buy_commission = commission_rate * commission_discount
        sell_commission = commission_rate * commission_discount
        total_cost = buy_commission + sell_commission + transaction_tax + slippage
        
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
                'conclusion': '數據不足，無法進行統計檢定',
                'strategy_mean': 0.0,
                'benchmark_mean': 0.0
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
        tk.Label(input_frame, text="同產業股票數:", bg='#2d2d2d', fg='#ffffff').grid(row=1, column=3, padx=5, pady=5, sticky='e')
        self.market_stocks_var = tk.StringVar(value="50")
        market_stocks_combo = ttk.Combobox(
            input_frame, 
            textvariable=self.market_stocks_var,
            values=["20", "30", "50", "80", "全部"],
            width=12,
            state='readonly'
        )
        market_stocks_combo.grid(row=1, column=4, padx=5, pady=5, sticky='w')
        tk.Label(input_frame, text="(從CSV讀取同產業股票)", bg='#2d2d2d', fg='#888888', font=('Arial', 8)).grid(row=1, column=5, padx=5, pady=5, sticky='w')
        
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
        Step 2: AE 預訓練 - 在同產業股票歷史資料上訓練，提取 Latent Space
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
                num_market_stocks = None  # 不限制數量
            else:
                num_market_stocks = int(market_stocks_str)
            
            # ========================================================================
            # Step 1: 特徵工程
            # 核心任務: 提取 10 種型態特徵 + 基礎量價指標
            # 注意事項: 務必進行標準化（Scaling），避免股價絕對值影響 AE 訓練
            # ========================================================================
            self.update_status("🔧 [Step 1/4] 特徵工程：提取 10 種型態特徵...")
            
            # 1-1. 從 CSV 尋找同產業股票（用於 AE 預訓練）
            same_industry_stocks, industry_name = self.data_processor.get_same_industry_stocks(
                stock_code, max_stocks=num_market_stocks
            )
            stock_list = same_industry_stocks
            
            self.update_status(f"📊 準備抓取 {len(stock_list)} 檔【{industry_name}】同產業股票資料用於 AE 預訓練...")
            print(f"\n{'='*60}")
            print(f"同產業（{industry_name}）股票：共 {len(stock_list)} 檔")
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
                f"{pattern_summary['down_patterns']} 個下跌型態 "
                f"(聚類樣本: 上漲 {pattern_summary['clustered_up_count']}, "
                f"下跌 {pattern_summary['clustered_down_count']})"
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
            # Step 2: AE 分別訓練
            # 核心任務: 分別為看漲和看跌型態訓練專用的 Autoencoder
            # 注意事項: 監控重構誤差（Reconstruction Error），確保特徵沒丟失
            # ========================================================================
            self.update_status("🧠 [Step 2/4] 分別 AE 訓練：為看漲和看跌型態建立專用模型...")
            
            # 2-1. 初始化模型引擎
            self.model_engine = ModelEngine(self.data_processor.feature_columns)
            
            # 2-2. 用 Step 1 篩選結果過濾訓練資料，只保留屬於型態群組的高品質樣本
            self.update_status("🔄 使用 Step 1 篩選後的型態樣本分別訓練看漲和看跌 AE...")
            
            filtered_train_data = pattern_selector.filter_training_data(train_stock_data)
            all_train_raw = pd.concat([df for df in train_stock_data.values()], ignore_index=True)
            
            if len(filtered_train_data) >= 100:
                all_train_data = filtered_train_data
                self.update_status(
                    f"📊 型態篩選後訓練樣本: {len(all_train_data)} 筆 "
                    f"(原始 {len(all_train_raw)} 筆，只保留聚類型態)"
                )
            else:
                all_train_data = all_train_raw
                self.update_status(
                    f"⚠️ 篩選後樣本不足 ({len(filtered_train_data)} 筆)，"
                    f"使用全部 {len(all_train_data)} 筆訓練"
                )
            
            # 訓練分別的 Autoencoder
            self.model_engine.train_autoencoder(all_train_data, epochs=50)
            
            # 2-3. 監控各自的損失
            up_msg = down_msg = ""
            if hasattr(self.model_engine, 'history_up') and self.model_engine.history_up is not None:
                up_loss = self.model_engine.history_up.history['loss'][-1]
                up_msg = f"看漲 AE: {up_loss:.6f}"
            if hasattr(self.model_engine, 'history_down') and self.model_engine.history_down is not None:
                down_loss = self.model_engine.history_down.history['loss'][-1]
                down_msg = f"看跌 AE: {down_loss:.6f}"
            
            if up_msg or down_msg:
                self.update_status(f"✅ 分別 AE 最終損失: {up_msg}, {down_msg}")
            
            self.update_status("✅ [Step 2 完成] 分別 AE 訓練完成，各型態擁有專用特徵空間")
            
            # ========================================================================
            # Step 3: OC-SVM 建模
            # 核心任務: 使用編碼後的特徵（Latent Code）訓練 OC-SVM
            # 注意事項: 調整 nu 參數（預期的異常比例），這會直接影響交易頻率
            # ========================================================================
            self.update_status("🤖 [Step 3/4] OC-SVM 建模：使用 Latent Code 訓練...")
            
            self.update_status("🔄 使用編碼後的 Latent Code 訓練 OC-SVM (nu=0.1)...")
            self.model_engine.train_one_class_svm(all_train_data)
            
            self.update_status(
                "✅ [Step 3 完成] 分別 OC-SVM 訓練完成 "
                "(各型態在專用 Latent Space 中訓練，特徵表示更精確)"
            )
            
            # ========================================================================
            # Step 4: 回測驗證
            # 核心任務: 驗證通過 OC-SVM 過濾後的看漲型態，勝率是否顯著提升
            # 注意事項: 警惕「存活者偏差」，確保回測包含已退市或下市的股票
            # ========================================================================
            self.update_status("📊 [Step 4/4] 回測驗證：評估 OC-SVM + 信心門檻 θ 的型態勝率...")
            
            # 4-1. OC-SVM 預測（含信心門檻 θ=0.1，差距不足視為雜訊，不進場）
            THETA = 0.1
            self.update_status(f"🔮 使用 OC-SVM+θ={THETA} 過濾型態（雜訊過濾中）...")
            pred_svm = self.model_engine.svm_predict(test_df, theta=THETA)
            
            n_long  = int((pred_svm == 1).sum())
            n_short = int((pred_svm == -1).sum())
            n_idle  = int((pred_svm == 0).sum())
            self.update_status(
                f"📋 訊號分布：做多 {n_long} 次, 做空 {n_short} 次, 觀望 {n_idle} 次 "
                f"(θ={THETA} 雜訊門檻有效過濾)"
            )
            
            # 4-2. 回測（含 0.1% 滑價 + 2% 追蹤止損）
            self.update_status("📈 執行回測（含滑價+追蹤止損）...")
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

        w, h = self._get_fig_size(self.loss_frame, fallback_w=10, fallback_h=5)
        fig, ax = plt.subplots(1, 1, figsize=(w, h), facecolor='#2d2d2d')
        fig.subplots_adjust(left=0.1, right=0.95, top=0.90, bottom=0.15)

        if hasattr(self.model_engine, 'history_up') or hasattr(self.model_engine, 'history_down'):
            ax.set_facecolor('#1e1e1e')
            
            # 繪製看漲 AE 訓練歷史
            if hasattr(self.model_engine, 'history_up') and self.model_engine.history_up is not None:
                history_up = self.model_engine.history_up.history
                ax.plot(history_up['loss'], color='#4caf50', linewidth=2, label='看漲 AE 訓練損失')
            
            # 繪製看跌 AE 訓練歷史
            if hasattr(self.model_engine, 'history_down') and self.model_engine.history_down is not None:
                history_down = self.model_engine.history_down.history
                ax.plot(history_down['loss'], color='#f44336', linewidth=2, label='看跌 AE 訓練損失')
            
            ax.set_title('分別 Autoencoder 訓練損失', color='white', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch', color='white', fontsize=10)
            ax.set_ylabel('Loss (MSE)', color='white', fontsize=10)
            ax.tick_params(colors='white', labelsize=9)
            ax.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white', fontsize=10)
            ax.grid(True, alpha=0.3, color='gray')
        else:
            ax.text(0.5, 0.5, '尚未訓練模型', transform=ax.transAxes, 
                   ha='center', va='center', color='white', fontsize=14)

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
    try:
        root.mainloop()
    except KeyboardInterrupt:
        # 在終端按 Ctrl+C 時優雅結束，不顯示 traceback
        print("\n使用者中斷執行，程式已安全結束。")
        try:
            root.destroy()
        except Exception:
            pass
