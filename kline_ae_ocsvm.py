"""
K線型態預測系統 - 三套特徵集 × Autoencoder + One-Class SVM
===========================================================
特徵集定義:
  1日 (7個特徵):  上影線、下影線、實體線、開盤型態、收盤型態、5日均量比、前五日趨勢
  2日 (10個特徵): 加上前日上/下影線/實體線
  3日 (12個特徵): 加上前兩日實體線、前一天開/收盤型態、3日累積漲跌、3日均量比、區間相對高低點

流程:
  1. yfinance 抓取台灣50檔股票 (2016-2025)
  2. 去除除權息日期產生的 K 線
  3. 標記3天後漲超5%（看漲）/ 跌超5%（看跌）
  4. 二次篩選：出現頻率 ≥ n 且平均報酬達標
  5. 各特徵集分別訓練 AE + OC-SVM (訓練: 2016-2024)
  6. 2025年回測，比較三套特徵集報酬率
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN

os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ──────────────────────────────────────────────
# 台灣50成分股（以yfinance格式）
# ──────────────────────────────────────────────
TW50_CODES = [
    '2330', '2317', '2454', '2412', '2308',
    '2882', '2881', '2886', '2891', '2892',
    '3008', '2303', '2002', '2301', '1301',
    '1303', '2884', '5880', '2357', '2382',
    '2207', '2801', '2885', '2395', '2912',
    '3711', '2327', '6505', '2379', '2408',
    '2474', '3034', '6669', '2409', '2324',
    '2345', '2347', '3045', '2354', '2356',
    '2377', '1216', '1326', '2609', '2603',
    '2615', '5871', '3231', '4904', '4938',
]

# ──────────────────────────────────────────────
# 特徵欄位名稱定義
# ──────────────────────────────────────────────
FEATURE_COLS = {
    '1day': [
        '上影線', '下影線', '實體線',
        '開盤型態', '收盤型態',
        '5日均量比', '趨勢',
    ],
    '2day': [
        '上影線', '下影線', '實體線',
        '前日上影線', '前日下影線', '前日實體線',
        '開盤型態', '收盤型態',
        '5日均量比', '趨勢',
    ],
    '3day': [
        '上影線', '下影線', '實體線',
        '前日實體線', '前兩日實體線',
        '開盤型態', '收盤型態',
        '前一天開盤型態', '前一天收盤型態',
        '3日累積漲跌', '3日均量比', '區間相對高低點',
    ],
}


# ══════════════════════════════════════════════
# Step 1: 資料下載與特徵計算
# ══════════════════════════════════════════════
def fetch_stock(code: str, start='2015-01-01', end='2026-01-01') -> pd.DataFrame:
    """下載單檔股票並去除除權息日期"""
    ticker_sym = f'{code}.TW'
    try:
        tk = yf.Ticker(ticker_sym)
        df = tk.history(start=start, end=end, auto_adjust=False)
        if df.empty:
            ticker_sym = f'{code}.TWO'
            tk = yf.Ticker(ticker_sym)
            df = tk.history(start=start, end=end, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()

        # 去除除權息日
        try:
            actions = tk.actions
            if not actions.empty:
                ex_dates = actions[
                    (actions.get('Dividends', pd.Series(dtype=float)) > 0) |
                    (actions.get('Stock Splits', pd.Series(dtype=float)) > 0)
                ].index
                df = df[~df.index.isin(ex_dates)]
        except Exception:
            pass

        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    except Exception as e:
        print(f'  ✗ {code} 下載失敗: {e}')
        return pd.DataFrame()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算全部特徵（共12個），供三套特徵集共用。
    同時計算標籤：3天後收盤漲跌幅。
    """
    d = df.copy()
    C = d['Close'].replace(0, 1e-10)
    O = d['Open']
    H = d['High']
    L = d['Low']
    V = d['Volume'].replace(0, 1e-10)

    # ── 當日三線 ──
    d['上影線'] = (H - d[['Open', 'Close']].max(axis=1)) / C * 100
    d['下影線'] = (d[['Open', 'Close']].min(axis=1) - L) / C * 100
    d['實體線'] = (C - O) / C * 100

    # ── 前一日三線 ──
    C1 = d['Close'].shift(1).replace(0, 1e-10)
    O1 = d['Open'].shift(1)
    H1 = d['High'].shift(1)
    L1 = d['Low'].shift(1)
    d['前日上影線'] = (H1 - d[['Open', 'Close']].shift(1).max(axis=1)) / C1 * 100
    d['前日下影線'] = (d[['Open', 'Close']].shift(1).min(axis=1) - L1) / C1 * 100
    d['前日實體線'] = (C1 - O1) / C1 * 100

    # ── 前兩日實體線 ──
    C2 = d['Close'].shift(2).replace(0, 1e-10)
    O2 = d['Open'].shift(2)
    d['前兩日實體線'] = (C2 - O2) / C2 * 100

    # ── 開盤/收盤型態 ──
    d['開盤型態'] = (O - C1) / C * 100          # 當日缺口
    d['收盤型態'] = (C - C1) / C * 100           # 當日漲跌

    # ── 前一天開盤/收盤型態 ──
    d['前一天開盤型態'] = (O1 - C2) / C1 * 100
    d['前一天收盤型態'] = (C1 - C2) / C1 * 100

    # ── 5日均量比 & 趨勢 ──
    vol_ma5 = V.rolling(5).mean().replace(0, 1e-10)
    d['5日均量比'] = (V - vol_ma5) / V
    d['趨勢'] = (d['Close'].shift(2) - d['Close'].shift(7).replace(0, 1e-10)) / d['Close'].shift(7).replace(0, 1e-10)

    # ── 3日特徵 ──
    d['3日累積漲跌'] = (C - C2) / C2 * 100
    vol_ma3 = V.rolling(3).mean().replace(0, 1e-10)
    d['3日均量比'] = (V - vol_ma3) / V

    # 3日區間相對高低點
    low3  = d['Low'].rolling(3).min()
    high3 = d['High'].rolling(3).max()
    rng = (high3 - low3).replace(0, 1e-10)
    d['區間相對高低點'] = (C - low3) / rng * 100

    # ── 標籤：3天後漲跌幅 ──
    future_ret = (d['Close'].shift(-3) - d['Close']) / d['Close'] * 100
    d['三天後漲跌'] = future_ret
    d['Label'] = 0
    d.loc[future_ret > 5,  'Label'] = 1    # 看漲
    d.loc[future_ret < -5, 'Label'] = -1   # 看跌

    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    return d


def load_all_stocks(codes=TW50_CODES, start='2015-01-01', end='2026-01-01'):
    """批次下載並計算特徵"""
    all_data = {}
    for i, code in enumerate(codes):
        print(f'  [{i+1}/{len(codes)}] {code}', end=' ')
        raw = fetch_stock(code, start, end)
        if raw.empty:
            print('- 跳過')
            continue
        feat = compute_features(raw)
        if len(feat) < 50:
            print(f'- 資料不足({len(feat)}筆)，跳過')
            continue
        all_data[code] = feat
        print(f'✓ {len(feat)}筆')
    return all_data


# ══════════════════════════════════════════════
# Step 2: 型態篩選（DBSCAN 二次篩選）
# ══════════════════════════════════════════════
def collect_patterns(all_data: dict, feat_key: str):
    """收集所有股票的看漲/看跌型態樣本"""
    cols = FEATURE_COLS[feat_key]
    up_patterns, down_patterns = [], []
    for code, df in all_data.items():
        train = df[df.index.year < 2025]
        for label, lst in [(1, up_patterns), (-1, down_patterns)]:
            sub = train[train['Label'] == label]
            if len(sub) == 0:
                continue
            feats = sub[cols].values
            rets  = sub['三天後漲跌'].values
            dates = sub.index.tolist()
            for j in range(len(sub)):
                lst.append({
                    'features': feats[j],
                    'return':   rets[j],
                    'stock':    code,
                    'date':     dates[j],
                })
    return up_patterns, down_patterns


def dbscan_filter(patterns: list, direction: str, min_freq=3, min_avg_ret=2.0) -> list:
    """
    DBSCAN 二次篩選：
    - 出現頻率 ≥ min_freq
    - 看漲平均漲幅 ≥ min_avg_ret%，看跌平均跌幅 ≥ min_avg_ret%
    回傳各群組中心特徵及統計資訊
    """
    if len(patterns) < min_freq:
        return []

    X = np.array([p['features'] for p in patterns])
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # 高維時先做 PCA 保留 80% 變異量
    from sklearn.decomposition import PCA
    n_comp = min(5, X_sc.shape[1], X_sc.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=SEED)
    X_pca = pca.fit_transform(X_sc)

    db = DBSCAN(eps=1.2, min_samples=min_freq).fit(X_pca)
    labels = db.labels_

    selected = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        idx = np.where(labels == lbl)[0]
        cluster = [patterns[i] for i in idx]
        avg_ret = np.mean([p['return'] for p in cluster])
        freq = len(cluster)
        if direction == 'up'   and avg_ret >= min_avg_ret:
            pass
        elif direction == 'down' and avg_ret <= -min_avg_ret:
            pass
        else:
            continue
        center = np.mean([p['features'] for p in cluster], axis=0)
        selected.append({
            'center':   center,
            'freq':     freq,
            'avg_ret':  avg_ret,
            'keys':     {(p['stock'], p['date']) for p in cluster},
        })

    # 依獲利排序
    if direction == 'up':
        selected.sort(key=lambda x: x['avg_ret'], reverse=True)
    else:
        selected.sort(key=lambda x: x['avg_ret'])
    return selected


def get_training_samples(all_data: dict, feat_key: str,
                          up_selected: list, down_selected: list):
    """
    從二次篩選結果取出 (features, label) 訓練矩陣
    """
    cols = FEATURE_COLS[feat_key]
    up_keys   = set().union(*(s['keys'] for s in up_selected))   if up_selected   else set()
    down_keys = set().union(*(s['keys'] for s in down_selected)) if down_selected else set()
    all_keys  = up_keys | down_keys

    rows = []
    for code, df in all_data.items():
        train = df[df.index.year < 2025]
        stock_keys = {d for (s, d) in all_keys if s == code}
        if not stock_keys:
            continue
        sub = train[train.index.isin(stock_keys)]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            rows.append({**{c: row[c] for c in cols}, 'Label': row['Label']})

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ══════════════════════════════════════════════
# Step 3: Autoencoder + One-Class SVM
# ══════════════════════════════════════════════
def build_autoencoder(input_dim: int, latent_dim: int = 4):
    """Input → 16 → latent_dim → 16 → Input"""
    from tensorflow.keras import layers, Model
    inp = layers.Input(shape=(input_dim,))
    x   = layers.Dense(16, activation='relu')(inp)
    lat = layers.Dense(latent_dim, activation='relu', name='latent')(x)
    x   = layers.Dense(16, activation='relu')(lat)
    out = layers.Dense(input_dim, activation='linear')(x)
    ae      = Model(inp, out)
    encoder = Model(inp, lat)
    ae.compile(optimizer='adam', loss='mse')
    return ae, encoder


class AE_OCSVM:
    """單一方向（看漲 or 看跌）的 AE + OC-SVM"""

    def __init__(self, feat_cols: list, direction: str, nu: float = 0.1):
        self.feat_cols  = feat_cols
        self.direction  = direction
        self.nu         = nu
        self.scaler     = StandardScaler()
        self.lat_scaler = StandardScaler()
        self.ae = None
        self.encoder = None
        self.svm = None
        self.trained = False

    def fit(self, train_df: pd.DataFrame, epochs: int = 80, batch_size: int = 32):
        label = 1 if self.direction == 'up' else -1
        sub = train_df[train_df['Label'] == label]
        if len(sub) < 10:
            print(f'    ⚠ {self.direction} 樣本不足({len(sub)})，跳過')
            return

        X = sub[self.feat_cols].values.astype(np.float32)
        X_sc = self.scaler.fit_transform(X)

        dim = X_sc.shape[1]
        self.ae, self.encoder = build_autoencoder(dim)
        self.ae.fit(X_sc, X_sc,
                    epochs=epochs,
                    batch_size=min(batch_size, len(X_sc)),
                    verbose=0)

        lat = self.encoder.predict(X_sc, verbose=0)
        lat_sc = self.lat_scaler.fit_transform(lat)

        self.svm = OneClassSVM(kernel='rbf', gamma='auto', nu=self.nu)
        self.svm.fit(lat_sc)
        self.trained = True
        print(f'    ✓ {self.direction} AE+OC-SVM 訓練完成 '
              f'(樣本:{len(X)}, 特徵:{dim}, latent:4)')

    def score(self, X_raw: np.ndarray) -> np.ndarray:
        """回傳 SVM decision_function 分數（越高越像正常型態）"""
        if not self.trained:
            return np.full(len(X_raw), -np.inf)
        X_sc  = self.scaler.transform(X_raw)
        lat   = self.encoder.predict(X_sc, verbose=0)
        lat_sc = self.lat_scaler.transform(lat)
        return self.svm.decision_function(lat_sc)


class FeatureSetModel:
    """封裝單一特徵集的完整訓練/預測流程"""

    def __init__(self, feat_key: str, nu: float = 0.1, theta: float = 0.0):
        self.feat_key = feat_key
        self.feat_cols = FEATURE_COLS[feat_key]
        self.theta = theta  # 信心門檻（差值 < theta 視為觀望）
        self.model_up   = AE_OCSVM(self.feat_cols, 'up',   nu)
        self.model_down = AE_OCSVM(self.feat_cols, 'down', nu)

    def fit(self, train_df: pd.DataFrame, **kwargs):
        self.model_up.fit(train_df, **kwargs)
        self.model_down.fit(train_df, **kwargs)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """回傳每筆資料的預測信號 1/0/-1"""
        X = df[self.feat_cols].values.astype(np.float32)
        score_up   = self.model_up.score(X)
        score_down = self.model_down.score(X)

        # Sigmoid 正規化，使兩個 space 的分數可比
        sig = lambda s: 1 / (1 + np.exp(-s))
        s_up   = sig(score_up)
        s_down = sig(score_down)
        diff   = s_up - s_down

        preds = np.where(diff >  self.theta,  1,
                np.where(diff < -self.theta, -1, 0))
        return preds


# ══════════════════════════════════════════════
# Step 4: 回測
# ══════════════════════════════════════════════
COMMISSION = 0.001425 * 0.6   # 手續費（6折）
TAX        = 0.003             # 證交稅（賣出）
SLIPPAGE   = 0.001             # 滑價
TOTAL_COST = COMMISSION * 2 + TAX + SLIPPAGE   # ≈ 0.52%


def backtest_single(test_df: pd.DataFrame, preds: np.ndarray,
                    hold_days: int = 3) -> dict:
    """
    簡單回測：訊號日隔天以開盤價進場，hold_days 天後以收盤價出場。
    含追蹤止損（2%）。
    """
    trailing_stop = 0.02
    returns = []
    equity  = [1.0]

    for i, pred in enumerate(preds):
        if pred == 0:
            continue
        if i + 1 + hold_days >= len(test_df):
            continue

        entry = test_df.iloc[i + 1]['Open']
        if entry <= 0:
            continue

        exit_price = test_df.iloc[i + 1 + hold_days]['Close']

        if pred == 1:   # 做多
            peak = entry
            for day in range(1, hold_days + 1):
                idx = i + 1 + day
                if idx >= len(test_df):
                    break
                close = test_df.iloc[idx]['Close']
                peak = max(peak, close)
                if (peak - close) / peak >= trailing_stop:
                    exit_price = close
                    break
                exit_price = close
            ret = (exit_price - entry) / entry - TOTAL_COST

        else:            # 做空
            trough = entry
            for day in range(1, hold_days + 1):
                idx = i + 1 + day
                if idx >= len(test_df):
                    break
                close = test_df.iloc[idx]['Close']
                trough = min(trough, close)
                if (close - trough) / trough >= trailing_stop:
                    exit_price = close
                    break
                exit_price = close
            ret = (entry - exit_price) / entry - TOTAL_COST

        returns.append(ret)
        equity.append(equity[-1] * (1 + ret))

    if not returns:
        return dict(total_return=0, win_rate=0, max_drawdown=0, trades=0,
                    equity=np.array([1.0]))

    r  = np.array(returns)
    eq = np.array(equity)
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max
    return {
        'total_return':  (eq[-1] - 1) * 100,
        'win_rate':      (r > 0).mean() * 100,
        'max_drawdown':  abs(dd.min()) * 100,
        'avg_return':    r.mean() * 100,
        'trades':        len(r),
        'equity':        eq,
    }


def run_backtest_all_stocks(all_data: dict, model: FeatureSetModel) -> dict:
    """
    對每檔股票的2025年資料執行回測，彙整總績效。

    權益曲線計算方式：
    - 等權重多股票組合：每筆訊號投入等額資金
    - 個別股票回測後平均，避免序列複利失真
    - 最終報酬 = 所有個別交易報酬率的平均累積
    """
    feat_key = model.feat_key
    cols = FEATURE_COLS[feat_key]
    per_stock_results = {}
    all_trade_returns = []
    up_count = down_count = 0

    for code, df in all_data.items():
        test = df[df.index.year >= 2025].copy()
        if len(test) < 5:
            continue
        missing = [c for c in cols if c not in test.columns]
        if missing:
            continue

        preds = model.predict(test)
        up_count   += (preds == 1).sum()
        down_count += (preds == -1).sum()

        result = backtest_single(test, preds)
        if result['trades'] == 0:
            continue
        per_stock_results[code] = result

        # 收集每筆交易報酬（含追蹤止損邏輯，與 backtest_single 一致）
        trailing_stop = 0.02
        for i, pred in enumerate(preds):
            if pred == 0:
                continue
            if i + 1 + 3 >= len(test):
                continue
            entry = test.iloc[i + 1]['Open']
            if entry <= 0:
                continue
            exit_price = test.iloc[i + 4]['Close']
            if pred == 1:
                peak = entry
                for day in range(1, 4):
                    idx = i + 1 + day
                    if idx >= len(test):
                        break
                    close = test.iloc[idx]['Close']
                    peak = max(peak, close)
                    if (peak - close) / peak >= trailing_stop:
                        exit_price = close
                        break
                    exit_price = close
                r = (exit_price - entry) / entry - TOTAL_COST
            else:
                trough = entry
                for day in range(1, 4):
                    idx = i + 1 + day
                    if idx >= len(test):
                        break
                    close = test.iloc[idx]['Close']
                    trough = min(trough, close)
                    if (close - trough) / trough >= trailing_stop:
                        exit_price = close
                        break
                    exit_price = close
                r = (entry - exit_price) / entry - TOTAL_COST
            all_trade_returns.append(r)

    print(f'  訊號分布 → 看漲: {up_count}  |  看跌: {down_count}')

    if not all_trade_returns:
        return dict(total_return=0, win_rate=0, max_drawdown=0,
                    avg_return=0, trades=0, equity=np.array([1.0]),
                    per_stock=per_stock_results)

    r = np.array(all_trade_returns)

    # 權益曲線：每筆交易以等額資金累積（等權重 portfolio）
    eq = np.concatenate([[1.0], np.cumprod(1 + r)])
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max

    # 各股平均總報酬
    stock_returns = [v['total_return'] for v in per_stock_results.values()]
    avg_stock_return = np.mean(stock_returns) if stock_returns else 0

    return {
        'total_return':      avg_stock_return,           # 各股平均總報酬
        'cumulative_return': (eq[-1] - 1) * 100,         # 等權重複利總報酬
        'win_rate':          (r > 0).mean() * 100,
        'max_drawdown':      abs(dd.min()) * 100,
        'avg_return':        r.mean() * 100,
        'trades':            len(r),
        'equity':            eq,
        'per_stock':         per_stock_results,
        'up_signals':        int(up_count),
        'down_signals':      int(down_count),
    }


# ══════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════
def main():
    print('=' * 60)
    print('K線型態預測系統：三套特徵集 AE + OC-SVM 比較')
    print('訓練期：2016-2024  |  回測期：2025')
    print('=' * 60)

    # ── 1. 下載資料 ──
    print('\n【Step 1】下載台灣50股票資料...')
    all_data = load_all_stocks(TW50_CODES, start='2015-06-01', end='2026-01-01')
    # 只保留 2016 年以後資料（確保特徵計算有足夠歷史）
    all_data = {k: v[v.index.year >= 2016] for k, v in all_data.items() if len(v[v.index.year >= 2016]) > 50}
    print(f'  成功載入 {len(all_data)} 檔股票')

    if len(all_data) == 0:
        print('無法載入任何股票資料，程式結束。')
        return

    results = {}

    for feat_key in ['1day', '2day', '3day']:
        print(f'\n{"="*60}')
        print(f'【特徵集: {feat_key}】{len(FEATURE_COLS[feat_key])} 個特徵')
        print('=' * 60)

        cols = FEATURE_COLS[feat_key]

        # ── 2. 型態篩選 ──
        print('\n【Step 2】型態篩選（DBSCAN 二次篩選）...')
        up_pats, down_pats = collect_patterns(all_data, feat_key)
        print(f'  原始大漲型態: {len(up_pats)}  |  大跌型態: {len(down_pats)}')

        up_sel   = dbscan_filter(up_pats,   'up',   min_freq=3, min_avg_ret=2.0)
        down_sel = dbscan_filter(down_pats, 'down', min_freq=3, min_avg_ret=2.0)
        print(f'  篩選後群組 -> 看漲: {len(up_sel)}  |  看跌: {len(down_sel)}')

        # 取出訓練樣本
        train_df = get_training_samples(all_data, feat_key, up_sel, down_sel)
        print(f'  訓練樣本筆數: {len(train_df)}')

        if len(train_df) < 20:
            print('  ⚠ 訓練樣本不足，跳過此特徵集')
            results[feat_key] = None
            continue

        # ── 3. 訓練 AE + OC-SVM ──
        print('\n【Step 3】訓練 Autoencoder + One-Class SVM...')
        model = FeatureSetModel(feat_key, nu=0.1, theta=0.10)
        model.fit(train_df, epochs=80, batch_size=32)

        # ── 4. 回測 ──
        print('\n【Step 4】2025年回測...')
        res = run_backtest_all_stocks(all_data, model)
        results[feat_key] = res

        print(f'\n  ── {feat_key} 回測結果 ──')
        print(f'  各股平均總報酬: {res["total_return"]:>8.2f}%')
        print(f'  複利總報酬:     {res["cumulative_return"]:>8.2f}%')
        print(f'  平均每筆:       {res["avg_return"]:>8.2f}%')
        print(f'  勝率:           {res["win_rate"]:>8.2f}%')
        print(f'  最大回撤:       {res["max_drawdown"]:>8.2f}%')
        print(f'  交易筆數:       {res["trades"]:>8d}')
        print(f'  看漲訊號:       {res["up_signals"]:>8d}')
        print(f'  看跌訊號:       {res["down_signals"]:>8d}')

    # ── 5. 彙整比較 ──
    print('\n' + '=' * 60)
    print('【彙整比較】三套特徵集回測績效')
    print('=' * 60)
    names = {'1day': '1日(7特徵)', '2day': '2日(10特徵)', '3day': '3日(12特徵)'}
    header = f"{'特徵集':<14} {'各股均報酬':>11} {'每筆均報酬':>11} {'勝率':>8} {'最大回撤':>10} {'交易次數':>10}"
    print(header)
    print('-' * (len(header) + 2))
    for key in ['1day', '2day', '3day']:
        res = results.get(key)
        if res is None:
            print(f'{names[key]:<14} {"N/A":>11}')
            continue
        print(f'{names[key]:<14} '
              f'{res["total_return"]:>10.2f}% '
              f'{res["avg_return"]:>10.2f}% '
              f'{res["win_rate"]:>7.2f}% '
              f'{res["max_drawdown"]:>9.2f}% '
              f'{res["trades"]:>10d}')

    # ── 6. 繪圖：權益曲線比較 ──
    _plot_equity_curves(results, names)
    print('\n圖表已儲存至 equity_curves.png')


def _plot_equity_curves(results: dict, names: dict):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = {'1day': 'steelblue', '2day': 'darkorange', '3day': 'seagreen'}
    keys = ['1day', '2day', '3day']

    # 上排：權益曲線
    for ax, key in zip(axes[0], keys):
        res = results.get(key)
        ax.set_title(f'{names[key]}\n2025年回測權益曲線', fontsize=11)
        ax.set_xlabel('交易次數（累積）')
        ax.set_ylabel('資產倍率（等權重）')
        if res is None or res['trades'] == 0:
            ax.text(0.5, 0.5, '無交易訊號', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='gray')
            continue
        eq = res['equity']
        ax.plot(eq, color=colors[key], linewidth=1.5)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
        ax.fill_between(range(len(eq)), 1, eq,
                        where=(eq >= 1), alpha=0.15, color='green')
        ax.fill_between(range(len(eq)), 1, eq,
                        where=(eq < 1),  alpha=0.15, color='red')
        info = (f'各股均報酬: {res["total_return"]:.1f}%\n'
                f'每筆均報酬: {res["avg_return"]:.2f}%\n'
                f'勝率: {res["win_rate"]:.1f}%\n'
                f'最大回撤: {res["max_drawdown"]:.1f}%\n'
                f'交易次數: {res["trades"]}')
        ax.text(0.02, 0.97, info, transform=ax.transAxes,
                fontsize=8, va='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 下排：各股報酬分布
    for ax, key in zip(axes[1], keys):
        res = results.get(key)
        ax.set_title(f'{names[key]}\n各股總報酬分布', fontsize=11)
        ax.set_xlabel('總報酬率 (%)')
        ax.set_ylabel('股票數量')
        if res is None or not res.get('per_stock'):
            ax.text(0.5, 0.5, '無資料', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            continue
        stock_rets = [v['total_return'] for v in res['per_stock'].values()]
        ax.hist(stock_rets, bins=15, color=colors[key], edgecolor='white', alpha=0.85)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.axvline(np.mean(stock_rets), color='red', linestyle='-', linewidth=1.5,
                   label=f'均值: {np.mean(stock_rets):.1f}%')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('equity_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
