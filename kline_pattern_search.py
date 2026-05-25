"""
K線型態分析系統
台灣前50大股票 | 1日 / 2日 / 3日 特徵集比較
流程：Euclidean Distance 找型態 → 重要性權重 → Autoencoder 壓縮 → One-Class SVM 判斷
"""

import warnings
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import OneClassSVM

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 設定中文字體（Windows 微軟正黑體）
plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 固定參數（不開放使用者調整）
# ══════════════════════════════════════════════════════════════════════════════
HOLD_DAYS    = 5
RISE_THRESH  = 5.0
FALL_THRESH  = -5.0
SIM_THRESH   = 0.95   # 已改用 Euclidean
DIST_THRESH  = 3.0
MIN_FREQ     = 5
MIN_BULL_RET = 2.0
MIN_BEAR_RET = -2.0
LATENT_DIM   = 4
EPOCHS       = 60
OCSVM_NU     = 0.1    # 越小邊界越緊，訊號越少越精準
# 交易成本（每筆來回）
BROKERAGE    = 0.06   # 手續費（買+賣各 0.06%，已折扣）
TAX          = 0.30   # 證券交易稅（賣出 0.3%）
TRADE_COST   = BROKERAGE * 2 + TAX   # = 0.42% per round trip
TRAIN_START  = '2016-01-01'
TRAIN_END    = '2024-12-31'
TEST_START   = '2025-01-01'
TEST_END     = '2025-12-31'

# ── 台灣前50大股票 ────────────────────────────────────────────────────────────
TW50 = {
    '2330.TW': '台積電',   '2317.TW': '鴻海',     '2454.TW': '聯發科',
    '2308.TW': '台達電',   '2303.TW': '聯電',     '2412.TW': '中華電',
    '2881.TW': '富邦金',   '2882.TW': '國泰金',   '2886.TW': '兆豐金',
    '2891.TW': '中信金',   '2884.TW': '玉山金',   '2885.TW': '元大金',
    '2892.TW': '第一金',   '5880.TW': '合庫金',   '2880.TW': '華南金',
    '2002.TW': '中鋼',     '1301.TW': '台塑',     '1303.TW': '南亞',
    '1326.TW': '台化',     '3711.TW': '日月光投控','2207.TW': '和泰車',
    '2382.TW': '廣達',     '2395.TW': '研華',     '3008.TW': '大立光',
    '2408.TW': '南亞科',   '2357.TW': '華碩',     '4938.TW': '和碩',
    '3045.TW': '台灣大',   '2327.TW': '國巨',     '6505.TW': '台塑化',
    '5871.TW': '中租控股', '2353.TW': '宏碁',     '2344.TW': '華邦電',
    '3034.TW': '聯詠',     '2301.TW': '光寶科',   '2912.TW': '統一超',
    '1216.TW': '統一',     '2615.TW': '萬海',     '2609.TW': '陽明',
    '2603.TW': '長榮',     '5876.TW': '上海商銀', '2474.TW': '可成',
    '1101.TW': '台泥',     '2337.TW': '旺宏',     '3037.TW': '欣興',
    '6669.TW': '緯穎',     '2049.TW': '上銀',     '2379.TW': '瑞昱',
    '2105.TW': '正新',     '1590.TW': '亞德客',
}


# ══════════════════════════════════════════════════════════════════════════════
# 特徵函式（三種天數定義）
# ══════════════════════════════════════════════════════════════════════════════
def detect_exdiv_mask(df: pd.DataFrame, tol: float = 0.005) -> pd.Series:
    if 'Adj Close' not in df.columns or 'Close' not in df.columns:
        return pd.Series(False, index=df.index)
    ratio = df['Adj Close'] / df['Close']
    return ratio.pct_change().abs() > tol


def features_1day(df: pd.DataFrame) -> pd.DataFrame:
    """1日K線特徵（7個）"""
    O, H, L, C, V = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
    Cp = C.shift(1)
    upper     = (H - np.maximum(O, C)) / C * 100
    lower     = (np.minimum(O, C) - L) / C * 100
    body      = (C - O) / C * 100
    gap       = (O - Cp) / C * 100
    close_chg = (C - Cp) / C * 100
    v5ma      = V.rolling(5, min_periods=5).mean()
    vol_ratio = (V - v5ma) / V.replace(0, np.nan)
    trend     = (C.shift(2) - C.shift(7)) / C.shift(7) * 100
    return pd.DataFrame({
        'upper': upper, 'lower': lower, 'body': body,
        'gap': gap, 'close_chg': close_chg,
        'vol_ratio': vol_ratio, 'trend': trend,
    }, index=df.index)


def features_2day(df: pd.DataFrame) -> pd.DataFrame:
    """2日K線特徵（10個）"""
    O, H, L, C, V = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
    O1, H1, L1, C1 = O.shift(1), H.shift(1), L.shift(1), C.shift(1)
    upper     = (H  - np.maximum(O,  C )) / C  * 100
    lower     = (np.minimum(O,  C ) - L ) / C  * 100
    body      = (C  - O ) / C  * 100
    p_upper   = (H1 - np.maximum(O1, C1)) / C1 * 100
    p_lower   = (np.minimum(O1, C1) - L1) / C1 * 100
    p_body    = (C1 - O1) / C1 * 100
    gap       = (O  - C1) / C  * 100
    close_chg = (C  - C1) / C  * 100
    v5ma      = V.rolling(5, min_periods=5).mean()
    vol_ratio = (V  - v5ma) / V.replace(0, np.nan)
    trend     = (C.shift(2) - C.shift(7)) / C.shift(7) * 100
    return pd.DataFrame({
        'upper': upper, 'lower': lower, 'body': body,
        'p_upper': p_upper, 'p_lower': p_lower, 'p_body': p_body,
        'gap': gap, 'close_chg': close_chg,
        'vol_ratio': vol_ratio, 'trend': trend,
    }, index=df.index)


def features_3day(df: pd.DataFrame) -> pd.DataFrame:
    """3日K線特徵（12個）"""
    O, H, L, C, V = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
    O1, H1, L1, C1 = O.shift(1), H.shift(1), L.shift(1), C.shift(1)
    O2, H2, L2, C2 = O.shift(2), H.shift(2), L.shift(2), C.shift(2)
    upper        = (H  - np.maximum(O,  C )) / C  * 100
    lower        = (np.minimum(O,  C ) - L ) / C  * 100
    body         = (C  - O ) / C  * 100
    p1_body      = (C1 - O1) / C1 * 100
    p2_body      = (C2 - O2) / C2 * 100
    gap          = (O  - C1) / C  * 100
    close_chg    = (C  - C1) / C  * 100
    p1_gap       = (O1 - C2) / C1 * 100
    p1_close_chg = (C1 - C2) / C1 * 100
    cum_return   = (C  - C2) / C2 * 100
    v3ma         = V.rolling(3, min_periods=3).mean()
    vol_ratio_3  = (V - v3ma) / V.replace(0, np.nan)
    h_max = pd.concat([H, H1, H2], axis=1).max(axis=1)
    l_min = pd.concat([L, L1, L2], axis=1).min(axis=1)
    denom = h_max - l_min
    range_pos = pd.Series(
        np.where(denom > 0, (C - l_min) / denom * 100, 50.0),
        index=df.index)
    return pd.DataFrame({
        'upper': upper, 'lower': lower, 'body': body,
        'p1_body': p1_body, 'p2_body': p2_body,
        'gap': gap, 'close_chg': close_chg,
        'p1_gap': p1_gap, 'p1_close_chg': p1_close_chg,
        'cum_return': cum_return,
        'vol_ratio_3': vol_ratio_3, 'range_pos': range_pos,
    }, index=df.index)


FEATURE_SETS = [
    ('1日-7特徵',  features_1day),
    ('2日-10特徵', features_2day),
    ('3日-12特徵', features_3day),
]


# ══════════════════════════════════════════════════════════════════════════════
# 下載原始資料
# ══════════════════════════════════════════════════════════════════════════════
def download_raw(log_fn: callable) -> dict:
    """下載所有股票原始OHLCV，回傳 {ticker: df}"""
    raw_data = {}
    tickers  = list(TW50.keys())
    for i, ticker in enumerate(tickers):
        log_fn(f"  [{i+1:02d}/{len(tickers)}] {ticker} {TW50[ticker]}...")
        try:
            df = yf.download(
                ticker, start='2015-01-01', end='2025-12-31',
                auto_adjust=False, progress=False, actions=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            required = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}
            if df.empty or len(df) < 100 or not required.issubset(df.columns):
                log_fn("       → 跳過"); continue
            raw_data[ticker] = df
            log_fn(f"       → {len(df)} 筆 OK")
        except Exception as e:
            log_fn(f"       → 錯誤: {e}")
    return raw_data


def build_dataset(raw_data: dict, feat_fn) -> pd.DataFrame:
    """用指定特徵函式建構全股票資料集"""
    parts = []
    for ticker, df in raw_data.items():
        try:
            exdiv = detect_exdiv_mask(df)
            feat  = feat_fn(df)
            fret  = (df['Close'].shift(-HOLD_DAYS) - df['Close']) / df['Close'] * 100
            combined = feat.copy()
            combined['future_ret'] = fret
            combined['exdiv']      = exdiv
            combined['ticker']     = ticker
            parts.append(combined)
        except Exception:
            continue
    full = pd.concat(parts, axis=0)
    full = full[~full['exdiv']].drop(columns='exdiv')
    full = full.replace([np.inf, -np.inf], np.nan).dropna()
    return full


# ══════════════════════════════════════════════════════════════════════════════
# Autoencoder
# ══════════════════════════════════════════════════════════════════════════════
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 4):
        super().__init__()
        h = max(16, input_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, 8), nn.ReLU(),
            nn.Linear(8, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, h), nn.ReLU(),
            nn.Linear(h, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


def train_ae(X: np.ndarray, input_dim: int, log_fn: callable) -> Autoencoder:
    X_t    = torch.FloatTensor(X)
    loader = DataLoader(TensorDataset(X_t),
                        batch_size=min(256, len(X)),
                        shuffle=True,
                        drop_last=len(X) > 256)
    model   = Autoencoder(input_dim, LATENT_DIM)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(1, EPOCHS + 1):
        ep_loss = 0.0
        for (xb,) in loader:
            opt.zero_grad()
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        sched.step()
        if ep % 10 == 0:
            log_fn(f"      Epoch {ep:3d}/{EPOCHS}  loss={ep_loss/len(loader):.5f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 找型態 + 訓練 AE + OCSVM（單一特徵集）
# ══════════════════════════════════════════════════════════════════════════════
def find_and_train(df: pd.DataFrame, feat_cols: list, log_fn: callable) -> dict:
    """
    老師要求流程：
    Step1 - Euclidean Distance 相似度比對（取代 Cosine Similarity）
    Step2 - 出現頻率 >= MIN_FREQ，平均報酬符合門檻
    Step3 - 重要性權重 = 出現頻率 × |平均報酬|
    Step4 - Autoencoder 壓縮至 Latent Space
    Step5 - One-Class SVM（含重要性 sample_weight）
    """
    train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    X = train_df[feat_cols].values.astype(np.float32)
    y = train_df['future_ret'].values.astype(np.float32)

    scaler   = StandardScaler()
    X_scaled = np.clip(scaler.fit_transform(X), -5, 5)

    bull_seeds = np.where(y > RISE_THRESH)[0]
    bear_seeds = np.where(y < FALL_THRESH)[0]
    log_fn(f"  看漲種子：{len(bull_seeds):,}  看跌種子：{len(bear_seeds):,}")

    models = {'scaler': scaler}

    for direction, seeds in [('bull', bull_seeds), ('bear', bear_seeds)]:
        label  = '看漲' if direction == 'bull' else '看跌'
        ret_ok = (lambda r: r > MIN_BULL_RET) if direction == 'bull' \
                 else (lambda r: r < MIN_BEAR_RET)
        if len(seeds) == 0:
            models[direction] = None; continue

        # Step1：Euclidean Distance 找有效型態（分批計算避免記憶體不足）
        idx_importance = {}   # {樣本idx: 重要性分數}
        valid_count    = 0
        top_patterns   = []
        BATCH = 200

        for batch_start in range(0, len(seeds), BATCH):
            batch_seeds = seeds[batch_start: batch_start + BATCH]
            batch_vecs  = X_scaled[batch_seeds]
            dist_batch  = euclidean_distances(batch_vecs, X_scaled)  # (BATCH, N)

            for j, seed_idx in enumerate(batch_seeds):
                dists        = dist_batch[j].copy()
                dists[seed_idx] = np.inf          # 排除自身
                similar_idx  = np.where(dists <= DIST_THRESH)[0]
                if len(similar_idx) < MIN_FREQ:
                    continue
                similar_rets = y[similar_idx]

                # 每一根相似K線都必須符合報酬條件（非平均）
                if direction == 'bull':
                    if not (similar_rets > MIN_BULL_RET).all():
                        continue
                else:
                    if not (similar_rets < MIN_BEAR_RET).all():
                        continue

                avg_ret    = float(similar_rets.mean())
                freq       = len(similar_idx)
                win_rate   = float((similar_rets > 0).mean()) if direction == 'bull' \
                             else float((similar_rets < 0).mean())
                importance = freq * abs(avg_ret)   # 老師要求：重要性 = 頻率 × |平均報酬|

                # 記錄有效型態（取重要性最高的那次）
                for idx in [seed_idx, *similar_idx.tolist()]:
                    if idx not in idx_importance or idx_importance[idx] < importance:
                        idx_importance[idx] = importance
                valid_count += 1
                top_patterns.append((freq, avg_ret, win_rate, importance))

        log_fn(f"  {label} 有效型態群：{valid_count}  有效樣本：{len(idx_importance):,}")

        # 顯示前10個重要型態
        top_patterns.sort(key=lambda x: x[3], reverse=True)
        if top_patterns:
            log_fn(f"  {'排名':>4}  {'頻率':>6}  {'平均報酬':>10}  {'勝率':>8}  {'重要性':>10}")
            for rank, (freq, avg, wr, imp) in enumerate(top_patterns[:10], 1):
                log_fn(f"  {rank:>4}  {freq:>6}  {avg:>+9.2f}%  {wr:>7.1%}  {imp:>10.2f}")

        if len(idx_importance) == 0:
            models[direction] = None; continue

        # Step2：用有效型態樣本訓練 Autoencoder
        sorted_idx    = sorted(idx_importance.keys())
        X_pattern     = X_scaled[sorted_idx]
        log_fn(f"\n  [{label}] 訓練 AE (input={len(feat_cols)}, latent={LATENT_DIM}, "
               f"樣本={len(X_pattern):,})...")
        ae = train_ae(X_pattern, len(feat_cols), log_fn)
        ae.eval()

        # Step3：Latent Code → One-Class SVM（含重要性 sample_weight）
        with torch.no_grad():
            Z = ae.encode(torch.FloatTensor(X_pattern)).numpy()

        weights = np.array([idx_importance[i] for i in sorted_idx], dtype=np.float32)
        weights = weights / weights.sum() * len(weights)   # 正規化

        log_fn(f"  [{label}] 訓練 OCSVM（含重要性權重）...")
        ocsvm = OneClassSVM(kernel='rbf', nu=OCSVM_NU, gamma='scale')
        ocsvm.fit(Z, sample_weight=weights)
        models[direction] = {'ae': ae, 'ocsvm': ocsvm}
        log_fn(f"  [{label}] 完成 ✓")

    return models


# ══════════════════════════════════════════════════════════════════════════════
# 回測（單一特徵集 × 選定股票）
# ══════════════════════════════════════════════════════════════════════════════
def backtest(df: pd.DataFrame, feat_cols: list,
             models: dict, ticker: str) -> dict:
    test_df = df[
        (df.index >= TEST_START) &
        (df.index <= TEST_END) &
        (df['ticker'] == ticker)
    ].copy()

    if test_df.empty:
        return None

    scaler   = models['scaler']
    X_test   = test_df[feat_cols].values.astype(np.float32)
    y_test   = test_df['future_ret'].values.astype(np.float32)
    X_scaled = np.clip(scaler.transform(X_test), -5, 5)
    signal   = np.zeros(len(X_test), dtype=np.float32)

    for direction, sig_val in [('bull', 1.0), ('bear', -1.0)]:
        if models.get(direction) is None:
            continue
        ae = models[direction]['ae']
        ocsvm = models[direction]['ocsvm']
        ae.eval()
        with torch.no_grad():
            Z = ae.encode(torch.FloatTensor(X_scaled)).numpy()
        pred = ocsvm.predict(Z)
        mask = pred == 1
        if direction == 'bull':
            signal[mask] = 1.0
        else:
            signal[mask & (signal != 1.0)] = -1.0

    strategy_ret = signal * y_test
    # 扣除交易成本（每筆來回：手續費×2 + 交易稅 = 0.42%）
    strategy_ret[signal != 0] -= TRADE_COST
    n_buy   = int((signal ==  1).sum())
    n_sell  = int((signal == -1).sum())
    n_total = int((signal !=  0).sum())

    if n_total > 0:
        act      = strategy_ret[signal != 0]
        avg_ret  = float(act.mean())
        avg_bull = float(strategy_ret[signal ==  1].mean()) if n_buy  > 0 else 0.0
        avg_bear = float(strategy_ret[signal == -1].mean()) if n_sell > 0 else 0.0
        win_rate = float((act > 0).mean())
        cum_ret  = float(act.sum())
    else:
        avg_ret = avg_bull = avg_bear = win_rate = cum_ret = 0.0

    return {
        'strategy_ret': strategy_ret,
        'signal':       signal,
        'n_buy':   n_buy,   'n_sell':  n_sell,  'n_total': n_total,
        'avg_ret': avg_ret, 'avg_bull': avg_bull,'avg_bear': avg_bear,
        'win_rate': win_rate, 'cum_ret': cum_ret,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIM_THRESH 參數搜尋
# ══════════════════════════════════════════════════════════════════════════════
GRID_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def grid_search_sim_thresh(df: pd.DataFrame, feat_cols: list,
                            feat_name: str, log_fn: callable) -> list:
    """
    固定訓練一次 AE（全部訓練資料），
    再對各 SIM_THRESH 分別跑 Cosine 過濾 + OCSVM，
    比較訓練集 Precision，找最佳門檻。
    """
    train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    X = train_df[feat_cols].values.astype(np.float32)
    y = train_df['future_ret'].values.astype(np.float32)

    scaler   = StandardScaler()
    X_scaled = np.clip(scaler.fit_transform(X), -5, 5)

    bull_seeds = np.where(y > RISE_THRESH)[0]
    bear_seeds = np.where(y < FALL_THRESH)[0]

    # ── AE 只訓練一次（全部資料）────────────────────────────────────────────
    log_fn(f"  [{feat_name}] AE 訓練中（全資料，{len(X_scaled):,} 筆）...")
    ae = train_ae(X_scaled, len(feat_cols), log_fn)
    ae.eval()
    with torch.no_grad():
        Z_all = ae.encode(torch.FloatTensor(X_scaled)).numpy()
    log_fn(f"  [{feat_name}] AE 完成，開始掃描門檻...")

    results = []

    for thresh in GRID_THRESHOLDS:
        bull_valid, bear_valid = set(), set()

        for seeds, valid_set, ret_ok in [
            (bull_seeds, bull_valid, lambda r: r > MIN_BULL_RET),
            (bear_seeds, bear_valid, lambda r: r < MIN_BEAR_RET),
        ]:
            for b in range(0, len(seeds), 200):
                batch = seeds[b: b + 200]
                sims  = cosine_similarity(X_scaled[batch], X_scaled)
                for j, idx in enumerate(batch):
                    row = sims[j].copy(); row[idx] = 0
                    similar = np.where(row >= thresh)[0]
                    if len(similar) < MIN_FREQ: continue
                    if not ret_ok(float(y[similar].mean())): continue
                    valid_set.add(idx)
                    valid_set.update(similar.tolist())

        precision_bull = precision_bear = 0.0

        if len(bull_valid) >= 10:
            ocsvm = OneClassSVM(kernel='rbf', nu=OCSVM_NU, gamma='scale')
            ocsvm.fit(Z_all[sorted(bull_valid)])
            pos = ocsvm.predict(Z_all) == 1
            precision_bull = float((y[pos] > MIN_BULL_RET).mean()) if pos.sum() > 0 else 0.0

        if len(bear_valid) >= 10:
            ocsvm = OneClassSVM(kernel='rbf', nu=OCSVM_NU, gamma='scale')
            ocsvm.fit(Z_all[sorted(bear_valid)])
            pos = ocsvm.predict(Z_all) == 1
            precision_bear = float((y[pos] < MIN_BEAR_RET).mean()) if pos.sum() > 0 else 0.0

        avg_prec = (precision_bull + precision_bear) / 2
        log_fn(f"  thresh={thresh:.2f}  "
               f"漲樣本:{len(bull_valid):>5,}  跌樣本:{len(bear_valid):>5,}  "
               f"P_漲:{precision_bull:.2%}  P_跌:{precision_bear:.2%}  "
               f"均Precision:{avg_prec:.2%}")

        results.append({
            'thresh':         thresh,
            'bull_samples':   len(bull_valid),
            'bear_samples':   len(bear_valid),
            'precision_bull': precision_bull,
            'precision_bear': precision_bear,
            'avg_precision':  avg_prec,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 特徵驗證（不回測）
# ══════════════════════════════════════════════════════════════════════════════
def validate_features(df: pd.DataFrame, feat_cols: list, log_fn: callable) -> dict:
    """
    訓練 AE + OCSVM，計算四大分類評估指標：
    Precision、Recall、Accuracy、F1-Score（看漲 & 看跌各一組）
    """
    train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    X = train_df[feat_cols].values.astype(np.float32)
    y = train_df['future_ret'].values.astype(np.float32)

    scaler   = StandardScaler()
    X_scaled = np.clip(scaler.fit_transform(X), -5, 5)

    log_fn(f"  [AE] 訓練中 (input={len(feat_cols)}, latent={LATENT_DIM}, "
           f"樣本={len(X_scaled):,})...")
    ae = train_ae(X_scaled, len(feat_cols), log_fn)
    ae.eval()

    with torch.no_grad():
        Z = ae.encode(torch.FloatTensor(X_scaled)).numpy()

    bull_mask = y > RISE_THRESH   # 實際正例（看漲）
    bear_mask = y < FALL_THRESH   # 實際正例（看跌）
    log_fn(f"  看漲種子：{bull_mask.sum():,}  看跌種子：{bear_mask.sum():,}")

    def calc_metrics(actual_pos: np.ndarray, pred_pos: np.ndarray,
                     label: str, log_fn) -> dict:
        """計算 TP/TN/FP/FN 及四大指標"""
        N   = len(actual_pos)
        TP  = int(( pred_pos &  actual_pos).sum())
        TN  = int((~pred_pos & ~actual_pos).sum())
        FP  = int(( pred_pos & ~actual_pos).sum())
        FN  = int((~pred_pos &  actual_pos).sum())

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        accuracy  = (TP + TN) / N  if N > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        log_fn(f"\n  [{label}] 混淆矩陣：")
        log_fn(f"    TP={TP:,}  TN={TN:,}  FP={FP:,}  FN={FN:,}")
        log_fn(f"    Precision : {precision:.4f}  ({precision:.2%})")
        log_fn(f"    Recall    : {recall:.4f}  ({recall:.2%})")
        log_fn(f"    Accuracy  : {accuracy:.4f}  ({accuracy:.2%})")
        log_fn(f"    F1-Score  : {f1:.4f}  ({f1:.2%})")

        return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
                'precision': precision, 'recall': recall,
                'accuracy': accuracy, 'f1': f1}

    metrics = {}

    if bull_mask.sum() >= 5:
        oc_bull = OneClassSVM(kernel='rbf', nu=OCSVM_NU, gamma='scale')
        oc_bull.fit(Z[bull_mask])
        pred_pos = oc_bull.predict(Z) == 1
        actual   = y > MIN_BULL_RET
        metrics['bull'] = calc_metrics(actual, pred_pos, '看漲', log_fn)
    else:
        metrics['bull'] = {}

    if bear_mask.sum() >= 5:
        oc_bear = OneClassSVM(kernel='rbf', nu=OCSVM_NU, gamma='scale')
        oc_bear.fit(Z[bear_mask])
        pred_pos = oc_bear.predict(Z) == 1
        actual   = y < MIN_BEAR_RET
        metrics['bear'] = calc_metrics(actual, pred_pos, '看跌', log_fn)
    else:
        metrics['bear'] = {}

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════════════════════
class KLineApp:
    def __init__(self, root: tk.Tk):
        self.root     = root
        self.root.title("K線型態分析系統")
        self.root.geometry("1280x800")
        self.root.resizable(True, True)

        self._raw_data = None   # {ticker: df}
        self._running  = False
        self._queue    = queue.Queue()

        self._build_ui()
        self._poll()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.columnconfigure(0, weight=0, minsize=210)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # 左側
        left = ttk.Frame(self.root, padding=14)
        left.grid(row=0, column=0, sticky='nsew')

        ttk.Label(left, text="K線型態分析系統",
                  font=('', 13, 'bold')).pack(pady=(0, 14))

        ttk.Label(left, text="回測股票代號",
                  font=('', 10, 'bold')).pack(anchor='w')
        ttk.Label(left, text="（例：2330.TW）",
                  font=('', 9), foreground='gray').pack(anchor='w')

        self._ticker_var = tk.StringVar(value='2330.TW')
        ttk.Entry(left, textvariable=self._ticker_var,
                  width=22, font=('', 11)).pack(fill='x', pady=(4, 16))

        ttk.Separator(left, orient='horizontal').pack(fill='x', pady=4)

        self._btn_dl = ttk.Button(left, text="① 下載全部資料",
                                  command=self._on_download)
        self._btn_dl.pack(fill='x', pady=4)

        self._btn_val = ttk.Button(left, text="② 特徵驗證（不回測）",
                                   command=self._on_validate, state='disabled')
        self._btn_val.pack(fill='x', pady=4)

        self._btn_grid = ttk.Button(left, text="③ 參數搜尋（SIM_THRESH）",
                                    command=self._on_grid_search, state='disabled')
        self._btn_grid.pack(fill='x', pady=4)

        self._btn_run = ttk.Button(left, text="④ 分析 + 回測（三特徵集）",
                                   command=self._on_run, state='disabled')
        self._btn_run.pack(fill='x', pady=4)

        ttk.Button(left, text="清除輸出",
                   command=self._clear).pack(fill='x', pady=4)

        ttk.Separator(left, orient='horizontal').pack(fill='x', pady=10)

        # 說明
        note = (
            "固定參數：\n"
            f"  持有天數  : {HOLD_DAYS} 日\n"
            f"  大漲門檻  : >{RISE_THRESH}%\n"
            f"  大跌門檻  : <{FALL_THRESH}%\n"
            f"  相似度門檻: {SIM_THRESH}\n"
            f"  最少出現  : {MIN_FREQ} 次\n"
            f"  AE 潛在維度: {LATENT_DIM}\n"
            f"  AE Epochs : {EPOCHS}\n"
            f"  OCSVM nu  : {OCSVM_NU}\n"
            f"  手續費    : {BROKERAGE}% × 2\n"
            f"  交易稅    : {TAX}%（賣出）\n"
            f"  每筆成本  : {TRADE_COST}%\n"
            f"  訓練期間  : 2016-2024\n"
            f"  測試期間  : 2025"
        )
        ttk.Label(left, text=note, font=('Consolas', 9),
                  foreground='gray', justify='left').pack(anchor='w')

        self._progress = ttk.Progressbar(left, mode='indeterminate')
        self._progress.pack(fill='x', pady=(16, 0))

        # 右側
        right = ttk.Frame(self.root, padding=10)
        right.grid(row=0, column=1, sticky='nsew')
        right.rowconfigure(0, weight=2)
        right.rowconfigure(1, weight=3)
        right.columnconfigure(0, weight=1)

        self._log_box = scrolledtext.ScrolledText(
            right, wrap=tk.WORD, font=('Consolas', 10))
        self._log_box.grid(row=0, column=0, sticky='nsew', pady=(0, 6))

        self._fig    = Figure(figsize=(11, 4), dpi=100)
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew')

    # ── 佇列輪詢 ──────────────────────────────────────────────────────────────
    def _poll(self):
        try:
            while True:
                kind, data = self._queue.get_nowait()
                if kind == 'log':
                    self._log_box.insert(tk.END, data + '\n')
                    self._log_box.see(tk.END)
                elif kind == 'done_dl':
                    self._progress.stop()
                    self._running = False
                    self._btn_val.config(state='normal')
                    self._btn_grid.config(state='normal')
                    self._btn_run.config(state='normal')
                elif kind == 'done_run':
                    self._progress.stop()
                    self._running = False
                elif kind == 'plot':
                    self._draw_plot(data)
                elif kind == 'plot_val':
                    pass
                elif kind == 'plot_cm':
                    self._draw_confusion_matrices(data)
                elif kind == 'plot_grid':
                    self._draw_grid_results(data)
                elif kind == 'apply_best_thresh':
                    self._apply_best_thresh(data)
                elif kind == 'error':
                    self._progress.stop()
                    self._running = False
                    messagebox.showerror("錯誤", data)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _log(self, text: str):
        self._queue.put(('log', text))

    # ── 按鈕 ──────────────────────────────────────────────────────────────────
    def _on_grid_search(self):
        if self._running or self._raw_data is None: return
        self._running = True
        self._progress.start()
        self._log("\n" + "=" * 58)
        self._log(f" [3] SIM_THRESH 參數搜尋（測試：{GRID_THRESHOLDS}）")
        self._log("=" * 58)
        threading.Thread(target=self._worker_grid_search, daemon=True).start()

    def _on_validate(self):
        if self._running or self._raw_data is None: return
        self._running = True
        self._progress.start()
        self._log("\n" + "=" * 58)
        self._log(" [2] 特徵驗證（訓練 AE + OCSVM，不跑回測）")
        self._log("=" * 58)
        threading.Thread(target=self._worker_validate, daemon=True).start()

    def _on_download(self):
        if self._running: return
        self._running = True
        self._progress.start()
        self._log("=" * 58)
        self._log(" [1] 下載台灣前50大股票原始資料")
        self._log("=" * 58)
        threading.Thread(target=self._worker_download, daemon=True).start()

    def _on_run(self):
        if self._running or self._raw_data is None: return
        self._running = True
        self._progress.start()
        ticker = self._ticker_var.get().strip().upper()
        if not ticker:
            messagebox.showwarning("提示", "請輸入股票代號")
            return
        self._log("\n" + "=" * 58)
        self._log(f" [2] 分析 + 回測：{ticker} {TW50.get(ticker, '')}")
        self._log("=" * 58)
        threading.Thread(target=self._worker_run,
                         args=(ticker,), daemon=True).start()

    def _clear(self):
        self._log_box.delete('1.0', tk.END)
        self._fig.clear()
        self._canvas.draw()

    # ── Workers ───────────────────────────────────────────────────────────────
    def _worker_grid_search(self):
        try:
            all_grid = {}   # {feat_name: [result_per_thresh]}

            for feat_name, feat_fn in FEATURE_SETS:
                self._log(f"\n{'─'*55}")
                self._log(f"  特徵集：{feat_name}")
                self._log(f"{'─'*55}")
                df        = build_dataset(self._raw_data, feat_fn)
                feat_cols = [c for c in df.columns
                             if c not in ('future_ret', 'ticker')]
                results   = grid_search_sim_thresh(df, feat_cols, feat_name, self._log)
                all_grid[feat_name] = results

            # 找最佳門檻（三特徵集平均 Precision 最高者）
            thresh_avg = {}
            for thresh in GRID_THRESHOLDS:
                avg = np.mean([
                    next(r['avg_precision'] for r in res if r['thresh'] == thresh)
                    for res in all_grid.values()
                ])
                thresh_avg[thresh] = avg

            best_thresh = max(thresh_avg, key=thresh_avg.get)
            self._log(f"\n{'='*58}")
            self._log("  各門檻三特徵集平均 Precision：")
            for t, p in thresh_avg.items():
                mark = " ← 最佳" if t == best_thresh else ""
                self._log(f"  SIM_THRESH={t:.2f}  均Precision: {p:.2%}{mark}")
            self._log(f"\n  ★ 最佳 SIM_THRESH = {best_thresh}")

            self._queue.put(('plot_grid', (all_grid, best_thresh)))
            self._queue.put(('apply_best_thresh', best_thresh))
            self._log("\n  ✓ 參數搜尋完成")
            self._queue.put(('done_run', None))
        except Exception as e:
            import traceback
            self._queue.put(('error', traceback.format_exc()))

    def _worker_validate(self):
        try:
            val_results = {}
            for feat_name, feat_fn in FEATURE_SETS:
                self._log(f"\n{'─'*55}")
                self._log(f"  特徵集：{feat_name}")
                self._log(f"{'─'*55}")
                df        = build_dataset(self._raw_data, feat_fn)
                feat_cols = [c for c in df.columns
                             if c not in ('future_ret', 'ticker')]
                result = validate_features(df, feat_cols, self._log)
                val_results[feat_name] = result

            self._log(f"\n{'='*58}")
            self._log("  三特徵集驗證摘要")
            self._log(f"{'='*58}")
            self._log(f"  {'特徵集':<12}  {'方向':<6}  {'Precision':>10}  {'Recall':>8}  {'Accuracy':>10}  {'F1-Score':>10}")
            self._log(f"  {'─'*64}")
            for name, r in val_results.items():
                for direction, label in [('bull', '看漲'), ('bear', '看跌')]:
                    m = r.get(direction, {})
                    if m:
                        self._log(
                            f"  {name:<12}  {label:<6}  "
                            f"{m['precision']:>9.2%}  {m['recall']:>7.2%}  "
                            f"{m['accuracy']:>9.2%}  {m['f1']:>9.2%}"
                        )

            self._queue.put(('plot_cm', val_results))
            self._log("\n  ✓ 特徵驗證完成")
            self._queue.put(('done_run', None))
        except Exception as e:
            import traceback
            self._queue.put(('error', traceback.format_exc()))

    def _worker_download(self):
        try:
            raw = download_raw(self._log)
            self._raw_data = raw
            self._log(f"\n  成功下載：{len(raw)} 檔")
            self._log("  ✓ 下載完成，請點擊「② 分析 + 回測」")
            self._queue.put(('done_dl', None))
        except Exception as e:
            self._queue.put(('error', str(e)))

    def _worker_run(self, ticker: str):
        try:
            # 若輸入的股票不在已下載清單中，補下載
            if ticker not in self._raw_data:
                self._log(f"  {ticker} 不在台灣50清單，補充下載...")
                try:
                    df_extra = yf.download(
                        ticker, start='2015-01-01', end='2025-12-31',
                        auto_adjust=False, progress=False, actions=False,
                    )
                    if isinstance(df_extra.columns, pd.MultiIndex):
                        df_extra.columns = [col[0] for col in df_extra.columns]
                    if df_extra.empty or len(df_extra) < 100:
                        self._queue.put(('error', f"{ticker} 無法下載或資料不足，請確認代號是否正確"))
                        return
                    self._raw_data[ticker] = df_extra
                    self._log(f"  {ticker} 下載完成：{len(df_extra)} 筆")
                except Exception as e:
                    self._queue.put(('error', f"{ticker} 下載失敗：{e}"))
                    return

            results = {}   # {feat_name: backtest_result}

            for feat_name, feat_fn in FEATURE_SETS:
                self._log(f"\n{'─'*55}")
                self._log(f"  特徵集：{feat_name}")
                self._log(f"{'─'*55}")

                df        = build_dataset(self._raw_data, feat_fn)
                feat_cols = [c for c in df.columns
                             if c not in ('future_ret', 'ticker')]
                train_n   = len(df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)])
                self._log(f"  訓練筆數：{train_n:,}  特徵數：{len(feat_cols)}")

                models = find_and_train(df, feat_cols, self._log)
                res    = backtest(df, feat_cols, models, ticker)

                if res is None:
                    self._log(f"  [警告] {ticker} 無測試資料"); continue

                results[feat_name] = res
                self._log(f"\n  ── 回測結果 ──────────────────────────────")
                self._log(f"  買進訊號：{res['n_buy']:,}  賣出訊號：{res['n_sell']:,}  "
                          f"總計：{res['n_total']:,}")
                if res['n_total'] > 0:
                    self._log(f"  平均報酬(全)：{res['avg_ret']:+.3f}%  "
                              f"勝率：{res['win_rate']:.2%}  "
                              f"累積報酬：{res['cum_ret']:+.2f}%")
                else:
                    self._log("  （無符合型態的訊號）")

            # 總結
            self._log(f"\n{'='*58}")
            self._log(f"  三特徵集比較總結 ({ticker} {TW50.get(ticker,'')} 2025)")
            self._log(f"{'='*58}")
            self._log(f"  {'特徵集':^12}  {'買進':>5}  {'賣出':>5}  "
                      f"{'勝率':>7}  {'累積報酬':>10}")
            for name, r in results.items():
                self._log(f"  {name:^12}  {r['n_buy']:>5}  {r['n_sell']:>5}  "
                          f"{r['win_rate']:>7.2%}  {r['cum_ret']:>+9.2f}%")

            self._queue.put(('plot', results))
            self._log("\n  ✓ 分析完成")
            self._queue.put(('done_run', None))
        except Exception as e:
            import traceback
            self._queue.put(('error', traceback.format_exc()))

    # ── 最佳門檻寫回全域變數並更新原始碼 ────────────────────────────────────
    def _apply_best_thresh(self, best_thresh: float):
        global SIM_THRESH
        SIM_THRESH = best_thresh
        self._log(f"\n  ✔ SIM_THRESH 已自動更新為 {best_thresh}（本次執行期間有效）")

        # 同步寫回 .py 原始碼，讓下次啟動也沿用最佳值
        try:
            src = __file__
            with open(src, 'r', encoding='utf-8') as f:
                code = f.read()
            import re
            new_code = re.sub(
                r'^SIM_THRESH\s*=\s*[\d.]+',
                f'SIM_THRESH   = {best_thresh}',
                code, flags=re.MULTILINE
            )
            with open(src, 'w', encoding='utf-8') as f:
                f.write(new_code)
            self._log(f"  ✔ 已將 SIM_THRESH={best_thresh} 寫入原始碼（永久生效）")
        except Exception as e:
            self._log(f"  [警告] 無法寫入原始碼：{e}")

    # ── 參數搜尋結果圖 ────────────────────────────────────────────────────────
    def _draw_grid_results(self, payload):
        all_grid, best_thresh = payload
        self._fig.clear()

        n_feat = len(all_grid)
        colors = ['steelblue', 'darkorange', 'green']

        for col, (feat_name, results) in enumerate(all_grid.items()):
            thresholds     = [r['thresh']         for r in results]
            prec_bull      = [r['precision_bull']  for r in results]
            prec_bear      = [r['precision_bear']  for r in results]
            avg_prec       = [r['avg_precision']   for r in results]
            bull_samples   = [r['bull_samples']    for r in results]

            # 上排：Precision vs Threshold
            ax1 = self._fig.add_subplot(2, n_feat, col + 1)
            ax1.plot(thresholds, prec_bull, 'o-', color='#2ecc71',
                     label='看漲P', linewidth=1.5)
            ax1.plot(thresholds, prec_bear, 's-', color='#e74c3c',
                     label='看跌P', linewidth=1.5)
            ax1.plot(thresholds, avg_prec, '^--', color=colors[col],
                     label='均P', linewidth=2)
            ax1.axvline(best_thresh, color='gold', linestyle='--',
                        linewidth=1.5, label=f'最佳={best_thresh}')
            ax1.set_title(f"{feat_name}\nPrecision vs SIM_THRESH", fontsize=9)
            ax1.set_xlabel('SIM_THRESH', fontsize=8)
            ax1.set_ylabel('Precision', fontsize=8)
            ax1.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
            ax1.legend(fontsize=7)
            ax1.grid(alpha=0.3)

            # 下排：有效樣本數 vs Threshold
            ax2 = self._fig.add_subplot(2, n_feat, n_feat + col + 1)
            ax2.bar([str(t) for t in thresholds], bull_samples,
                    color=colors[col], alpha=0.7)
            best_idx = thresholds.index(best_thresh)
            ax2.bar(str(best_thresh), bull_samples[best_idx],
                    color='gold', alpha=0.9)
            ax2.set_title('看漲有效樣本數', fontsize=9)
            ax2.set_xlabel('SIM_THRESH', fontsize=8)
            ax2.set_ylabel('樣本數', fontsize=8)
            ax2.grid(alpha=0.3, axis='y')

        self._fig.suptitle(
            f'SIM_THRESH 參數搜尋結果  ★ 最佳門檻 = {best_thresh}',
            fontsize=11, fontweight='bold'
        )
        self._fig.tight_layout()
        self._canvas.draw()

    # ── 混淆矩陣圖（2列×3行：上看漲、下看跌）────────────────────────────────
    def _draw_confusion_matrices(self, val_results: dict):
        self._fig.clear()
        feat_names = list(val_results.keys())
        n = len(feat_names)
        directions = [('bull', '看漲'), ('bear', '看跌')]

        for row, (direction, dir_label) in enumerate(directions):
            for col, feat_name in enumerate(feat_names):
                ax = self._fig.add_subplot(2, n, row * n + col + 1)
                m  = val_results[feat_name].get(direction, {})
                if not m:
                    ax.axis('off'); continue

                TP, FN = m['TP'], m['FN']
                FP, TN = m['FP'], m['TN']
                cm = np.array([[TP, FN], [FP, TN]])

                im = ax.imshow(cm, cmap='Blues')
                self._fig.colorbar(im, ax=ax, shrink=0.8)

                labels = [['TP', 'FN'], ['FP', 'TN']]
                for i in range(2):
                    for j in range(2):
                        val = cm[i, j]
                        ax.text(j, i,
                                f"{labels[i][j]}\n{val:,}",
                                ha='center', va='center', fontsize=9,
                                color='white' if val > cm.max() * 0.5 else 'black')

                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(['預測正例', '預測負例'], fontsize=8)
                ax.set_yticklabels(['實際正例', '實際負例'], fontsize=8)
                ax.set_title(
                    f"{feat_name}  {dir_label}\n"
                    f"P:{m['precision']:.2%}  R:{m['recall']:.2%}  "
                    f"Acc:{m['accuracy']:.2%}  F1:{m['f1']:.2%}",
                    fontsize=8
                )

        self._fig.suptitle('混淆矩陣（上：看漲 / 下：看跌）', fontsize=11, fontweight='bold')
        self._fig.tight_layout()
        self._canvas.draw()

    # ── 繪圖（三特徵集累積報酬對比）────────────────────────────────────────
    def _draw_plot(self, results: dict):
        self._fig.clear()
        n = len(results)
        if n == 0: return

        colors = ['steelblue', 'darkorange', 'green']
        axes   = []

        # 上排：各特徵集累積報酬
        for i, (name, res) in enumerate(results.items()):
            ax = self._fig.add_subplot(1, n, i + 1)
            axes.append(ax)
            active = res['signal'] != 0
            if active.sum() > 0:
                cum = pd.Series(
                    res['strategy_ret'][active]
                ).cumsum().reset_index(drop=True)
                ax.plot(cum.values, color=colors[i], linewidth=1.5)
                ax.axhline(0, color='red', linestyle='--', linewidth=0.8)
                ax.fill_between(range(len(cum)), cum.values, 0,
                                where=cum.values >= 0, alpha=0.2, color='green')
                ax.fill_between(range(len(cum)), cum.values, 0,
                                where=cum.values <  0, alpha=0.2, color='red')
                ax.set_title(
                    f"{name}\n"
                    f"累積:{res['cum_ret']:+.1f}%  勝率:{res['win_rate']:.1%}",
                    fontsize=10)
            else:
                ax.set_title(f"{name}\n（無訊號）", fontsize=10)

            ax.set_xlabel('交易次數')
            ax.set_ylabel('累積報酬 (%)')
            ax.grid(alpha=0.3)

        self._fig.tight_layout()
        self._canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    root = tk.Tk()
    app  = KLineApp(root)
    root.mainloop()
