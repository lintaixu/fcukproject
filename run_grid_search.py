"""
SIM_THRESH 參數搜尋 — 獨立執行腳本（不需要 GUI）
完成後自動將最佳值寫入 kline_pattern_search.py
"""
import warnings
warnings.filterwarnings('ignore')

import re, sys
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import OneClassSVM

torch.manual_seed(42)
np.random.seed(42)

# ── 參數（與主程式一致）────────────────────────────────────────────────────────
HOLD_DAYS    = 3
RISE_THRESH  = 5.0
FALL_THRESH  = -5.0
MIN_FREQ     = 3
MIN_BULL_RET = 2.0
MIN_BEAR_RET = -2.0
LATENT_DIM   = 4
EPOCHS       = 60
OCSVM_NU     = 0.3
TRAIN_START  = '2016-01-01'
TRAIN_END    = '2024-12-31'

GRID_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

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

# ── 特徵函式 ──────────────────────────────────────────────────────────────────
def detect_exdiv_mask(df, tol=0.005):
    if 'Adj Close' not in df.columns or 'Close' not in df.columns:
        return pd.Series(False, index=df.index)
    return (df['Adj Close'] / df['Close']).pct_change().abs() > tol

def features_1day(df):
    O,H,L,C,V = df['Open'],df['High'],df['Low'],df['Close'],df['Volume']
    Cp = C.shift(1)
    return pd.DataFrame({
        'upper': (H-np.maximum(O,C))/C*100, 'lower': (np.minimum(O,C)-L)/C*100,
        'body': (C-O)/C*100, 'gap': (O-Cp)/C*100, 'close_chg': (C-Cp)/C*100,
        'vol_ratio': (V-V.rolling(5,min_periods=5).mean())/V.replace(0,np.nan),
        'trend': (C.shift(2)-C.shift(7))/C.shift(7)*100,
    }, index=df.index)

def features_2day(df):
    O,H,L,C,V = df['Open'],df['High'],df['Low'],df['Close'],df['Volume']
    O1,H1,L1,C1 = O.shift(1),H.shift(1),L.shift(1),C.shift(1)
    return pd.DataFrame({
        'upper': (H-np.maximum(O,C))/C*100, 'lower': (np.minimum(O,C)-L)/C*100,
        'body': (C-O)/C*100,
        'p_upper': (H1-np.maximum(O1,C1))/C1*100,
        'p_lower': (np.minimum(O1,C1)-L1)/C1*100, 'p_body': (C1-O1)/C1*100,
        'gap': (O-C1)/C*100, 'close_chg': (C-C1)/C*100,
        'vol_ratio': (V-V.rolling(5,min_periods=5).mean())/V.replace(0,np.nan),
        'trend': (C.shift(2)-C.shift(7))/C.shift(7)*100,
    }, index=df.index)

def features_3day(df):
    O,H,L,C,V = df['Open'],df['High'],df['Low'],df['Close'],df['Volume']
    O1,H1,L1,C1 = O.shift(1),H.shift(1),L.shift(1),C.shift(1)
    O2,H2,L2,C2 = O.shift(2),H.shift(2),L.shift(2),C.shift(2)
    h_max = pd.concat([H,H1,H2],axis=1).max(axis=1)
    l_min = pd.concat([L,L1,L2],axis=1).min(axis=1)
    denom = h_max - l_min
    return pd.DataFrame({
        'upper': (H-np.maximum(O,C))/C*100, 'lower': (np.minimum(O,C)-L)/C*100,
        'body': (C-O)/C*100, 'p1_body': (C1-O1)/C1*100, 'p2_body': (C2-O2)/C2*100,
        'gap': (O-C1)/C*100, 'close_chg': (C-C1)/C*100,
        'p1_gap': (O1-C2)/C1*100, 'p1_close_chg': (C1-C2)/C1*100,
        'cum_return': (C-C2)/C2*100,
        'vol_ratio_3': (V-V.rolling(3,min_periods=3).mean())/V.replace(0,np.nan),
        'range_pos': pd.Series(np.where(denom>0,(C-l_min)/denom*100,50.0),index=df.index),
    }, index=df.index)

FEATURE_SETS = [
    ('1日-7特徵',  features_1day),
    ('2日-10特徵', features_2day),
    ('3日-12特徵', features_3day),
]

# ── Autoencoder ───────────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super().__init__()
        h = max(16, input_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, 8), nn.ReLU(), nn.Linear(8, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, h), nn.ReLU(), nn.Linear(h, input_dim))
    def forward(self, x):
        z = self.encoder(x); return self.decoder(z), z
    def encode(self, x): return self.encoder(x)

def train_ae(X, input_dim):
    loader  = DataLoader(TensorDataset(torch.FloatTensor(X)),
                         batch_size=min(256,len(X)), shuffle=True,
                         drop_last=len(X)>256)
    model   = Autoencoder(input_dim, LATENT_DIM)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(1, EPOCHS+1):
        for (xb,) in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb)[0], xb)
            loss.backward(); opt.step()
        sched.step()
        if ep % 20 == 0:
            print(f"      Epoch {ep}/{EPOCHS}", flush=True)
    return model

# ── 資料下載 ──────────────────────────────────────────────────────────────────
def download_raw():
    raw = {}
    tickers = list(TW50.keys())
    for i, t in enumerate(tickers):
        print(f"  [{i+1:02d}/{len(tickers)}] {t} {TW50[t]}...", end=' ', flush=True)
        try:
            df = yf.download(t, start='2015-01-01', end='2025-12-31',
                             auto_adjust=False, progress=False, actions=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            if df.empty or len(df) < 100: print("skip"); continue
            raw[t] = df; print(f"{len(df)} 筆 OK")
        except Exception as e:
            print(f"ERR: {e}")
    return raw

def build_dataset(raw, feat_fn):
    parts = []
    for t, df in raw.items():
        try:
            exdiv = detect_exdiv_mask(df)
            feat  = feat_fn(df)
            fret  = (df['Close'].shift(-HOLD_DAYS) - df['Close']) / df['Close'] * 100
            c = feat.copy(); c['future_ret'] = fret
            c['exdiv'] = exdiv; c['ticker'] = t
            parts.append(c)
        except: continue
    full = pd.concat(parts)
    full = full[~full['exdiv']].drop(columns='exdiv')
    return full.replace([np.inf,-np.inf], np.nan).dropna()

# ── Grid Search ───────────────────────────────────────────────────────────────
def run_grid_search(df, feat_cols, feat_name):
    print(f"\n  訓練 AE（全部資料）...", flush=True)
    train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    X = train_df[feat_cols].values.astype(np.float32)
    y = train_df['future_ret'].values.astype(np.float32)

    scaler   = StandardScaler()
    X_scaled = np.clip(scaler.fit_transform(X), -5, 5)

    ae = train_ae(X_scaled, len(feat_cols))
    ae.eval()
    with torch.no_grad():
        Z_all = ae.encode(torch.FloatTensor(X_scaled)).numpy()
    print(f"  AE 完成，Z shape: {Z_all.shape}", flush=True)

    bull_seeds = np.where(y > RISE_THRESH)[0]
    bear_seeds = np.where(y < FALL_THRESH)[0]
    print(f"  看漲種子: {len(bull_seeds):,}  看跌種子: {len(bear_seeds):,}", flush=True)

    results = []
    for thresh in GRID_THRESHOLDS:
        bull_valid, bear_valid = set(), set()
        for seeds, valid_set, ret_ok in [
            (bull_seeds, bull_valid, lambda r: r > MIN_BULL_RET),
            (bear_seeds, bear_valid, lambda r: r < MIN_BEAR_RET),
        ]:
            for b in range(0, len(seeds), 200):
                batch = seeds[b:b+200]
                sims  = cosine_similarity(X_scaled[batch], X_scaled)
                for j, idx in enumerate(batch):
                    row = sims[j].copy(); row[idx] = 0
                    similar = np.where(row >= thresh)[0]
                    if len(similar) < MIN_FREQ: continue
                    if not ret_ok(float(y[similar].mean())): continue
                    valid_set.add(idx); valid_set.update(similar.tolist())

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

        avg = (precision_bull + precision_bear) / 2
        print(f"  thresh={thresh:.2f}  漲樣本:{len(bull_valid):>5,}  "
              f"跌樣本:{len(bear_valid):>5,}  "
              f"P_漲:{precision_bull:.2%}  P_跌:{precision_bear:.2%}  均:{avg:.2%}",
              flush=True)
        results.append({'thresh': thresh, 'bull_samples': len(bull_valid),
                        'bear_samples': len(bear_valid),
                        'precision_bull': precision_bull,
                        'precision_bear': precision_bear, 'avg_precision': avg})
    return results

# ── 主程式 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print(" SIM_THRESH 參數搜尋")
    print("=" * 60)

    print("\n[1] 下載資料...")
    raw = download_raw()
    print(f"  下載完成：{len(raw)} 檔\n")

    all_results = {}
    for feat_name, feat_fn in FEATURE_SETS:
        print(f"\n{'='*60}")
        print(f"  特徵集：{feat_name}")
        print(f"{'='*60}")
        df        = build_dataset(raw, feat_fn)
        feat_cols = [c for c in df.columns if c not in ('future_ret','ticker')]
        all_results[feat_name] = run_grid_search(df, feat_cols, feat_name)

    # ── 找最佳門檻 ────────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  總結：三特徵集各門檻均 Precision")
    print(f"{'='*60}")
    thresh_avg = {}
    for thresh in GRID_THRESHOLDS:
        avg = np.mean([
            next(r['avg_precision'] for r in res if r['thresh'] == thresh)
            for res in all_results.values()
        ])
        thresh_avg[thresh] = avg
        print(f"  SIM_THRESH={thresh:.2f}  三特徵集均Precision: {avg:.2%}")

    best = max(thresh_avg, key=thresh_avg.get)
    print(f"\n  ★ 最佳 SIM_THRESH = {best}  (均Precision={thresh_avg[best]:.2%})")

    # ── 寫入 kline_pattern_search.py ─────────────────────────────────────────
    target = 'C:/Users/alex/Desktop/專題進度/0401/kline_pattern_search.py'
    with open(target, 'r', encoding='utf-8') as f:
        code = f.read()
    new_code = re.sub(
        r'^SIM_THRESH\s*=\s*[\d.]+',
        f'SIM_THRESH   = {best}',
        code, flags=re.MULTILINE
    )
    with open(target, 'w', encoding='utf-8') as f:
        f.write(new_code)
    print(f"\n  ✔ 已將 SIM_THRESH={best} 寫入 kline_pattern_search.py")
    print("  完成！")
