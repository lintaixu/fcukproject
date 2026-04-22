"""
全參數 Grid Search 腳本
搜尋：HOLD_DAYS × RISE_THRESH × DIST_THRESH × MIN_FREQ
速度優先：只用 1日-7特徵 跑完所有組合，找出最佳參數後寫入主程式
預估時間：約 30~50 分鐘
"""
import warnings, re, time, itertools
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import OneClassSVM

torch.manual_seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 固定不搜尋的參數
# ══════════════════════════════════════════════════════════════════════════════
MIN_BULL_RET = 2.0
MIN_BEAR_RET = -2.0
LATENT_DIM   = 4
EPOCHS       = 60
OCSVM_NU     = 0.1
TRAIN_START  = '2016-01-01'
TRAIN_END    = '2024-12-31'

# ══════════════════════════════════════════════════════════════════════════════
# 搜尋空間（共 3×3×3×2 = 54 組）
# ══════════════════════════════════════════════════════════════════════════════
GRID = {
    'hold_days':   [3, 5, 10],
    'rise_thresh': [2.0, 3.0, 5.0],   # fall_thresh = -rise_thresh
    'dist_thresh': [2.0, 2.5, 3.0],
    'min_freq':    [3, 5],
}

# ══════════════════════════════════════════════════════════════════════════════
# 台灣前50大股票
# ══════════════════════════════════════════════════════════════════════════════
TW50 = {
    '2330.TW':'台積電','2317.TW':'鴻海','2454.TW':'聯發科',
    '2308.TW':'台達電','2303.TW':'聯電','2412.TW':'中華電',
    '2881.TW':'富邦金','2882.TW':'國泰金','2886.TW':'兆豐金',
    '2891.TW':'中信金','2884.TW':'玉山金','2885.TW':'元大金',
    '2892.TW':'第一金','5880.TW':'合庫金','2880.TW':'華南金',
    '2002.TW':'中鋼','1301.TW':'台塑','1303.TW':'南亞',
    '1326.TW':'台化','3711.TW':'日月光投控','2207.TW':'和泰車',
    '2382.TW':'廣達','2395.TW':'研華','3008.TW':'大立光',
    '2408.TW':'南亞科','2357.TW':'華碩','4938.TW':'和碩',
    '3045.TW':'台灣大','2327.TW':'國巨','6505.TW':'台塑化',
    '5871.TW':'中租控股','2353.TW':'宏碁','2344.TW':'華邦電',
    '3034.TW':'聯詠','2301.TW':'光寶科','2912.TW':'統一超',
    '1216.TW':'統一','2615.TW':'萬海','2609.TW':'陽明',
    '2603.TW':'長榮','5876.TW':'上海商銀','2474.TW':'可成',
    '1101.TW':'台泥','2337.TW':'旺宏','3037.TW':'欣興',
    '6669.TW':'緯穎','2049.TW':'上銀','2379.TW':'瑞昱',
    '2105.TW':'正新','1590.TW':'亞德客',
}

# ══════════════════════════════════════════════════════════════════════════════
# 特徵函式（只用 1日-7特徵，速度最快）
# ══════════════════════════════════════════════════════════════════════════════
def detect_exdiv(df, tol=0.005):
    if 'Adj Close' not in df.columns or 'Close' not in df.columns:
        return pd.Series(False, index=df.index)
    return (df['Adj Close'] / df['Close']).pct_change().abs() > tol

def features_1day(df):
    O, H, L, C, V = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
    Cp = C.shift(1)
    return pd.DataFrame({
        'upper':     (H - np.maximum(O, C)) / C * 100,
        'lower':     (np.minimum(O, C) - L) / C * 100,
        'body':      (C - O) / C * 100,
        'gap':       (O - Cp) / C * 100,
        'close_chg': (C - Cp) / C * 100,
        'vol_ratio': (V - V.rolling(5, min_periods=5).mean()) / V.replace(0, np.nan),
        'trend':     (C.shift(2) - C.shift(7)) / C.shift(7) * 100,
    }, index=df.index)

def build_dataset(raw, hold_days):
    """根據 hold_days 建構資料集"""
    parts = []
    for t, df in raw.items():
        try:
            exdiv = detect_exdiv(df)
            feat  = features_1day(df)
            fret  = (df['Close'].shift(-hold_days) - df['Close']) / df['Close'] * 100
            c = feat.copy()
            c['future_ret'] = fret
            c['exdiv']      = exdiv
            c['ticker']     = t
            parts.append(c)
        except:
            continue
    full = pd.concat(parts)
    full = full[~full['exdiv']].drop(columns='exdiv')
    return full.replace([np.inf, -np.inf], np.nan).dropna()

# ══════════════════════════════════════════════════════════════════════════════
# Autoencoder
# ══════════════════════════════════════════════════════════════════════════════
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
    def encode(self, x):
        return self.encoder(x)

def train_ae(X, input_dim):
    loader = DataLoader(TensorDataset(torch.FloatTensor(X)),
                        batch_size=min(256, len(X)), shuffle=True,
                        drop_last=len(X) > 256)
    model   = Autoencoder(input_dim, LATENT_DIM)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(1, EPOCHS + 1):
        for (xb,) in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb)[0], xb)
            loss.backward(); opt.step()
        sched.step()
    return model

# ══════════════════════════════════════════════════════════════════════════════
# 單次評估函式
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_params(X_scaled, y, feat_cols,
                    rise_thresh, dist_thresh, min_freq):
    """
    對一組參數評估 Precision。
    回傳 (precision_bull, precision_bear, bull_samples, bear_samples)
    """
    bull_seeds = np.where(y >  rise_thresh)[0]
    bear_seeds = np.where(y < -rise_thresh)[0]

    if len(bull_seeds) == 0 or len(bear_seeds) == 0:
        return 0.0, 0.0, 0, 0

    results = {}
    for direction, seeds, ret_ok in [
        ('bull', bull_seeds, lambda r: r >  MIN_BULL_RET),
        ('bear', bear_seeds, lambda r: r < MIN_BEAR_RET),
    ]:
        idx_importance = {}
        BATCH = 200
        for b in range(0, len(seeds), BATCH):
            batch     = seeds[b:b + BATCH]
            dists_mat = euclidean_distances(X_scaled[batch], X_scaled)
            for j, seed_idx in enumerate(batch):
                row = dists_mat[j].copy()
                row[seed_idx] = np.inf
                sim_idx = np.where(row <= dist_thresh)[0]
                if len(sim_idx) < min_freq: continue
                avg_ret = float(y[sim_idx].mean())
                if not ret_ok(avg_ret): continue
                importance = len(sim_idx) * abs(avg_ret)
                for idx in [seed_idx, *sim_idx.tolist()]:
                    if idx not in idx_importance or idx_importance[idx] < importance:
                        idx_importance[idx] = importance

        results[direction] = idx_importance

    precision = {}
    for direction, idx_importance in results.items():
        if len(idx_importance) < 10:
            precision[direction] = 0.0
            continue

        sorted_idx = sorted(idx_importance.keys())
        X_pat = X_scaled[sorted_idx]

        # 訓練 AE
        ae = train_ae(X_pat, len(feat_cols))
        ae.eval()
        with torch.no_grad():
            Z_pat = ae.encode(torch.FloatTensor(X_pat)).numpy()
            Z_all = ae.encode(torch.FloatTensor(X_scaled)).numpy()

        # 訓練 OCSVM（含重要性權重）
        sw = np.array([idx_importance[i] for i in sorted_idx], dtype=np.float32)
        sw = sw / sw.sum() * len(sw)
        ocsvm = OneClassSVM(kernel='rbf', nu=OCSVM_NU, gamma='scale')
        ocsvm.fit(Z_pat, sample_weight=sw)
        pos = ocsvm.predict(Z_all) == 1

        if pos.sum() == 0:
            precision[direction] = 0.0
        elif direction == 'bull':
            precision[direction] = float((y[pos] > MIN_BULL_RET).mean())
        else:
            precision[direction] = float((y[pos] < MIN_BEAR_RET).mean())

    bull_s = len(results['bull'])
    bear_s = len(results['bear'])
    return precision.get('bull', 0.0), precision.get('bear', 0.0), bull_s, bear_s

# ══════════════════════════════════════════════════════════════════════════════
# 下載資料
# ══════════════════════════════════════════════════════════════════════════════
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
            if df.empty or len(df) < 100:
                print("skip"); continue
            raw[t] = df
            print(f"{len(df)}筆 OK")
        except Exception as e:
            print(f"ERR:{e}")
    return raw

# ══════════════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    total_combos = (len(GRID['hold_days']) *
                    len(GRID['rise_thresh']) *
                    len(GRID['dist_thresh']) *
                    len(GRID['min_freq']))

    print("=" * 65)
    print(f"  全參數 Grid Search — 共 {total_combos} 組合（1日-7特徵）")
    print(f"  搜尋空間：")
    for k, v in GRID.items():
        print(f"    {k:15s} = {v}")
    print("=" * 65)

    # ── 1. 下載資料 ───────────────────────────────────────────────────────────
    print("\n[Step 1] 下載原始資料...", flush=True)
    raw = download_raw()
    print(f"  完成：{len(raw)} 檔\n", flush=True)

    # ── 2. 預先建立各 hold_days 的 dataset（節省重複計算）────────────────────
    print("[Step 2] 建構各 HOLD_DAYS 資料集...", flush=True)
    datasets = {}
    for hold in GRID['hold_days']:
        print(f"  HOLD_DAYS={hold}...", end=' ', flush=True)
        df = build_dataset(raw, hold)
        feat_cols = [c for c in df.columns if c not in ('future_ret', 'ticker')]
        train_df  = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
        X_raw     = train_df[feat_cols].values.astype(np.float32)
        y         = train_df['future_ret'].values.astype(np.float32)
        scaler    = StandardScaler()
        X_scaled  = np.clip(scaler.fit_transform(X_raw), -5, 5)
        datasets[hold] = {'X_scaled': X_scaled, 'y': y, 'feat_cols': feat_cols}
        print(f"{len(X_raw):,} 筆 OK", flush=True)

    # ── 3. Grid Search ────────────────────────────────────────────────────────
    print(f"\n[Step 3] 開始搜尋（共 {total_combos} 組）...\n", flush=True)
    print(f"  {'#':>4}  {'HOLD':>4}  {'RISE':>5}  {'DIST':>5}  {'FREQ':>4}  "
          f"{'P_漲':>8}  {'P_跌':>8}  {'均P':>8}  {'漲樣本':>7}  {'跌樣本':>7}",
          flush=True)
    print("  " + "-" * 72, flush=True)

    all_results = []
    combo_num   = 0

    for hold, rise, dist, freq in itertools.product(
            GRID['hold_days'], GRID['rise_thresh'],
            GRID['dist_thresh'], GRID['min_freq']):

        combo_num += 1
        data = datasets[hold]
        X_scaled  = data['X_scaled']
        y         = data['y']
        feat_cols = data['feat_cols']

        p_bull, p_bear, bull_s, bear_s = evaluate_params(
            X_scaled, y, feat_cols, rise, dist, freq)

        avg_p = (p_bull + p_bear) / 2

        print(f"  {combo_num:>4}  {hold:>4}  {rise:>5.1f}  {dist:>5.1f}  {freq:>4}  "
              f"{p_bull:>7.2%}  {p_bear:>7.2%}  {avg_p:>7.2%}  "
              f"{bull_s:>7,}  {bear_s:>7,}",
              flush=True)

        all_results.append({
            'hold_days':   hold,
            'rise_thresh': rise,
            'dist_thresh': dist,
            'min_freq':    freq,
            'precision_bull': p_bull,
            'precision_bear': p_bear,
            'avg_precision':  avg_p,
            'bull_samples':   bull_s,
            'bear_samples':   bear_s,
        })

    # ── 4. 結果排序 ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results).sort_values(
        'avg_precision', ascending=False).reset_index(drop=True)

    elapsed = time.time() - t0
    print(f"\n\n{'='*65}")
    print(f"  搜尋完成！耗時 {elapsed/60:.1f} 分鐘")
    print(f"{'='*65}")
    print("\n  Top-10 最佳組合：")
    print(f"  {'排名':>4}  {'HOLD':>4}  {'RISE':>5}  {'DIST':>5}  {'FREQ':>4}  "
          f"{'均Precision':>12}")
    print("  " + "-" * 50)
    for i, row in results_df.head(10).iterrows():
        print(f"  {i+1:>4}  {int(row.hold_days):>4}  {row.rise_thresh:>5.1f}  "
              f"{row.dist_thresh:>5.1f}  {int(row.min_freq):>4}  "
              f"{row.avg_precision:>11.2%}")

    # ── 5. 儲存完整結果 ───────────────────────────────────────────────────────
    out_csv = 'C:/Users/alex/Desktop/專題進度/0401/grid_search_results.csv'
    results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n  完整結果已存至：{out_csv}")

    # ── 6. 寫入最佳參數 ───────────────────────────────────────────────────────
    best = results_df.iloc[0]
    print(f"\n  ★ 最佳組合：")
    print(f"     HOLD_DAYS   = {int(best.hold_days)}")
    print(f"     RISE_THRESH = {best.rise_thresh}")
    print(f"     DIST_THRESH = {best.dist_thresh}")
    print(f"     MIN_FREQ    = {int(best.min_freq)}")
    print(f"     均Precision = {best.avg_precision:.2%}")

    target = 'C:/Users/alex/Desktop/專題進度/0401/kline_pattern_search.py'
    with open(target, 'r', encoding='utf-8') as f:
        code = f.read()

    code = re.sub(r'^HOLD_DAYS\s*=\s*\d+',
                  f'HOLD_DAYS    = {int(best.hold_days)}',
                  code, flags=re.MULTILINE)
    code = re.sub(r'^RISE_THRESH\s*=\s*[\d.]+',
                  f'RISE_THRESH  = {best.rise_thresh}',
                  code, flags=re.MULTILINE)
    code = re.sub(r'^FALL_THRESH\s*=\s*[-\d.]+',
                  f'FALL_THRESH  = {-best.rise_thresh}',
                  code, flags=re.MULTILINE)
    code = re.sub(r'^DIST_THRESH\s*=\s*[\d.]+',
                  f'DIST_THRESH  = {best.dist_thresh}',
                  code, flags=re.MULTILINE)
    code = re.sub(r'^MIN_FREQ\s*=\s*\d+',
                  f'MIN_FREQ     = {int(best.min_freq)}',
                  code, flags=re.MULTILINE)

    with open(target, 'w', encoding='utf-8') as f:
        f.write(code)

    print(f"\n  ✔ 已將最佳參數寫入 kline_pattern_search.py")
    print("  完成！")
