"""
OCSVM_NU 逐步收緊測試
測試 nu = [0.1, 0.05, 0.01]
觀察 Precision / Recall / Accuracy / F1 的變化
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

torch.manual_seed(42)
np.random.seed(42)

# ── 固定參數（與主程式一致）────────────────────────────────────────────────────
HOLD_DAYS    = 5
RISE_THRESH  = 5.0
FALL_THRESH  = -5.0
MIN_BULL_RET = 2.0
MIN_BEAR_RET = -2.0
LATENT_DIM   = 4
EPOCHS       = 60
TRAIN_START  = '2016-01-01'
TRAIN_END    = '2024-12-31'

NU_LIST = [0.1, 0.05, 0.01]   # 逐步收緊

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

# ── 特徵（只用 1日-7特徵，速度最快）────────────────────────────────────────────
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

def build_dataset(raw):
    parts = []
    for t, df in raw.items():
        try:
            exdiv = detect_exdiv(df)
            feat  = features_1day(df)
            fret  = (df['Close'].shift(-HOLD_DAYS) - df['Close']) / df['Close'] * 100
            c = feat.copy(); c['future_ret'] = fret
            c['exdiv'] = exdiv; c['ticker'] = t
            parts.append(c)
        except: continue
    full = pd.concat(parts)
    full = full[~full['exdiv']].drop(columns='exdiv')
    return full.replace([np.inf, -np.inf], np.nan).dropna()

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
    def forward(self, x): z = self.encoder(x); return self.decoder(z), z
    def encode(self, x): return self.encoder(x)

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
            opt.zero_grad(); loss = loss_fn(model(xb)[0], xb)
            loss.backward(); opt.step()
        sched.step()
        if ep % 20 == 0: print(f"    Epoch {ep}/{EPOCHS}", flush=True)
    return model

# ── 計算四大指標 ──────────────────────────────────────────────────────────────
def calc_metrics(actual_pos, pred_pos):
    N  = len(actual_pos)
    TP = int(( pred_pos &  actual_pos).sum())
    TN = int((~pred_pos & ~actual_pos).sum())
    FP = int(( pred_pos & ~actual_pos).sum())
    FN = int((~pred_pos &  actual_pos).sum())
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy  = (TP + TN) / N  if N > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                precision=precision, recall=recall,
                accuracy=accuracy, f1=f1)

# ── 下載 ─────────────────────────────────────────────────────────────────────
def download_raw():
    raw = {}
    for i, (t, name) in enumerate(TW50.items()):
        print(f"  [{i+1:02d}/{len(TW50)}] {t} {name}...", end=' ', flush=True)
        try:
            df = yf.download(t, start='2015-01-01', end='2025-12-31',
                             auto_adjust=False, progress=False, actions=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            if df.empty or len(df) < 100: print("skip"); continue
            raw[t] = df; print(f"{len(df)}筆 OK")
        except Exception as e: print(f"ERR:{e}")
    return raw

# ── 主程式 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print(f"  OCSVM_NU 收緊測試  nu = {NU_LIST}")
    print("=" * 65)

    print("\n[1] 下載資料...", flush=True)
    raw = download_raw()
    print(f"  完成：{len(raw)} 檔\n")

    print("[2] 建構資料集 + 訓練 AE（只需一次）...", flush=True)
    df        = build_dataset(raw)
    feat_cols = [c for c in df.columns if c not in ('future_ret', 'ticker')]
    train_df  = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    X = train_df[feat_cols].values.astype(np.float32)
    y = train_df['future_ret'].values.astype(np.float32)
    scaler    = StandardScaler()
    X_scaled  = np.clip(scaler.fit_transform(X), -5, 5)

    ae = train_ae(X_scaled, len(feat_cols))
    ae.eval()
    with torch.no_grad():
        Z = ae.encode(torch.FloatTensor(X_scaled)).numpy()
    print(f"  AE 完成，Z shape: {Z.shape}\n")

    bull_mask = y > RISE_THRESH
    bear_mask = y < FALL_THRESH
    print(f"  看漲種子：{bull_mask.sum():,}  看跌種子：{bear_mask.sum():,}\n")

    # ── 逐一測試 nu ───────────────────────────────────────────────────────────
    all_results = []

    for nu in NU_LIST:
        print(f"\n{'='*65}")
        print(f"  OCSVM_NU = {nu}")
        print(f"{'='*65}")

        row = {'nu': nu}

        for direction, mask, actual_thresh, label in [
            ('bull', bull_mask, lambda y: y > MIN_BULL_RET, '看漲'),
            ('bear', bear_mask, lambda y: y < MIN_BEAR_RET, '看跌'),
        ]:
            ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
            ocsvm.fit(Z[mask])
            pred_pos   = ocsvm.predict(Z) == 1
            actual_pos = actual_thresh(y)
            m = calc_metrics(actual_pos, pred_pos)

            n_pred = pred_pos.sum()
            print(f"\n  [{label}]  預測正例：{n_pred:,}（佔全部 {n_pred/len(y):.1%}）")
            print(f"    TP={m['TP']:,}  TN={m['TN']:,}  FP={m['FP']:,}  FN={m['FN']:,}")
            print(f"    Precision : {m['precision']:.4f}  ({m['precision']:.2%})")
            print(f"    Recall    : {m['recall']:.4f}  ({m['recall']:.2%})")
            print(f"    Accuracy  : {m['accuracy']:.4f}  ({m['accuracy']:.2%})")
            print(f"    F1-Score  : {m['f1']:.4f}  ({m['f1']:.2%})")

            row[f'{direction}_precision'] = m['precision']
            row[f'{direction}_recall']    = m['recall']
            row[f'{direction}_f1']        = m['f1']
            row[f'{direction}_fp']        = m['FP']
            row[f'{direction}_pred_pct']  = n_pred / len(y)

        all_results.append(row)

    # ── 總結對比表 ────────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  總結對比（1日-7特徵，看漲）")
    print(f"{'='*65}")
    print(f"  {'nu':>6}  {'預測正例%':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'FP':>8}")
    print(f"  {'─'*60}")
    for r in all_results:
        print(f"  {r['nu']:>6}  {r['bull_pred_pct']:>9.1%}  "
              f"{r['bull_precision']:>9.2%}  {r['bull_recall']:>7.2%}  "
              f"{r['bull_f1']:>7.2%}  {r['bull_fp']:>8,}")

    print(f"\n  總結對比（看跌）")
    print(f"  {'nu':>6}  {'預測正例%':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'FP':>8}")
    print(f"  {'─'*60}")
    for r in all_results:
        print(f"  {r['nu']:>6}  {r['bear_pred_pct']:>9.1%}  "
              f"{r['bear_precision']:>9.2%}  {r['bear_recall']:>7.2%}  "
              f"{r['bear_f1']:>7.2%}  {r['bear_fp']:>8,}")

    print("\n  完成！")
