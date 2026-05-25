#!/usr/bin/env python3
"""
K線型態分析：Autoencoder + One-Class SVM
=============================================
特徵集：1日(7特徵)、2日(10特徵)、3日(12特徵)
訓練資料：2016-2024
回測資料：2025
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

torch.manual_seed(42)
np.random.seed(42)

# ── 台灣50成分股 ──────────────────────────────────────────────────────────────
TW50 = [
    '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2303.TW',
    '2412.TW', '2881.TW', '2882.TW', '2886.TW', '2891.TW',
    '2884.TW', '2885.TW', '2892.TW', '5880.TW', '2880.TW',
    '2002.TW', '1301.TW', '1303.TW', '1326.TW', '3711.TW',
    '2207.TW', '2382.TW', '2395.TW', '3008.TW', '2408.TW',
    '2357.TW', '4938.TW', '3045.TW', '2327.TW', '6505.TW',
    '5871.TW', '2353.TW', '2344.TW', '3034.TW', '2301.TW',
    '2912.TW', '1216.TW', '2615.TW', '2609.TW', '2603.TW',
    '5876.TW', '2474.TW', '1101.TW', '2337.TW', '3037.TW',
    '6669.TW', '2049.TW', '2379.TW', '2105.TW', '1590.TW',
]
TW50 = list(dict.fromkeys(TW50))  # 去重

# ── 全域參數 ──────────────────────────────────────────────────────────────────
RISE_THRESH = 5.0    # 大漲門檻 (%)
FALL_THRESH = -5.0   # 大跌門檻 (%)
HOLD_DAYS   = 3      # 持有天數
TRAIN_START = '2016-01-01'
TRAIN_END   = '2024-12-31'
TEST_START  = '2025-01-01'
TEST_END    = '2025-12-31'
LATENT_DIM  = 4
BATCH_SIZE  = 256
EPOCHS      = 60
LR          = 1e-3


# ══════════════════════════════════════════════════════════════════════════════
# 除權息偵測
# ══════════════════════════════════════════════════════════════════════════════
def detect_exdiv_mask(df: pd.DataFrame, tol: float = 0.005) -> pd.Series:
    """
    偵測除權息日。
    比較 Adj Close / Close 的比率前後變化，變動超過 tol 則標記為除權息日。
    """
    if 'Adj Close' not in df.columns or 'Close' not in df.columns:
        return pd.Series(False, index=df.index)
    ratio = df['Adj Close'] / df['Close']
    ratio_chg = ratio.pct_change().abs()
    return ratio_chg > tol


# ══════════════════════════════════════════════════════════════════════════════
# K線特徵計算
# ══════════════════════════════════════════════════════════════════════════════
def features_1day(df: pd.DataFrame) -> pd.DataFrame:
    """
    1日K線特徵 (7個)
    上影線、下影線、實體線、開盤缺口、收盤漲跌、5日均量比、前五日趨勢
    """
    O, H, L, C, V = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
    Cp = C.shift(1)

    upper     = (H - np.maximum(O, C)) / C * 100
    lower     = (np.minimum(O, C) - L) / C * 100
    body      = (C - O) / C * 100
    gap       = (O - Cp) / C * 100
    close_chg = (C - Cp) / C * 100
    v5ma      = V.rolling(5, min_periods=5).mean()
    vol_ratio = (V - v5ma) / V
    trend     = (C.shift(2) - C.shift(7)) / C.shift(7) * 100

    return pd.DataFrame({
        'upper': upper, 'lower': lower, 'body': body,
        'gap': gap, 'close_chg': close_chg,
        'vol_ratio': vol_ratio, 'trend': trend
    }, index=df.index)


def features_2day(df: pd.DataFrame) -> pd.DataFrame:
    """
    2日K線特徵 (10個)
    當日3個 + 前日3個 + 開盤缺口、收盤漲跌、5日均量比、前五日趨勢
    """
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
    vol_ratio = (V  - v5ma) / V
    trend     = (C.shift(2) - C.shift(7)) / C.shift(7) * 100

    return pd.DataFrame({
        'upper': upper, 'lower': lower, 'body': body,
        'p_upper': p_upper, 'p_lower': p_lower, 'p_body': p_body,
        'gap': gap, 'close_chg': close_chg,
        'vol_ratio': vol_ratio, 'trend': trend
    }, index=df.index)


def features_3day(df: pd.DataFrame) -> pd.DataFrame:
    """
    3日K線特徵 (12個)
    當日實體線3個 + 前日/前兩日實體線 + 開盤/收盤型態×2日
    + 3日累積漲跌、3日均量比、3日區間位階
    """
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
    vol_ratio_3  = (V - v3ma) / V

    # 3日區間位階
    h_max = pd.concat([H, H1, H2], axis=1).max(axis=1)
    l_min = pd.concat([L, L1, L2], axis=1).min(axis=1)
    denom = h_max - l_min
    range_pos = np.where(denom > 0, (C - l_min) / denom * 100, 50.0)
    range_pos = pd.Series(range_pos, index=df.index)

    return pd.DataFrame({
        'upper': upper, 'lower': lower, 'body': body,
        'p1_body': p1_body, 'p2_body': p2_body,
        'gap': gap, 'close_chg': close_chg,
        'p1_gap': p1_gap, 'p1_close_chg': p1_close_chg,
        'cum_return': cum_return,
        'vol_ratio_3': vol_ratio_3,
        'range_pos': range_pos
    }, index=df.index)


def get_future_return(df: pd.DataFrame, n: int = 3) -> pd.Series:
    """n日後收盤漲跌幅 (%)"""
    return (df['Close'].shift(-n) - df['Close']) / df['Close'] * 100


# ══════════════════════════════════════════════════════════════════════════════
# Autoencoder
# ══════════════════════════════════════════════════════════════════════════════
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 4):
        super().__init__()
        h = max(16, input_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Linear(h, 8),         nn.ReLU(),
            nn.Linear(8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, h),          nn.ReLU(),
            nn.Linear(h, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(X: np.ndarray, input_dim: int,
                      latent_dim: int = LATENT_DIM,
                      epochs: int = EPOCHS, lr: float = LR) -> Autoencoder:
    X_t    = torch.FloatTensor(X)
    loader = DataLoader(TensorDataset(X_t), batch_size=BATCH_SIZE, shuffle=True)
    model  = Autoencoder(input_dim, latent_dim)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        for (xb,) in loader:
            opt.zero_grad()
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        if ep % 10 == 0:
            print(f"    Epoch {ep:3d}/{epochs}  loss={epoch_loss/len(loader):.5f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 資料建構
# ══════════════════════════════════════════════════════════════════════════════
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """處理 yfinance 多層欄位名稱"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def build_dataset(feat_fn, verbose: bool = True):
    """下載台灣50資料，計算特徵與標籤，分割訓練/測試集"""
    all_parts = []
    success, fail = 0, 0

    for ticker in TW50:
        try:
            raw = yf.download(
                ticker,
                start='2015-01-01',   # 多抓一年讓 rolling 指標暖身
                end='2025-12-31',
                auto_adjust=False,
                progress=False,
                actions=False
            )
            raw = flatten_columns(raw)
            if raw.empty or len(raw) < 100:
                fail += 1; continue
            if not {'Open','High','Low','Close','Volume','Adj Close'}.issubset(raw.columns):
                fail += 1; continue

            exdiv_mask = detect_exdiv_mask(raw)
            feat       = feat_fn(raw)
            fret       = get_future_return(raw, HOLD_DAYS)

            combined = feat.copy()
            combined['future_ret'] = fret
            combined['exdiv']      = exdiv_mask
            combined['ticker']     = ticker
            all_parts.append(combined)
            success += 1
        except Exception as e:
            fail += 1
            if verbose:
                print(f"  [SKIP] {ticker}: {e}")

    if verbose:
        print(f"  下載完成：{success} 檔成功，{fail} 檔失敗")

    df = pd.concat(all_parts, axis=0)
    df = df[~df['exdiv']].drop(columns='exdiv')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    feat_cols = [c for c in df.columns if c not in ('future_ret', 'ticker')]

    train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    test_df  = df[(df.index >= TEST_START)  & (df.index <= TEST_END)].copy()

    return train_df, test_df, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# 完整流程
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(feat_fn, feat_name: str) -> dict:
    print(f"\n{'='*65}")
    print(f"  特徵集：{feat_name}")
    print(f"{'='*65}")

    # ── 1. 資料 ──────────────────────────────────────────────────
    train_df, test_df, feat_cols = build_dataset(feat_fn)
    n_feat = len(feat_cols)
    print(f"  訓練筆數：{len(train_df):,}  測試筆數：{len(test_df):,}  特徵數：{n_feat}")

    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df['future_ret'].values.astype(np.float32)
    X_test  = test_df[feat_cols].values.astype(np.float32)
    y_test  = test_df['future_ret'].values.astype(np.float32)

    # ── 2. 標準化 ─────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── 3. Autoencoder 訓練 ────────────────────────────────────────
    print(f"  [Autoencoder] 訓練中 (input={n_feat}, latent={LATENT_DIM})...")
    ae = train_autoencoder(X_train, n_feat)
    ae.eval()
    with torch.no_grad():
        Z_train = ae.encode(torch.FloatTensor(X_train)).numpy()
        Z_test  = ae.encode(torch.FloatTensor(X_test)).numpy()

    # ── 4. One-Class SVM ──────────────────────────────────────────
    bull_mask = y_train > RISE_THRESH
    bear_mask = y_train < FALL_THRESH
    print(f"  看漲樣本：{bull_mask.sum():,}  看跌樣本：{bear_mask.sum():,}")

    if bull_mask.sum() < 5 or bear_mask.sum() < 5:
        print("  [WARNING] 樣本數不足，跳過此特徵集")
        return {}

    ocsvm_bull = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
    ocsvm_bear = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
    ocsvm_bull.fit(Z_train[bull_mask])
    ocsvm_bear.fit(Z_train[bear_mask])

    pred_bull = ocsvm_bull.predict(Z_test)   # +1 = inlier (類似看漲型態)
    pred_bear = ocsvm_bear.predict(Z_test)   # +1 = inlier (類似看跌型態)

    # ── 5. 訊號產生 ───────────────────────────────────────────────
    # 買進：看漲SVM為inlier 且 看跌SVM為outlier
    # 賣出：看跌SVM為inlier 且 看漲SVM為outlier
    signal = np.zeros(len(Z_test), dtype=np.float32)
    signal[(pred_bull ==  1) & (pred_bear != 1)] =  1.0
    signal[(pred_bear ==  1) & (pred_bull != 1)] = -1.0

    # ── 6. 報酬率計算 ─────────────────────────────────────────────
    # 買進：持有3日，報酬 = future_ret
    # 賣出：放空3日，報酬 = -future_ret
    strategy_ret = signal * y_test

    n_buy     = int((signal ==  1).sum())
    n_sell    = int((signal == -1).sum())
    n_total   = int((signal !=  0).sum())

    act_ret   = strategy_ret[signal != 0]
    avg_ret   = float(act_ret.mean())           if n_total > 0 else 0.0
    avg_bull  = float(strategy_ret[signal ==  1].mean()) if n_buy  > 0 else 0.0
    avg_bear  = float(strategy_ret[signal == -1].mean()) if n_sell > 0 else 0.0
    win_rate  = float((act_ret > 0).mean())     if n_total > 0 else 0.0
    cum_final = float(act_ret.sum())

    print(f"\n  ── 回測結果 (2025) ──────────────────────────────────────")
    print(f"  買進訊號數 : {n_buy:,}")
    print(f"  賣出訊號數 : {n_sell:,}")
    print(f"  總訊號數   : {n_total:,}")
    print(f"  平均報酬率 (全部)  : {avg_ret:+.3f}%")
    print(f"  平均報酬率 (買進)  : {avg_bull:+.3f}%")
    print(f"  平均報酬率 (賣出)  : {avg_bear:+.3f}%")
    print(f"  勝率               : {win_rate:.2%}")
    print(f"  累積報酬率 (加總)  : {cum_final:+.2f}%")

    # ── 7. 繪圖 ───────────────────────────────────────────────────
    safe_name = feat_name.replace(' ', '_').replace('(', '').replace(')', '')
    _plot_results(strategy_ret, signal, test_df, safe_name, feat_name)

    return {
        'feature_set': feat_name,
        'n_features':  n_feat,
        'n_buy':       n_buy,
        'n_sell':      n_sell,
        'n_total':     n_total,
        'avg_ret_%':   round(avg_ret,  4),
        'avg_buy_%':   round(avg_bull, 4),
        'avg_sell_%':  round(avg_bear, 4),
        'win_rate':    round(win_rate, 4),
        'cum_ret_%':   round(cum_final, 4),
    }


def _plot_results(strategy_ret, signal, test_df, safe_name, feat_name):
    """繪製累積報酬率與訊號分布圖"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'{feat_name} — 2025回測結果', fontsize=14)

    # 上圖：累積報酬率
    ax1 = axes[0]
    if (signal != 0).sum() > 0:
        cum = pd.Series(strategy_ret[signal != 0]).cumsum().reset_index(drop=True)
        ax1.plot(cum.values, color='steelblue', linewidth=1.5, label='Cumulative Return')
        ax1.axhline(0, color='red', linestyle='--', linewidth=0.8)
        ax1.fill_between(range(len(cum)), cum.values, 0,
                         where=cum.values >= 0, alpha=0.2, color='green')
        ax1.fill_between(range(len(cum)), cum.values, 0,
                         where=cum.values < 0,  alpha=0.2, color='red')
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.set_xlabel('Trade #')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 下圖：單次報酬率分布
    ax2 = axes[1]
    if (signal != 0).sum() > 0:
        returns_nonzero = strategy_ret[signal != 0]
        ax2.hist(returns_nonzero, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
        ax2.axvline(0, color='red', linestyle='--', linewidth=1)
        ax2.axvline(returns_nonzero.mean(), color='orange', linestyle='-',
                    linewidth=1.5, label=f'Mean={returns_nonzero.mean():.3f}%')
    ax2.set_xlabel('Return per Trade (%)')
    ax2.set_ylabel('Frequency')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f'C:/Users/alex/Desktop/專題進度/0401/backtest_{safe_name}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  圖表已儲存：backtest_{safe_name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    FEATURE_SETS = [
        (features_1day, '1日-7特徵'),
        (features_2day, '2日-10特徵'),
        (features_3day, '3日-12特徵'),
    ]

    all_results = []
    for fn, name in FEATURE_SETS:
        result = run_pipeline(fn, name)
        if result:
            all_results.append(result)

    # ── 總結報告 ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print(" 三特徵集 回測總結 (2025)")
    print(f"{'='*65}")
    summary = pd.DataFrame(all_results).set_index('feature_set')
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(summary.to_string())

    out_csv = 'C:/Users/alex/Desktop/專題進度/0401/backtest_summary.csv'
    summary.to_csv(out_csv, encoding='utf-8-sig')
    print(f"\n結果已儲存：backtest_summary.csv")
    print("圖表已儲存：backtest_*.png")
