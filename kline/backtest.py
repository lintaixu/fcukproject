"""
回測。

以訓練好的模型（scaler + 看漲/看跌 AE + OCSVM）對測試期產生買賣訊號，
計算含交易成本後的策略報酬、勝率與累積報酬。
"""

import numpy as np
import torch

from kline.config import TEST_START, TEST_END, TRADE_COST, HOLD_DAYS


def backtest(df, feat_cols: list, models: dict, ticker: str) -> dict:
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

    # 非重疊持倉：每筆報酬為未來 HOLD_DAYS 日報酬，
    # 進場後持有期間內的新訊號一律忽略，避免同一段行情被重複計算
    executed  = np.zeros_like(signal)
    next_free = 0
    for i in range(len(signal)):
        if signal[i] != 0 and i >= next_free:
            executed[i] = signal[i]
            next_free = i + HOLD_DAYS

    strategy_ret = executed * y_test
    strategy_ret[executed != 0] -= TRADE_COST
    n_buy   = int((executed ==  1).sum())
    n_sell  = int((executed == -1).sum())
    n_total = int((executed !=  0).sum())

    if n_total > 0:
        act      = strategy_ret[executed != 0]
        avg_ret  = float(act.mean())
        avg_bull = float(strategy_ret[executed ==  1].mean()) if n_buy  > 0 else 0.0
        avg_bear = float(strategy_ret[executed == -1].mean()) if n_sell > 0 else 0.0
        win_rate = float((act > 0).mean())
        cum_ret  = float(act.sum())
    else:
        avg_ret = avg_bull = avg_bear = win_rate = cum_ret = 0.0

    return {
        'strategy_ret': strategy_ret,
        'signal':       executed,
        'raw_signal':   signal,
        'n_buy':   n_buy,   'n_sell':  n_sell,  'n_total': n_total,
        'avg_ret': avg_ret, 'avg_bull': avg_bull,'avg_bear': avg_bear,
        'win_rate': win_rate, 'cum_ret': cum_ret,
    }


if __name__ == '__main__':
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    n = 300
    rng = np.random.default_rng(0)
    cols = ['upper', 'lower', 'body', 'gap', 'close_chg', 'vol_ratio', 'trend']
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in cols}, index=None)
    df.index = pd.date_range('2025-01-01', periods=n, freq='B')
    df['future_ret'] = rng.normal(0, 3, n)
    df['ticker'] = '2330.TW'

    scaler = StandardScaler().fit(df[cols].values)
    models = {'scaler': scaler, 'bull': None, 'bear': None}   # 無型態模型 → 0 訊號
    res = backtest(df, cols, models, '2330.TW')
    print(f"backtest（無模型基準）: n_buy={res['n_buy']}  n_sell={res['n_sell']}  "
          f"n_total={res['n_total']}  win_rate={res['win_rate']:.2%}  "
          f"cum_ret={res['cum_ret']:+.2f}%")
    print("  → 無型態模型時訊號應為 0，函式路徑正常")
