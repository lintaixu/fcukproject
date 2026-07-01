"""
交易模擬 — 嚴格對齊論文 Section 6.3 (Li et al., KBS 2022).

策略邏輯:
  - Long signal:  模型預測「漲」→ 若無持倉則買入
  - Short signal: 模型預測「跌」→ 若有持倉則賣出
  - MACD 輔助:    若 MACD < 0, 亦執行賣出 (論文: "reduce volatility risk")
  - Benchmark: 等權持有所有股票 (buy-and-hold)

變更:
  - 移除 A (模型不再需要 adjacency matrix)
  - 推論改為同日批次 (對齊訓練時的 Self-Attention 語意)
  - MACD 的信號線平滑使用 n 期 EMA (對齊論文 Table 1)
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "core"))
from model import ChartGCN
from dataset import ChartGCNDataset
from train import DateGroupedBatchSampler


def compute_macd_signal(close: pd.Series, n: int = 100):
    """
    論文 MACD: DIFF = EMA(12) − EMA(26), MACD = EMA(n) of DIFF.
    回傳 MACD 信號線 (用於出場條件).
    """
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    return diff.ewm(span=n, adjust=False).mean()


def get_per_stock_predictions(model, test_ds, device='cpu'):
    """
    使用 DateGroupedBatchSampler 批次推論 (對齊 Self-Attention 語意).
    同一天的所有股票一起送入模型, 讓 attention 跨股票計算.

    Returns:
        dict: {ticker: [(date, pred_label), ...]}
    """
    model.eval()

    # 使用 DateGroupedBatchSampler 確保同日推論
    sampler = DateGroupedBatchSampler(test_ds, shuffle=False)
    all_preds = np.zeros(len(test_ds), dtype=np.int64)

    with torch.no_grad():
        for indices in sampler:
            X_batch = torch.from_numpy(test_ds.X[indices]).float().to(device)
            logits = model(X_batch)
            preds = logits.argmax(1).cpu().numpy()
            for i, idx in enumerate(indices):
                all_preds[idx] = preds[i]

    # 整理成 per-stock 結構
    results = {}
    for idx in range(len(test_ds)):
        ticker, date = test_ds.meta[idx]
        if ticker not in results:
            results[ticker] = []
        results[ticker].append((date, int(all_preds[idx])))

    for tk in results:
        results[tk].sort(key=lambda x: x[0])

    return results


def simulate_one_stock(
    ticker: str,
    predictions: list,
    price_df: pd.DataFrame,
    indicator_n: int = 100,
):
    """
    對單檔股票執行交易模擬 (論文 Section 6.3).
    """
    close = price_df['close']
    macd = compute_macd_signal(close, n=indicator_n)

    position = 0
    cash = 1.0
    shares = 0.0
    n_trades = 0

    dates = []
    net_values = []

    for date, pred in predictions:
        if date not in close.index:
            continue
        price = close.loc[date]
        macd_val = macd.loc[date] if date in macd.index else 0.0

        # Long: 預測漲 + 空手 → 買入
        if pred == 1 and position == 0:
            shares = cash / price
            cash = 0.0
            position = 1
            n_trades += 1

        # Short: (預測跌 or MACD<0) + 持倉 → 賣出
        elif position == 1 and (pred == 0 or macd_val < 0):
            cash = shares * price
            shares = 0.0
            position = 0
            n_trades += 1

        nv = cash + shares * price if position == 1 else cash
        dates.append(date)
        net_values.append(nv)

    return {
        'ticker': ticker,
        'dates': dates,
        'net_values': net_values,
        'n_trades': n_trades,
        'final_nv': net_values[-1] if net_values else 1.0,
        'total_return': (net_values[-1] - 1.0) * 100 if net_values else 0.0,
    }


def compute_benchmark(stock_data: dict, test_dates: list):
    """Benchmark: 等權持有所有股票 (buy-and-hold)."""
    all_returns = []
    for tk, df in stock_data.items():
        close = df['close']
        available = [d for d in test_dates if d in close.index]
        if len(available) < 2:
            continue
        start_price = close.loc[available[0]]
        nv = [close.loc[d] / start_price for d in available]
        all_returns.append(nv)

    if not all_returns:
        return test_dates, [1.0] * len(test_dates)

    min_len = min(len(r) for r in all_returns)
    avg_nv = np.mean([r[:min_len] for r in all_returns], axis=0)
    return test_dates[:min_len], avg_nv.tolist()


def run_backtest(model, test_ds, stock_data, indicator_n=100,
                 device='cpu', verbose=True):
    """完整交易模擬流程."""
    if verbose:
        print("\n" + "=" * 60)
        print("交易模擬 (對應論文 Section 6.3)")
        print("=" * 60)

    predictions = get_per_stock_predictions(model, test_ds, device)

    if verbose:
        print(f"\n共 {len(predictions)} 檔股票有預測結果")

    stock_results = []
    for tk, preds in predictions.items():
        if tk not in stock_data:
            continue
        result = simulate_one_stock(tk, preds, stock_data[tk],
                                    indicator_n=indicator_n)
        stock_results.append(result)

    if stock_results:
        min_len = min(len(r['net_values']) for r in stock_results)
        avg_nv = np.mean(
            [r['net_values'][:min_len] for r in stock_results], axis=0
        )
        avg_final = avg_nv[-1]
    else:
        avg_final = 1.0
        avg_nv = [1.0]

    all_dates = []
    for r in stock_results:
        all_dates.extend(r['dates'])
    unique_dates = sorted(set(all_dates))
    bench_dates, bench_nv = compute_benchmark(stock_data, unique_dates)

    if verbose:
        print(f"\n{'股票':>10s} | {'最終淨值':>8s} | {'報酬率':>8s} | {'交易次數':>6s}")
        print("-" * 50)
        sorted_results = sorted(stock_results,
                                key=lambda x: x['final_nv'], reverse=True)
        for r in sorted_results[:10]:
            print(f"{r['ticker']:>10s} | {r['final_nv']:>8.4f} | "
                  f"{r['total_return']:>+7.2f}% | {r['n_trades']:>6d}")
        if len(sorted_results) > 10:
            print(f"  ... 共 {len(sorted_results)} 檔 (僅顯示前 10)")

        print(f"\n{'=' * 50}")
        print(f"Chart GCN 策略平均淨值:  {avg_final:.4f} "
              f"({(avg_final-1)*100:+.2f}%)")
        bench_final = bench_nv[-1] if bench_nv else 1.0
        print(f"Benchmark (等權持有):    {bench_final:.4f} "
              f"({(bench_final-1)*100:+.2f}%)")
        print(f"超額報酬:               {(avg_final - bench_final)*100:+.2f}%")
        print(f"平均交易次數:           "
              f"{np.mean([r['n_trades'] for r in stock_results]):.1f} 次/股")

    return {
        'stock_results': stock_results,
        'avg_net_values': avg_nv.tolist() if isinstance(avg_nv, np.ndarray) else avg_nv,
        'avg_final_nv': avg_final,
        'benchmark_nv': bench_nv,
        'benchmark_final': bench_nv[-1] if bench_nv else 1.0,
    }


def plot_backtest(results, save_path='backtest_result.png'):
    """繪製淨值曲線圖."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("[WARN] matplotlib 未安裝, 跳過繪圖")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    avg_nv = results['avg_net_values']
    ax.plot(range(len(avg_nv)), avg_nv,
            label=f"Chart GCN ({avg_nv[-1]:.4f})", color='#e74c3c', linewidth=2)

    bench_nv = results['benchmark_nv']
    min_len = min(len(avg_nv), len(bench_nv))
    ax.plot(range(min_len), bench_nv[:min_len],
            label=f"Benchmark ({bench_nv[min_len-1]:.4f})", color='#3498db',
            linewidth=2, linestyle='--')

    for r in results['stock_results']:
        nv = r['net_values'][:min_len]
        ax.plot(range(len(nv)), nv, color='gray', alpha=0.1, linewidth=0.5)

    ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.3)
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Net Value')
    ax.set_title('Chart GCN Trading Simulation vs Benchmark (Buy & Hold)')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n淨值曲線已儲存: {save_path}")
    plt.close()
    return save_path
