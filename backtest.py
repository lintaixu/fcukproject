"""
交易模擬 (對應論文 Section 6.3).

策略邏輯:
  - Long signal:  模型預測「漲」→ 若無持倉則買入, 收盤前持有
  - Short signal: 模型預測「跌」→ 若有持倉則賣出
  - MACD 輔助:    若 MACD < 0, 亦執行賣出 (避免波動風險)
  - 計算每檔股票的淨值曲線, 以及所有股票的平均淨值
  - Benchmark: 等權持有所有股票 (buy-and-hold)
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import ChartGCN
from dataset import ChartGCNDataset
from indicators import compute_indicators


def compute_macd_signal(close: pd.Series):
    """計算 MACD 值 (12-day EMA - 26-day EMA)."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    return ema12 - ema26


def get_per_stock_predictions(model, test_ds, device='cpu'):
    """
    從 test dataset 取出每檔股票的逐日預測.

    Returns:
        dict: {ticker: [(date, pred_label), ...]}
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for i in range(len(test_ds)):
            # 因為 test_ds 可能是 Subset, 取底層 dataset 的 meta
            if hasattr(test_ds, 'dataset'):
                # Subset
                real_idx = test_ds.indices[i]
                base_ds = test_ds.dataset
            else:
                real_idx = i
                base_ds = test_ds

            X = torch.from_numpy(base_ds.X[real_idx]).float().unsqueeze(0).to(device)
            A = torch.from_numpy(base_ds.A[real_idx]).float().unsqueeze(0).to(device)
            logit = model(X, A)
            pred = logit.argmax(1).item()

            ticker, date = base_ds.meta[real_idx]
            if ticker not in results:
                results[ticker] = []
            results[ticker].append((date, pred))

    # 按日期排序
    for tk in results:
        results[tk].sort(key=lambda x: x[0])

    return results


def simulate_one_stock(
    ticker: str,
    predictions: list,
    price_df: pd.DataFrame,
):
    """
    對單檔股票執行交易模擬.

    Args:
        ticker:      股票代號
        predictions: [(date, pred_label), ...]
        price_df:    該股票的完整 DataFrame (含 close)

    Returns:
        dict with keys:
          'dates':       交易日列表
          'net_values':  淨值序列 (起始=1.0)
          'n_trades':    交易次數
          'final_nv':    最終淨值
          'total_return': 總報酬率 (%)
    """
    close = price_df['close']
    macd = compute_macd_signal(close)

    position = 0       # 0=空手, 1=持倉
    entry_price = 0.0
    cash = 1.0          # 起始淨值
    shares = 0.0
    n_trades = 0

    dates = []
    net_values = []

    for date, pred in predictions:
        if date not in close.index:
            continue
        price = close.loc[date]
        macd_val = macd.loc[date] if date in macd.index else 0.0

        # Long signal: 預測漲 + 目前空手 → 買入
        if pred == 1 and position == 0:
            shares = cash / price
            cash = 0.0
            position = 1
            entry_price = price
            n_trades += 1

        # Short signal: (預測跌 or MACD<0) + 目前持倉 → 賣出
        elif position == 1 and (pred == 0 or macd_val < 0):
            cash = shares * price
            shares = 0.0
            position = 0
            n_trades += 1

        # 計算當日淨值
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
    """
    Benchmark: 等權持有所有股票 (buy-and-hold).
    從第一個測試日買入, 計算平均淨值.
    """
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

    # 對齊長度 (取最短)
    min_len = min(len(r) for r in all_returns)
    avg_nv = np.mean([r[:min_len] for r in all_returns], axis=0)

    return test_dates[:min_len], avg_nv.tolist()


def run_backtest(model, test_ds, stock_data, device='cpu', verbose=True):
    """
    完整交易模擬流程.

    Returns:
        results: dict 包含所有結果
    """
    if verbose:
        print("\n" + "=" * 60)
        print("交易模擬 (對應論文 Section 6.3)")
        print("=" * 60)

    # 1. 取得逐股逐日預測
    predictions = get_per_stock_predictions(model, test_ds, device)

    if verbose:
        print(f"\n共 {len(predictions)} 檔股票有預測結果")

    # 2. 逐股模擬
    stock_results = []
    for tk, preds in predictions.items():
        if tk not in stock_data:
            continue
        result = simulate_one_stock(tk, preds, stock_data[tk])
        stock_results.append(result)

    # 3. 計算平均淨值
    if stock_results:
        min_len = min(len(r['net_values']) for r in stock_results)
        avg_nv = np.mean(
            [r['net_values'][:min_len] for r in stock_results], axis=0
        )
        avg_final = avg_nv[-1]
    else:
        avg_final = 1.0
        avg_nv = [1.0]

    # 4. Benchmark
    all_dates = []
    for r in stock_results:
        all_dates.extend(r['dates'])
    unique_dates = sorted(set(all_dates))
    bench_dates, bench_nv = compute_benchmark(stock_data, unique_dates)

    # 5. 輸出結果
    if verbose:
        print(f"\n{'股票':>10s} | {'最終淨值':>8s} | {'報酬率':>8s} | {'交易次數':>6s}")
        print("-" * 50)

        # 按最終淨值排序
        sorted_results = sorted(stock_results, key=lambda x: x['final_nv'], reverse=True)
        for r in sorted_results[:10]:  # 顯示前 10 名
            print(f"{r['ticker']:>10s} | {r['final_nv']:>8.4f} | "
                  f"{r['total_return']:>+7.2f}% | {r['n_trades']:>6d}")
        if len(sorted_results) > 10:
            print(f"  ... 共 {len(sorted_results)} 檔 (僅顯示前 10)")

        print(f"\n{'=' * 50}")
        print(f"Chart GCN 策略平均淨值:  {avg_final:.4f} ({(avg_final-1)*100:+.2f}%)")
        bench_final = bench_nv[-1] if bench_nv else 1.0
        print(f"Benchmark (等權持有):    {bench_final:.4f} ({(bench_final-1)*100:+.2f}%)")
        print(f"超額報酬:               {(avg_final - bench_final)*100:+.2f}%")
        print(f"平均交易次數:           {np.mean([r['n_trades'] for r in stock_results]):.1f} 次/股")

    return {
        'stock_results': stock_results,
        'avg_net_values': avg_nv.tolist() if isinstance(avg_nv, np.ndarray) else avg_nv,
        'avg_final_nv': avg_final,
        'benchmark_nv': bench_nv,
        'benchmark_final': bench_nv[-1] if bench_nv else 1.0,
    }


def plot_backtest(results, save_path='backtest_result.png'):
    """
    繪製淨值曲線圖 (Chart GCN vs Benchmark).
    """
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

    # Chart GCN 平均淨值
    avg_nv = results['avg_net_values']
    ax.plot(range(len(avg_nv)), avg_nv,
            label=f"Chart GCN ({avg_nv[-1]:.4f})", color='#e74c3c', linewidth=2)

    # Benchmark
    bench_nv = results['benchmark_nv']
    min_len = min(len(avg_nv), len(bench_nv))
    ax.plot(range(min_len), bench_nv[:min_len],
            label=f"Benchmark ({bench_nv[min_len-1]:.4f})", color='#3498db',
            linewidth=2, linestyle='--')

    # 個股淨值 (淡色背景)
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


if __name__ == "__main__":
    print("backtest.py — 需從 main.py 呼叫, 或獨立測試如下:")
    print("  python main.py --start 2010-01-01 --end 2024-12-31 "
          "--train-end 2023-12-31 --epochs 30 --stride 3")
