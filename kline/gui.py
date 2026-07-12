"""
Tkinter 圖形介面。

KLineApp 只負責介面與流程調度：按鈕 → 背景執行緒 → 佇列回報 → 更新畫面。
所有運算邏輯都來自 kline 套件其他模組，GUI 本身不含演算法。
"""

import queue
import threading

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from kline.config import (
    HOLD_DAYS, RISE_THRESH, FALL_THRESH, DIST_THRESH, MIN_FREQ,
    LATENT_DIM, EPOCHS, OCSVM_NU, BROKERAGE, TAX, TRADE_COST,
    TRAIN_START, TRAIN_END, TW50,
)
from kline.features import FEATURE_SETS
from kline.data_loader import download_raw, build_dataset, download_one
from kline.pattern_model import find_and_train
from kline.backtest import backtest
from kline.validate import validate_features

plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False


class KLineApp:
    def __init__(self, root: tk.Tk):
        self.root     = root
        self.root.title("K線型態分析系統")
        self.root.geometry("1280x800")
        self.root.resizable(True, True)

        self._raw_data = None
        self._running  = False
        self._queue    = queue.Queue()

        self._build_ui()
        self._poll()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=0, minsize=210)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

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

        self._btn_run = ttk.Button(left, text="③ 分析 + 回測（三特徵集）",
                                   command=self._on_run, state='disabled')
        self._btn_run.pack(fill='x', pady=4)

        ttk.Button(left, text="清除輸出",
                   command=self._clear).pack(fill='x', pady=4)

        ttk.Separator(left, orient='horizontal').pack(fill='x', pady=10)

        note = (
            "固定參數：\n"
            f"  持有天數  : {HOLD_DAYS} 日\n"
            f"  大漲門檻  : >{RISE_THRESH}%\n"
            f"  大跌門檻  : <{FALL_THRESH}%\n"
            f"  距離門檻  : {DIST_THRESH}\n"
            f"  最少出現  : {MIN_FREQ} 次\n"
            f"  AE 潛在維度: {LATENT_DIM}\n"
            f"  AE Epochs : {EPOCHS}\n"
            f"  OCSVM nu  : {OCSVM_NU}\n"
            f"  手續費    : {BROKERAGE}% x 2\n"
            f"  交易稅    : {TAX}%（賣出）\n"
            f"  每筆成本  : {TRADE_COST}%\n"
            f"  訓練期間  : 2016-2024\n"
            f"  測試期間  : 2025"
        )
        ttk.Label(left, text=note, font=('Consolas', 9),
                  foreground='gray', justify='left').pack(anchor='w')

        self._progress = ttk.Progressbar(left, mode='determinate',
                                         maximum=100)
        self._progress.pack(fill='x', pady=(16, 0))

        self._progress_label = ttk.Label(left, text="", font=('', 9),
                                         foreground='gray')
        self._progress_label.pack(anchor='w', pady=(2, 0))

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
                elif kind == 'progress':
                    done, total = data
                    pct = int(done / total * 100)
                    self._progress['value'] = pct
                    self._progress_label.config(text=f"{done}/{total} ({pct}%)")
                elif kind == 'done_dl':
                    self._progress['value'] = 100
                    self._progress_label.config(text="下載完成")
                    self._running = False
                    self._btn_val.config(state='normal')
                    self._btn_run.config(state='normal')
                elif kind == 'done_run':
                    self._progress['value'] = 100
                    self._progress_label.config(text="完成")
                    self._running = False
                elif kind == 'plot':
                    self._draw_plot(data)
                elif kind == 'plot_cm':
                    self._draw_confusion_matrices(data)
                elif kind == 'error':
                    self._progress['value'] = 0
                    self._progress_label.config(text="")
                    self._running = False
                    messagebox.showerror("錯誤", data)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _log(self, text: str):
        self._queue.put(('log', text))

    def _set_progress(self, done: int, total: int):
        self._queue.put(('progress', (done, total)))

    # ── 按鈕 ──────────────────────────────────────────────────────────────────
    def _on_validate(self):
        if self._running or self._raw_data is None: return
        self._running = True
        self._progress['value'] = 0
        self._progress_label.config(text="驗證中...")
        self._log("\n" + "=" * 58)
        self._log(" [2] 特徵驗證（訓練 AE + OCSVM，不跑回測）")
        self._log("=" * 58)
        threading.Thread(target=self._worker_validate, daemon=True).start()

    def _on_download(self):
        if self._running: return
        self._running = True
        self._progress['value'] = 0
        self._progress_label.config(text="下載中...")
        self._log("=" * 58)
        self._log(" [1] 下載台灣前50大股票原始資料")
        self._log("=" * 58)
        threading.Thread(target=self._worker_download, daemon=True).start()

    def _on_run(self):
        if self._running or self._raw_data is None: return
        self._running = True
        self._progress['value'] = 0
        ticker = self._ticker_var.get().strip().upper()
        if not ticker:
            messagebox.showwarning("提示", "請輸入股票代號")
            self._running = False
            return
        self._progress_label.config(text="分析中...")
        self._log("\n" + "=" * 58)
        self._log(f" [3] 分析 + 回測：{ticker} {TW50.get(ticker, '')}")
        self._log("=" * 58)
        threading.Thread(target=self._worker_run,
                         args=(ticker,), daemon=True).start()

    def _clear(self):
        self._log_box.delete('1.0', tk.END)
        self._fig.clear()
        self._canvas.draw()
        self._progress['value'] = 0
        self._progress_label.config(text="")

    # ── Workers ───────────────────────────────────────────────────────────────
    def _worker_validate(self):
        try:
            val_results = {}
            total_sets = len(FEATURE_SETS)
            for i, (feat_name, feat_fn) in enumerate(FEATURE_SETS):
                self._log(f"\n{'─'*55}")
                self._log(f"  特徵集：{feat_name}")
                self._log(f"{'─'*55}")
                df        = build_dataset(self._raw_data, feat_fn)
                feat_cols = [c for c in df.columns
                             if c not in ('future_ret', 'ticker')]
                result = validate_features(df, feat_cols, self._log)
                val_results[feat_name] = result
                self._set_progress(i + 1, total_sets)

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
            self._log("\n  完成")
            self._queue.put(('done_run', None))
        except Exception as e:
            import traceback
            self._queue.put(('error', traceback.format_exc()))

    def _worker_download(self):
        try:
            raw = download_raw(self._log, self._set_progress)
            self._raw_data = raw
            self._log(f"\n  成功下載：{len(raw)} 檔")
            self._log("  下載完成，可開始分析")
            self._queue.put(('done_dl', None))
        except Exception as e:
            self._queue.put(('error', str(e)))

    def _worker_run(self, ticker: str):
        try:
            if ticker not in self._raw_data:
                self._log(f"  {ticker} 不在台灣50清單，補充下載...")
                _, df_extra = download_one(ticker)
                if df_extra is None:
                    self._queue.put(('error',
                        f"{ticker} 無法下載、資料不足或缺少必要欄位"
                        f"（Open/High/Low/Close/Volume/Adj Close），請確認代號是否正確"))
                    return
                self._raw_data[ticker] = df_extra
                self._log(f"  {ticker} 下載完成：{len(df_extra)} 筆")

            results = {}
            total_sets = len(FEATURE_SETS)

            for i, (feat_name, feat_fn) in enumerate(FEATURE_SETS):
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

                self._set_progress(i + 1, total_sets)

            self._log(f"\n{'='*58}")
            self._log(f"  三特徵集比較總結 ({ticker} {TW50.get(ticker,'')} 2025)")
            self._log(f"{'='*58}")
            self._log(f"  {'特徵集':^12}  {'買進':>5}  {'賣出':>5}  "
                      f"{'勝率':>7}  {'累積報酬':>10}")
            for name, r in results.items():
                self._log(f"  {name:^12}  {r['n_buy']:>5}  {r['n_sell']:>5}  "
                          f"{r['win_rate']:>7.2%}  {r['cum_ret']:>+9.2f}%")

            self._queue.put(('plot', results))
            self._log("\n  分析完成")
            self._queue.put(('done_run', None))
        except Exception as e:
            import traceback
            self._queue.put(('error', traceback.format_exc()))

    # ── 混淆矩陣圖 ────────────────────────────────────────────────────────────
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

    # ── 繪圖（累積報酬對比）──────────────────────────────────────────────────
    def _draw_plot(self, results: dict):
        self._fig.clear()
        n = len(results)
        if n == 0: return

        colors = ['steelblue', 'darkorange', 'green']

        for i, (name, res) in enumerate(results.items()):
            ax = self._fig.add_subplot(1, n, i + 1)
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


if __name__ == '__main__':
    # 建構視窗 → 更新一次 → 關閉（不進 mainloop 以免阻塞）
    try:
        root = tk.Tk()
        app  = KLineApp(root)
        root.update_idletasks(); root.update()
        print(f"GUI 建構成功: 視窗標題='{root.title()}'")
        print(f"  初始按鈕狀態: 驗證={str(app._btn_val['state'])}  "
              f"回測={str(app._btn_run['state'])}  (下載前應為 disabled)")
        root.destroy()
        print("GUI 正常關閉")
    except tk.TclError as e:
        print("（無顯示環境，略過 GUI 實體測試）:", e)
