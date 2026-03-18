"""
K線型態回測分析系統 - GUI界面
包含數據獲取、型態識別、回測分析、可視化等完整功能
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from data_fetcher import DataFetcher
from pattern_recognizer import CandlePattern
from backtester import Backtester
from analyzer import PatternAnalyzer, generate_html_report
from visualizer import CandleVisualizer
from config import STOCK_SYMBOLS, ANALYSIS_CONFIG, DATA_CONFIG, ANALYSIS_PERIOD

# 型態名稱翻譯字典
PATTERN_NAME_ZH = {
    'doji': '十字線',
    'hammer': '錘子線',
    'hanging_man': '上吊線',
    'inverted_hammer': '倒錘線',
    'bullish_harami': '看漲孕線',
    'bearish_harami': '看跌孕線',
    'bullish_engulfing': '看漲吞沒',
    'bearish_engulfing': '看跌吞沒',
    'piercing_line': '穿刺線',
    'dark_cloud_cover': '烏雲蓋頂',
    'morning_star': '晨星',
    'evening_star': '晚星',
    'three_black_crows': '三根烏鴉',
    'three_white_soldiers': '三個白兵',
    'bullish_flag': '牛旗形',
    'bearish_flag': '熊旗形',
    'triple_top': '三重頂',
    'triple_bottom': '三重底',
}

def get_pattern_name_zh(pattern_name: str) -> str:
    """獲取中文型態名稱"""
    return PATTERN_NAME_ZH.get(pattern_name, pattern_name)

# 設置matplotlib中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KLineAnalysisGUI:
    """K線型態分析系統GUI"""
    
    def __init__(self, root):
        """初始化GUI"""
        self.root = root
        self.root.title("K線型態回測分析系統 - 台灣前50大公司綜合分析")
        self.root.geometry("1200x800")
        
        # 初始化模塊
        self.fetcher = DataFetcher(DATA_CONFIG["cache_dir"])
        self.analyzer = PatternAnalyzer()
        self.backtester = None  # 在獲取數據後初始化
        self.visualizer = CandleVisualizer()
        
        # 存儲分析結果
        self.analysis_results = {}
        self.current_df = None
        self.current_symbol = None
        
        # 創建UI
        self._create_widgets()
        
        # 禁用運行中標誌
        self.is_running = False
        
        # 自動載入綜合分析結果
        self._load_combined_analysis()
    
    def _normalize_price(self, df_subset):
        """
        正規化價格為相對百分比 (%)
        取第一天開盤價為基準 (0%)，後續高低點相對於基準的漲跌幅
        """
        base_price = df_subset.iloc[0]['Open']
        df_normalized = df_subset.copy()
        df_normalized['Open'] = (df_subset['Open'] - base_price) / base_price * 100
        df_normalized['Close'] = (df_subset['Close'] - base_price) / base_price * 100
        df_normalized['High'] = (df_subset['High'] - base_price) / base_price * 100
        df_normalized['Low'] = (df_subset['Low'] - base_price) / base_price * 100
        return df_normalized
    
    def _normalize_volume(self, df_subset):
        """
        正規化成交量為相對均量比
        用5天平均成交量作為1.0，爆量是1.5，量縮是0.6
        """
        avg_volume = df_subset['Volume'].mean()
        if avg_volume == 0:
            return df_subset['Volume'].values
        return (df_subset['Volume'] / avg_volume).values
    
    def _load_combined_analysis(self):
        """自動載入綜合分析結果（所有50大公司）並預快取第一個股票數據"""
        try:
            import json
            from pathlib import Path
            
            # 嘗試讀取 backtest_results.json
            json_path = Path("backtest_results.json")
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # 轉換為 GUI 需要的格式
                combined_metrics = {}
                for pattern_name, pattern_data in results.items():
                    metrics = pattern_data.get('metrics', {})
                    combined_metrics[pattern_name] = {
                        'total_trades': metrics.get('total_trades', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'profit_factor': metrics.get('profit_factor', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'avg_profit': metrics.get('avg_profit', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                    }
                
                # 存儲結果
                self.backtest_results = combined_metrics
                
                # 更新排名表
                self._update_ranking(combined_metrics)
                
                # 顯示狀態
                self._set_status("✓ 已載入台灣前50大公司的綜合K線型態回測結果")
                logger.info("✓ 綜合分析結果已成功載入")
                
                # 後臺預快取第一個股票的數據（加快後續查看圖表的速度）
                thread = threading.Thread(target=self._preload_first_stock_data)
                thread.daemon = True
                thread.start()
            else:
                self._set_status("未找到回測結果，請先運行 python main.py")
                logger.warning("backtest_results.json 不存在")
                
        except Exception as e:
            logger.error(f"載入綜合分析失敗: {e}")
            self._set_status(f"載入失敗: {str(e)}")
    
    def _preload_first_stock_data(self):
        """預快取第一個股票（2330）的數據，加快後續查看圖表的速度"""
        try:
            symbol = STOCK_SYMBOLS[0]  # 通常是 '2330.TW'
            start_date = ANALYSIS_PERIOD['start_date']
            end_date = ANALYSIS_PERIOD['end_date']
            
            # 獲取數據（使用cache）
            df = self.fetcher.fetch_data(symbol, start_date, end_date)
            
            if df is not None and not df.empty:
                # 識別型態
                pattern_recognizer = CandlePattern(df)
                patterns = pattern_recognizer.detect_all_patterns()
                
                # 存儲在實例變數中供後用
                self.current_df = df
                self.current_symbol = symbol
                self.current_patterns = patterns
                
                logger.info(f"✓ 已預快取 {symbol} 的數據，後續查看圖表將快速加載")
        except Exception as e:
            logger.warning(f"預快取失敗（非關鍵）: {e}")
    
    def _run_all_analysis(self):
        """回測所有50大公司 - 調用 main.py"""
        import subprocess
        import threading
        
        if self.is_running:
            self._set_status("⚠️ 程式正在運行中，請稍候")
            return
        
        def run_in_thread():
            try:
                self._start_progress()
                self._set_status("⏳ 正在回測所有50大公司... 這可能需要幾分鐘")
                
                # 調用 main.py
                result = subprocess.run(
                    ["python", "main.py"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                if result.returncode == 0:
                    # 回測成功，自動載入結果
                    self._load_combined_analysis()
                    self._set_status("✓ 回測完成！已刷新排名結果")
                else:
                    self._set_status(f"❌ 回測失敗: {result.stderr}")
                    logger.error(f"main.py 錯誤: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                self._set_status("❌ 回測超時（超過10分鐘）")
                logger.error("回測超時")
            except Exception as e:
                self._set_status(f"❌ 錯誤: {str(e)}")
                logger.error(f"執行 main.py 失敗: {e}")
            finally:
                self._stop_progress()
                self.is_running = False
        
        # 在後台線程運行
        thread = threading.Thread(target=run_in_thread, daemon=True)
        self.is_running = True
        thread.start()
    
    def _create_widgets(self):
        """創建UI元件"""
        
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== 左側面板：參數設置 =====
        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # 標題
        ttk.Label(left_frame, text="K線型態回測分析", font=("Arial", 14, "bold")).pack(pady=10)
        ttk.Label(left_frame, text="台灣前50大公司", font=("Arial", 12)).pack()
        
        # 分隔線
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        # 主要按鈕
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="▶ 回測所有50大公司", command=self._run_all_analysis).pack(pady=5, fill=tk.X)
        
        # 查看報告按鈕
        ttk.Button(button_frame, text="📊 查看綜合報告", command=self._open_html_report).pack(pady=5, fill=tk.X)
        
        # 分隔線
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
        
        # 狀態指示
        ttk.Label(left_frame, text="狀態", font=("Arial", 10, "bold")).pack()
        self.status_var = tk.StringVar(value="準備就緒")
        status_label = ttk.Label(left_frame, textvariable=self.status_var,
                                foreground="blue", wraplength=280)
        status_label.pack(pady=(5, 10), fill=tk.X)
        
        # 進度條
        self.progress = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        # ===== 右側面板：結果顯示 =====
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 標籤頁
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 標籤1：數據信息
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="數據信息")
        self.data_text = tk.Text(data_frame, height=25, width=80, wrap=tk.WORD)
        self.data_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 標籤2：型態統計
        pattern_frame = ttk.Frame(self.notebook)
        self.notebook.add(pattern_frame, text="型態統計")
        
        # 型態表格
        pattern_scroll = ttk.Scrollbar(pattern_frame)
        pattern_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.pattern_tree = ttk.Treeview(pattern_frame, 
                                        columns=("Pattern", "Count", "First", "Last"),
                                        height=25, yscrollcommand=pattern_scroll.set)
        pattern_scroll.config(command=self.pattern_tree.yview)
        
        self.pattern_tree.column("#0", width=0, stretch=tk.NO)
        self.pattern_tree.column("Pattern", anchor=tk.W, width=200)
        self.pattern_tree.column("Count", anchor=tk.CENTER, width=80)
        self.pattern_tree.column("First", anchor=tk.CENTER, width=100)
        self.pattern_tree.column("Last", anchor=tk.CENTER, width=100)
        
        self.pattern_tree.heading("#0", text="", anchor=tk.W)
        self.pattern_tree.heading("Pattern", text="型態名稱")
        self.pattern_tree.heading("Count", text="出現次數")
        self.pattern_tree.heading("First", text="首次索引")
        self.pattern_tree.heading("Last", text="最後索引")
        
        self.pattern_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 標籤3：回測結果
        result_frame = ttk.Frame(self.notebook)
        self.notebook.add(result_frame, text="回測結果")
        
        result_scroll = ttk.Scrollbar(result_frame)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_tree = ttk.Treeview(result_frame,
                                       columns=("Pattern", "Trades", "WinRate", "Ratio", "Sharpe"),
                                       height=25, yscrollcommand=result_scroll.set)
        result_scroll.config(command=self.result_tree.yview)
        
        self.result_tree.column("#0", width=0, stretch=tk.NO)
        self.result_tree.column("Pattern", anchor=tk.W, width=150)
        self.result_tree.column("Trades", anchor=tk.CENTER, width=80)
        self.result_tree.column("WinRate", anchor=tk.CENTER, width=100)
        self.result_tree.column("Ratio", anchor=tk.CENTER, width=100)
        self.result_tree.column("Sharpe", anchor=tk.CENTER, width=100)
        
        self.result_tree.heading("#0", text="", anchor=tk.W)
        self.result_tree.heading("Pattern", text="型態")
        self.result_tree.heading("Trades", text="交易數")
        self.result_tree.heading("WinRate", text="勝率 %")
        self.result_tree.heading("Ratio", text="盈虧比")
        self.result_tree.heading("Sharpe", text="夏普比")
        
        self.result_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 標籤4：K線圖表
        chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(chart_frame, text="K線圖表")
        
        # 圖表控制面板
        chart_control = ttk.Frame(chart_frame)
        chart_control.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(chart_control, text="選擇圖表：").pack(side=tk.LEFT, padx=5)
        self.chart_var = tk.StringVar(value="candlestick")
        chart_combo = ttk.Combobox(chart_control, textvariable=self.chart_var,
                                   values=["K線+MA", "Hammer", "Bullish Harami", "Bearish Engulfing"],
                                   state="readonly", width=20)
        chart_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(chart_control, text="📊 載入圖表", 
                   command=self._load_chart).pack(side=tk.LEFT, padx=5)
        
        # 圖表顯示區域
        self.chart_canvas_frame = ttk.Frame(chart_frame)
        self.chart_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 標籤5：排名
        ranking_frame = ttk.Frame(self.notebook)
        self.notebook.add(ranking_frame, text="型態排名")
        
        ranking_scroll = ttk.Scrollbar(ranking_frame)
        ranking_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.ranking_tree = ttk.Treeview(ranking_frame,
                                        columns=("Rank", "Pattern", "Score", "WinRate", "Trades"),
                                        height=25, yscrollcommand=ranking_scroll.set)
        ranking_scroll.config(command=self.ranking_tree.yview)
        
        self.ranking_tree.column("#0", width=0, stretch=tk.NO)
        self.ranking_tree.column("Rank", anchor=tk.CENTER, width=50)
        self.ranking_tree.column("Pattern", anchor=tk.W, width=150)
        self.ranking_tree.column("Score", anchor=tk.CENTER, width=80)
        self.ranking_tree.column("WinRate", anchor=tk.CENTER, width=100)
        self.ranking_tree.column("Trades", anchor=tk.CENTER, width=100)
        
        self.ranking_tree.heading("#0", text="", anchor=tk.W)
        self.ranking_tree.heading("Rank", text="排名")
        self.ranking_tree.heading("Pattern", text="型態名稱 (雙擊查看K線圖)")
        self.ranking_tree.heading("Score", text="評分")
        self.ranking_tree.heading("WinRate", text="勝率%")
        self.ranking_tree.heading("Trades", text="交易筆數")
        
        # 綁定點擊事件
        self.ranking_tree.bind("<Double-1>", lambda e: self._on_ranking_click(e))
        
        self.ranking_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加提示標籤
        ttk.Label(ranking_frame, text="💡 提示：雙擊任何型態可查看對應的K線圖", 
                 foreground="blue").pack(padx=5, pady=2, anchor=tk.W)
    
    def _set_status(self, message: str):
        """更新狀態信息"""
        self.status_var.set(message)
        self.root.update()
    
    def _start_progress(self):
        """開始進度條"""
        self.progress.start()
        self.is_running = True
    
    def _stop_progress(self):
        """停止進度條"""
        self.progress.stop()
        self.is_running = False
    
    def _fetch_data(self):
        """獲取數據"""
        if self.is_running:
            messagebox.showwarning("提示", "程式正在運行中，請稍候")
            return
        
        thread = threading.Thread(target=self._fetch_data_thread)
        thread.start()
    
    def _fetch_data_thread(self):
        """獲取數據（線程化）"""
        try:
            self._start_progress()
            self._set_status("正在獲取數據...")
            
            symbol = self.stock_var.get()
            start_date = self.start_date_var.get()
            end_date = self.end_date_var.get()
            
            # 驗證日期格式
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                messagebox.showerror("錯誤", "日期格式應為 YYYY-MM-DD")
                self._stop_progress()
                self._set_status("日期格式錯誤")
                return
            
            # 獲取數據
            df = self.fetcher.fetch_data(symbol, start_date, end_date, "1d")
            
            if df is None or len(df) == 0:
                messagebox.showerror("錯誤", f"無法獲取 {symbol} 的數據")
                self._stop_progress()
                self._set_status("數據獲取失敗")
                return
            
            self.current_df = df
            self.current_symbol = symbol
            
            # 初始化Backtester
            self.backtester = Backtester(df)
            
            # 顯示數據信息
            info_text = f"""
【數據信息】
股票代碼: {symbol}
日期範圍: {start_date} 到 {end_date}
K線數量: {len(df)}
收盤價範圍: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}
平均成交量: {df['Volume'].mean():.0f}

【OHLCV 數據预览】
{df.tail(10).to_string()}
            """
            
            self.data_text.delete(1.0, tk.END)
            self.data_text.insert(tk.END, info_text)
            
            self._set_status(f"✓ 成功獲取 {len(df)} 根K線數據")
            messagebox.showinfo("成功", f"已獲取 {len(df)} 根K線數據")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"獲取數據失敗: {str(e)}")
            self._set_status(f"錯誤: {str(e)}")
        finally:
            self._stop_progress()
    
    def _recognize_patterns(self):
        """識別型態"""
        if self.current_df is None:
            messagebox.showwarning("提示", "請先獲取數據")
            return
        
        thread = threading.Thread(target=self._recognize_patterns_thread)
        thread.start()
    
    def _recognize_patterns_thread(self):
        """識別型態（線程化）"""
        try:
            self._start_progress()
            self._set_status("正在識別型態...")
            
            # 識別型態
            pattern_recognizer = CandlePattern(self.current_df)
            patterns = pattern_recognizer.detect_all_patterns()
            
            # 清空表格
            for item in self.pattern_tree.get_children():
                self.pattern_tree.delete(item)
            
            # 填充表格
            total_patterns = 0
            for pattern_name, indices in patterns.items():
                if indices:
                    count = len(indices)
                    total_patterns += count
                    first_idx = min(indices)
                    last_idx = max(indices)
                    
                    self.pattern_tree.insert("", tk.END, values=(
                        pattern_name,
                        count,
                        first_idx,
                        last_idx
                    ))
            
            self._set_status(f"✓ 識別完成：發現 {len(patterns)} 種型態，共 {total_patterns} 筆")
            messagebox.showinfo("成功", f"識別完成：發現 {len(patterns)} 種型態")
            
            # 存儲模式供後續使用
            self.current_patterns = patterns
            
        except Exception as e:
            messagebox.showerror("錯誤", f"識別型態失敗: {str(e)}")
            self._set_status(f"錯誤: {str(e)}")
        finally:
            self._stop_progress()
    
    def _run_backtest(self):
        """執行回測"""
        if self.current_df is None:
            messagebox.showwarning("提示", "請先獲取數據")
            return
        
        if not hasattr(self, 'current_patterns'):
            messagebox.showwarning("提示", "請先識別型態")
            return
        
        if self.backtester is None:
            messagebox.showwarning("提示", "回測引擎未初始化，請重新獲取數據")
            return
        
        thread = threading.Thread(target=self._run_backtest_thread)
        thread.start()
    
    def _run_backtest_thread(self):
        """執行回測（線程化）"""
        try:
            self._start_progress()
            self._set_status("正在執行回測...")
            
            all_metrics = {}
            
            # 對每個型態執行回測
            for pattern_name, indices in self.current_patterns.items():
                if not indices:
                    continue
                
                trades = self.backtester.backtest_pattern(
                    indices,
                    pattern_name
                )
                
                metrics = self.backtester.calculate_metrics(trades)
                all_metrics[pattern_name] = metrics
            
            # 清空回測結果表格
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)
            
            # 填充回測結果
            for pattern, metrics in all_metrics.items():
                self.result_tree.insert("", tk.END, values=(
                    pattern,
                    metrics.get('total_trades', 0),
                    f"{metrics.get('win_rate', 0):.1%}",
                    f"{metrics.get('profit_ratio', 0):.2f}",
                    f"{metrics.get('sharpe_ratio', 0):.2f}"
                ))
            
            # 生成排名
            self._update_ranking(all_metrics)
            
            self._set_status(f"✓ 回測完成：分析了 {len(all_metrics)} 種型態")
            messagebox.showinfo("成功", f"回測完成：分析了 {len(all_metrics)} 種型態")
            
            # 存儲結果供後續使用
            self.backtest_results = all_metrics
            
        except Exception as e:
            messagebox.showerror("錯誤", f"執行回測失敗: {str(e)}")
            self._set_status(f"錯誤: {str(e)}")
        finally:
            self._stop_progress()
    
    def _update_ranking(self, metrics_dict: Dict):
        """更新排名表格"""
        # 計算評分 (0-100分制，與analysis_report.html統一)
        rankings = []
        for pattern, metrics in metrics_dict.items():
            total_trades = metrics.get('total_trades', 0)
            
            if total_trades < 10:
                # 樣本數不足則評分為0
                score = 0
            else:
                # 綜合評分 (0-100分制)
                win_score = metrics.get('win_rate', 0) * 50  # 勝率50分
                profit_score = min(metrics.get('profit_factor', 0) / 2 * 30, 30)  # 盈虧比30分
                sharpe_score = min(max(metrics.get('sharpe_ratio', 0) / 2, 0) * 20, 20)  # 夏普比20分
                score = round(win_score + profit_score + sharpe_score, 2)
            
            rankings.append({
                'pattern': pattern,
                'score': score,
                'win_rate': metrics.get('win_rate', 0),
                'trades': total_trades
            })
        
        # 排序
        rankings = sorted(rankings, key=lambda x: x['score'], reverse=True)
        
        # 清空排名表格
        for item in self.ranking_tree.get_children():
            self.ranking_tree.delete(item)
        
        # 填充排名表格
        for rank, item in enumerate(rankings[:20], 1):  # 顯示前20名
            pattern_zh = get_pattern_name_zh(item['pattern'])
            self.ranking_tree.insert("", tk.END, 
                                    values=(
                                        rank,
                                        pattern_zh,  # 顯示中文名稱
                                        f"{item['score']:.2f}",
                                        f"{item['win_rate']:.1%}",
                                        item['trades']
                                    ),
                                    tags=(item['pattern'],))  # tag保持原始名稱以供龜擊
    
    def _run_complete_analysis(self):
        """完整分析 - 合併所有步驟（數據→型態→回測→報告）"""
        if self.is_running:
            messagebox.showwarning("提示", "程式正在運行中，請稍候")
            return
        
        thread = threading.Thread(target=self._complete_analysis_thread)
        thread.start()
    
    def _complete_analysis_thread(self):
        """完整分析線程化執行"""
        try:
            self._start_progress()
            
            # ==================== 清空舊數據 ====================
            # 清空所有舊的分析結果
            for item in self.data_text.get(1.0, tk.END):
                self.data_text.delete(1.0, tk.END)
            for item in self.pattern_tree.get_children():
                self.pattern_tree.delete(item)
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)
            for item in self.ranking_tree.get_children():
                self.ranking_tree.delete(item)
            
            # 重置內部狀態
            self.current_df = None
            self.current_symbol = None
            self.backtester = None
            self.current_patterns = {}
            self.backtest_results = {}
            
            # ==================== 第1步：獲取數據 ====================
            self._set_status("📥 步驟1/4: 正在獲取數據...")
            
            symbol = self.stock_var.get()
            start_date = self.start_date_var.get()
            end_date = self.end_date_var.get()
            
            # 驗證日期格式
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                messagebox.showerror("錯誤", "日期格式應為 YYYY-MM-DD")
                self._set_status("日期格式錯誤")
                self._stop_progress()
                return
            
            # 獲取數據
            df = self.fetcher.fetch_data(symbol, start_date, end_date, "1d")
            
            if df is None or len(df) == 0:
                messagebox.showerror("錯誤", f"無法獲取 {symbol} 的數據")
                self._set_status("數據獲取失敗")
                self._stop_progress()
                return
            
            # 深複製數據，避免被後續操作修改
            self.current_df = df.copy()
            self.current_symbol = symbol
            self.backtester = Backtester(self.current_df.copy())
            
            # 顯示數據信息
            info_text = f"""
【數據信息】
股票代碼: {symbol}
日期範圍: {start_date} 到 {end_date}
K線數量: {len(df)}
收盤價範圍: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}
平均成交量: {df['Volume'].mean():.0f}

【OHLCV 數據預覽】
{df.tail(10).to_string()}
            """
            
            self.data_text.delete(1.0, tk.END)
            self.data_text.insert(tk.END, info_text)
            
            # ==================== 第2步：識別型態 ====================
            self._set_status("🔍 步驟2/4: 正在識別型態...")
            
            # 識別型態
            pattern_recognizer = CandlePattern(self.current_df)
            patterns = pattern_recognizer.detect_all_patterns()
            
            # 清空表格
            for item in self.pattern_tree.get_children():
                self.pattern_tree.delete(item)
            
            # 填充表格
            total_patterns = 0
            for pattern_name, indices in patterns.items():
                if indices:
                    count = len(indices)
                    total_patterns += count
                    first_idx = min(indices)
                    last_idx = max(indices)
                    
                    self.pattern_tree.insert("", tk.END, values=(
                        pattern_name,
                        count,
                        first_idx,
                        last_idx
                    ))
            
            # 存儲模式供後續使用
            self.current_patterns = patterns
            
            # ==================== 第3步：執行回測 ====================
            self._set_status("📊 步驟3/4: 正在執行回測...")
            
            all_metrics = {}
            
            # 對每個型態執行回測
            for pattern_name, indices in self.current_patterns.items():
                if not indices:
                    continue
                
                trades = self.backtester.backtest_pattern(indices, pattern_name)
                metrics = self.backtester.calculate_metrics(trades)
                all_metrics[pattern_name] = metrics
            
            # 清空回測結果表格
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)
            
            # 填充回測結果
            for pattern, metrics in all_metrics.items():
                self.result_tree.insert("", tk.END, values=(
                    pattern,
                    metrics.get('total_trades', 0),
                    f"{metrics.get('win_rate', 0):.1%}",
                    f"{metrics.get('profit_ratio', 0):.2f}",
                    f"{metrics.get('sharpe_ratio', 0):.2f}"
                ))
            
            # 生成排名
            self._update_ranking(all_metrics)
            
            # 存儲結果供後續使用
            self.backtest_results = all_metrics
            
            # ==================== 第4步：生成報告 ====================
            self._set_status("📝 步驟4/4: 正在生成報告...")
            
            # 轉換數據結構以匹配 generate_html_report 期望的格式
            analysis_dict = {}
            for pattern_name, metrics in self.backtest_results.items():
                # 計算評分 (0-100分制，與analyze.py一致)
                total_trades = metrics.get('total_trades', 0)
                
                if total_trades < 10:
                    score = 0
                else:
                    win_score = metrics.get('win_rate', 0) * 50
                    profit_score = min(metrics.get('profit_factor', 0) / 2 * 30, 30)
                    sharpe_score = min(max(metrics.get('sharpe_ratio', 0) / 2, 0) * 20, 20)
                    score = round(win_score + profit_score + sharpe_score, 2)
                
                # 構建正確的結構
                analysis_dict[pattern_name] = {
                    'score': score,
                    'metrics': metrics
                }
            
            # 生成HTML報告
            output_file = f"analysis_report_{self.current_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            generate_html_report(
                analysis_dict, 
                f"{self.current_symbol} ({start_date} 至 {end_date})", 
                output_file
            )
            
            # ==================== 完成 ====================
            self._set_status(f"✅ 完整分析完成！識別{len(patterns)}種型態，分析{len(all_metrics)}種型態，報告已保存: {output_file}")
            messagebox.showinfo("成功", 
                f"✅ 完整分析完成！\n\n"
                f"📊 識別型態: {len(patterns)} 種，共 {total_patterns} 筆\n"
                f"📈 分析型態: {len(all_metrics)} 種\n\n"
                f"雙擊排名表選擇型態查看K線圖表")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"分析失敗: {str(e)}")
            self._set_status(f"❌ 分析失敗: {str(e)}")
        finally:
            self._stop_progress()
    
    def _show_ranking(self):
        """顯示排名 - 提示用戶雙擊查看型態"""
        if not hasattr(self, 'backtest_results'):
            messagebox.showwarning("提示", "請先執行完整分析")
            return
        
        # 切換到排名標籤並顯示提示
        self.notebook.select(3)  # 排名標籤通常是第4個(索引3)
        messagebox.showinfo("提示", "👆 在排名表中雙擊任意型態以查看對應的K線圖表")
    
    def _generate_report(self):
        """生成報告"""
        if self.current_df is None:
            messagebox.showwarning("提示", "請先獲取並分析數據")
            return
        
        if not hasattr(self, 'backtest_results'):
            messagebox.showwarning("提示", "請先執行回測")
            return
        
        try:
            self._set_status("正在生成報告...")
            
            # 轉換數據結構以匹配 generate_html_report 期望的格式
            analysis_dict = {}
            for pattern_name, metrics in self.backtest_results.items():
                # 計算評分 (0-100分制，與analyze.py一致)
                total_trades = metrics.get('total_trades', 0)
                
                if total_trades < 10:
                    score = 0
                else:
                    win_score = metrics.get('win_rate', 0) * 50
                    profit_score = min(metrics.get('profit_factor', 0) / 2 * 30, 30)
                    sharpe_score = min(max(metrics.get('sharpe_ratio', 0) / 2, 0) * 20, 20)
                    score = round(win_score + profit_score + sharpe_score, 2)
                
                # 構建正確的結構
                analysis_dict[pattern_name] = {
                    'score': score,
                    'metrics': metrics
                }
            
            # 生成HTML報告
            output_file = f"analysis_report_{self.current_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            generate_html_report(
                analysis_dict,
                self.current_symbol,
                output_file
            )
            
            # 打開報告
            import webbrowser
            webbrowser.open(f"file:///{Path(output_file).absolute()}")
            
            self._set_status(f"✓ 報告已生成: {output_file}")
            messagebox.showinfo("成功", f"報告已生成並在瀏覽器中打開")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"生成報告失敗: {str(e)}")
            self._set_status(f"錯誤: {str(e)}")
    
    def _show_candlestick(self):
        """顯示K線圖表"""
        if self.current_df is None:
            messagebox.showwarning("提示", "請先獲取數據")
            return
        
        if not hasattr(self, 'current_patterns'):
            messagebox.showwarning("提示", "請先識別型態")
            return
        
        thread = threading.Thread(target=self._show_candlestick_thread)
        thread.start()
    
    def _show_candlestick_thread(self):
        """顯示K線圖表（線程化）"""
        try:
            self._start_progress()
            self._set_status("正在生成圖表...")
            
            output_dir = Path(f"pattern_visualizations/{self.current_symbol.replace('.', '_')}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 繪製K線圖
            self.visualizer.plot_candle_with_ma(
                self.current_df,
                ma_periods=[5, 20, 50],
                output_path=str(output_dir / "candlestick_with_ma.png")
            )
            
            # 繪製型態圖表
            self.visualizer.plot_patterns(
                self.current_df,
                self.current_patterns,
                output_dir=str(output_dir / "patterns")
            )
            
            self._set_status(f"✓ 圖表已保存到: {output_dir}")
            messagebox.showinfo("成功", f"所有圖表已保存到:\n{output_dir}")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"生成圖表失敗: {str(e)}")
            self._set_status(f"錯誤: {str(e)}")
        finally:
            self._stop_progress()
    
    def _load_chart(self):
        """載入圖表到GUI"""
        if self.current_df is None:
            messagebox.showwarning("提示", "請先獲取數據")
            return
        
        chart_type = self.chart_var.get()
        
        if chart_type == "K線+MA":
            self._plot_candlestick_chart()
        else:
            # 如果選擇型態圖表，需要先識別型態
            if not hasattr(self, 'current_patterns'):
                messagebox.showwarning("提示", "請先識別型態")
                return
            
            # 映射型態名稱
            pattern_map = {
                "Hammer": "hammer",
                "Bullish Harami": "bullish_harami",
                "Bearish Engulfing": "bearish_engulfing"
            }
            pattern_name = pattern_map.get(chart_type, "hammer")
            self._plot_pattern_chart(pattern_name)
    
    def _open_html_report(self):
        """打開HTML報告"""
        try:
            import webbrowser
            from pathlib import Path
            
            report_path = Path("analysis_report.html")
            if report_path.exists():
                # 轉換為絕對路徑
                abs_path = report_path.resolve()
                webbrowser.open(f"file:///{abs_path}")
                self._set_status(f"✓ 已打開報告: {report_path.name}")
            else:
                messagebox.showwarning("提示", "報告文件不存在，請先運行回測")
                self._set_status("❌ 報告文件不存在")
                
        except Exception as e:
            messagebox.showerror("錯誤", f"打開報告失敗: {str(e)}")
            self._set_status(f"錯誤: {str(e)}")
    
    def _on_ranking_click(self, event):
        """排名表格點擊事件 - 雙擊查看型態K線圖"""
        try:
            # 獲取被點擊的項目
            item = self.ranking_tree.selection()[0]
            values = self.ranking_tree.item(item, 'values')
            
            if not values or len(values) < 2:
                return
            
            # 從表格讀取型態名稱（第2列）
            pattern_name = values[1] if len(values) > 1 else None
            
            if not pattern_name:
                messagebox.showwarning("提示", "無法讀取型態名稱")
                return
            
            # 將中文型態名映射回英文
            pattern_mapping = {
                '十字線': 'doji',
                '錘子線': 'hammer',
                '上吊線': 'hanging_man',
                '倒錘線': 'inverted_hammer',
                '看漲孕線': 'bullish_harami',
                '看跌孕線': 'bearish_harami',
                '看漲吞沒': 'bullish_engulfing',
                '看跌吞沒': 'bearish_engulfing',
                '穿刺線': 'piercing_line',
                '烏雲蓋頂': 'dark_cloud_cover',
                '晨星': 'morning_star',
                '晚星': 'evening_star',
                '三根烏鴉': 'three_black_crows',
                '三個白兵': 'three_white_soldiers',
                '牛旗形': 'bullish_flag',
                '熊旗形': 'bearish_flag',
                '三重頂': 'triple_top',
                '三重底': 'triple_bottom',
            }
            
            pattern_name_en = pattern_mapping.get(pattern_name, pattern_name.lower())
            
            # 後臺線程加載首個股票(2330)的數據並顯示該型態的圖表
            thread = threading.Thread(target=self._load_pattern_chart_thread, args=(pattern_name_en, pattern_name))
            thread.daemon = True
            thread.start()
            
        except IndexError:
            # 沒有選中任何項目
            return
        except Exception as e:
            messagebox.showerror("錯誤", f"加載圖表失敗: {str(e)}")
            self._set_status(f"錯誤: {str(e)}")
    
    def _load_pattern_chart_thread(self, pattern_name_en: str, pattern_name_zh: str):
        """後臺加載型態圖表（自動獲取第一個股票的數據）"""
        try:
            # 在主線程中啟動進度
            self.root.after(0, self._start_progress)
            self.root.after(0, lambda: self._set_status(f"⏳ 正在加載 {pattern_name_zh} 的K線圖..."))
            
            # 使用第一個股票 (2330 TSMC)
            symbol = STOCK_SYMBOLS[0]  # 通常是 '2330.TW'
            start_date = ANALYSIS_PERIOD['start_date']
            end_date = ANALYSIS_PERIOD['end_date']
            
            # 優化1：如果已經加載過該股票就重用數據（快速路徑）
            if self.current_symbol == symbol and self.current_df is not None:
                df = self.current_df
                patterns = self.current_patterns
                self.root.after(0, lambda: self._set_status(f"✓ 使用快取數據 ({symbol})"))
            else:
                # 否則重新獲取
                self.root.after(0, lambda: self._set_status(f"⏳ 正在獲取 {symbol} 的數據..."))
                df = self.fetcher.fetch_data(symbol, start_date, end_date)
                
                if df is None or df.empty:
                    def show_error():
                        messagebox.showwarning("提示", f"無法獲取 {symbol} 的數據")
                        self._set_status("❌ 無法獲取數據")
                        self._stop_progress()
                    self.root.after(0, show_error)
                    return
                
                # 識別所有型態
                self.root.after(0, lambda: self._set_status(f"⏳ 正在識別型態..."))
                pattern_recognizer = CandlePattern(df)
                patterns = pattern_recognizer.detect_all_patterns()
            
            # 存儲當前數據和型態
            self.current_df = df
            self.current_symbol = symbol
            self.current_patterns = patterns
            
            # 檢查該型態是否存在
            if pattern_name_en not in patterns or not patterns[pattern_name_en]:
                def show_not_found():
                    messagebox.showinfo("提示", f"股票 {symbol} 中未發現 {pattern_name_zh} 型態\n請點擊 '📊 查看綜合報告' 查看其他股票的該型態")
                    self._set_status(f"⚠️ {symbol} 中未找到 {pattern_name_zh}")
                    self._stop_progress()
                self.root.after(0, show_not_found)
                return
            
            # 在主線程中執行 GUI 更新
            def update_gui():
                try:
                    # 跳轉到K線圖表標籤
                    self.notebook.select(3)  # K線圖表是第4個標籤 (索引3)
                    
                    # 繪製圖表
                    self._plot_pattern_chart(pattern_name_en)
                    
                    self._set_status(f"✓ 已加載 {symbol} 的 {pattern_name_zh} K線圖 (共 {len(patterns[pattern_name_en])} 筆)")
                except Exception as e:
                    messagebox.showerror("錯誤", f"繪製圖表失敗: {str(e)}")
                    self._set_status(f"❌ 繪製失敗: {str(e)}")
                    logger.error(f"繪製圖表失敗: {e}")
                finally:
                    self._stop_progress()
            
            self.root.after(0, update_gui)
            
        except Exception as e:
            def show_error():
                messagebox.showerror("錯誤", f"加載圖表失敗: {str(e)}")
                self._set_status(f"❌ 錯誤: {str(e)}")
                logger.error(f"加載型態圖表失敗: {e}")
                self._stop_progress()
            self.root.after(0, show_error)
    
    def _load_chart_with_pattern(self, pattern_name: str):
        """指定型態加載圖表"""
        if self.current_df is None:
            messagebox.showwarning("提示", "請先獲取數據")
            return
        
        if not hasattr(self, 'current_patterns'):
            messagebox.showwarning("提示", "請先識別型態")
            return
        
        self._plot_pattern_chart(pattern_name)
    
    def _load_chart(self):
        """載入圖表到GUI"""
        if self.current_df is None:
            messagebox.showwarning("提示", "請先獲取數據")
            return
        
        chart_type = self.chart_var.get()
        
        if chart_type == "K線+MA":
            self._plot_candlestick_chart()
        else:
            # 如果選擇型態圖表，需要先識別型態
            if not hasattr(self, 'current_patterns'):
                messagebox.showwarning("提示", "請先識別型態")
                return
            
            # 映射型態名稱
            pattern_map = {
                "Hammer": "hammer",
                "Bullish Harami": "bullish_harami",
                "Bearish Engulfing": "bearish_engulfing"
            }
            pattern_name = pattern_map.get(chart_type, "hammer")
            self._plot_pattern_chart(pattern_name)
    
    def _plot_candlestick_chart(self):
        """在GUI中繪製K線圖表"""
        try:
            # 清空前一個圖表
            for widget in self.chart_canvas_frame.winfo_children():
                widget.destroy()
            
            # 創建圖表
            fig = Figure(figsize=(12, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            # 繪製K線
            width = 0.6
            for i in range(len(self.current_df)):
                row = self.current_df.iloc[i]
                
                # 判斷顏色
                color = '#FF0000' if row['Close'] >= row['Open'] else '#00AA00'
                
                # 繪製影線
                ax.plot([i, i], [row['Low'], row['High']], color=color, linewidth=1)
                
                # 繪製實體
                height = abs(row['Close'] - row['Open'])
                bottom = min(row['Open'], row['Close'])
                rect = patches.Rectangle((i - width/2, bottom), width, height,
                                        linewidth=0.5, edgecolor=color, facecolor=color, alpha=0.7)
                ax.add_patch(rect)
            
            # 繪製移動平均線
            for period, color in zip([5, 20, 50], ['blue', 'orange', 'purple']):
                ma = self.current_df['Close'].rolling(window=period).mean()
                ax.plot(range(len(ma)), ma, label=f'MA{period}', color=color, linewidth=2, alpha=0.7)
            
            ax.set_xlim(-1, len(self.current_df))
            ax.set_ylim(self.current_df['Low'].min() * 0.98, self.current_df['High'].max() * 1.02)
            ax.set_xlabel('時間')
            ax.set_ylabel('價格')
            ax.set_title(f'{self.current_symbol} K線圖 + 移動平均線')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # 嵌入到tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.chart_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self._set_status("✓ K線圖已加載")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"繪製圖表失敗: {str(e)}")
            self._set_status(f"錯誤: {str(e)}")
    
    def _plot_pattern_chart(self, pattern_name: str):
        """在GUI中繪製型態圖表（正規化版本 - 相對時間序列 + 正規化數值）"""
        try:
            if pattern_name not in self.current_patterns:
                messagebox.showwarning("提示", f"未發現型態: {pattern_name}")
                return
            
            indices = self.current_patterns[pattern_name]
            if not indices:
                messagebox.showwarning("提示", f"型態 {pattern_name} 沒有實例")
                return
            
            # 清空前一個圖表
            for widget in self.chart_canvas_frame.winfo_children():
                widget.destroy()
            
            # 根據型態天數計算顯示範圍
            pattern_days_map = {
                'doji': 1, 'hammer': 1, 'hanging_man': 1, 'inverted_hammer': 1,
                'bullish_harami': 2, 'bearish_harami': 2, 'bullish_engulfing': 2,
                'bearish_engulfing': 2, 'piercing_line': 2, 'dark_cloud_cover': 2,
                'morning_star': 3, 'evening_star': 3, 'three_black_crows': 3,
                'three_white_soldiers': 3, 'bullish_flag': 5, 'bearish_flag': 5,
                'triple_top': 5, 'triple_bottom': 5,
            }
            pattern_days = pattern_days_map.get(pattern_name, 3)
            pattern_zh = get_pattern_name_zh(pattern_name)  # 中文型態名
            
            # 取第一個型態實例
            candle_idx = indices[0]
            start = max(0, candle_idx - pattern_days + 1)
            end = min(len(self.current_df), candle_idx + 1)
            
            # 創建相對時間標籤 (T-n, T0 格式或 Day 1-5 格式)
            num_days = end - start
            relative_time_labels = [f"Day {i+1}" for i in range(num_days)]
            
            # 取數據並正規化
            subset = self.current_df.iloc[start:end].copy()
            subset = subset.reset_index(drop=True)
            
            # 正規化價格（相對百分比）
            subset_price_norm = self._normalize_price(subset)
            
            # 正規化成交量（相對均量比）
            volume_normalized = self._normalize_volume(subset)
            
            # 創建圖表（上K線，下成交量）
            fig = Figure(figsize=(12, 6), dpi=100)
            ax_price = fig.add_subplot(211)  # 上面K線
            ax_volume = fig.add_subplot(212)  # 下面成交量
            
            width = 0.6
            
            # ========== 繪製K線（正規化價格）==========
            for i in range(len(subset_price_norm)):
                row = subset_price_norm.iloc[i]
                actual_idx = start + i
                
                # 根據漲跌決定顏色
                is_up = subset.iloc[i]['Close'] >= subset.iloc[i]['Open']
                color = '#FF0000' if is_up else '#00AA00'  # 紅漲綠跌
                
                # 決定邊框寬度（信號K線加粗）
                is_signal = (actual_idx == candle_idx)
                edge_width = 3.0 if is_signal else 0.5
                edge_color = '#FFD700' if is_signal else color  # 信號K線用金色邊框
                
                # 繪製影線
                line_width = 2 if is_signal else 1
                ax_price.plot([i, i], [row['Low'], row['High']], color=color, linewidth=line_width)
                
                # 繪製實體
                height = abs(row['Close'] - row['Open'])
                bottom = min(row['Open'], row['Close'])
                rect = patches.Rectangle((i - width/2, bottom), width, height,
                                        linewidth=edge_width, edgecolor=edge_color, facecolor=color, alpha=0.7)
                ax_price.add_patch(rect)
            
            ax_price.set_xlim(-1, len(subset_price_norm))
            ax_price.set_ylim(subset_price_norm['Low'].min() * 1.1, subset_price_norm['High'].max() * 1.1)
            ax_price.set_ylabel('相對漲跌幅 (%)')
            ax_price.set_xlabel('相對時間序列')
            ax_price.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)  # 基準線
            ax_price.set_title(f"型態: {pattern_zh} ({pattern_days}根K線) | 信號K線 = Day {pattern_days} (黃框)")
            ax_price.set_xticks(range(len(subset)))
            ax_price.set_xticklabels(relative_time_labels, rotation=0)
            ax_price.grid(True, alpha=0.3)
            
            # ========== 繪製成交量（正規化成交量比）==========
            for i in range(len(volume_normalized)):
                is_up = subset.iloc[i]['Close'] >= subset.iloc[i]['Open']
                vol_color = '#FF0000' if is_up else '#00AA00'  # 紅漲綠跌
                ax_volume.bar(i, volume_normalized[i], width=width, color=vol_color, alpha=0.7)
            
            ax_volume.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)  # 平均線
            ax_volume.set_xlim(-1, len(volume_normalized))
            ax_volume.set_xticks(range(len(subset)))
            ax_volume.set_xticklabels(relative_time_labels, rotation=0)
            ax_volume.set_xlabel('相對時間序列')
            ax_volume.set_ylabel('相對均量比 (Ratio)')
            ax_volume.grid(True, alpha=0.3)
            
            # 嵌入到tkinter
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.chart_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self._set_status(f"✓ 型態圖已加載: {pattern_zh}")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"繪製型態圖失敗: {str(e)}")
            self._set_status(f"錯誤: {str(e)}")


def main():
    """主函數"""
    root = tk.Tk()
    app = KLineAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
