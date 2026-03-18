"""
主程序 - 完整K線型態回測流程
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analyzer import PatternAnalyzer, generate_html_report
from backtester import Backtester
from config import (
    ANALYSIS_CONFIG,
    ANALYSIS_PERIOD,
    BACKTEST_CONFIG,
    DATA_CONFIG,
    PATTERN_CONFIG,
    STOCK_SYMBOLS,
)
from data_fetcher import DataFetcher
from pattern_recognizer import CandlePattern

# 配置日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """主函數"""

    logger.info("=" * 60)
    logger.info("開始 K線型態回測分析")
    logger.info("=" * 60)

    # 初始化模塊
    fetcher = DataFetcher(DATA_CONFIG["cache_dir"])
    analyzer = PatternAnalyzer()

    # 存儲全部結果
    all_pattern_metrics = {}
    all_trades = {}

    # 遍歷股票
    symbols_to_test = STOCK_SYMBOLS[:3]  # 測試前3個股票以加快速度

    for symbol in symbols_to_test:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"分析股票: {symbol}")
        logger.info(f"{'=' * 60}")

        try:
            # 1. 獲取數據
            logger.info(
                f"獲取歷史數據 ({ANALYSIS_PERIOD['start_date']} 到 {ANALYSIS_PERIOD['end_date']})"
            )
            df = fetcher.fetch_data(
                symbol, ANALYSIS_PERIOD["start_date"], ANALYSIS_PERIOD["end_date"], "1d"
            )

            if df.empty or len(df) < 100:
                logger.warning(f"{symbol} 數據不足，跳過")
                continue

            if not fetcher.validate_data(df):
                logger.warning(f"{symbol} 數據驗證失敗，跳過")
                continue

            # 2. 識別型態
            logger.info(f"識別K線型態...")
            recognizer = CandlePattern(df)
            patterns = recognizer.detect_all_patterns()

            pattern_summary = ", ".join(
                [f"{p}({len(idx)})強]" for p, idx in patterns.items() if idx]
            )
            logger.info(f"發現型態: {pattern_summary}")

            # 3. 回測
            logger.info(f"執行回測...")
            backtester = Backtester(
                df,
                initial_capital=BACKTEST_CONFIG["initial_capital"],
                commission=BACKTEST_CONFIG["commission"],
                slippage=BACKTEST_CONFIG["slippage"],
                stop_loss=BACKTEST_CONFIG["stop_loss_ratio"],
                take_profit=BACKTEST_CONFIG["take_profit_ratio"],
            )

            pattern_metrics = {}
            symbol_trades = {}

            for pattern_name, indices in patterns.items():
                if len(indices) >= 5:  # 至少有5個訊號
                    trades = backtester.backtest_pattern(
                        indices, pattern_name, direction="long", exit_bars=5
                    )

                    metrics = backtester.calculate_metrics(trades)
                    pattern_metrics[pattern_name] = metrics
                    symbol_trades[pattern_name] = trades

                    logger.info(
                        f"  {pattern_name}: {len(trades)} 筆, "
                        f"勝率 {metrics['win_rate']:.1%}, "
                        f"盈虧比 {metrics['profit_factor']:.2f}"
                    )

            # 4. 存儲結果
            all_pattern_metrics[symbol] = pattern_metrics
            all_trades[symbol] = symbol_trades

        except Exception as e:
            logger.error(f"處理 {symbol} 時出錯: {e}", exc_info=True)
            continue

    # 5. 生成報告
    logger.info(f"\n{'=' * 60}")
    logger.info("生成分析報告")
    logger.info(f"{'=' * 60}")

    # 統計全局結果
    combined_metrics = {}
    for symbol, metrics_dict in all_pattern_metrics.items():
        for pattern, metrics in metrics_dict.items():
            if pattern not in combined_metrics:
                combined_metrics[pattern] = {
                    "total_trades": 0,
                    "total_wins": 0,
                    "total_loss": 0,
                    "total_pnl": 0,
                    "symbols": set(),
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "sharpe_ratio": 0.0,
                    "avg_profit": 0.0,
                    "max_drawdown": 0.0,
                }

            combined_metrics[pattern]["total_trades"] += metrics.get("total_trades", 0)
            combined_metrics[pattern]["total_wins"] += metrics.get("wins", 0)
            combined_metrics[pattern]["total_loss"] += metrics.get("losses", 0)
            combined_metrics[pattern]["total_pnl"] += metrics.get("total_profit", 0)
            combined_metrics[pattern]["symbols"].add(symbol)

            # Simple averaging of these metrics for the combined result
            n = len(combined_metrics[pattern]["symbols"])
            combined_metrics[pattern]["win_rate"] = (
                combined_metrics[pattern]["win_rate"] * (n - 1)
                + metrics.get("win_rate", 0)
            ) / n
            combined_metrics[pattern]["profit_factor"] = (
                combined_metrics[pattern]["profit_factor"] * (n - 1)
                + metrics.get("profit_factor", 0)
            ) / n
            combined_metrics[pattern]["sharpe_ratio"] = (
                combined_metrics[pattern]["sharpe_ratio"] * (n - 1)
                + metrics.get("sharpe_ratio", 0)
            ) / n
            combined_metrics[pattern]["avg_profit"] = (
                combined_metrics[pattern]["avg_profit"] * (n - 1)
                + metrics.get("avg_profit", 0)
            ) / n
            combined_metrics[pattern]["max_drawdown"] = max(
                combined_metrics[pattern]["max_drawdown"],
                metrics.get("max_drawdown", 0),
            )

    # 轉換為分析格式
    combined_analysis = analyzer.analyze_pattern_performance(combined_metrics)

    # 打印文本報告
    if combined_metrics:
        report_text = analyzer.generate_summary_report(combined_metrics, "台灣股市樣本")
        logger.info(report_text)

        # 匯出結果
        analyzer.export_csv(combined_analysis, "patterns_stats.csv")
        analyzer.export_json(combined_analysis, "backtest_results.json")
        generate_html_report(
            combined_analysis, "台灣股市樣本 (2022-2025)", "analysis_report.html"
        )

        logger.info("\n✓ 已生成以下文件:")
        logger.info("  - patterns_stats.csv (型態統計)")
        logger.info("  - backtest_results.json (詳細結果)")
        logger.info("  - analysis_report.html (互動報告)")
    else:
        logger.warning("沒有足夠的交易訊號進行分析")

    logger.info(f"\n{'=' * 60}")
    logger.info("分析完成！")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
