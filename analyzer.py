"""
分析與統計模塊 - 型態有效性評估與排名
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import json
import logging

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """型態分析與統計"""
    
    def __init__(self):
        self.results = {}
    
    def analyze_pattern_performance(self, pattern_metrics: Dict) -> Dict:
        """
        分析型態績效
        
        Args:
            pattern_metrics: {pattern_name: metrics_dict}
            
        Returns:
            評分和排名
        """
        analysis = {}
        
        for pattern, metrics in pattern_metrics.items():
            if metrics['total_trades'] < 10:
                score = 0
                reason = "樣本數不足"
            else:
                # 綜合評分 (0-100)
                win_score = metrics['win_rate'] * 50  # 勝率50分
                profit_score = min(metrics['profit_factor'] / 2 * 30, 30)  # 盈虧比30分
                sharpe_score = min(max(metrics['sharpe_ratio'] / 2, 0) * 20, 20)  # 夏普比20分
                
                score = win_score + profit_score + sharpe_score
                reason = f"勝率{metrics['win_rate']:.1%}, 盈虧比{metrics['profit_factor']:.2f}"
            
            analysis[pattern] = {
                'score': round(score, 2),
                'reason': reason,
                'metrics': metrics,
            }
        
        return analysis
    
    def rank_patterns(self, analysis: Dict) -> List[Tuple[str, float]]:
        """排名型態"""
        ranked = sorted(
            [(p, a['score']) for p, a in analysis.items()],
            key=lambda x: x[1],
            reverse=True
        )
        return ranked
    
    def generate_summary_report(self, 
                               pattern_metrics: Dict,
                               symbol: str) -> str:
        """生成文本報告"""
        analysis = self.analyze_pattern_performance(pattern_metrics)
        ranked = self.rank_patterns(analysis)
        
        report = f"""
╔═══════════════════════════════════════════════════════════╗
║          K線型態回測分析報告 - {symbol}
╚═══════════════════════════════════════════════════════════╝

【型態排名】
"""
        
        for rank, (pattern, score) in enumerate(ranked, 1):
            metrics = analysis[pattern]['metrics']
            
            report += f"""
{rank}. {pattern.upper()} (評分: {score:.2f})
   ├─ 交易數: {metrics.get('total_trades', 0)}
   ├─ 勝率: {metrics.get('win_rate', 0):.1%}
   ├─ 盈虧比: {metrics.get('profit_factor', 0):.2f}
   ├─ 平均損益: ${metrics.get('avg_profit', 0):.2f}
   ├─ 夏普比: {metrics.get('sharpe_ratio', 0):.2f}
   └─ 最大回撤: {metrics.get('max_drawdown', 0):.1%}
"""
        
        report += "\n【統計摘要】\n"
        total_trades = sum(m.get('total_trades', 0) for m in pattern_metrics.values())
        valid_patterns = len([m for m in pattern_metrics.values() if m.get('total_trades', 0) >= 10])
        
        report += f"""
   ├─ 總交易數: {total_trades}
   ├─ 有效型態: {valid_patterns}/11
   └─ 分析期間: 252日
"""
        
        return report
    
    def export_csv(self, analysis: Dict, filename: str = 'patterns_stats.csv'):
        """匯出CSV"""
        rows = []
        for pattern, data in analysis.items():
            metrics = data['metrics']
            rows.append({
                '型態': pattern,
                '評分': data['score'],
                '交易數': metrics.get('total_trades', 0),
                '勝率': f"{metrics.get('win_rate', 0):.1%}",
                '盈虧比': f"{metrics.get('profit_factor', 0):.2f}",
                '平均損益': f"${metrics.get('avg_profit', 0):.2f}",
                '夏普比': f"{metrics.get('sharpe_ratio', 0):.2f}",
                '最大回撤': f"{metrics.get('max_drawdown', 0):.1%}",
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"已匯出 {filename}")
        
        return df
    
    def export_json(self, analysis: Dict, filename: str = 'backtest_results.json'):
        """匯出JSON"""
        output = {}
        for pattern, data in analysis.items():
            metrics = data['metrics'].copy()
            if 'trades_df' in metrics:
                del metrics['trades_df']
            
            output[pattern] = {
                'score': data['score'],
                'reason': data['reason'],
                'metrics': metrics,
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"已匯出 {filename}")


def generate_html_report(analysis: Dict, 
                        symbol: str, 
                        filename: str = 'analysis_report.html') -> str:
    """生成互動HTML報告"""
    
    ranked = sorted(
        [(p, a['score']) for p, a in analysis.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # 構建表格行
    table_rows = ""
    for rank, (pattern, score) in enumerate(ranked, 1):
        metrics = analysis[pattern]['metrics']
        table_rows += f"""
    <tr>
        <td>{rank}</td>
        <td><strong>{pattern}</strong></td>
        <td>{score:.2f}</td>
        <td>{metrics.get('total_trades', 0)}</td>
        <td>{metrics.get('win_rate', 0):.1%}</td>
        <td>{metrics.get('profit_factor', 0):.2f}</td>
        <td>${metrics.get('avg_profit', 0):.2f}</td>
        <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
        <td>{metrics.get('max_drawdown', 0):.1%}</td>
    </tr>
"""
    
    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>K線型態回測分析報告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background-color: #007bff; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 12px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background-color: #f9f9f9; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; padding: 10px 15px; background: #e9ecef; border-radius: 5px; }}
        .metric-label {{ font-weight: bold; color: #555; }}
        .metric-value {{ color: #007bff; font-size: 18px; }}
        .highlight {{ background-color: #fff3cd; }}
        footer {{ margin-top: 40px; text-align: center; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 K線型態回測分析報告</h1>
        <p><strong>標的：</strong> {symbol}</p>
        <p><strong>生成時間：</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📈 型態排名</h2>
        <table>
            <thead>
                <tr>
                    <th>排名</th>
                    <th>型態</th>
                    <th>評分</th>
                    <th>交易數</th>
                    <th>勝率</th>
                    <th>盈虧比</th>
                    <th>平均損益</th>
                    <th>夏普比</th>
                    <th>最大回撤</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        
        <h2>📋 關鍵指標說明</h2>
        <ul>
            <li><strong>評分：</strong>綜合勝率、盈虧比、風險調整後收益(0-100)</li>
            <li><strong>勝率：</strong>獲利交易 / 總交易數</li>
            <li><strong>盈虧比：</strong>平均獲利 / 平均虧損(越高越好)</li>
            <li><strong>夏普比：</strong>風險調整後收益(>1.0表現佳)</li>
            <li><strong>最大回撤：</strong>從高峰到谷底的最大跌幅</li>
        </ul>
        
        <h2>💡 建議</h2>
        <ul>
            <li>評分在60以上的型態較值得信賴</li>
            <li>優先使用交易數足夠(>30)的型態進行交易</li>
            <li>結合多個型態可提高系統穩定性</li>
            <li>定期重新回測以適應市場變化</li>
        </ul>
        
        <footer>
            <p>此報告僅供參考，不構成投資建議。過往績效不代表未來表現。</p>
        </footer>
    </div>
</body>
</html>
"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"已生成 {filename}")
    return filename


if __name__ == '__main__':
    analyzer = PatternAnalyzer()
    print("分析模塊已初始化")
