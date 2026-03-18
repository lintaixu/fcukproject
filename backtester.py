"""
回測框架 - 模擬交易並計算績效
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """交易記錄"""
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    pattern: str
    direction: str  # 'long' or 'short'
    quantity: float
    commission: float
    pnl: float
    pnl_pct: float


class Backtester:
    """回測引擎"""
    
    def __init__(self, 
                 df: pd.DataFrame,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 stop_loss: float = 0.05,
                 take_profit: float = 0.10):
        """
        Args:
            df: OHLCV DataFrame
            initial_capital: 初始資金
            commission: 手續費比例
            slippage: 滑點比例
            stop_loss: 止損比例
            take_profit: 止盈比例
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        
    def simulate_trade(self, 
                       entry_idx: int, 
                       pattern: str,
                       direction: str = 'long',
                       exit_bars: int = 5) -> Trade:
        """
        模擬一筆交易
        
        Args:
            entry_idx: 進場K線索引
            pattern: 型態名稱
            direction: 交易方向 ('long' or 'short')
            exit_bars: 持有天數
            
        Returns:
            Trade object
        """
        if entry_idx + exit_bars >= len(self.df):
            return None
        
        entry_bar = self.df.iloc[entry_idx]
        exit_bar = self.df.iloc[entry_idx + exit_bars]
        
        # 進場價格（加入滑點）
        if direction == 'long':
            entry_price = entry_bar['Close'] * (1 + self.slippage)
        else:
            entry_price = entry_bar['Close'] * (1 - self.slippage)
        
        # 計算手續費
        commission_cost = entry_price * self.commission
        
        # 出場價格
        exit_price = exit_bar['Close']
        
        # 檢查止損/止盈
        high_price = self.df.iloc[entry_idx:entry_idx + exit_bars]['High'].max()
        low_price = self.df.iloc[entry_idx:entry_idx + exit_bars]['Low'].min()
        
        if direction == 'long':
            # 檢查止損
            if low_price <= entry_price * (1 - self.stop_loss):
                exit_price = entry_price * (1 - self.stop_loss)
            # 檢查止盈
            elif high_price >= entry_price * (1 + self.take_profit):
                exit_price = entry_price * (1 + self.take_profit)
        else:
            # 空單邏輯
            if high_price >= entry_price * (1 + self.stop_loss):
                exit_price = entry_price * (1 + self.stop_loss)
            elif low_price <= entry_price * (1 - self.take_profit):
                exit_price = entry_price * (1 - self.take_profit)
        
        # 計算損益
        if direction == 'long':
            pnl = (exit_price - entry_price) * 1 - commission_cost * 2
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) * 1 - commission_cost * 2
            pnl_pct = (entry_price - exit_price) / entry_price
        
        trade = Trade(
            entry_date=self.df.index[entry_idx],
            entry_price=entry_price,
            exit_date=self.df.index[entry_idx + exit_bars],
            exit_price=exit_price,
            pattern=pattern,
            direction=direction,
            quantity=1,
            commission=commission_cost * 2,
            pnl=pnl,
            pnl_pct=pnl_pct
        )
        
        return trade
    
    def backtest_pattern(self, 
                        pattern_indices: List[int], 
                        pattern_name: str,
                        direction: str = 'long',
                        exit_bars: int = 5) -> List[Trade]:
        """
        回測單一型態
        
        Returns:
            List of Trade objects
        """
        trades = []
        for idx in pattern_indices:
            trade = self.simulate_trade(idx, pattern_name, direction, exit_bars)
            if trade:
                trades.append(trade)
        
        return trades
    
    def calculate_metrics(self, trades: List[Trade]) -> Dict:
        """計算績效指標"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_profit': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
            }
        
        df_trades = pd.DataFrame([
            {
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'date': t.exit_date,
            }
            for t in trades
        ])
        
        wins = (df_trades['pnl'] > 0).sum()
        losses = (df_trades['pnl'] < 0).sum()
        
        win_rate = wins / len(trades) if len(trades) > 0 else 0
        
        total_gains = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
        total_losses = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
        
        profit_factor = total_gains / total_losses if total_losses > 0 else 0
        
        avg_profit = df_trades['pnl'].mean()
        
        # 計算夏普比率
        returns = df_trades['pnl_pct'].values
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # 計算最大回撤
        equity = self.initial_capital + df_trades['pnl'].cumsum().values
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'wins': wins,
            'losses': losses,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'total_profit': df_trades['pnl'].sum(),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades_df': df_trades,
        }


if __name__ == '__main__':
    from data_fetcher import DataFetcher
    
    fetcher = DataFetcher()
    df = fetcher.fetch_data('2330.TW', '2023-01-01', '2024-01-01')
    
    backtester = Backtester(df, initial_capital=100000)
    print("回測框架已初始化")
