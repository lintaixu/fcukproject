import numpy as np

class PortfolioOptimizer:
    def __init__(self, max_sector_weight=0.3, max_stock_weight=0.1):
        self.max_sector_weight = max_sector_weight
        self.max_stock_weight = max_stock_weight

    def optimize(self, day_df, top_n=10):
        candidates = day_df.sort_values('pred_rank', ascending=False).head(top_n * 2)
        candidates['inv_vol'] = 1.0 / candidates['atr_percentile'].replace(0, 0.01)
        candidates['weight'] = candidates['inv_vol'] / candidates['inv_vol'].sum()
        candidates['weight'] = candidates['weight'].clip(upper=self.max_stock_weight)
        return candidates.sort_values('weight', ascending=False).head(top_n)

class RiskEngine:
    def calculate_exposure(self, market_stress, current_drawdown):
        exposure = 0.9
        if market_stress > 0.8: exposure *= 0.5
        if current_drawdown < -0.2: exposure *= 0.2
        return max(0.0, exposure)
