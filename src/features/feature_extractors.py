import pandas as pd
import glob
import os
import numpy as np
import pandas_ta as ta

class TrendFeatureExtractor:
    def __init__(self, price_df: pd.DataFrame):
        self.df = price_df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')

    def extract(self) -> pd.DataFrame:
        df = self.df
        df['ma5'] = ta.sma(df['close'], length=5)
        df['ma20'] = ta.sma(df['close'], length=20)
        df['ma60'] = ta.sma(df['close'], length=60)
        df['ma5_slope'] = (df['ma5'] - df['ma5'].shift(1)) / df['ma5'].shift(1)
        df['ma20_slope'] = (df['ma20'] - df['ma20'].shift(1)) / df['ma20'].shift(1)
        df['ma60_slope'] = (df['ma60'] - df['ma60'].shift(1)) / df['ma60'].shift(1)
        macd = ta.macd(df['close'])
        df = pd.concat([df, macd], axis=1)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['roc'] = ta.roc(df['close'], length=10)
        adx = ta.adx(df['max'], df['min'], df['close'], length=14)
        df = pd.concat([df, adx], axis=1)
        st = ta.supertrend(df['max'], df['min'], df['close'], length=7, multiplier=3)
        df = pd.concat([df, st], axis=1)
        df['dist_ma20'] = (df['close'] - df['ma20']) / df['ma20']
        return df

class FlowFeatureExtractor:
    def __init__(self, price_df: pd.DataFrame, inst_df: pd.DataFrame = None, margin_df: pd.DataFrame = None):
        self.price_df = price_df.copy()
        self.inst_df = inst_df.copy() if inst_df is not None else pd.DataFrame()
        self.margin_df = margin_df.copy() if margin_df is not None else pd.DataFrame()

    def extract(self) -> pd.DataFrame:
        df = self.price_df
        df['obv'] = ta.obv(df['close'], df['Trading_Volume'])
        df['cmf'] = ta.cmf(df['open'], df['max'], df['min'], df['close'], df['Trading_Volume'], length=20)
        if not self.inst_df.empty:
            self.inst_df['net_buy'] = self.inst_df['buy'] - self.inst_df['sell']
            pivot = self.inst_df.pivot_table(index='date', columns='name', values='net_buy', aggfunc='sum')
            pivot.rename(columns={'Foreign_Investor': 'foreign_net_buy', 'Investment_Trust': 'trust_net_buy'}, inplace=True)
            for col in ['foreign_net_buy', 'trust_net_buy']:
                if col in pivot.columns:
                    pivot[f'{col}_20d'] = pivot[col].rolling(20).sum()
            df = pd.merge(df, pivot.reset_index(), on='date', how='left')
        return df.fillna(0)

class VolatilityFeatureExtractor:
    def __init__(self, price_df: pd.DataFrame):
        self.df = price_df.copy()

    def extract(self) -> pd.DataFrame:
        df = self.df
        df['atr'] = ta.atr(df['max'], df['min'], df['close'], length=14)
        df['atr_percentile'] = df['atr'].rolling(window=252).rank(pct=True)
        df['parkinson'] = np.sqrt(1 / (4 * np.log(2)) * (np.log(df['max'] / df['min']))**2)
        df['gk'] = np.sqrt(0.5 * (np.log(df['max'] / df['min']))**2 - (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']))**2)
        bb = ta.bbands(df['close'], length=20, std=2)
        df['bb_width'] = bb[[c for c in bb.columns if c.startswith('BBB')]].iloc[:, 0]
        return df
