"""
K線型態識別引擎
識別10+種經典K線型態及組合型態
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CandlePattern:
    """K線型態識別基類"""

    def __init__(self, df: pd.DataFrame, window: int = 3):
        """
        Args:
            df: OHLCV DataFrame (必須包含 Open, High, Low, Close, Volume)
            window: 識別窗口大小
        """
        self.df = df.copy()
        self.window = window
        self._normalize_data()

    def _normalize_data(self):
        """數據歸一化"""
        self.df["Body"] = abs(self.df["Close"] - self.df["Open"])
        self.df["HighShadow"] = self.df["High"] - self.df[["Open", "Close"]].max(axis=1)
        self.df["LowShadow"] = self.df[["Open", "Close"]].min(axis=1) - self.df["Low"]
        self.df["Range"] = self.df["High"] - self.df["Low"]
        self.df["IsGreen"] = self.df["Close"] >= self.df["Open"]  # True=漲, False=跌

    def _is_hammer(self, idx: int) -> bool:
        """錘子線：小實體，下影線長"""
        if idx < 1 or idx >= len(self.df):
            return False

        prev = self.df.iloc[idx - 1]
        curr = self.df.iloc[idx]

        body_ratio = curr["Body"] / curr["Range"]
        shadow_ratio = curr["LowShadow"] / curr["Range"]

        # 錘子線：實體小，下影線為實體3倍以上
        return (
            body_ratio < 0.3
            and shadow_ratio > 0.6
            and curr["HighShadow"] < 0.1 * curr["Range"]
            and prev["Close"] > curr["Close"]
        )  # 前一根下跌

    def _is_hanging_man(self, idx: int) -> bool:
        """上吊線：與錘子線形狀相同，但出現在上升趨勢"""
        if idx < 1 or idx >= len(self.df):
            return False

        prev = self.df.iloc[idx - 1]
        curr = self.df.iloc[idx]

        body_ratio = curr["Body"] / curr["Range"]
        shadow_ratio = curr["LowShadow"] / curr["Range"]

        return (
            body_ratio < 0.3
            and shadow_ratio > 0.6
            and curr["HighShadow"] < 0.1 * curr["Range"]
            and prev["Close"] < curr["Close"]
        )  # 前一根上漲

    def _is_inverted_hammer(self, idx: int) -> bool:
        """倒錘線：小實體，上影線長，出現在下跌趨勢"""
        if idx < 1 or idx >= len(self.df):
            return False

        prev = self.df.iloc[idx - 1]
        curr = self.df.iloc[idx]

        if curr["Range"] == 0:
            return False
        body_ratio = curr["Body"] / curr["Range"]
        shadow_ratio = curr["HighShadow"] / curr["Range"]

        return (
            body_ratio < 0.3
            and shadow_ratio > 0.6
            and curr["LowShadow"] < 0.1 * curr["Range"]
            and prev["Close"] > curr["Close"]
        )  # 前一根下跌

    def _is_bullish_harami(self, idx: int) -> bool:
        """孕線（正/看漲）：前大陰線，後小實體完全被包含"""
        if idx < 1 or idx >= len(self.df):
            return False

        prev = self.df.iloc[idx - 1]
        curr = self.df.iloc[idx]

        # 前跌，當前可漲可跌但通常是漲
        if prev["IsGreen"]:
            return False

        # 當前實體被前一根實體完全包含
        return (
            curr["Open"] > prev["Close"]
            and curr["Close"] < prev["Open"]
            and curr["High"] < prev["Open"]
            and curr["Low"] > prev["Close"]
        )

    def _is_bearish_harami(self, idx: int) -> bool:
        """孕線（反/看跌）：前大陽線，後小實體完全被包含"""
        if idx < 1 or idx >= len(self.df):
            return False

        prev = self.df.iloc[idx - 1]
        curr = self.df.iloc[idx]

        # 前漲
        if not prev["IsGreen"]:
            return False

        # 當前實體被前一根實體完全包含
        return (
            curr["Open"] < prev["Close"]
            and curr["Close"] > prev["Open"]
            and curr["High"] < prev["Close"]
            and curr["Low"] > prev["Open"]
        )

    def _is_bullish_engulfing(self, idx: int) -> bool:
        """吞沒線（上升）：第二根K線吞沒第一根下跌K線"""
        if idx < 1 or idx >= len(self.df):
            return False

        prev = self.df.iloc[idx - 1]
        curr = self.df.iloc[idx]

        # 前根下跌，當根上漲
        if prev["IsGreen"] or not curr["IsGreen"]:
            return False

        # 當根O < 前根C 且 當根C > 前根O
        return curr["Open"] < prev["Close"] and curr["Close"] > prev["Open"]

    def _is_bearish_engulfing(self, idx: int) -> bool:
        """吞沒線（下降）：第二根K線吞沒第一根上漲K線"""
        if idx < 1 or idx >= len(self.df):
            return False

        prev = self.df.iloc[idx - 1]
        curr = self.df.iloc[idx]

        # 前根上漲，當根下跌
        if not prev["IsGreen"] or curr["IsGreen"]:
            return False

        # 當根O > 前根C 且 當根C < 前根O
        return curr["Open"] > prev["Close"] and curr["Close"] < prev["Open"]

    def _is_morning_star(self, idx: int) -> bool:
        """晨星型態（三根K線）"""
        if idx < 2 or idx >= len(self.df):
            return False

        k1 = self.df.iloc[idx - 2]
        k2 = self.df.iloc[idx - 1]
        k3 = self.df.iloc[idx]

        # 1.下跌 2.縮小(gap或小實體) 3.上漲
        return (
            not k1["IsGreen"]
            and k2["Body"] < 0.3 * k1["Range"]
            and k3["IsGreen"]
            and k3["Close"] > k1["Close"]
        )

    def _is_evening_star(self, idx: int) -> bool:
        """晚星型態（三根K線）"""
        if idx < 2 or idx >= len(self.df):
            return False

        k1 = self.df.iloc[idx - 2]
        k2 = self.df.iloc[idx - 1]
        k3 = self.df.iloc[idx]

        # 1.上漲 2.縮小(gap或小實體) 3.下跌
        return (
            k1["IsGreen"]
            and k2["Body"] < 0.3 * k1["Range"]
            and not k3["IsGreen"]
            and k3["Close"] < k1["Close"]
        )

    def _is_piercing_line(self, idx: int) -> bool:
        """穿刺線：下跌後上升，收盤價在前根實體中點以上"""
        if idx < 1 or idx >= len(self.df):
            return False

        prev = self.df.iloc[idx - 1]
        curr = self.df.iloc[idx]

        if prev["IsGreen"] or not curr["IsGreen"]:
            return False

        midpoint = prev["Open"] - prev["Body"] / 2
        return curr["Close"] > midpoint and curr["Close"] < prev["Open"]

    def _is_dark_cloud_cover(self, idx: int) -> bool:
        """烏雲蓋頂：上漲後下降，收盤價在前根實體中點以下"""
        if idx < 1 or idx >= len(self.df):
            return False

        prev = self.df.iloc[idx - 1]
        curr = self.df.iloc[idx]

        if not prev["IsGreen"] or curr["IsGreen"]:
            return False

        midpoint = prev["Close"] - prev["Body"] / 2
        return curr["Close"] < midpoint and curr["Close"] > prev["Open"]

    def _is_doji(self, idx: int) -> bool:
        """十字線：開盤等於收盤或極小實體"""
        if idx < 0 or idx >= len(self.df):
            return False

        curr = self.df.iloc[idx]
        body_ratio = curr["Body"] / curr["Range"]

        return body_ratio < 0.1 and curr["Range"] > 0

    def _is_three_black_crows(self, idx: int) -> bool:
        """三根烏鴉（三個連續下跌K線）"""
        if idx < 2 or idx >= len(self.df):
            return False

        k1 = self.df.iloc[idx - 2]
        k2 = self.df.iloc[idx - 1]
        k3 = self.df.iloc[idx]

        return (
            not k1["IsGreen"]
            and not k2["IsGreen"]
            and not k3["IsGreen"]
            and k1["Close"] > k2["Close"] > k3["Close"]
        )

    def _is_three_white_soldiers(self, idx: int) -> bool:
        """三個白兵（三個連續上漲K線）"""
        if idx < 2 or idx >= len(self.df):
            return False

        k1 = self.df.iloc[idx - 2]
        k2 = self.df.iloc[idx - 1]
        k3 = self.df.iloc[idx]

        return (
            k1["IsGreen"]
            and k2["IsGreen"]
            and k3["IsGreen"]
            and k1["Close"] < k2["Close"] < k3["Close"]
        )

    def detect_all_patterns(self) -> Dict[str, List[int]]:
        """
        檢測所有型態

        Returns:
            {pattern_name: [indices]}
        """
        patterns = {
            "hammer": [],
            "hanging_man": [],
            "inverted_hammer": [],
            "bullish_harami": [],
            "bearish_harami": [],
            "bullish_engulfing": [],
            "bearish_engulfing": [],
            "morning_star": [],
            "evening_star": [],
            "piercing_line": [],
            "dark_cloud_cover": [],
            "doji": [],
            "three_black_crows": [],
            "three_white_soldiers": [],
        }

        pattern_methods = {
            "hammer": self._is_hammer,
            "hanging_man": self._is_hanging_man,
            "inverted_hammer": self._is_inverted_hammer,
            "bullish_harami": self._is_bullish_harami,
            "bearish_harami": self._is_bearish_harami,
            "bullish_engulfing": self._is_bullish_engulfing,
            "bearish_engulfing": self._is_bearish_engulfing,
            "morning_star": self._is_morning_star,
            "evening_star": self._is_evening_star,
            "piercing_line": self._is_piercing_line,
            "dark_cloud_cover": self._is_dark_cloud_cover,
            "doji": self._is_doji,
            "three_black_crows": self._is_three_black_crows,
            "three_white_soldiers": self._is_three_white_soldiers,
        }

        for idx in range(len(self.df)):
            for pattern_name, method in pattern_methods.items():
                if method(idx):
                    patterns[pattern_name].append(idx)

        return patterns

    def get_pattern_details(self, pattern_name: str, idx: int) -> Dict:
        """獲取型態詳細信息"""
        row = self.df.iloc[idx]

        return {
            "date": self.df.index[idx],
            "pattern": pattern_name,
            "open": row["Open"],
            "high": row["High"],
            "low": row["Low"],
            "close": row["Close"],
            "volume": row["Volume"],
            "body_size": row["Body"],
            "range": row["Range"],
        }


if __name__ == "__main__":
    from data_fetcher import DataFetcher

    fetcher = DataFetcher()
    df = fetcher.fetch_data("2330.TW", "2023-01-01", "2024-01-01")

    recognizer = CandlePattern(df)
    patterns = recognizer.detect_all_patterns()

    print("型態檢測結果:")
    for pattern, indices in patterns.items():
        if indices:
            print(f"  {pattern}: {len(indices)} 個")
