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

        if curr["Range"] == 0: return False
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

        if curr["Range"] == 0: return False
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
        if curr["Range"] == 0: return False
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
        if curr["Range"] == 0: return False
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

    def _is_triple_top(self, idx: int, lookback: int = 20) -> bool:
        """三重頂: 過去lookback天內有三個相近的高點"""
        if idx < lookback:
            return False

        data = self.df["High"].iloc[idx - lookback : idx + 1].values
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                peaks.append(data[i])

        if len(peaks) >= 3:
            last_3_peaks = peaks[-3:]
            avg_peak = sum(last_3_peaks) / 3
            if all(abs(p - avg_peak) / avg_peak < 0.015 for p in last_3_peaks):
                if self.df["Close"].iloc[idx] < self.df["Close"].iloc[idx - 1]:
                    return True
        return False

    def _is_triple_bottom(self, idx: int, lookback: int = 20) -> bool:
        """三重底: 過去lookback天內有三個相近的低點"""
        if idx < lookback:
            return False

        data = self.df["Low"].iloc[idx - lookback : idx + 1].values
        valleys = []
        for i in range(1, len(data) - 1):
            if data[i] < data[i - 1] and data[i] < data[i + 1]:
                valleys.append(data[i])

        if len(valleys) >= 3:
            last_3_valleys = valleys[-3:]
            avg_valley = sum(last_3_valleys) / 3
            if all(abs(v - avg_valley) / avg_valley < 0.015 for v in last_3_valleys):
                if self.df["Close"].iloc[idx] > self.df["Close"].iloc[idx - 1]:
                    return True
        return False

    def _is_bullish_flag(
        self, idx: int, flagpole_len: int = 5, flag_len: int = 5
    ) -> bool:
        """牛旗形: 急漲後的小幅回調"""
        lookback = flagpole_len + flag_len
        if idx < lookback:
            return False

        pole_start = idx - flag_len - flagpole_len
        pole_end = idx - flag_len

        pole_start_price = self.df["Close"].iloc[pole_start]
        pole_end_price = self.df["Close"].iloc[pole_end]

        if pole_end_price <= pole_start_price * 1.03:
            return False

        flag_data = self.df["Close"].iloc[pole_end : idx + 1]

        if flag_data.iloc[-1] > pole_end_price:
            return False

        retracement = (pole_end_price - flag_data.iloc[-1]) / (
            pole_end_price - pole_start_price
        )
        if retracement > 0.5 or retracement < 0.1:
            return False

        if self.df["IsGreen"].iloc[idx]:
            return True
        return False

    def _is_bearish_flag(
        self, idx: int, flagpole_len: int = 5, flag_len: int = 5
    ) -> bool:
        """熊旗形: 急跌後的小幅反彈"""
        lookback = flagpole_len + flag_len
        if idx < lookback:
            return False

        pole_start = idx - flag_len - flagpole_len
        pole_end = idx - flag_len

        pole_start_price = self.df["Close"].iloc[pole_start]
        pole_end_price = self.df["Close"].iloc[pole_end]

        if pole_end_price >= pole_start_price * 0.97:
            return False

        flag_data = self.df["Close"].iloc[pole_end : idx + 1]

        if flag_data.iloc[-1] < pole_end_price:
            return False

        retracement = (flag_data.iloc[-1] - pole_end_price) / (
            pole_start_price - pole_end_price
        )
        if retracement > 0.5 or retracement < 0.1:
            return False

        if not self.df["IsGreen"].iloc[idx]:
            return True
        return False

    def _is_bullish_wedge(self, idx: int, wedge_len: int = 10) -> bool:
        """下降楔形 (看漲): 高點和低點都在降低"""
        if idx < wedge_len + 1:
            return False

        data_high = self.df["High"].iloc[idx - wedge_len : idx + 1].values
        data_low = self.df["Low"].iloc[idx - wedge_len : idx + 1].values

        if data_high[-1] > data_high[0] or data_low[-1] > data_low[0]:
            return False

        range_start = data_high[0] - data_low[0]
        range_end = data_high[-1] - data_low[-1]

        if range_start == 0 or range_end >= range_start * 0.8:
            return False

        if (
            self.df["IsGreen"].iloc[idx]
            and self.df["Close"].iloc[idx] > self.df["Open"].iloc[idx - 1]
        ):
            return True
        return False

    def _is_bearish_wedge(self, idx: int, wedge_len: int = 10) -> bool:
        """上升楔形 (看跌): 高點和低點都在升高"""
        if idx < wedge_len + 1:
            return False

        data_high = self.df["High"].iloc[idx - wedge_len : idx + 1].values
        data_low = self.df["Low"].iloc[idx - wedge_len : idx + 1].values

        if data_high[-1] < data_high[0] or data_low[-1] < data_low[0]:
            return False

        range_start = data_high[0] - data_low[0]
        range_end = data_high[-1] - data_low[-1]

        if range_start == 0 or range_end >= range_start * 0.8:
            return False

        if (
            not self.df["IsGreen"].iloc[idx]
            and self.df["Close"].iloc[idx] < self.df["Open"].iloc[idx - 1]
        ):
            return True
        return False

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
            "triple_top": [],
            "triple_bottom": [],
            "bullish_flag": [],
            "bearish_flag": [],
            "bullish_wedge": [],
            "bearish_wedge": [],
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
            "triple_top": self._is_triple_top,
            "triple_bottom": self._is_triple_bottom,
            "bullish_flag": self._is_bullish_flag,
            "bearish_flag": self._is_bearish_flag,
            "bullish_wedge": self._is_bullish_wedge,
            "bearish_wedge": self._is_bearish_wedge,
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
