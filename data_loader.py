"""
台股資料下載器 (使用 yfinance, 需 .TW suffix).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd


# 台灣 50 ETF 成分股 (完整清單)
TW50_TICKERS = [
    "2330.TW",  # 台積電
    "2317.TW",  # 鴻海
    "2454.TW",  # 聯發科
    "2412.TW",  # 中華電
    "2308.TW",  # 台達電
    "2382.TW",  # 廣達
    "2881.TW",  # 富邦金
    "2882.TW",  # 國泰金
    "2891.TW",  # 中信金
    "2303.TW",  # 聯電
    "1301.TW",  # 台塑
    "1303.TW",  # 南亞
    "2002.TW",  # 中鋼
    "3711.TW",  # 日月光投控
    "2207.TW",  # 和泰車
    "2886.TW",  # 兆豐金
    "2884.TW",  # 玉山金
    "2892.TW",  # 第一金
    "2880.TW",  # 華南金
    "2885.TW",  # 元大金
    "2887.TW",  # 台新金
    "5880.TW",  # 合庫金
    "2890.TW",  # 永豐金
    "1326.TW",  # 台化
    "1101.TW",  # 台泥
    "1102.TW",  # 亞泥
    "2912.TW",  # 統一超
    "2301.TW",  # 光寶科
    "2357.TW",  # 華碩
    "2395.TW",  # 研華
    "3008.TW",  # 大立光
    "2327.TW",  # 國巨
    "3045.TW",  # 台灣大
    "4904.TW",  # 遠傳
    "2379.TW",  # 瑞昱
    "3034.TW",  # 聯詠
    "2345.TW",  # 智邦
    "2603.TW",  # 長榮
    "2609.TW",  # 陽明
    "2615.TW",  # 萬海
    "5871.TW",  # 中租-KY
    "2105.TW",  # 正新
    "1216.TW",  # 統一
    "2633.TW",  # 台灣高鐵
    "9910.TW",  # 豐泰
    "6505.TW",  # 台塑化
    "2474.TW",  # 可成
    "4938.TW",  # 和碩
    "2347.TW",  # 聯強
    "8046.TW",  # 南電
]

# 台灣前 100 大市值股票 (台灣50 + 中型100 精選)
TW100_TICKERS = TW50_TICKERS + [
    "2888.TW",  # 新光金
    "2883.TW",  # 開發金
    "3037.TW",  # 欣興
    "2356.TW",  # 英業達
    "3231.TW",  # 緯創
    "6669.TW",  # 緯穎
    "3443.TW",  # 創意
    "2408.TW",  # 南亞科
    "3661.TW",  # 世芯-KY
    "5347.TW",  # 世界
    "6415.TW",  # 矽力-KY
    "2377.TW",  # 微星
    "6488.TW",  # 環球晶
    "3653.TW",  # 健策
    "2352.TW",  # 佳世達
    "4961.TW",  # 天鈺
    "2360.TW",  # 致茂
    "3702.TW",  # 大聯大
    "6446.TW",  # 藥華藥
    "1402.TW",  # 遠東新
    "1590.TW",  # 亞德客-KY
    "2201.TW",  # 裕隆
    "9945.TW",  # 潤泰新
    "2324.TW",  # 仁寶
    "2376.TW",  # 技嘉
    "3706.TW",  # 神達
    "2344.TW",  # 華邦電
    "2823.TW",  # 中壽
    "6770.TW",  # 力積電
    "3533.TW",  # 嘉澤
    "2049.TW",  # 上銀
    "1477.TW",  # 聚陽
    "2368.TW",  # 金像電
    "6239.TW",  # 力成
    "5876.TW",  # 上海商銀
    "2383.TW",  # 台光電
    "3044.TW",  # 健鼎
    "2542.TW",  # 興富發
    "9904.TW",  # 寶成
    "1504.TW",  # 東元
    "6547.TW",  # 高端疫苗
    "2903.TW",  # 遠百
    "9914.TW",  # 美利達
    "2801.TW",  # 彰銀
    "6409.TW",  # 旭隼
    "3023.TW",  # 信邦
    "8454.TW",  # 富邦媒
    "2101.TW",  # 南港
    "1802.TW",  # 台玻
    "2227.TW",  # 裕日車
]


def fetch_yfinance(tickers, start: str, end: str, cache_dir: str = "./cache"):
    """
    用 yfinance 下載資料, 並 cache 在本地以避免重複下載。
    """
    import yfinance as yf
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    out = {}
    for tk in tickers:
        cache_file = os.path.join(cache_dir, f"{tk}_{start}_{end}.parquet")
        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
        else:
            print(f"下載 {tk} ...")
            df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                continue
            # 攤平 multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.columns = [c.lower() for c in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
            df.to_parquet(cache_file)
        out[tk] = df
    return out


def fetch_tw_stocks(tickers=None, start="2018-01-01", end="2024-12-31"):
    """
    主接口: 用 yfinance 下載台股資料。
    """
    if tickers is None:
        tickers = TW50_TICKERS

    data = fetch_yfinance(tickers, start, end)
    if not data:
        raise RuntimeError("yfinance 回傳空，請確認網路連線與股票代碼是否正確")
    return data


if __name__ == "__main__":
    data = fetch_tw_stocks(
        tickers=TW50_TICKERS[:3],
        start="2020-01-01",
        end="2024-12-31",
    )
    for tk, df in data.items():
        print(f"{tk}: {len(df)} 天, 範圍 {df.index[0].date()} ~ {df.index[-1].date()}")
        print(df.head(3))
        print()
