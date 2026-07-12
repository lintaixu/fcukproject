"""
K線型態分析系統（相容層）

原本的單檔實作已重構為 `kline/` 套件，本檔僅為向後相容而保留：
    - 重新匯出原有的公開名稱，讓既有的 `from kline_pattern_search import ...` 不會壞掉。
    - 直接執行時照舊開啟 GUI。

新程式請改用套件與新入口：
    python main.py
"""

import warnings
warnings.filterwarnings('ignore')

# ── 重新匯出公開 API（維持舊有匯入路徑）──────────────────────────────────────
from kline.config import (                       # noqa: F401
    HOLD_DAYS, RISE_THRESH, FALL_THRESH, DIST_THRESH, MIN_FREQ,
    MIN_BULL_RET, MIN_BEAR_RET, LATENT_DIM, EPOCHS, OCSVM_NU,
    BROKERAGE, TAX, TRADE_COST,
    TRAIN_START, TRAIN_END, TEST_START, TEST_END, TW50,
)
from kline.features import (                      # noqa: F401
    detect_exdiv_mask, features_1day, features_2day, features_3day, FEATURE_SETS,
)
from kline.data_loader import (                   # noqa: F401
    download_one, _download_one, download_raw, build_dataset,
)
from kline.autoencoder import Autoencoder, train_ae          # noqa: F401
from kline.pattern_model import find_and_train               # noqa: F401
from kline.backtest import backtest                          # noqa: F401
from kline.validate import validate_features                 # noqa: F401
from kline.gui import KLineApp                                # noqa: F401


if __name__ == '__main__':
    from main import main
    main()
