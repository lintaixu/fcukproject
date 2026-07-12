"""
K線型態分析系統 — 程式進入點。

用法：
    python main.py
"""

import warnings
warnings.filterwarnings('ignore')

import tkinter as tk

from kline.gui import KLineApp


def main():
    root = tk.Tk()
    app  = KLineApp(root)          # noqa: F841  (保持參考避免被回收)
    root.mainloop()


if __name__ == '__main__':
    main()
