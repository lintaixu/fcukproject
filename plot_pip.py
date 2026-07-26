"""
視覺化 PIP pipeline：
  上排: 原始收盤價 / PIP 關鍵點 / Visibility Graph
  下方: N 個子圖各自獨立一格

用法:
    python plot_pip.py
    python plot_pip.py --ticker 2330.TW --m-pips 40 --window 100 --offset 300
"""
import argparse
import math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import networkx as nx

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "core"))
from data_loader import fetch_tw_stocks
from pip_algorithm import extract_pips
from vg_graph import build_visibility_graph
from subgraph import select_top_nodes, bfs_subgraph

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#d35400", "#c0392b",
    "#16a085", "#8e44ad", "#2980b9", "#27ae60", "#f1c40f",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="2330.TW")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--m-pips", type=int, default=40)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--N", type=int, default=15)
    p.add_argument("--g", type=int, default=5)
    p.add_argument("--offset", type=int, default=0)
    args = p.parse_args()

    print(f"下載 {args.ticker} ...")
    data = fetch_tw_stocks(tickers=[args.ticker], start=args.start, end=args.end)
    df = data[args.ticker]
    close = df["close"].values
    print(f"共 {len(close)} 個交易日")

    start = min(args.offset, len(close) - args.window - 1)
    series = close[start : start + args.window]
    x = np.arange(start, start + args.window)

    pips, scores = extract_pips(series, m=args.m_pips)
    G = build_visibility_graph(series, pips)
    pip_x = x[pips]
    pip_y = series[pips]

    top_nodes = select_top_nodes(G, scores, args.N)
    n_subs = len(top_nodes)

    # ── 版面配置 ──
    # 上排 3 格 (原始 / PIP / VG)，下方 n_subs 格子圖
    sub_cols = 5
    sub_rows = math.ceil(n_subs / sub_cols)
    total_rows = 1 + sub_rows  # 上排 1 列 + 子圖列

    fig = plt.figure(figsize=(20, 4 + sub_rows * 3.5))
    gs = fig.add_gridspec(total_rows, sub_cols, hspace=0.45, wspace=0.3)

    # ── 上排: 原始收盤價 ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, series, color="#2c3e50", linewidth=1)
    ax1.set_title("(1) 原始收盤價", fontsize=10)
    ax1.set_ylabel("Price", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.grid(True, alpha=0.3)

    # ── 上排: PIP 關鍵點 ──
    ax2 = fig.add_subplot(gs[0, 1:3])
    ax2.plot(x, series, color="gray", linewidth=0.6, alpha=0.5)
    ax2.plot(pip_x, pip_y, "r-o", markersize=4, linewidth=1, label=f"PIP m={args.m_pips}")
    ax2.set_title(f"(2) PIP 關鍵點  m={args.m_pips}", fontsize=10)
    ax2.legend(fontsize=7, loc="upper left")
    ax2.tick_params(labelsize=7)
    ax2.grid(True, alpha=0.3)

    # ── 上排: Visibility Graph ──
    ax3 = fig.add_subplot(gs[0, 3:5])
    ax3.plot(x, series, color="gray", linewidth=0.4, alpha=0.3)
    ax3.scatter(pip_x, pip_y, c="black", s=15, zorder=5)
    for u, v in G.edges():
        tu, tv = G.nodes[u]["time"], G.nodes[v]["time"]
        ax3.plot([x[tu], x[tv]], [series[tu], series[tv]],
                 color="#3498db", linewidth=0.6, alpha=0.6)
    ax3.set_title(f"(3) VG  nodes={G.number_of_nodes()}, edges={G.number_of_edges()}", fontsize=10)
    ax3.tick_params(labelsize=7)
    ax3.grid(True, alpha=0.3)

    # ── 下方: 每個子圖獨立一格 ──
    for i, root in enumerate(top_nodes):
        row = 1 + i // sub_cols
        col = i % sub_cols
        ax = fig.add_subplot(gs[row, col])

        # bfs_subgraph 回傳 (node_id, depth) tuple 清單, 取 node_id
        sub_nodes = [n for n, _d in bfs_subgraph(G, root, args.g)]
        color = COLORS[i % len(COLORS)]

        # 背景: 完整收盤價
        ax.plot(x, series, color="gray", linewidth=0.4, alpha=0.25)

        # 子圖邊
        sub_set = set(sub_nodes)
        for u in sub_nodes:
            for v in G.neighbors(u):
                if v in sub_set:
                    tu, tv = G.nodes[u]["time"], G.nodes[v]["time"]
                    ax.plot([x[tu], x[tv]], [series[tu], series[tv]],
                            color=color, linewidth=1.5, alpha=0.7)

        # 子圖節點
        sub_x = [x[G.nodes[n]["time"]] for n in sub_nodes]
        sub_y = [series[G.nodes[n]["time"]] for n in sub_nodes]
        ax.scatter(sub_x, sub_y, c=color, s=40, zorder=5,
                   edgecolors="white", linewidths=0.5)

        # 核心節點星號
        root_t = G.nodes[root]["time"]
        ax.scatter([x[root_t]], [series[root_t]], c=color, s=120,
                   marker="*", zorder=6, edgecolors="black", linewidths=0.5)

        node_ids = [G.nodes[n]["time"] for n in sub_nodes]
        ax.set_title(f"Sub {i+1}  root={root_t}  nodes={node_ids}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    # 空白格清除
    for i in range(n_subs, sub_rows * sub_cols):
        row = 1 + i // sub_cols
        col = i % sub_cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis("off")

    fig.suptitle(
        f"{args.ticker}  window=[{start}, {start+args.window})  "
        f"m_pips={args.m_pips}  N={args.N}  g={args.g}",
        fontsize=13, fontweight="bold",
    )
    plt.show()


if __name__ == "__main__":
    main()
