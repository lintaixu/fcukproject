"""
Visibility Graph (VG) construction.

把 PIP 關鍵點序列轉成圖：
若兩點之間的連線「沒有被任何中間點的價格擋住」即建立一條邊。
"""
import numpy as np
import networkx as nx


def build_visibility_graph(series: np.ndarray, pip_indices):
    """
    Args:
        series:      1D numpy array, 完整收盤價序列
        pip_indices: List[int], PIP 點的 index (時間軸)

    Returns:
        G: networkx.Graph, 節點為 0..len(pip_indices)-1,
           節點屬性 'time' 為原序列 index, 'price' 為價格
    """
    pts = [(idx, float(series[idx])) for idx in pip_indices]
    n = len(pts)

    G = nx.Graph()
    for i, (t, c) in enumerate(pts):
        G.add_node(i, time=t, price=c)

    for a in range(n):
        for b in range(a + 1, n):
            ta, ca = pts[a]
            tb, cb = pts[b]
            if a + 1 == b:
                # 相鄰點一定可見
                G.add_edge(a, b)
                continue

            visible = True
            for c_idx in range(a + 1, b):
                tc, cc = pts[c_idx]
                # Eq.(3): 若中間點 cc 高於 a→b 連線在該時刻的高度即遮擋
                threshold = ca + (cb - ca) * (tc - ta) / (tb - ta)
                if cc >= threshold:
                    visible = False
                    break

            if visible:
                G.add_edge(a, b)

    return G


if __name__ == "__main__":
    from pip_algorithm import extract_pips

    rng = np.random.default_rng(42)
    t = np.linspace(0, 4 * np.pi, 100)
    s = np.sin(t) * 10 + 100 + rng.normal(0, 0.5, 100)

    pips, scores = extract_pips(s, m=15)
    G = build_visibility_graph(s, pips)
    print(f"PIPs={len(pips)}, Nodes={G.number_of_nodes()}, "
          f"Edges={G.number_of_edges()}")
    print(f"平均度數: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
