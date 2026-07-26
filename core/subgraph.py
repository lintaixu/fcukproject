"""
Subgraph Generation & Normalization — 嚴格對齊論文 (Li et al., KBS 2022).

論文 Section 3.2.1–3.2.2 (依循 PATCHY-SAN, Niepert et al. ICML 2016 [22]):
  1. 核心節點選取: 依 chart importance score 排序, 同分依 degree,
     再同分依「鄰居平均 importance score」, 選前 N 個
  2. BFS 展開: 以核心節點為起點, 「整層」擴展直到節點數 >= g
     (論文: "If the size of the current depth subgraph is insufficient,
      we expand one layer to add more nodes")
  3. 正規化排序: 先按 BFS 深度, 同深度按 importance score, 同分按 degree
  4. 節點 > g: 依上述排序取前 g 個 (truncate); 節點 < g: 補 dummy node
"""
from collections import deque
import numpy as np
import networkx as nx


def select_top_nodes(G: nx.Graph, scores: np.ndarray, N: int):
    """
    論文 3.2.1: importance score 降序 → 同分依 degree 降序
    → 再同分依鄰居平均 importance score 降序, 選前 N 個.
    """
    nodes = list(G.nodes())

    def sort_key(node):
        time_idx = G.nodes[node]['time']
        own_score = scores[time_idx]
        deg = G.degree(node)
        nb_scores = [scores[G.nodes[nb]['time']] for nb in G.neighbors(node)]
        nb_mean = np.mean(nb_scores) if nb_scores else 0.0
        return (-own_score, -deg, -nb_mean)

    nodes_sorted = sorted(nodes, key=sort_key)
    return nodes_sorted[:N]


def bfs_subgraph(G: nx.Graph, root: int, g: int):
    """
    BFS 從 root 逐層展開: 每次加入「一整層」節點, 直到總數 >= g
    或無節點可加 (論文 3.2.1 / PATCHY-SAN neighborhood assembly).
    候選數可能 > g, 由 normalize_subgraph 的排序決定最終留下哪 g 個.

    Returns:
        List[(node_id, depth)]
    """
    visited = [(root, 0)]
    visited_set = {root}
    frontier = [root]
    depth = 0

    while len(visited) < g and frontier:
        depth += 1
        next_frontier = []
        for node in frontier:
            for nb in G.neighbors(node):
                if nb not in visited_set:
                    visited_set.add(nb)
                    next_frontier.append(nb)
        visited.extend((nb, depth) for nb in next_frontier)
        frontier = next_frontier
    return visited


def normalize_subgraph(
    G: nx.Graph,
    sub_nodes_with_depth: list,
    scores: np.ndarray,
    g: int,
):
    """
    論文正規化排序 (Section 3.2.2):
      1. BFS 深度 (升序 — root 排最前)
      2. 同深度: chart importance score (降序)
      3. 同分: node degree (降序)
      4. 節點 > g: 依排序取前 g 個 (truncate — PATCHY-SAN 作法,
         對應論文 "filter out the subgraphs whose number of nodes
         is more than G")
      5. 節點 < g: 補 dummy node (-1), 特徵為 0 (論文: "add dummy nodes
         that have no connection with any points")

    Returns:
        ordered_nodes: List[int] (長度 g), -1 = dummy
        adj:           np.ndarray (g, g) — 供分析用; 論文模型無顯式 GCN 層,
                       訓練時不使用此矩陣
    """

    def sort_key(item):
        node, depth = item
        time_idx = G.nodes[node]['time']
        importance = scores[time_idx]
        return (depth, -importance, -G.degree(node))

    ordered_items = sorted(sub_nodes_with_depth, key=sort_key)
    ordered_nodes = [item[0] for item in ordered_items]

    # Padding
    while len(ordered_nodes) < g:
        ordered_nodes.append(-1)
    ordered_nodes = ordered_nodes[:g]

    # 建立 adjacency matrix
    adj = np.zeros((g, g), dtype=np.float32)
    pos = {nid: i for i, nid in enumerate(ordered_nodes) if nid != -1}
    for u in pos:
        for v in pos:
            if u != v and G.has_edge(u, v):
                adj[pos[u], pos[v]] = 1.0

    return ordered_nodes, adj


def attach_features(
    ordered_nodes: list,
    G: nx.Graph,
    feature_matrix: np.ndarray,
    g: int,
    F: int,
):
    """
    把每個節點對應到原 series index, 取出該日的 F 個技術指標.
    Dummy node 補 0.
    """
    feat = np.zeros((g, F), dtype=np.float32)
    for i, node in enumerate(ordered_nodes):
        if node == -1:
            continue
        time_idx = G.nodes[node]['time']
        if time_idx < feature_matrix.shape[0]:
            feat[i] = feature_matrix[time_idx]
    return feat


def build_3d_feature(
    series: np.ndarray,
    pip_indices: list,
    scores: np.ndarray,
    feature_matrix: np.ndarray,
    G: nx.Graph,
    N: int,
    g: int,
):
    """
    完整 pipeline: VG → 選核心節點 → BFS 子圖 → 正規化 → 附加技術指標.
    輸出 3D 特徵 X of shape (N, g, F).
    """
    F = feature_matrix.shape[1]
    top_nodes = select_top_nodes(G, scores, N)

    X = np.zeros((N, g, F), dtype=np.float32)

    for i, root in enumerate(top_nodes):
        sub_nodes_with_depth = bfs_subgraph(G, root, g)
        ordered, _adj = normalize_subgraph(G, sub_nodes_with_depth, scores, g)
        X[i] = attach_features(ordered, G, feature_matrix, g, F)

    return X


if __name__ == "__main__":
    from pip_algorithm import extract_pips
    from vg_graph import build_visibility_graph

    rng = np.random.default_rng(42)
    s = np.sin(np.linspace(0, 4 * np.pi, 120)) * 10 + 100 + rng.normal(0, 0.5, 120)

    pips, scores = extract_pips(s, m=40)
    G = build_visibility_graph(s, pips)

    F_dim = 9
    feat = rng.normal(0, 1, (120, F_dim)).astype(np.float32)

    X = build_3d_feature(s, pips, scores, feat, G, N=15, g=5)
    print(f"X shape: {X.shape}")   # (15, 5, 9)
