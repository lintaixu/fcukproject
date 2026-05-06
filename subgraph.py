"""
Subgraph Generation & Normalization.

從 VG 圖中以 importance score 排序選出 N 個核心節點,
對每個節點用 BFS 展開到 g 個節點的子圖,
再將子圖節點按 (importance, degree, neighbor mean) 排序並 padding 到固定大小。
"""
from collections import deque
import numpy as np
import networkx as nx


def select_top_nodes(G: nx.Graph, scores: np.ndarray, N: int):
    """
    依「圖表重要性分數 → 節點度數 → 鄰居平均分數」排序，
    選出前 N 個節點。

    Args:
        G:      VG 圖, 節點屬性需含 'time' (對應到 series index)
        scores: shape (n_series,) 的 importance score 陣列
        N:      要選取的核心節點數

    Returns:
        List[int], 排序後的前 N 個節點 ID
    """
    nodes = list(G.nodes())

    def sort_key(node):
        time_idx = G.nodes[node]['time']
        own_score = scores[time_idx]
        deg = G.degree(node)
        neighbors = list(G.neighbors(node))
        if neighbors:
            n_score = float(np.mean([scores[G.nodes[nb]['time']] for nb in neighbors]))
        else:
            n_score = 0.0
        # 越大越重要 → 用負號讓 sort ascending 等同 descending
        return (-own_score, -deg, -n_score)

    nodes_sorted = sorted(nodes, key=sort_key)
    return nodes_sorted[:N]


def bfs_subgraph(G: nx.Graph, root: int, g: int):
    """
    從 root 開始 BFS, 直到收集到 g 個節點 (或圖內沒節點可加為止).

    Returns:
        List[int]: BFS 順序展開的節點 list (長度 <= g)
    """
    visited = [root]
    visited_set = {root}
    queue = deque([root])
    while queue and len(visited) < g:
        node = queue.popleft()
        for nb in G.neighbors(node):
            if nb not in visited_set:
                visited.append(nb)
                visited_set.add(nb)
                queue.append(nb)
                if len(visited) >= g:
                    break
    return visited


def normalize_subgraph(
    G: nx.Graph,
    sub_nodes,
    scores: np.ndarray,
    g: int,
):
    """
    將子圖節點按 (importance, degree, neighbor score) 排序,
    若節點數 < g 則補 dummy node (-1).

    Returns:
        ordered_nodes: List[int] (長度 g), -1 表 dummy
        adj:           np.ndarray of shape (g, g), padding 後的 adjacency matrix
    """
    sub_set = set(sub_nodes)

    def sort_key(node):
        time_idx = G.nodes[node]['time']
        own_score = scores[time_idx]
        deg_in_sub = sum(1 for nb in G.neighbors(node) if nb in sub_set)
        sub_neighbors = [nb for nb in G.neighbors(node) if nb in sub_set]
        if sub_neighbors:
            n_score = float(np.mean([scores[G.nodes[nb]['time']] for nb in sub_neighbors]))
        else:
            n_score = 0.0
        return (-own_score, -deg_in_sub, -n_score)

    ordered = sorted(sub_nodes, key=sort_key)
    while len(ordered) < g:
        ordered.append(-1)  # dummy
    ordered = ordered[:g]

    # 建立 adjacency
    adj = np.zeros((g, g), dtype=np.float32)
    pos = {nid: i for i, nid in enumerate(ordered) if nid != -1}
    for u in pos:
        for v in pos:
            if u != v and G.has_edge(u, v):
                adj[pos[u], pos[v]] = 1.0

    return ordered, adj


def attach_features(
    ordered_nodes,
    G: nx.Graph,
    feature_matrix: np.ndarray,
    g: int,
    F: int,
):
    """
    把每個節點對應到原 series index, 取出該日的 F 個技術指標。
    Dummy node 補 0。

    Args:
        ordered_nodes: List[int] (長度 g)
        G:             VG 圖
        feature_matrix: shape (T, F) — 原序列每日的技術指標
        g, F:          固定維度

    Returns:
        np.ndarray of shape (g, F)
    """
    feat = np.zeros((g, F), dtype=np.float32)
    for i, node in enumerate(ordered_nodes):
        if node == -1:
            continue
        time_idx = G.nodes[node]['time']
        feat[i] = feature_matrix[time_idx]
    return feat


def build_3d_feature(
    series: np.ndarray,
    pip_indices,
    scores: np.ndarray,
    feature_matrix: np.ndarray,
    G: nx.Graph,
    N: int,
    g: int,
):
    """
    完整的「VG → 子圖 → 正規化 → 附加技術指標」流程,
    輸出 3D 特徵 X of shape (N, g, F).
    """
    F = feature_matrix.shape[1]
    top_nodes = select_top_nodes(G, scores, N)

    X = np.zeros((N, g, F), dtype=np.float32)
    A = np.zeros((N, g, g), dtype=np.float32)

    for i, root in enumerate(top_nodes):
        sub_nodes = bfs_subgraph(G, root, g)
        ordered, adj = normalize_subgraph(G, sub_nodes, scores, g)
        X[i] = attach_features(ordered, G, feature_matrix, g, F)
        A[i] = adj

    # 若 top_nodes < N 則 X[i:] 維持 0
    return X, A


if __name__ == "__main__":
    from pip_algorithm import extract_pips
    from vg_graph import build_visibility_graph

    rng = np.random.default_rng(42)
    s = np.sin(np.linspace(0, 4 * np.pi, 120)) * 10 + 100 + rng.normal(0, 0.5, 120)

    pips, scores = extract_pips(s, m=40)
    G = build_visibility_graph(s, pips)

    # 假的特徵矩陣 (T x F)
    F = 9
    feat = rng.normal(0, 1, (120, F)).astype(np.float32)

    X, A = build_3d_feature(s, pips, scores, feat, G, N=15, g=5)
    print(f"X shape: {X.shape}")  # (15, 5, 9)
    print(f"A shape: {A.shape}")  # (15, 5, 5)
    print(f"Mean adj density per subgraph: {A.mean(axis=(1, 2)).mean():.3f}")
