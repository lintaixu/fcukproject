"""
Perceptually Important Points (PIP) algorithm — 論文 Algorithm 1 (Li et al., KBS 2022).

從股價序列中萃取最具代表性的 m 個關鍵點，並計算每個點的
chart importance score（加入時與兩側相鄰 PIP 構成的最大 Euclidean 距離,
論文 Eq.(1)(2)）; 起點/終點分數 = max(IS)+1 (Algorithm 1 line 23)。

實作註記: 與逐點重算的樸素寫法完全等價 (每個候選點的距離只取決於
其左右相鄰 PIP), 但只重算被新 PIP 分割的線段並以 numpy 向量化,
速度快約 20~50 倍。tie-break 與樸素寫法一致 (取最左邊的最大值)。
"""
import bisect
import numpy as np


def extract_pips(series: np.ndarray, m: int):
    """
    Args:
        series: 1D numpy array, 收盤價序列, 長度 n
        m:      想保留的關鍵點數, m < n

    Returns:
        indices: List[int], 排序後的 m 個 PIP index
        scores:  np.ndarray of shape (n,), 每個點的 importance score
                 (非 PIP 的點為 0; 起點與終點為 max(IS)+1)
    """
    n = len(series)
    if m >= n:
        scores = np.full(n, np.inf)
        return list(range(n)), scores

    s = np.asarray(series, dtype=np.float64)
    scores = np.zeros(n, dtype=np.float64)

    # d[k] = 候選點 k 到其左右相鄰 PIP 的 Euclidean 距離和 (Eq.1);
    # 已選為 PIP 或無效的位置為 -inf
    d = np.full(n, -np.inf)

    def fill_segment(L: int, R: int):
        """重算線段 (L, R) 內所有候選點的距離."""
        if R - L < 2:
            return
        k = np.arange(L + 1, R)
        d_left = np.sqrt((k - L) ** 2 + (s[k] - s[L]) ** 2)
        d_right = np.sqrt((R - k) ** 2 + (s[R] - s[k]) ** 2)
        d[L + 1:R] = d_left + d_right

    indices = [0, n - 1]          # 保持排序
    fill_segment(0, n - 1)

    while len(indices) < m:
        best_k = int(np.argmax(d))          # tie → 最左 (同樸素掃描)
        if not np.isfinite(d[best_k]):
            break
        scores[best_k] = d[best_k]

        pos = bisect.bisect_left(indices, best_k)
        L, R = indices[pos - 1], indices[pos]
        bisect.insort(indices, best_k)

        d[best_k] = -np.inf
        fill_segment(L, best_k)             # 只重算被分割的兩段
        fill_segment(best_k, R)

    # 起終點分數 = 比所有實際分數都高 (Algorithm 1 line 23)
    positive = scores[scores > 0]
    finite_max = positive.max() if len(positive) else 1.0
    scores[0] = finite_max + 1
    scores[-1] = finite_max + 1

    return sorted(indices), scores


if __name__ == "__main__":
    # 等價性驗證: 與樸素 O(m·n·m) 寫法逐一比對
    def extract_pips_naive(series, m):
        n = len(series)
        indices = [0, n - 1]
        scores = np.zeros(n)
        while len(indices) < m:
            idx_sorted = sorted(indices)
            max_dist, best_k = -1.0, -1
            for i in range(len(idx_sorted) - 1):
                left, right = idx_sorted[i], idx_sorted[i + 1]
                for k in range(left + 1, right):
                    dl = np.sqrt((k - left) ** 2 + (series[k] - series[left]) ** 2)
                    dr = np.sqrt((right - k) ** 2 + (series[right] - series[k]) ** 2)
                    if dl + dr > max_dist:
                        max_dist, best_k = dl + dr, k
            if best_k == -1:
                break
            indices.append(best_k)
            scores[best_k] = max_dist
        fm = scores[scores > 0].max() if (scores > 0).any() else 1.0
        scores[0] = scores[-1] = fm + 1
        return sorted(indices), scores

    rng = np.random.default_rng(42)
    for trial in range(20):
        s = 100 + np.cumsum(rng.normal(0, 1, 140))
        p1, s1 = extract_pips(s, 60)
        p2, s2 = extract_pips_naive(s, 60)
        assert p1 == p2, f"trial {trial}: indices differ"
        assert np.allclose(s1, s2), f"trial {trial}: scores differ"
    print("OK: 向量化版與樸素版 20 次隨機測試完全一致")

    import time
    s = 100 + np.cumsum(rng.normal(0, 1, 140))
    t0 = time.time()
    for _ in range(100):
        extract_pips(s, 60)
    print(f"vectorized: {(time.time()-t0)*10:.2f} ms/call")
