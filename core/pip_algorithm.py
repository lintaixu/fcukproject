"""
Perceptually Important Points (PIP) algorithm.

從股價序列中萃取最具代表性的 m 個關鍵點，並計算每個點的
chart importance score（加入時與兩端點構成的最大 Euclidean 距離）。
"""
import numpy as np


def extract_pips(series: np.ndarray, m: int):
    """
    Args:
        series: 1D numpy array, 收盤價序列, 長度 n
        m:      想保留的關鍵點數, m < n

    Returns:
        indices: List[int], 排序後的 m 個 PIP index
        scores:  np.ndarray of shape (n,), 每個點的 importance score
                 (非 PIP 的點為 0; 起點與終點為 inf)
    """
    n = len(series)
    if m >= n:
        scores = np.full(n, np.inf)
        return list(range(n)), scores

    indices = [0, n - 1]
    scores = np.zeros(n, dtype=np.float64)
    scores[0] = np.inf
    scores[-1] = np.inf

    while len(indices) < m:
        indices_sorted = sorted(indices)
        max_dist = -1.0
        best_k = -1

        # 對每對相鄰 PIP 之間的所有候選點計算距離
        for i in range(len(indices_sorted) - 1):
            left = indices_sorted[i]
            right = indices_sorted[i + 1]
            for k in range(left + 1, right):
                d_left = np.sqrt(
                    (k - left) ** 2 + (series[k] - series[left]) ** 2
                )
                d_right = np.sqrt(
                    (right - k) ** 2 + (series[right] - series[k]) ** 2
                )
                d = d_left + d_right
                if d > max_dist:
                    max_dist = d
                    best_k = k

        if best_k == -1:
            break
        indices.append(best_k)
        scores[best_k] = max_dist

    # 起終點分數設為比所有實際分數都高
    finite_max = scores[np.isfinite(scores)].max() if np.any(np.isfinite(scores) & (scores > 0)) else 1.0
    scores[0] = finite_max + 1
    scores[-1] = finite_max + 1

    return sorted(indices), scores


if __name__ == "__main__":
    # 簡單測試: 用一段含雜訊的 sine wave
    rng = np.random.default_rng(42)
    t = np.linspace(0, 4 * np.pi, 100)
    s = np.sin(t) * 10 + 100 + rng.normal(0, 0.5, 100)

    pips, scores = extract_pips(s, m=10)
    print(f"找到 {len(pips)} 個 PIPs (前後): {pips}")
    print(f"前 5 大 importance score 的 index: "
          f"{np.argsort(scores)[::-1][:5].tolist()}")
