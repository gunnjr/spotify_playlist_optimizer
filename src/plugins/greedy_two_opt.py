from typing import List
import numpy as np

def greedy_cycle(S: np.ndarray, start_idx: int = 0) -> List[int]:
    n = S.shape[0]
    unvisited = set(range(n))
    tour = [start_idx]
    unvisited.remove(start_idx)
    current = start_idx
    while unvisited:
        nxt = max(unvisited, key=lambda j: S[current, j])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tour

def cycle_score(S: np.ndarray, tour: List[int]) -> float:
    total = sum(S[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))
    return total / len(tour) if tour else 0.0

def two_opt_cycle(S: np.ndarray, tour: List[int], max_iter: int = 2000) -> List[int]:
    n = len(tour)
    improved = True
    count = 0
    while improved and count < max_iter:
        improved = False
        count += 1
        for i in range(n - 1):
            for k in range(i + 2, n if i > 0 else n - 1):
                a, b = tour[i], tour[(i + 1) % n]
                c, d = tour[k], tour[(k + 1) % n]
                current = S[a, b] + S[c, d]
                new = S[a, c] + S[b, d]
                if new > current + 1e-9:
                    tour[i + 1 : k + 1] = reversed(tour[i + 1 : k + 1])
                    improved = True
    return tour

def optimize_cycle(S: np.ndarray, starts: int = 10, seed: int = 42) -> list[int]:
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    start_idxs = rng.choice(n, size=min(starts, n), replace=False)
    best, best_score = None, -1.0
    for s in start_idxs:
        t = greedy_cycle(S, int(s))
        t = two_opt_cycle(S, t, max_iter=2000)
        sc = cycle_score(S, t)
        if sc > best_score:
            best, best_score = t, sc
    return best
