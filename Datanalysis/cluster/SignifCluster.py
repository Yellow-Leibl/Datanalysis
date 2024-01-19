import numpy as np


def signif_weighted_sum(S, d):
    q = 0.0
    for s in S:
        x_ = np.mean(s, axis=0)
        for x in s:
            q += d(x, x_) ** 2
    return q


def signif_pair_sum(S, d):
    q = 0.0
    for s in S:
        for j in range(len(s) - 1):
            for h in range(j + 1, len(s)):
                q += d(s[j], s[h])
    return q


def signif_general_dispersion(S: np.ndarray):
    v_sum = np.zeros(S.shape[0])
    for s in S:
        V = s @ s.T
        N = len(s)
        v_sum += N * V
    return np.linalg.det(v_sum)

# TODO: Відношення функціоналів
