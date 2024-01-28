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


def signif_general_dispersion(S: list[np.ndarray]):
    m = S[0].shape[1]
    v_sum = np.zeros((m, m))
    for s in S:
        V = s.T @ s
        N = len(s)
        v_sum += N * V
    return np.linalg.det(v_sum)


def signif_relation_functionals(S: list[np.ndarray], d):
    Q_4_div = 0.0
    for s in S:
        N = len(s)
        Q_4_div += N * (N - 1) / 2
    Q_4 = signif_pair_sum(S, d) / Q_4_div
    K = len(S)
    Q__4 = 0.0
    for j in range(K - 1):
        Nj = len(S[j])
        for ll in range(Nj):
            for m in range(j + 1, K):
                Nm = len(S[m])
                for h in range(Nm):
                    Q__4 += d(S[j][ll], S[m][h])
    for j in range(K):
        Nj = len(S[j])
        Q__4 /= Nj
    return Q_4 / Q__4
