import numpy as np
import Datanalysis.cluster.distances as DIST


def linkage(X: np.ndarray, n_clusters, linkage="median", metric="euclidean"):
    d = DIST.get_distance_metric_one_interface(metric, X)
    linkage_consts_f = DIST.get_distance_method_consts(linkage)

    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))

    distances = distance_matrix(X, d)

    N = X.shape[0]
    clusters_cnt = np.ones(N, np.int64)
    dendrogram_items = np.empty((N - 1, 4))
    clusters = [[i] for i in range(N)]
    for i in range(N - 1):
        min_dist, min_j, min_k = get_min_distance(distances, N)

        d12 = distances[min_j, min_k]
        d13 = distances[min_j, :]
        d23 = distances[min_k, :]
        n1 = clusters_cnt[min_j]
        n2 = clusters_cnt[min_k]
        n3 = clusters_cnt
        col_d = DIST.lance_williams_distance(d12, d13, d23,
                                             *linkage_consts_f(n1, n2, n3))
        col_d[min_k] = np.nan
        distances[min_k, :] = np.nan
        distances[:, min_k] = np.nan
        distances[min_j, :] = col_d
        distances[:, min_j] = col_d

        dendrogram_items[i, 0] = min_j
        dendrogram_items[i, 1] = min_k
        dendrogram_items[i, 2] = min_dist
        dendrogram_items[i, 3] = clusters_cnt[min_j] + clusters_cnt[min_k]
        clusters_cnt[min_j] = dendrogram_items[i, 3]
        clusters_cnt[min_k] = -99999

        if i < N - n_clusters:
            merged_cluster = clusters[min_j] + clusters[min_k]
            clusters[min_j] = merged_cluster
            clusters[min_k] = []

    out_clusters = []
    for cluster in clusters:
        if len(cluster) > 0:
            out_clusters.append(list(cluster))

    return out_clusters, dendrogram_items


def distance_matrix(X: np.ndarray, d):
    N = len(X)
    distances = np.empty((N, N))
    m = X.shape[1]
    for i in range(N):
        distances[i, i] = 0
        for j in range(i + 1, N):
            xi = X[i].reshape((1, m))
            xj = X[j].reshape((1, m))
            distances[i, j] = distances[j, i] = d(xi, xj)
    if np.isnan(distances).any() or np.isinf(distances).any():
        raise ValueError('Distance function returned NaN or Inf')
    return distances


def get_min_distance(distances: np.ndarray, N: int):
    min_dist = np.inf
    for j in range(N):
        for k in range(j + 1, N):
            dist = distances[j, k]
            if dist < min_dist and j != k and dist >= 0:
                min_dist = dist
                min_j = j
                min_k = k
    return min_dist, min_j, min_k
