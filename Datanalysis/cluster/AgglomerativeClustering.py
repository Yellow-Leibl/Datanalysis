import numpy as np
from Datanalysis.cluster.distances import (
    get_distance_method_consts, get_distance_metric, lance_williams_distance)


class AgglomerativeClustering:
    def __init__(self, n_clusters, linkage="median", metric="euclidean"):
        self.n_clusters = n_clusters
        self.d = get_distance_metric(metric)
        self.linkage_consts = get_distance_method_consts(linkage)

    def fit(self, X: np.ndarray):
        if len(X.shape) == 1:
            X = X.reshape((len(X), 1))

        alpha1, alpha2, beta, eps = self.linkage_consts

        distances = distance_matrix(X, self.d)

        N = X.shape[0]
        clusters_cnt = np.ones(N, np.int64)
        dendrogram_items = np.empty((N - 1, 4))
        clusters = [{i} for i in range(N)]
        for i in range(N - self.n_clusters):
            min_dist = np.inf
            for j in range(N):
                for k in range(j + 1, N):
                    dist = distances[j, k]
                    if dist < min_dist and j != k and dist >= 0:
                        min_dist = dist
                        min_j = j
                        min_k = k

            col_d = lance_williams_distance(distances[min_j, min_k],
                                            distances[min_j, :],
                                            distances[min_k, :],
                                            alpha1, alpha2, beta, eps)
            col_d[min_k] = -1
            distances[min_k, :] = -1
            distances[:, min_k] = -1
            distances[min_j, :] = col_d
            distances[:, min_j] = col_d

            dendrogram_items[i, 0] = min_j
            dendrogram_items[i, 1] = min_k
            dendrogram_items[i, 2] = min_dist
            dendrogram_items[i, 3] = clusters_cnt[min_j] + clusters_cnt[min_k]
            clusters_cnt[min_j] = dendrogram_items[i, 3]
            clusters_cnt[min_k] = -1

            merged_cluster = clusters[min_j] | clusters[min_k]
            clusters[min_j] = merged_cluster
            clusters[min_k] = set()

        self.clusters = []
        for cluster in clusters:
            if len(cluster) > 0:
                self.clusters.append(list(cluster))

        return self.clusters


def distance_matrix(X, d):
    N = len(X)
    distances = np.empty((N, N))
    m = X.shape[1]
    for i in range(N):
        distances[i, i] = 0
        for j in range(i + 1, N):
            xi = X[i].reshape((1, m))
            xj = X[j].reshape((1, m))
            distances[i, j] = d(xi, xj)
            if np.isnan(distances[i, j]) or np.isinf(distances[i, j]):
                raise ValueError('Distance function returned NaN or Inf')
            distances[j, i] = distances[i, j]
    return distances
