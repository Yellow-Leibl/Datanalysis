import Datanalysis.cluster.distances as DIST
import numpy as np


class KMeans:
    def __init__(self, n_clusters, init='random', metric="euclidean"):
        self.n_clusters = n_clusters
        self.init = init
        self.metric = metric

    def fit(self, X):
        d = DIST.get_distance_metric_one_interface(self.metric, X)
        k_indexes = self.get_init_points(X)
        clusters = [X[i].reshape((1, len(X[i]))) for i in k_indexes]
        clusters_indexes = [[i] for i in k_indexes]

        for j, x in enumerate(X):
            if j in k_indexes:
                continue
            s1 = x.reshape((1, len(x)))
            min_dist = np.inf
            min_i = -1
            for i in range(len(clusters)):
                dist = DIST.centroid_distance(s1, clusters[i], d)
                if dist < min_dist:
                    min_dist = dist
                    min_i = i
            clusters[min_i] = np.append(clusters[min_i], s1, axis=0)
            clusters_indexes[min_i].append(j)
        return clusters_indexes

    def get_init_points(self, X, method="random"):
        if method == "random":
            return np.random.randint(0, len(X), self.n_clusters)
        elif method == "first":
            return [i for i in range(self.n_clusters)]
