# from sklearn.cluster import KMeans
from Datanalysis.cluster.distances import (
    get_distance_metric, centroid_distance)
import numpy as np


class KMeans:
    def __init__(self, n_clusters, init='random'):
        self.n_clusters = n_clusters
        self.init = init

    def fit(self, X):
        d = get_distance_metric("euclidean")
        k_indexes = self.get_init_points(X)
        clusters = [X[i].reshape((1, len(X[i]))) for i in k_indexes]
        X = np.delete(X, k_indexes, axis=0)

        for x in X:
            s1 = x.reshape((1, len(x)))
            min_dist = np.inf
            min_i = -1
            for i in range(len(clusters)):
                dist = centroid_distance(s1, clusters[i], d)
                if dist < min_dist:
                    min_dist = dist
                    min_i = i
            clusters[min_i] = np.append(clusters[min_i], s1, axis=0)
        return clusters

    def get_init_points(self, X):
        return random(X, self.n_clusters)


def random(X, n):
    return np.random.randint(0, len(X), n)
