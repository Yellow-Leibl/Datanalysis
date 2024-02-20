import numpy as np
from Datanalysis.cluster.distances import get_distance_metric_one_interface


class NeighborsModClassifier:
    def __init__(self, metric="euclidean"):
        self.metric = metric
        self.d = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.d = get_distance_metric_one_interface(self.metric, X)
        self.X = X
        self.Y = Y
        n = Y.max() - Y.min() + 1
        self.d_max = np.zeros(n)
        clusters = [[] for _ in range(n)]
        for x, y in zip(X, Y):
            clusters[y].append(x)
        for i in range(n):
            max_dist = 0
            N = len(clusters[i])
            for j in range(N):
                for k in range(j + 1, N):
                    max_dist = max(max_dist, self.d(X[j], X[k]))
            self.d_max[i] = max_dist

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def predict_one(self, x):
        distances = [self.d(x, xi) for xi in self.X]
        index_min = np.argmin(distances)
        y = self.Y[index_min]

        distances_to_cluster = distances[self.Y == y]
        d0_max = np.max(distances_to_cluster)
        if d0_max < self.d_max[y]:
            return index_min

        return np.nan
