import numpy as np
from Datanalysis.cluster.distances import get_distance_metric_one_interface


class KNeighborsClassifier:
    def __init__(self, n_neighbors, metric="euclidean"):
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.d = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.d = get_distance_metric_one_interface(self.metric, X)
        self.X = X
        self.y = y

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def predict_one(self, x):
        distances = [self.d(x, xi) for xi in self.X]
        indexes = np.argsort(distances)[:self.n_neighbors]
        return np.argmax(np.bincount(self.y[indexes]))
