import numpy as np


class LogisticClassifier:
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray,
            alpha=0.01, max_iter=1000):
        X = np.c_[np.ones(X.shape[0]), X]
        N = X.shape[1]
        self.teta = np.zeros(X.shape[1])
        for _ in range(max_iter):
            z = X @ self.teta
            y_pred = self.sigmoid(z)
            dteta = (y_pred - y) @ X / N
            self.teta -= alpha * dteta

    def predict(self, X: np.ndarray, alpha=0.5):
        X = np.c_[np.ones(X.shape[0]), X]
        z = X @ self.teta
        y_pred = self.sigmoid(z)
        return np.where(y_pred > alpha, 1, 0)
