import numpy as np


class BayesClassifier:
    def fit(self, X: np.ndarray, y: np.ndarray):
        classes = np.unique(y)
        self.N = np.array([np.count_nonzero(y == c) for c in classes])
        self.p = self.N / len(y)
        self.X = [X[y == c] for c in classes]
        self.X_ = [np.mean(x, axis=0) for x in self.X]
        self.DC = [np.cov(x.T) for x in self.X]
        self.DC_1 = [np.linalg.inv(dc) for dc in self.DC]
        self.ln_det_DC = [np.log(np.linalg.det(dc)) for dc in self.DC]

    def predict(self, X: np.ndarray):
        return np.array([self.predict_one(x) for x in X], dtype=int)

    def predict_one(self, x: np.ndarray):
        K = len(self.X)
        g = [self.g(x, j) for j in range(K)]
        return np.argmax(g)

    def g(self, x: np.ndarray, j):
        return np.log(self.p[j]) - 1/2 * self.ln_det_DC[j] - \
            1/2 * (x - self.X_[j]) @ self.DC_1[j] @ (x - self.X_[j])
