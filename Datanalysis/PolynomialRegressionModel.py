import numpy as np


class PolynomialRegressionModel:
    def __init__(self, degree=1):
        self.degree = degree

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.X_poly = self.__polynomial_features(X)
        self.theta = self.__normal_equation(self.X_poly, y)

    def predict(self, X):
        X_poly = self.__polynomial_features(X)
        return X_poly @ self.theta

    def __polynomial_features(self, X: np.ndarray):
        m = 1
        n = len(X)
        if len(X.shape) == 2:
            m = X.shape[0]
            n = X.shape[1]
        X_poly = np.ones((n, 1))
        for i in range(1, self.degree + 1):
            if m > 1:
                X_poly = np.concatenate((X_poly, np.power(X.T, i)), axis=1)
            else:
                X_poly = np.c_[X_poly, np.power(X, i)]
        return X_poly

    def __normal_equation(self, X: np.ndarray, y: np.ndarray):
        return np.linalg.inv(X.T @ X) @ X.T @ y

    def score(self, x, y) -> float:
        N = len(x)
        y_ = self.predict(x)
        S_2 = 0.0
        for i in range(N):
            S_2 += (y[i] - y_[i]) ** 2
        return S_2 / N

    def train_test_score(self, x_test, y_test):
        train_score = self.score(self.X, self.y)
        test_score = self.score(x_test, y_test)
        return train_score, test_score
