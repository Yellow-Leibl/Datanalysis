from Datanalysis import SamplingData
from Datanalysis.SamplesTools import MED, static_vars
import Datanalysis.functions as func
import math
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TimeSeriesData:
    def __init__(self, d: SamplingData):
        self.sampling_data = d
        self.to_calculate_characteristic()

    def set_series(self, x):
        self.sampling_data.setSeries(x)
        self.to_calculate_characteristic()

    def auto_remove_anomalys(self, k=3):
        def x_(i):
            return sum(self.x[:i]) / i

        def S_2(i):
            sum_1 = 0.0
            for j in range(1, i + 1):
                sum_1 += (self.x[j - 1] - x_(i)) ** 2
            return sum_1 / (i - 1)

        to_del_indexes = []
        for i in range(2, len(self.x)):
            if not (x_(i) - k * S_2(i) < self.x[i] < x_(i) + k * S_2(i)):
                to_del_indexes.append(i)

        if len(to_del_indexes) != 0:
            for i in to_del_indexes:
                self.x[i] = 2 * self.x[i - 1] - self.x[i - 2]
            self.set_series(self.x)
            logger.debug(f"Replaced observations {len(to_del_indexes)}")
        return bool(len(to_del_indexes))

    def to_calculate_characteristic(self):
        self.trust = self.sampling_data.trust
        N = len(self.sampling_data.raw)
        self.x = self.sampling_data.raw
        self.m = self.sampling_data.x_
        self.Dispersion = self.sampling_data.S

        def auto_cov(teta):
            sum_1 = 0.0
            for t in range(N - teta):
                sum_1 += (self.x[t] - self.m) * (self.x[t + teta] - self.m)
            return 1 / (N - teta) * sum_1
        self.auto_cov_f = auto_cov

        def auto_cor(teta):
            return auto_cov(teta) / self.Dispersion
        self.auto_cor_f = auto_cor

        self.sign_criterion_for_trend()
        self.manns_criterion_for_trend()
        self.series_criterion_for_trend()
        self.rises_descending_series_criterion_for_trend()
        self.abbe_criterion()

    def sign_criterion_for_trend(self):
        N = len(self.x)
        c = 0
        for i in range(N - 1):
            if self.x[i + 1] > self.x[i]:
                c += 1

        E_c = 1/2 * (N - 1)
        D_c = 1/12 * (N + 1)

        self.critetion_sign = (c - E_c) / D_c ** 0.5
        self.critetion_sign_signif = func.QuantileNorm(1 - self.trust / 2)

    def manns_criterion_for_trend(self):
        N = len(self.x)
        t = 0.0
        for i in range(N - 1):
            for j in range(i + 1, N):
                if self.x[i] < self.x[j]:
                    t += 1
                elif self.x[i] == self.x[j]:
                    t += 0.5

        E_t = 1/4 * N * (N - 1)
        D_t = 1/72 * (2 * N + 5) * (N - 1) * N

        self.critetion_mann = (t + 0.5 - E_t) / D_t ** 0.5
        self.critetion_mann_signif = func.QuantileNorm(1 - self.trust / 2)

    def series_criterion_for_trend(self):
        N = len(self.x)
        x_m = MED(self.x)

        nu_N = 0
        d_N = 0

        series = []
        if self.x[0] >= x_m:
            series.append(1)
        else:
            series.append(-1)
        for i in range(1, N):
            if self.x[i] >= x_m:
                if series[-1] != 1:
                    nu_N += 1
                    d_N = max(d_N, len(series))
                    series = []
                series.append(1)
            else:
                if series[-1] != -1:
                    nu_N += 1
                    d_N = max(d_N, len(series))
                    series = []
                series.append(-1)
        nu_N += 1
        d_N = max(d_N, len(series))

        self.critetion_series_nu = nu_N
        self.critetion_series_d = d_N
        u_1_a_2 = func.QuantileNorm(1 - self.trust / 2)
        self.critetion_series_nu_signif = int(
            1/2 * (N + 1 - u_1_a_2 * (N - 1) ** 0.5))
        self.critetion_series_d_signif = int(3.3 * math.log10(N + 1))

    def rises_descending_series_criterion_for_trend(self):
        N = len(self.x)

        nu_N = 0
        d_N = 0

        series = []
        if self.x[1] >= self.x[0]:
            series.append(1)
        else:
            series.append(-1)
        for i in range(1, N - 1):
            if self.x[i+1] >= self.x[i]:
                if series[-1] != 1:
                    nu_N += 1
                    d_N = max(d_N, len(series))
                    series = []
                series.append(1)
            else:
                if series[-1] != -1:
                    nu_N += 1
                    d_N = max(d_N, len(series))
                    series = []
                series.append(-1)
        nu_N += 1
        d_N = max(d_N, len(series))

        self.critetion_series_nu = nu_N
        self.critetion_series_d = d_N
        u_1_a_2 = func.QuantileNorm(1 - self.trust / 2)
        self.critetion_series_nu_signif = int(
            1/3 * (2 * N - 1) - u_1_a_2 * (1/90 * (16 * N - 29)) ** 0.5)
        if N <= 26:
            self.critetion_series_d_signif = 5
        elif N <= 153:
            self.critetion_series_d_signif = 6
        else:
            self.critetion_series_d_signif = 7

    def abbe_criterion(self):
        N = len(self.x)
        q_2 = 0.0
        for i in range(N - 1):
            q_2 += (self.x[i] - self.x[i+1]) ** 2
        q_2 /= N - 1

        s_2 = 0.0
        for i in range(N):
            s_2 += (self.x[i] - self.m) ** 2
        s_2 /= N - 1

        gamma = q_2 / (2 * s_2)
        u = (gamma - 1) * ((N**2 - 1) / (N - 2)) ** 0.5
        self.critetion_abbe = abs(u)
        self.critetion_abbe_signif = func.QuantileNorm(1 - self.trust / 2)

    def moving_average_method(self, p=3, k=5):
        N = len(self.x)
        if k % 2 == 0:
            logger.error("k must be odd")
            return
        m = k // 2
        arr_t = np.zeros(p*2 + 1)
        arr_t[0] = k
        for i in range(2, len(arr_t), 2):
            arr_t[i] = sum([t ** i for t in range(-m, m + 1)])

        t_mat = np.empty((p + 1, p + 1))
        for i in range(p + 1):
            for j in range(p + 1):
                t_mat[i, j] = arr_t[i + j]

        x = self.x.copy()

        def f(t, a):
            return sum([a[j] * t ** j for j in range(p + 1)])

        for i in range(2, N - 2, m*2):
            if i + m >= N:
                break
            b = np.zeros(p + 1)
            for j in range(p + 1):
                for t in range(-m, m + 1):
                    b[j] += x[i + t] * t ** j
            a = func.gauss_method(t_mat, b)

            for t in range(-m, m + 1):
                x[i + t] = f(t, a)
        return x

    def median_method(self):
        N = len(self.x)
        x = self.x.copy()
        alpha = 1/3

        def x_(i):
            if i == 0:
                return 1/3 * (x0 + x[i] + x[i+1])
            if i == N - 1:
                return 1/3 * (x[i-1] + x[i] + xN_1)
            return 1/3 * (x[i-1] + x[i] + x[i+1])

        def ll(i):
            return (1 + (x[i] - x[i-1]) ** 2) ** 0.5

        def l_(i):
            return (1 + (x_(i) - x_(i-1)) ** 2) ** 0.5

        qt_i = 1
        pt_i = 0

        def qx(i):
            if i == 0:
                return x[i] - x0
            if i == N:
                return xN_1 - x[i - 1]
            return x[i] - x[i - 1]

        def px(i):
            return qx(i+1) - 2 * qx(i) + qx(i-1)

        def T(i):
            return 1/(2*l_(i)) * (qt_i * pt_i + qx(i) * px(i))

        iter_i = 0
        while True:
            x0 = 1/3 * (4*x[0] + x[1] - 2*x[2])
            xN_1 = 1/3 * (4*x[N-1] + x[N-2] - 2*x[N-3])

            for i in range(N):
                x[i] = x_(i)

            L = 0.0
            for i in range(1, N):
                L += ll(i)
            L_ = 0.0
            for i in range(1, N):
                L_ += l_(i)

            A = 0.0
            for i in range(1, N):
                A += l_(i) * T(i)
            iter_i += 1
            if abs(L - L_) <= abs(A) * alpha * (L_/N) ** 2:
                break
            if iter_i > 100:
                logger.error("Too many iterations")
                break

        return x

    def sma_method(self, n=8):
        N = len(self.x)
        x = self.x.copy()

        def sma(i, sma_1=0.0):
            if i == n:
                return sma0(x, n)
            return sma_1 + (x[i] - x[i - n]) / n

        sma_1 = 0.0
        for i in range(n, N):
            x[i] = sma(i, sma_1)
            sma_1 = x[i]

        return x

    def wma_method(self, n=8):
        N = len(self.x)
        x = self.x.copy()

        def wma(i):
            sum_n_j = 0.0
            for j in range(n):
                sum_n_j += (n - j) * x[i - j]
            return 2/(n*(n+1)) * sum_n_j

        for i in range(n, N):
            x[i] = wma(i)

        return x

    def ema_method(self, n=8):
        N = len(self.x)
        x = self.x.copy()

        for i in range(n, N):
            x[i] = ema(i, x, n)

        return x

    def dma_method(self, n=8):
        N = len(self.x)
        x = self.x.copy()

        for i in range(n, N):
            x[i] = dma(i, x, n)

        return x

    def tma_method(self, n=8):
        N = len(self.x)
        x = self.x.copy()

        for i in range(n, N):
            x[i] = tma(i, x, n)

        return x

    def remove_poly_trend(self, k=2):
        N = len(self.x)

        t_ = np.zeros((k * 2 + 1))
        for j in range(1, k * 2 + 1):
            for t in range(N):
                t_[j] += t ** j
        t_ /= N
        t_[0] = 1

        t_mat = np.empty((k + 1, k + 1))
        for i in range(k + 1):
            for j in range(k + 1):
                t_mat[i, j] = t_[i + j]

        b = np.zeros(k + 1)
        for j in range(k + 1):
            for t in range(N):
                b[j] += self.x[t] * t ** j
        b /= N

        a = func.gauss_method(t_mat, b)

        def m(t):
            val = 0.0
            for j in range(k + 1):
                val += a[j] * t ** j
            return val

        x = np.empty(N)
        for t in range(N):
            x[t] = m(t)

        return x

    def components_ssa_method(self, M):
        components = np.empty((M, len(self.x)))
        A, Y = self.ssa_decomposition(self.x, M)

        for v in range(M):
            A_c = np.zeros(A.shape)
            A_c[:, v] = A[:, v]
            X, _ = self.ssa_reproduction_trajectory_matrix(A_c, Y, M)
            components[v] = ssa_diagonal_averaging(X)

        return components

    def reconstruction_ssa_method(self, M, n_components):
        A, Y = self.ssa_decomposition(self.x, M)
        X, _ = self.ssa_reproduction_trajectory_matrix(A, Y, n_components)
        p = ssa_diagonal_averaging(X)
        return p

    def test_forecast_ssa_method(self, M, n_components, count_for_forecast):
        N = len(self.x)
        if not (0 < count_for_forecast < N - 3):
            count_for_forecast = N // 2
        p = self.x[:-count_for_forecast]

        return self.forecast_ssa_method(p, M, n_components, count_for_forecast)

    def forecast_ssa_method(self, p, M, n_components, count_forecast):
        for i in range(count_forecast):
            A, Y = self.ssa_decomposition(p, M)

            X, V = self.ssa_reproduction_trajectory_matrix(A, Y, n_components)

            X = self.ssa_append_forecasting_vector(V, X)

            p = ssa_diagonal_averaging(X)

            logger.debug(f"Forecasting count: {i+1}")
        return p

    def ssa_decomposition(self, p, M):
        X = self.ssa_trajectory_matrix(p, M)

        A = ssa_eigen_vectors(X)

        Y = A.T @ X
        return A, Y

    def ssa_trajectory_matrix(self, p, M):
        N = len(p)

        X = np.empty((M, N - M + 1))
        for k in range(M):
            for i in range(N - M + 1):
                X[k, i] = p[k + i]
        return X

    def ssa_reproduction_trajectory_matrix(self,
                                           A: np.ndarray,
                                           Y: np.ndarray,
                                           n_components: int):
        M = A.shape[0]
        V = A.copy()
        for i in range(n_components, M):
            V[:, i] = 0.0

        X = V @ Y
        return X, V

    def ssa_append_forecasting_vector(self,
                                      A: np.ndarray,
                                      X: np.ndarray) -> np.ndarray:
        a = A[:-1, :-1]
        b = X[1:, -1]
        y = func.gauss_method(a, b)
        p_forecast = 0.0
        for v in range(len(y)):
            p_forecast += A[-1, v] * y[v]
        last_x = np.append(b, p_forecast)
        return np.c_[X, last_x]


def ssa_eigen_vectors(X: np.ndarray):
    DC = X @ X.T
    vals, A = func.EigenvalueJacob(DC)
    _, A = func.sort_evects_n_evals(vals, A)
    return A


def ssa_diagonal_averaging(X: np.ndarray):
    M = X.shape[0]
    N = M + X.shape[1] - 1
    p = np.zeros(N)
    for i in range(M):
        for j in range(i + 1):
            p[i] += X[j, i - j]
        p[i] /= (i + 1)

    for i in range(M, N - M):
        for j in range(M):
            p[i] += X[j, i - j]
        p[i] /= M

    for i in range(N - M, N):
        for j in range(i - (N - M), M):
            p[i] += X[j, i - j]
        p[i] /= (N - i)
    return p


def sma0(x, n):
    sma_0 = 0.0
    for j in range(n - 1):
        sma_0 += x[j]
    return sma_0 / n


@static_vars(ema_1=0.0, alpha=0.0)
def ema(i, x, n):
    if i == n:
        ema.alpha = 2/(n + 1)
        ema.ema_1 = sma0(x, n)
    else:
        ema.ema_1 = ema.alpha * x[i] + (1 - ema.alpha) * ema.ema_1
    return ema.ema_1


@static_vars(dma_1=0.0, alpha=0.0)
def dma(i, x, n):
    if i == n:
        dma.alpha = 2/(n + 1)
        dma.dma_1 = ema(i, x, n)
    else:
        dma.dma_1 = dma.alpha * ema(i, x, n) + (1 - dma.alpha) * dma.dma_1
    return dma.dma_1


@static_vars(tma_1=0.0, alpha=0.0)
def tma(i, x, n):
    if i == n:
        tma.alpha = 2/(n + 1)
        tma.tma_1 = dma(i, x, n)
    else:
        tma.tma_1 = tma.alpha * dma(i, x, n) + (1 - tma.alpha) * tma.tma_1
    return tma.tma_1
