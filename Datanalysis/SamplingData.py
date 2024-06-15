import numpy as np
import math
from Datanalysis.functions import (
    QuantileNorm, QuantileTStudent, QuantilePearson,
    FNorm, fNorm, fNorm_d_m, fNorm_d_sigma,
    FUniform, fUniform,
    FArcsin, fArcsin,
    FWeibull, fWeibull, fWeibull_d_alpha, fWeibull_d_beta,
    FExp, fExp, fExp_d_lamda,
    DF1Parametr, DF2Parametr)
from Datanalysis.SamplesTools import median, calculate_m
from copy import deepcopy


class SamplingData:
    def __init__(self, not_ranked_series_x: np.ndarray, trust: float = 0.05,
                 move_data=False, name='', ticks=None):
        self.name = name
        self.ticks = ticks
        self.trust = trust
        self.clusters = None
        self.metric = None
        self.set_data(not_ranked_series_x, move_data)
        self.init_characteristic()

    def set_clusters(self, clusters, metric):
        self.clusters = clusters
        self.metric = metric

    def remove_clusters(self):
        self.set_clusters(None, None)

    def set_data(self, not_ranked_series_x: np.ndarray, move_data):
        if move_data:
            self.raw = not_ranked_series_x
        else:
            self.raw = not_ranked_series_x.copy()
        self._x = self.raw.copy()

    def init_characteristic(self):
        self.min = 0.0
        self.max = 0.0

        self.MED = 0.0
        self.MAD = 0.0

        self.x_a = 0.0

        self.MED_Walsh = 0.0

        self.x_ = 0.0

        self.S_slide = 0.0
        self.Sigma_slide = 0.0
        self.u2 = 0.0
        self.u3 = 0.0
        self.S = 0.0
        self.Sigma = 0.0

        self.A = 0.0
        self.E = 0.0
        self.c_E = 0.0

        self.W_ = 0.0
        self.Wp = 0.0

        self.quant = []
        self.inter_range = 0.0

        self.det_x_ = 0.0
        self.det_Sigma = 0.0
        self.det_S = 0.0

        self.det_A = 0.0
        self.det_E = 0.0
        self.det_c_E = 0.0
        self.det_W_ = 0.0
        self.vanga_x_ = 0.0

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, i: int) -> float:
        return self._x[i]

    def remove_observations(self, i):
        new_clusters = []
        for cluster in self.clusters:
            new_cluster = cluster[~np.isin(cluster, i)]

            for ind in i[0]:
                for k in range(len(new_cluster)):
                    if new_cluster[k] > ind:
                        new_cluster[k] -= 1
            new_clusters.append(new_cluster)
        metric = self.metric

        self.setSeries(np.delete(self.raw, i))
        self.set_clusters(new_clusters, metric)

    def copy(self):
        return deepcopy(self)

    def toRanking(self):
        self._x = self._x[~np.isnan(self._x)]
        self._x.sort()
        self.probabilityX = np.zeros(len(self._x), dtype=float)
        j = -1
        prev = None
        for x in self._x:
            if prev != x:
                j += 1
                prev = x
            self.probabilityX[j] += 1.0
        self.probabilityX = self.probabilityX[:j + 1] / len(self._x)
        self._x = np.unique(self._x)

    def set_trust(self, trust: float):
        self.trust = trust

    def toCalculateCharacteristic(self, MED_Walsh=True):
        self.calculated = True
        N = len(self._x)
        if N == 0 or N == 1:
            return

        self.min = self._x[0]
        self.max = self._x[-1]

        self.MED = median(self._x)
        self.MAD = 1.483 * self.MED

        k = int(self.trust * N)

        self.x_a = 0.0
        for i in np.arange(k + 1, N - k):
            self.x_a += self._x[i]
        self.x_a /= N - 2 * k

        if MED_Walsh:
            self.MED_Walsh = self.toCalcMEDWalsh()

        self.x_ = sum(self.raw) / len(self.raw)

        nu2 = 0.0
        u2 = u3 = u4 = u5 = u6 = u8 = 0.0
        for i in range(N):
            nu2 += self._x[i] ** 2 * self.probabilityX[i]
            x_x_ = self._x[i] - self.x_
            u2 += x_x_ ** 2 * self.probabilityX[i]
            u3 += x_x_ ** 3 * self.probabilityX[i]
            u4 += x_x_ ** 4 * self.probabilityX[i]
            u5 += x_x_ ** 5 * self.probabilityX[i]
            u6 += x_x_ ** 6 * self.probabilityX[i]
            u8 += x_x_ ** 8 * self.probabilityX[i]

        # u2 -= self.x_ ** 2
        self.S_slide = nu2 - self.x_ ** 2
        self.Sigma_slide = self.S_slide ** 0.5
        self.u2 = u2
        self.u3 = u3
        sigma_u2 = math.sqrt(u2)
        self.S = u2 * N / (N - 1)
        self.Sigma = math.sqrt(self.S)

        if N > 3:
            self.A = u3 * math.sqrt(N * (N - 1)) / ((N - 2) * sigma_u2 ** 3)
            self.E = ((N ** 2 - 1) / ((N - 2) * (N - 3))) * (
                (u4 / sigma_u2 ** 4 - 3) + 6 / (N + 1))

            self.c_E = 1.0 / math.sqrt(abs(self.E))
        else:
            self.A = math.inf
            self.E = math.inf
            self.c_E = math.inf

        if self.x_:
            self.W_ = self.Sigma / self.x_
        else:
            self.W_ = math.inf

        if self.MED != 0:
            self.Wp = self.MAD / self.MED
        else:
            self.Wp = math.inf

        ip = 0.0
        self.quant = []
        p = 0.0
        step_quant = 0.025
        # 0.025     0.05    0.075   0.1     0.125
        # 0.15      0.175   0.2     0.225   0.25
        # ...       ...     ...     ...     ...
        # 0.825     0.85    0.875   0.9     0.925
        # 0.95      0.975   1.000
        for i in np.arange(N):
            p += self.probabilityX[i]
            while ip + step_quant < p:
                ip += step_quant
                self.quant.append(self._x[i])
        ind75 = math.floor(0.75 / step_quant) - 1
        ind25 = math.floor(0.25 / step_quant) - 1
        self.inter_range = self.quant[ind75] - self.quant[ind25]

        if N > 60:
            QUANT_I = QuantileNorm(1 - self.trust / 2)
        else:
            QUANT_I = QuantileTStudent(1 - self.trust / 2, N)

        self.det_x_ = self.Sigma / math.sqrt(N) * QUANT_I
        self.det_Sigma = self.Sigma / math.sqrt(2 * N) * QUANT_I
        self.det_S = 2 * self.S / (N - 1)

        B1 = u3 * u3 / (u2 ** 3)
        B2 = u4 / (u2 ** 2)
        B3 = u3 * u5 / (u2 ** 4)
        B4 = u6 / (u2 ** 3)
        B6 = u8 / (u2 ** 4)

        # det_A is negative
        self.det_A = math.sqrt(abs(1.0 / (4 * N) * (4 * B4 - 12 * B3 - 24 * B2
                                                    + 9 * B2 * B1 + 35 * B1 -
                                                    36))) * QUANT_I

        self.det_A = math.sqrt(6 * (N - 2) / ((N + 1) * (N + 3)))

        self.det_E = math.sqrt(1.0 / N * (B6 - 4 * B4 * B2 - 8 * B3 +
                                          4 * B2 ** 3 - B2 ** 2 +
                                          16 * B2 * B1 + 16 * B1)) * QUANT_I

        self.det_c_E = math.sqrt(abs(u4 / sigma_u2 ** 4) / (29 * N)) * (
            abs(u4 / sigma_u2 ** 4 - 1) ** 3) ** 0.25 * QUANT_I

        self.det_W_ = self.W_ * math.sqrt((1 + 2 * self.W_ ** 2) / (2 * N)
                                          ) * QUANT_I

        self.vanga_x_ = self.Sigma * math.sqrt(1 + 1 / N) * QUANT_I

    def toCalcMEDWalsh(self):
        N = len(self._x)
        if N < 2:
            return self._x[0]
        xl = np.empty(N * (N - 1) // 2, dtype=float)
        ll = 0
        for i in np.arange(N):
            for j in np.arange(i, N - 1):
                xl[ll] = 0.5 * (self._x[i] * self._x[j])
                ll += 1

        return median(xl)

    def setSeries(self, not_ranked_series_x: np.ndarray):
        self.set_data(not_ranked_series_x, False)
        self.init_characteristic()
        self.toRanking()
        self.toCalculateCharacteristic()

# edit sample
    def remove(self, minimum: float, maximum: float):
        new_raw_x = [x for x in self.raw if minimum <= x <= maximum]
        if len(new_raw_x) != len(self.raw):
            self.setSeries(np.array(new_raw_x))

    def auto_remove_anomalys(self) -> bool:
        is_del = False
        while self.auto_remove_anomaly():
            is_del = True
        return is_del

    def auto_remove_anomaly(self) -> bool:
        N = len(self.raw)
        is_item_del = False

        t1 = 2 + 0.2 * math.log10(0.04 * N)
        t2 = (19 * (self.E + 2) ** 0.5 + 1) ** 0.5
        a = 0.0
        b = 0.0
        if self.A < -0.2:
            a = self.x_ - t2 * self.S
            b = self.x_ + t1 * self.S
        elif self.A > 0.2:
            a = self.x_ - t1 * self.S
            b = self.x_ + t2 * self.S
        else:
            a = self.x_ - t1 * self.S
            b = self.x_ + t1 * self.S

        raw_x = list(self.raw)
        for i in range(N):
            if not (a < self.raw[i] < b):
                raw_x.remove(self.raw[i])
                is_item_del = True
                break

        if is_item_del:
            self.setSeries(np.array(raw_x))
        return is_item_del

    def to_log10(self):
        self.to_log(10)

    def to_log(self, base):
        self.setSeries(np.emath.logn(base, self.raw))

    def toExp(self):
        self.setSeries(np.exp(self.raw))

    def to_standardization(self):
        self.setSeries((self.raw - self.x_) / self.Sigma)

    def to_slide(self, value):
        self.setSeries(self.raw + value)

    def toMultiply(self, value):
        self.setSeries(self.raw * value)

    def toBinarization(self, x_):
        self.setSeries(np.array([1 if x > x_ else 0 for x in self.raw]))

    def toTransform(self, f_tr):
        self.setSeries(f_tr(self.raw))

    def to_centralization(self):
        self.to_slide(-self.x_)
# end edit

    def toCreateNormalFunc(self) -> tuple:
        N = len(self._x)
        m = self.x_
        sigma = self.Sigma
        def f(x): return fNorm(x, m, sigma)

        def F(x): return FNorm(x, m, sigma)

        def DF(x): return DF2Parametr(fNorm_d_m(x, m, sigma),
                                      fNorm_d_sigma(x, m, sigma),
                                      sigma ** 2 / N, sigma ** 2 / (2 * N),
                                      0)
        return f, F, DF

    def toCreateUniformFunc(self) -> tuple:
        # MM
        a = self.x_ - math.sqrt(3 * self.u2)
        b = self.x_ + math.sqrt(3 * self.u2)

        def f(x): return fUniform(a, b)

        def F(x): return FUniform(x, a, b)

        def DF(x): return (x - b) ** 2 / (b - a) ** 4
        return f, F, DF

    def to_create_exp_func(self) -> tuple:
        # MM
        N = len(self._x)
        lamd_a = 1 / self.x_

        def f(x): return fExp(x, lamd_a)

        def F(x): return FExp(x, lamd_a)

        d_theta = lamd_a ** 2 / N
        def DF(x): return DF1Parametr(fExp_d_lamda(x, lamd_a), d_theta)
        return f, F, DF

    def toCreateWeibullFunc(self) -> tuple:
        # MHK
        N = len(self._x)
        a11 = N - 1
        a12 = a21 = 0.0
        a22 = 0.0
        b1 = b2 = 0.0
        emp_func = 0.0
        for i in range(N - 1):
            emp_func += self.probabilityX[i]
            a12 += math.log(self._x[i])
            a22 += math.log(self._x[i]) ** 2
            b1 += math.log(math.log(1 / (1 - emp_func)))
            b2 += math.log(math.log(1 / (1 - emp_func))) * math.log(self._x[i])
        a21 = a12

        alpha = math.exp(-(b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21))
        beta = (b2 * a11 - b1 * a21) / (a11 * a22 - a12 * a21)

        def f(x): return fWeibull(x, alpha, beta)

        def F(x): return FWeibull(x, alpha, beta)

        S_2 = 0.0
        emp_func = 0.0
        for i in range(N - 1):
            emp_func += self.probabilityX[i]
            S_2 += (math.log(math.log(1 / (1 - emp_func))) +
                    math.log(alpha) - beta * math.log(self._x[i])) ** 2

        S_2 /= N - 3

        D_A_ = a22 * S_2 / (a11 * a22 - a12 * a21)
        D_alpha = math.exp(2 * math.log(alpha)) * D_A_

        D_beta = a11 * S_2 / (a11 * a22 - a12 * a21)

        cov_A_beta = -a12 * S_2 / (a11 * a22 - a12 * a21)
        cov_alpha_beta = -math.exp(-math.log(alpha)) * cov_A_beta

        def DF(x): return DF2Parametr(fWeibull_d_alpha(x, alpha, beta),
                                      fWeibull_d_beta(x, alpha, beta),
                                      D_alpha, D_beta, cov_alpha_beta)
        return f, F, DF

    def toCreateArcsinFunc(self) -> tuple:
        N = len(self._x)
        a_ = math.sqrt(2 * self.u2)
        def f(x): return fArcsin(x, a_)

        def F(x): return FArcsin(x, a_)

        def DF(x): return DF1Parametr(-x / (math.pi * a_ *
                                            math.sqrt(a_ ** 2 - x ** 2)),
                                      a_ ** 4 / (8 * N))
        return f, F, DF

    def toCreateTrustIntervals(self, f, F, DF, h):
        u = QuantileNorm(1 - self.trust / 2)

        def limit(x):
            return u * math.sqrt(DF(x))

        def hist_f(x): return f(x) * h
        def lw_limit_F(x): return F(x) - limit(x)
        def hg_limit_F(x): return F(x) + limit(x)

        return hist_f, lw_limit_F, F, hg_limit_F

    def is_normal(self) -> bool:
        return self.kolmogorov_test(self.toCreateNormalFunc()[1])

    def kolmogorov_test(self, func_reproduction) -> bool:
        N = len(self._x)

        D = 0.0
        emp_func = 0.0
        for i in range(N):
            emp_func += self.probabilityX[i]
            DN_plus = abs(emp_func - func_reproduction(self._x[i]))
            DN_minus = abs(emp_func - func_reproduction(self._x[i - 1]))
            if DN_plus > D:
                D = DN_plus
            if i > 0 and DN_minus > D:
                D = DN_minus

        z = math.sqrt(N) * D
        Kz = 0.0
        for k in range(1, 5):
            f1 = k ** 2 - 0.5 * (1 - (-1) ** k)
            f2 = 5 * k ** 2 + 22 - 7.5 * (1 - (-1) ** k)
            Kz += (-1) ** k * math.exp(-2 * k ** 2 * z ** 2) * (
                1 - 2 * k ** 2 * z / (3 * math.sqrt(N)) -
                1 / (18 * N) * ((f1 - 4 * (f1 + 3)) * k ** 2 * z ** 2 +
                                8 * k ** 4 * z ** 4) +
                k ** 2 * z / (27 * math.sqrt(N ** 3)) *
                (f2 ** 2 / 5 - 4 * (f2 + 45) * k ** 2 * z ** 2 / 15 +
                 8 * k ** 4 * z ** 4))
        Kz = 1 + 2 * Kz
        Pz = 1 - Kz
        alpha_zgodi = 0.3
        if N > 100:
            alpha_zgodi = 0.05
        self.kolmogorov_pz = Pz
        self.kolmogorov_alpha_zgodi = alpha_zgodi
        return Pz >= alpha_zgodi

    def xiXiTest(self,
                 func_reproduction,
                 hist_list: list):  # Pearson test
        hist_num = []
        M = len(hist_list)
        h = abs(self.max - self.min) / M
        N = len(self._x)
        Xi = 0.0
        xi = self.min
        j = 0
        for i in range(M):
            hist_num.append(0)
            while j < N and \
                    h * i <= self._x[j] - self.min <= h * (i + 1):
                hist_num[i] += 1
                j += 1
            xi += h
            ni_o = N * (func_reproduction(xi) - func_reproduction(xi - h))
            if ni_o == 0:
                return False
            Xi += (hist_num[i] - ni_o) ** 2 / ni_o

        Xi2 = QuantilePearson(1 - self.trust, M - 1)
        self.xixitest_x_2 = Xi
        self.xixitest_quant = Xi2
        return Xi < Xi2

    def get_histogram_data(self, column_number=0):
        if column_number <= 0:
            column_number = calculate_m(len(self.raw))
            column_number = min(column_number, len(self._x))

        n = len(self._x)
        h = (self.max - self.min) / column_number
        hist_list = np.zeros(column_number)
        for i in range(n - 1):
            j = math.floor((self._x[i] - self.min) / h)
            hist_list[j] += self.probabilityX[i]
        hist_list[-1] += self.probabilityX[-1]
        return hist_list

    def critetion_abbe(self) -> float:
        N = len(self.raw)
        d2 = 1 / (N - 1) * sum(
            [(self.raw[i + 1] - self.raw[i]) ** 2 for i in range(N - 1)])

        q = d2 / (2 * self.S)

        E_q = 1
        D_q = (N - 2) / (N ** 2 - 1)
        U = (q - E_q) / D_q ** 0.5
        P = FNorm(U)
        return P
