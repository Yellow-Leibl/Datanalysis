import math
import numpy as np

from Datanalysis.SamplingData import SamplingData
from Datanalysis.PolynomialRegressionModel import PolynomialRegressionModel
from Datanalysis.SamplesTools import median

import Datanalysis.functions as func


class DoubleSampleRegression:
    def __init__(self, x: SamplingData, y: SamplingData, trust: float = 0.05):
        self.x = x
        self.y = y
        self.trust = trust

    # implemented in DoubleSampleData class
    def identDispersionBarlet(self, samples, trust: float = 0.05) -> bool:
        print("AbstractMethod")
        raise NotImplementedError

    def toCreateLinearRegressionMNK(self):
        b = self.r * self.y.Sigma / self.x.Sigma
        a = self.y.x_ - b * self.x.x_

        self.line_a = a
        self.line_b = b
        self.accuracyLineParameters(a, b)

        def f(x): return a + b * x

        return *self.linearTolerantIntervals(f), \
            *self.linearTrustIntervals(f), f

    def initial_condition_for_line_regr(self) -> bool:
        if not (self.x.is_normal() and self.y.is_normal()):
            print("Початкова умова 1 лінійного "
                  "регресійного аналізу не виконується")
            return False

        if not (self.identDispersionBarlet([self.x, self.y], self.trust)
                and self.x.critetion_abbe() and self.y.critetion_abbe()):
            print("Початкова умова 2 лінійного "
                  "регресійного аналізу не виконується")
            return False

        return True

    def toCreateLinearRegressionMethodTeila(self):
        if self.initial_condition_for_line_regr():
            pass

        x = self.x.raw.copy()
        y = self.y.raw.copy()
        x.sort()
        y.sort()

        N = len(x)
        b_l = np.empty(N * (N - 1) // 2, dtype=float)
        ll = 0
        for i in np.arange(N):
            for j in np.arange(i + 1, N):
                b_l[ll] = (y[i] - y[j]) / (x[i] - x[j])
                ll += 1
        b = median(b_l)

        a_l = np.empty(N, dtype=float)
        for i in range(N):
            a_l[i] = y[i] - b * x[i]
        a = median(a_l)

        self.line_a = a
        self.line_b = b
        self.accuracyLineParameters(a, b)

        def f(x): return a + b * x

        return *self.linearTolerantIntervals(f), \
            *self.linearTrustIntervals(f), f

    def accuracyLineParameters(self, a, b):
        N = len(self)
        x = self.x.raw
        y = self.y.raw
        def f(x): return a + b * x

        S_zal = sum([(y[i] - f(x[i])) ** 2 for i in range(N)]
                    ) / (N - 2)

        S_a = S_zal ** 0.5 * (1 / N + self.x.x_ ** 2 /
                              (self.x.S * (N - 1))) ** 0.5
        S_b = S_zal ** 0.5 / (self.x.S * (N - 1) ** 0.5)
        t_a = a / S_a
        t_b = b / S_b
        t_stud = func.QuantileTStudent(1 - self.trust / 2, N - 2)

        if not (abs(t_a) > t_stud or abs(t_b) > t_stud):
            return

        self.det_line_a = t_stud * S_a
        self.det_line_b = t_stud * S_b

    def linearTolerantIntervals(self, f):
        N = len(self)
        sigma_eps = self.y.Sigma * (
            (1 - self.r ** 2) * (N - 1) / (N - 2)) ** 0.5

        def less_f(x): return f(x) - func.QuantileTStudent(
            1 - self.trust / 2, N - 2) * sigma_eps

        def more_f(x): return f(x) + func.QuantileTStudent(
            1 - self.trust / 2, N - 2) * sigma_eps
        return less_f, more_f

    def linearTrustIntervals(self, f):
        N = len(self)
        xl = self.x.raw
        y = self.y.raw
        x_ = self.x.x_

        S_zal = sum([(y[i] - f(xl[i])) ** 2 for i in range(N)]
                    ) / (N - 2)
        S_b = S_zal ** 0.5 / (self.x.S * (N - 1) ** 0.5)
        def S_y_(x): return (self.sigma_eps ** 2 / N +
                             S_b ** 2 * (x - self.x.x_) ** 2) ** 0.5
        t = func.QuantileTStudent(1 - self.trust / 2, N - 2)

        def tr_lf(x): return f(x) - t * S_y_(x)

        def tr_mf(x): return f(x) + t * S_y_(x)

        def S_y_x0(x): return (self.sigma_eps ** 2 * (
            1 + 1 / N) + S_b ** 2 * (x - x_)) ** 0.5

        def tr_f_lf(x): return f(x) - t * S_y_x0(x)

        def tr_f_mf(x): return f(x) + t * S_y_x0(x)

        return tr_lf, tr_mf, tr_f_lf, tr_f_mf

    def toCreateParabolicRegression(self):
        x_ = self.x.x_
        y_ = self.y.x_
        xl = self.x.raw
        y = self.y.raw
        N = len(self)
        a = y_
        b = sum([(xl[i] - x_) * y[i] for i in range(N)]) / sum(
            [(xl[i] - x_) ** 2 for i in range(N)])

        def phi1(x): return x - x_

        phi2_k = sum([xl[i] ** 3 - x_ * xl[i] ** 2 for i in range(N)]
                     ) / (sum([xl[i] ** 2 for i in range(N)]) - N * x_ ** 2)

        x_2 = sum([xl[i] ** 2 for i in range(N)]) / N

        def phi2(x): return x ** 2 - phi2_k * (x - x_) - x_2

        c = sum([phi2(xl[i]) * y[i] for i in range(N)]) / sum(
            [phi2(xl[i]) ** 2 for i in range(N)])

        self.parab_a = a
        self.parab_b = b
        self.parab_c = c

        def f(x): return a + b * phi1(x) + c * phi2(x)

        self.accuracyParabolaParameters(a, b, c, phi2)

        return *self.parabolaTolerantIntervals(f), \
            *self.parabolaTrustIntervals(f, phi1, phi2), f

    def accuracyParabolaParameters(self, a, b, c, phi2):
        N = len(self)
        xl = self.x.raw
        y = self.y.raw
        a = self.parab_a
        b = self.parab_b
        c = self.parab_c
        sigma_x = self.x.Sigma

        def f(x): return a + b * x + c * x ** 2

        S_zal = (sum([(y[i] - f(xl[i])) ** 2 for i in range(N)])
                 / (N - 2)) ** 0.5
        self.parab_a_t = abs(a / S_zal * N ** 0.5)
        self.parab_b_t = abs(b * sigma_x / S_zal * N ** 0.5)
        self.parab_c_t = abs(c / S_zal *
                             sum([phi2(xl[i]) ** 2 for i in range(N)]) ** 0.5)
        t = func.QuantileTStudent(1 - self.trust / 2, N - 3)

        S_a = S_zal / N ** 0.5
        S_b = S_zal / (sigma_x * N ** 0.5)
        S_c = S_zal / (N * sum([phi2(xl[i]) ** 2 for i in range(N)])) ** 0.5
        self.det_parab_a = t * S_a
        self.det_parab_b = t * S_b
        self.det_parab_c = t * S_c

    def parabolaTolerantIntervals(self, f):
        N = len(self)
        t = func.QuantileTStudent(1 - self.trust / 2, N - 3)
        x = self.x.raw
        y = self.y.raw
        S_zal = (sum([(y[i] - f(x[i])) ** 2 for i in range(N)]
                     ) / (N - 2)) ** 0.5

        def less_f(x): return f(x) - t * S_zal

        def more_f(x): return f(x) + t * S_zal
        return less_f, more_f

    def parabolaTrustIntervals(self, f, phi1, phi2):
        N = len(self)
        t = func.QuantileTStudent(1 - self.trust / 2, N - 3)
        x = self.x.raw
        y = self.y.raw
        sigma_x = self.x.Sigma
        S_zal = (sum([(y[i] - f(x[i])) ** 2 for i in range(N)]
                     ) / (N - 2)) ** 0.5
        t = func.QuantileTStudent(1 - self.trust / 2, N - 3)

        phi2_2 = sum([phi2(x[i]) ** 2 for i in range(N)])

        def S_y_x(x): return S_zal / N ** 0.5 * (
            1 + phi1(x) ** 2 / sigma_x ** 2 +
            phi2(x) ** 2 / phi2_2) ** 0.5

        def tr_lf(x): return f(x) - t * S_y_x(x)

        def tr_mf(x): return f(x) + t * S_y_x(x)

        def S_y_x0(x): return S_zal / N ** 0.5 * (
            N + 1 + phi1(x) ** 2 / sigma_x ** 2 +
            phi2(x) ** 2 / phi2_2) ** 0.5

        def tr_f_lf(x): return f(x) - t * S_y_x0(x)

        def tr_f_mf(x): return f(x) + t * S_y_x0(x)

        return tr_lf, tr_mf, tr_f_lf, tr_f_mf

    def toCreateKvazi8(self):  # y = a * exp(b * x)
        N = len(self)
        x = self.x.raw
        y = self.y.raw
        def phi(i): return x[i]
        def kappa(i): return math.log(y[i])
        def lambda_(i): return y[i] ** 2

        phi_ = sum([phi(i) * lambda_(i) for i in range(N)]
                   ) / sum([lambda_(i) for i in range(N)])

        kappa_ = sum([kappa(i) * lambda_(i) for i in range(N)]
                     ) / sum([lambda_(i) for i in range(N)])

        phi_2 = sum([phi(i) ** 2 * lambda_(i) for i in range(N)]
                    ) / sum([lambda_(i) for i in range(N)])

        phi_kappa = sum(
            [phi(i) * kappa(i) * lambda_(i) for i in range(N)]
            ) / sum([lambda_(i) for i in range(N)])

        B = (phi_kappa - phi_ * kappa_) / (phi_2 - phi_ ** 2)
        A = kappa_ - B * phi_

        self.kvaz_a = math.exp(A)
        self.kvaz_b = B

        def f(x): return math.exp(A) * math.exp(x * B)

        self.accuracyKvaziParameters(A, B)

        return *self.kvaziTolerantIntervals(f), *self.kvaziTrustIntervals(f), f

    def accuracyKvaziParameters(self, A, B):
        N = len(self)
        x = self.x.raw
        y = [math.log(yi) for yi in self.y.raw]
        def f(x): return A * math.exp(x * B)

        S_zal = sum([(y[i] - f(x[i])) ** 2 for i in range(N)]
                    ) / (N - 2)

        S_a = S_zal ** 0.5 * (1 / N + self.x.x_ ** 2 /
                              (self.x.S * (N - 1))) ** 0.5
        S_b = S_zal ** 0.5 / (self.x.S * (N - 1) ** 0.5)
        t_a = A / S_a
        t_b = B / S_b
        t_stud = func.QuantileTStudent(1 - self.trust / 2, N - 2)

        if not (abs(t_a) > t_stud or abs(t_b) > t_stud):
            print("не значущі проміжки")

        self.det_kvaz_a = t_stud * S_a
        self.det_kvaz_b = t_stud * S_b

    def kvaziTolerantIntervals(self, f):
        N = len(self)
        sigma_eps = math.log(self.y.Sigma) * (
            (1 - self.r ** 2) * (N - 1) / (N - 2)) ** 0.5

        def less_f(x): return f(x) - func.QuantileTStudent(
            1 - self.trust / 2, N - 2) * sigma_eps

        def more_f(x): return f(x) + func.QuantileTStudent(
            1 - self.trust / 2, N - 2) * sigma_eps
        return less_f, more_f

    def kvaziTrustIntervals(self, f):
        N = len(self)
        xl = self.x.raw
        y = [math.log(yi) for yi in self.y.raw]
        x_ = self.x.x_

        S_zal = sum([(y[i] - f(xl[i])) ** 2 for i in range(N)]
                    ) / (N - 2)
        S_b = S_zal ** 0.5 / (self.x.S * (N - 1) ** 0.5)
        def S_y_(x): return (self.sigma_eps ** 2 / N +
                             S_b ** 2 * (x - self.x.x_) ** 2) ** 0.5
        t = func.QuantileTStudent(1 - self.trust / 2, N - 2)

        def tr_lf(x): return f(x) - t * S_y_(x)

        def tr_mf(x): return f(x) + t * S_y_(x)

        def S_y_x0(x): return (self.sigma_eps ** 2 * (
            1 + 1 / N) + S_b ** 2 * (x - x_)) ** 0.5

        def tr_f_lf(x): return f(x) - t * S_y_x0(x)

        def tr_f_mf(x): return f(x) + t * S_y_x0(x)

        return tr_lf, tr_mf, tr_f_lf, tr_f_mf

    def coefficientOfDetermination(self, f):
        S_zal = 0
        N = len(self)
        x = self.x.raw
        y = self.y.raw
        for i in range(N):
            S_zal += (y[i] - f(x[i])) ** 2
        S_zal /= N - 2

        self.R_2 = (1 - S_zal / self.y.Sigma) * 100
        print(f"{self.R_2} {self.r * 100}")

    def to_create_polynomial_regression(self, degree):
        model = PolynomialRegressionModel(degree)
        Y = self.y.raw
        X = self.x.raw
        model.fit(X, Y)

        def f(X):
            return model.predict(X)

        def none(X):
            return None

        ret = [f] * 7

        return ret
