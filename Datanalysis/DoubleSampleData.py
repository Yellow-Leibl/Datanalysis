import math

import numpy as np

import functions as func
from Datanalysis.SamplingData import (SamplingData, formRow3V, formRow4V,
                                      toMakeRange, calc_reproduction_dx, MED)


def printHistogram(hist, N):
    hist = [[j / N for j in i] for i in hist]
    hist = np.array(hist[::-1])
    print(hist)


class DoubleSampleData(SamplingData):
    def __init__(self, x: SamplingData, y: SamplingData, trust: float = 0.05):
        if len(x.getRaw()) != len(y.getRaw()):
            raise Exception('X and Y are independet samples')
        self.x = x
        self.y = y
        self.trust = trust
        self.probability_table = []

    def __len__(self):
        return len(self.x.getRaw())

    def toCalculateCharacteristic(self):
        self.x_ = (self.x.x_, self.y.x_)

        N = len(self)
        xy_ = sum([self.x.raw_x[i] * self.y.raw_x[i]
                   for i in range(len(self.x.raw_x))]) / N

        self.r = N / (N - 1) * (xy_ - self.x.x_ * self.y.x_) / (
            self.x.Sigma * self.y.Sigma)

        self.r_signif = self.r * (N - 2) ** 0.5 / (
            1 - self.r ** 2) ** 0.5

        self.r_det_v = self.r + self.r * (1 - self.r ** 2) / (2 * N)
        self.det_r = func.QuantileNorm(1 - self.trust / 2
                                       ) * (1 - self.r ** 2) / (N - 1) ** 0.5

        self.coeficientOfCorrelation()
        try:
            self.rangeCorrelation()
        except ZeroDivisionError:
            print('Error in range correlation')
        self.coefficientsOfCombinationsOfTables()
        self.linearCorrelationParametrs()

    def generateMas3Dot3(self, k):
        y = []
        N = len(self)
        dx = calc_reproduction_dx(self.x.min, self.x.max, k)
        def x(i): return self.x.min + (i - 0.5) * dx
        xy = [(self.x.getRaw()[i], self.y.getRaw()[i]) for i in range(N)]
        for i in range(1, k + 1):
            yi = []
            rm_i = 0
            for j in range(len(xy)):
                if x(i) - 0.5 * dx <= xy[j - rm_i][0] <= x(i) + 0.5 * dx:
                    yi.append(xy[j - rm_i][1])
                    xy.pop(j - rm_i)
                    rm_i += 1
            y.append(yi)
        return y

    def coeficientOfCorrelation(self):
        N = len(self)
        k = N // 2
        y = self.generateMas3Dot3(k)
        def m(i): return len(y[i])
        def _y_(i): return sum(y[i]) / m(i) if m(i) != 0 else 0
        y_ = self.y.x_
        self.po_2 = sum([m(i) * (_y_(i) - y_) ** 2 for i in range(k)]) / sum(
            [sum([(y[i][j] - y_) ** 2 for j in range(m(i))])
             for i in range(k)])

        nu1 = round((k - 1 + N * self.po_2) ** 2 /
                    (k - 1 + 2 * N * self.po_2))
        nu2 = N - k
        self.det_less_po = (N - k) * self.po_2 / (
            N * (1 - self.po_2) * func.QuantileFisher(1 - self.trust, nu1, nu2)
            ) - (k - 1) / N
        self.det_more_po = (N - k) * self.po_2 / (
            N * (1 - self.po_2) * func.QuantileFisher(self.trust, nu1, nu2)
            ) - (k - 1) / N
        self.po_signif_t = self.po_2 ** 0.5 * (N - 2) ** 0.5 / (
            1 - self.po_2) ** 0.5
        self.po_signif_f = self.po_2 / (1 - self.po_2) * (N - k) / (k - 1)
        self.po_k = k

    def coefficientsOfCombinationsOfTables(self):
        N = self.get_histogram_data(2)
        self.ind_F = (N[0][0] + N[1][1] - N[1][0] - N[0][1]) / (
             N[0][0] + N[1][1] + N[1][0] + N[0][1])

        N0 = N[0][0] + N[0][1]
        N1 = N[1][1] + N[1][0]
        M0 = N[0][0] + N[1][0]
        M1 = N[0][1] + N[1][1]
        self.ind_Fi = (N[0][0] * N[1][1] - N[0][1] * N[1][0]) / (
            N0 * N1 * M0 * M1) ** 0.5
        N_g = N0 + N1

        if len(self.x) < 40:
            self.ind_F_signif = (N[0][0] * N[1][1] - N[0][1] * N[1][0] - 0.5
                                 ) ** 2 / (N0 * N1 * M0 * M1)
        else:
            self.ind_F_signif = N_g * self.ind_F ** 2

        self.ind_Y = ((N[0][0] * N[1][1]) ** 0.5 -
                      (N[0][1] * N[1][0]) ** 0.5) / (
                     (N[0][0] * N[1][1]) ** 0.5 +
                     (N[0][1] * N[1][0]) ** 0.5)
        self.ind_Q = 2 * self.ind_Y / (1 + self.ind_Y) ** 2
        if N[0][0] == 0 or N[1][0] == 0 or N[0][1] == 0 or N[1][1] == 0:
            S_Q = math.inf
            S_Y = math.inf
        else:
            S_Q = 1 / 2 * (1 - self.ind_Q ** 2) * (
                1 / N[0][0] + 1 / N[0][1] + 1 / N[1][0] + 1 / N[1][1])
            S_Y = 1 / 2 * (1 - self.ind_Y ** 2) * (
                1 / N[0][0] + 1 / N[0][1] + 1 / N[1][0] + 1 / N[1][1])

        self.ind_Q_signif = self.ind_Q / S_Q
        self.ind_Y_signif = self.ind_Y / S_Y

        m_g = 9
        n_g = m_g
        nn = self.get_histogram_data(n_g)
        def n(i): return sum(nn[i])
        def m(j): return sum([nn[i][j] for i in range(m_g)])
        N_g = sum([n(i) for i in range(m_g)])

        def N(i, j): return n(i) * m(j) / N_g

        X_2 = 0
        for i in range(n_g):
            for j in range(m_g):
                if N(i, j) != 0:
                    X_2 += (nn[i][j] - N(i, j)) ** 2 / N(i, j)

        self.C_Pearson = (X_2 / (N_g + X_2)) ** 0.5
        self.C_Pearson_signif = X_2

        P = 0
        Q = 0
        T1 = 0
        T2 = 0
        for i in range(n_g):
            T1 += n(i) * (n(i) - 1)
            T2 += m(i) * (m(i) - 1)
            for j in range(m_g):
                P += nn[i][j] * sum(
                    [sum([nn[k][ll] for ll in range(j + 1, m_g)])
                        for k in range(i + 1, n_g)])
                Q += nn[i][j] * sum(
                    [sum([nn[k][ll] for ll in range(j - 1)])
                        for k in range(i + 1, n_g)])
        T1 /= 2
        T2 /= 2

        self.teta_b = math.inf
        self.teta_b_signif = math.inf
        self.det_teta_b = math.inf

        if n_g == m_g:
            self.teta_b = (P - Q) / (
                (1 / 2 * N_g * (N_g - 1) - T1) *
                1 / 2 * N_g * (N_g - 1) - T2) ** 0.5

            self.teta_b_signif = 3 * self.teta_b * (N_g * (N_g - 1)) ** 0.5 / (
                2 * (2 * N_g + 5)) ** 0.5
            sigma_teta_b = ((4 * N_g + 10) / (9 * (N_g ** 2 - N_g))) ** 0.5
            self.det_teta_b = func.QuantileNorm(1 - self.trust / 2
                                                ) * sigma_teta_b
        else:
            self.teta_st = 2 * (P - Q) * min(m_g, n_g) / (
                N_g ** 2 * (min(m_g, n_g) - 1))

            def A(i, j):
                return sum([sum([nn[k][ll] for ll in range(j + 1, m_g)])
                            for k in range(i + 1, n_g)]) + sum(
                                [sum([nn[k][ll] for ll in range(j - 1)])
                                 for k in range(i - 1)])

            def B(i, j):
                return sum([sum([nn[k][ll] for ll in range(j - 1)])
                            for k in range(i + 1, n_g)]) + sum(
                                [sum([nn[k][ll] for ll in range(j + 1, m_g)])
                                 for k in range(i - 1)])

            sigma_teta_st = 2 * min(m_g, n_g) / (
                N_g ** 3 * (min(m_g, n_g) - 1)) * (
                N_g ** 2 * sum(
                    [sum([nn[i][j] * (A(i, j) - B(i, j)) ** 2
                          for j in range(m_g)]) for i in range(n_g)]) -
                4 * N_g * (P - Q)) ** 0.5

            print(f"{sigma_teta_st} <= "
                  f"{func.QuantileTStudent(1 - self.trust, n_g * m_g - 2)}")

    def rangeCorrelation(self):
        self.teta_c = 0.0
        self.det_teta_c = 0.0
        self.teta_c_signif = 0.0
        self.teta_k = 0.0
        self.teta_k_signif = 0.0
        self.det_teta_k = 0.0

        N = len(self)
        x = [[i, 0] for i in self.x.getRaw()]
        y = [[i, 0] for i in self.y.getRaw()]
        x.sort()
        y.sort()
        toMakeRange(x)
        toMakeRange(y)

        def binaryFind(ranking_array, v):
            left = 0
            right = N - 1
            if ranking_array[left][0] == v:
                return left
            if ranking_array[right][0] == v:
                return right
            while left != right:
                m = (left + right) // 2
                if ranking_array[m][0] == v:
                    return m
                elif ranking_array[m][0] > v:
                    right = m
                else:
                    left = m
            return -1

        r = []
        for i in range(N):
            rxi = binaryFind(x, self.x.getRaw()[i])
            ryi = binaryFind(y, self.y.getRaw()[i])
            r.append([x[rxi][1], y[ryi][1]])

        def rx(i): return r[i][0]
        def ry(i): return r[i][1]
        def d(i): return rx(i) - ry(i)

        self.teta_c = 1 - 6 / (N * (N ** 2 - 1)) * sum(
            [d(i) ** 2 for i in range(N)])

        # A = 0
        # prev = x[0][0] - 1
        # Aj = 0
        # for i in x:
        #     if prev == i[0]:
        #         Aj += 1
        #     else:
        #         prev = i[0]
        #         A += Aj ** 3 - Aj
        #         Aj = 0
        # A /= 12
        # B = 0
        # prev = y[0][0] - 1
        # Bj = 0
        # for i in x:
        #     if prev == i[0]:
        #         Bj += 1
        #     else:
        #         prev = i[0]
        #         B += Bj ** 3 - Bj
        #         Bj = 0
        # B /= 12

        # sperman = (1 / 6 * N * (N ** 2 - 1) -
        #            sum([d(i) ** 2 for i in range(N)]) - A - B) / (
        #                (1 / 6 * N * (N ** 2 - 1) - 2 * A) *
        #                (1 / 6 * N * (N ** 2 - 1) - 2 * B)) ** 0.5
        # print(f"Sperman={sperman}")

        self.teta_c_signif = self.teta_c * (N - 2) ** 0.5 / (
            1 - self.teta_c ** 2) ** 0.5

        sigma_teta_c = ((1 - self.teta_c ** 2) / (N - 2)) ** 0.5
        self.det_teta_c = func.QuantileTStudent(1 - self.trust / 2, N - 2
                                                ) * sigma_teta_c

        def v(i, j):
            return 1 if ry(i) < ry(j) else (-1 if ry(i) > ry(j) else 0)
        S = sum([sum([v(i, j) for j in range(N)]) for i in range(N - 1)])
        self.teta_k = 2 * S / (N * (N - 1))
        self.teta_k_signif = 3 * self.teta_k * (N * (N - 1)) ** 0.5 / (
            2 * (2 * N + 5)) ** 0.5
        sigma_teta_k = ((4 * N + 10) / (9 * (N ** 2 - N))) ** 0.5
        self.det_teta_k = func.QuantileNorm(1 - self.trust / 2) * sigma_teta_k

    def linearCorrelationParametrs(self):
        N = len(self)
        self.sigma_eps = self.y.Sigma * (
            (1 - self.r ** 2) * (N - 1) / (N - 2)) ** 0.5
        self.line_f_signif = self.sigma_eps ** 2 / self.y.Sigma ** 2

    def setTrust(self, trust: float):
        self.trust = trust

    def get_histogram_data(self, column_number: int = 0) -> list:
        if column_number <= 0:
            column_number = SamplingData.calculateM(len(self.x))

        self.probability_table = [[0 for x in range(column_number)]
                                  for y in range(column_number)]

        N = len(self)
        h_x = (self.x.max - self.x.min) / column_number
        h_y = (self.y.max - self.y.min) / column_number

        def point(i): return (self.x.getRaw()[i], self.y.getRaw()[i])

        for i in range(N):
            p = point(i)
            s_x = self.x.min
            s_y = self.y.min
            c = 0
            r = 0
            while not s_x <= p[0] <= s_x + h_x and c + 1 < column_number:
                c += 1
                s_x += h_x
            while not s_y <= p[1] <= s_y + h_y and r + 1 < column_number:
                r += 1
                s_y += h_y
            self.probability_table[r][c] += 1

        printHistogram(self.probability_table, N)
        return self.probability_table

    def remove(self, x, y, w, h):
        self.fastRemove(x, y, w, h)

        self.x.setSeries(self.x.raw_x)
        self.y.setSeries(self.y.raw_x)

    def fastRemove(self, x, y, w, h):
        N = len(self)
        del_ind = []
        for i in range(N):
            if (x <= self.x.raw_x[i] <= x + w) and\
               (y <= self.y.raw_x[i] <= y + h):
                del_ind.append(i)

        for i in range(len(del_ind)):
            self.x.raw_x.pop(del_ind[-1 - i])
            self.y.raw_x.pop(del_ind[-1 - i])

    def autoRemoveAnomaly(self, hist_data):
        is_item_del = False

        N = len(self)
        column_number = len(hist_data)
        h_x = (self.x.max - self.x.min) / column_number
        h_y = (self.y.max - self.y.min) / column_number

        for i in range(column_number):
            for j in range(column_number):
                n = hist_data[i][j]
                p = n / N
                if p <= self.trust and n != 0:
                    self.fastRemove(self.x.min + i * h_x,
                                    self.y.min + j * h_y,
                                    h_x, h_y,)
                    is_item_del = True

        self.x.setSeries(self.x.raw_x)
        self.y.setSeries(self.y.raw_x)

        return is_item_del

    def toCreateNormalFunc(self):
        x_ = self.x.x_
        y_ = self.y.x_
        sigma_x = self.x.Sigma
        sigma_y = self.y.Sigma

        def f(x, y): return 1 / (2 * math.pi * sigma_x * sigma_y) * math.exp(
            - 1 / 2 * (((x - x_) / sigma_x) ** 2 + ((y - y_) / sigma_y) ** 2))

        return f

    def identDispersionBarlet(self, samples, trust: float = 0.05) -> bool:
        k = len(samples)

        def N(i): return len(samples[i])
        def S(i): return samples[i].S

        S_2 = sum([(N(i) - 1) * S(i) for i in range(k)])\
            / sum([N(i) - 1 for i in range(k)])

        B = - sum([(N(i) - 1) * math.log(S(i) / S_2) for i in range(k)])
        C = 1 + 1 / (3 * (k - 1)) * (sum([1 / (N(i) - 1) for i in range(k)])
                                     - 1 / sum([(N(i) - 1) for i in range(k)]))
        X = B / C

        print("дисперсії Бартлета: "
              f"{X} <= {func.QuantilePearson(1 - trust / 2, k - 1)}")
        return X <= func.QuantilePearson(1 - trust / 2, k - 1)

# line regression
    def initialConditionLine(self) -> bool:
        if not (self.x.isNormal() and self.y.isNormal()):
            print("Початкова умова 1 лінійного"
                  " регресійного аналізу не виконується")
            if not (self.identDispersionBarlet([self.x, self.y], self.trust)
                    and self.x.critetionAbbe() and self.y.critetionAbbe()):
                print("The initial conditions arent correct for linear regression")
                return False

    def toCreateLinearRegressionMNK(self):

        b = self.r * self.y.Sigma / self.x.Sigma
        a = self.y.x_ - b * self.x.x_

        self.line_a = a
        self.line_b = b
        self.accuracyLineParameters(a, b)

        def f(x): return a + b * x

        tl_lf, tl_mf = self.linearTolerantIntervals(f)
        tr_lf, tr_mf, tr_f_lf, tr_f_mf = self.linearTrustIntervals(f)

        return tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f

    def toCreateLinearRegressionMethodTeila(self):
        if not (self.x.isNormal() and self.y.isNormal() and
                self.identDispersionBarlet([self.x, self.y], self.trust) and
                self.x.critetionAbbe() > self.trust and
                self.y.critetionAbbe() > self.trust):
            print("The initial conditions arent correct for linear regression")
            return

        x = self.x.x
        y = self.y.x

        N = len(self)
        b_l = [0] * (N * (N - 1) / 2)
        ll = 0
        for i in range(N):
            for j in range(i + 1, N):
                b_l[ll] = (y[j] - y[i]) / (x[j] - x[i])
                ll += 1
        b = MED(b_l)

        a_l = [y[i] - b * x[i] for i in range(N)]
        a = MED(a_l)

        self.line_a = a
        self.line_b = b
        self.accuracyLineParameters(a, b)

        def f(x): return a + b * x

        tl_lf, tl_mf = self.linearTolerantIntervals(f)
        tr_lf, tr_mf, tr_f_lf, tr_f_mf = self.linearTrustIntervals(f)

        return tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f

    def accuracyLineParameters(self, a, b):
        N = len(self)
        x = self.x.getRaw()
        y = self.y.getRaw()
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
        xl = self.x.getRaw()
        y = self.y.getRaw()
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

# parabolic regression
    def toCreateParabolicRegression(self):
        x_ = self.x.x_
        y_ = self.y.x_
        xl = self.x.getRaw()
        y = self.y.getRaw()
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

        tl_lf, tl_mf = self.parabolaTolerantIntervals(f)
        tr_lf, tr_mf, tr_f_lf, tr_f_mf = \
            self.parabolaTrustIntervals(f, phi1, phi2)

        return tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f

    def accuracyParabolaParameters(self, a, b, c, phi2):
        N = len(self)
        xl = self.x.getRaw()
        y = self.y.getRaw()
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
        x = self.x.getRaw()
        y = self.y.getRaw()
        S_zal = (sum([(y[i] - f(x[i])) ** 2 for i in range(N)]
                     ) / (N - 2)) ** 0.5

        def less_f(x): return f(x) - t * S_zal

        def more_f(x): return f(x) + t * S_zal
        return less_f, more_f

    def parabolaTrustIntervals(self, f, phi1, phi2):
        N = len(self)
        t = func.QuantileTStudent(1 - self.trust / 2, N - 3)
        x = self.x.getRaw()
        y = self.y.getRaw()
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

# y = a * exp(b * x)

    def toCreateKvazi8(self):
        N = len(self)
        x = self.x.getRaw()
        y = self.y.getRaw()
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

        tl_lf, tl_mf = self.kvaziTolerantIntervals(f)
        tr_lf, tr_mf, tr_f_lf, tr_f_mf = self.kvaziTrustIntervals(f)

        return tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f

    def accuracyKvaziParameters(self, A, B):
        N = len(self)
        x = self.x.getRaw()
        y = [math.log(yi) for yi in self.y.getRaw()]
        def f(x): return A * math.exp(x * B)

        S_zal = sum([(y[i] - f(x[i])) ** 2 for i in range(N)]
                    ) / (N - 2)

        S_a = S_zal ** 0.5 * (1 / N + self.x.x_ ** 2 /
                              (self.x.S * (N - 1))) ** 0.5
        S_b = S_zal ** 0.5 / (self.x.S * (N - 1) ** 0.5)
        t_a = A / S_a
        t_b = B / S_b
        t_stud = func.QuantileTStudent(1 - self.trust / 2, N - 2)

        # if not (abs(t_a) > t_stud or abs(t_b) > t_stud):
        #     return

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
        xl = self.x.getRaw()
        y = [math.log(yi) for yi in self.y.getRaw()]
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
        x = self.x.getRaw()
        y = self.y.getRaw()
        for i in range(N):
            S_zal += (y[i] - f(x[i])) ** 2
        S_zal /= N - 2

        self.R_2 = (1 - S_zal / self.y.Sigma) * 100
        print(f"{self.R_2} {self.r * 100}")

    def toGenerateReproduction(self, f) -> list:
        x_gen = []
        dx = calc_reproduction_dx(self.x.min, self.x.max)
        x = self.x.min
        while x < self.x.max:
            if f(x) is not None:
                x_gen.append(x)
            x += dx

        if len(x_gen) > 0 and x_gen[-1] != self.x.max:
            x = self.x.max
            if f(x) is not None:
                x_gen.append(x)

        return x_gen

    def xiXiTest(self, hist_data: list) -> bool:
        f = self.toCreateNormalFunc()
        x_2 = 0
        N = len(self)
        column_number = len(hist_data)
        h_x = (self.x.max - self.x.min) / column_number
        h_y = (self.y.max - self.y.min) / column_number

        def p(i, j): return f(h_x * (2 * i + 1) / 2,
                              h_y * (2 * j + 1) / 2) * h_x * h_y

        for i in range(column_number):
            for j in range(column_number):
                if p(i, j) != 0:
                    x_2 += (hist_data[i][j] / N - p(i, j)) ** 2 / p(i, j)

        quant = func.QuantilePearson(1 - self.trust, column_number ** 2 - 2)
        values = f"x_2={x_2:.5} <= {quant:.5}"
        return x_2 <= func.QuantilePearson(
            1 - self.trust, column_number ** 2 - 2), values

    def getProtocol(self) -> str:
        info_protocol = []

        def addIn(text): info_protocol.append(text)
        addIn("-" * 44 + "ПРОТОКОЛ" + "-" * 44 + "\n")
        addIn(formRow4V('Характеристика',
                        'INF', 'Значення', 'SUP', 'SKV') + "\n")

        addIn(formRow4V("Сер арифметичне X",
                        f"{self.x.x_-self.x.det_x_:.5}",
                        f"{self.x.x_:.5}",
                        f"{self.x.x_+self.x.det_x_:.5}",
                        f"{self.x.det_x_:.5}"))

        addIn(formRow4V("Сер арифметичне Y",
                        f"{self.y.x_-self.y.det_x_:.5}",
                        f"{self.y.x_:.5}",
                        f"{self.y.x_+self.y.det_x_:.5}",
                        f"{self.y.det_x_:.5}"))

        addIn("")

        addIn(formRow4V("Коефіціент кореляції",
                        f"{self.r_det_v - self.det_r:.5}",
                        f"{self.r:.5}",
                        f"{self.r_det_v + self.det_r:.5}",
                        f"{self.det_r:.5}"))

        addIn(formRow4V("Коеф кореляційного відношення p",
                        f"{self.det_less_po:.5}",
                        f"{self.po_2:.5}",
                        f"{self.det_more_po:.5}", ""))

        addIn("")

        addIn(formRow4V("Ранговий коеф кореляції Спірмена",
                        f"{self.teta_c - self.det_teta_c:.5}",
                        f"{self.teta_c:.5}",
                        f"{self.teta_c + self.det_teta_c:.5}",
                        f"{self.det_teta_c:.5}"))

        addIn(formRow4V("Ранговий коефіцієнт Кендалла",
                        f"{self.teta_k - self.det_teta_k:.5}",
                        f"{self.teta_k:.5}",
                        f"{self.teta_k + self.det_teta_k:.5}",
                        f"{self.det_teta_k:.5}"))

        addIn(formRow4V("Коефіцієнт сполучень Пірсона",
                        "", f"{self.C_Pearson:.5}", "", ""))

        addIn(formRow4V("Міра звʼязку Кендалла",
                        f"{self.teta_b - self.det_teta_b:.5}",
                        f"{self.teta_b:.5}",
                        f"{self.teta_b + self.det_teta_b:.5}",
                        f"{self.det_teta_b:.5}"))

        addIn("")

        addIn(formRow3V("Індекс Фехнера", "",
                        f"{self.ind_F:.5}", ""))
        addIn(formRow3V("Індекс сполучень", "",
                        f"{self.ind_Fi:.5}", ""))

        addIn(formRow3V("Коефіцієнт зв’язку Юла Q", "",
                        f"{self.ind_Q:.5}", ""))

        addIn(formRow3V("Коефіцієнт зв’язку Юла Y", "",
                        f"{self.ind_Y:.5}", ""))

        addIn("")

        N = len(self.x)
        nu = N - 2
        alpha = 1 - self.trust
        addIn(
            formRow3V("Значимість коефіцієнта кореліції",
                      f"{self.r_signif:.5}",
                      "<=",
                      f"{func.QuantileTStudent(alpha, nu):.5}"))
        addIn(
            formRow3V("Значимість коефіцієнта p, t-test",
                      f"{abs(self.po_signif_t):.5}",
                      "<=",
                      f"{func.QuantileTStudent(1 - self.trust, nu):.5}"))
        try:
            f_po = func.QuantileFisher(1 - self.trust,
                                       self.po_k - 1,
                                       N - self.po_k)
        except:
            f_po = 0.0
        addIn(
            formRow3V("Значимість коефіцієнта p, f-test",
                      f"{self.po_signif_f:.5}",
                      "<=",
                      f"{f_po:.5}"))
        addIn(formRow3V("Значимість коефіцієнта Фі",
                        f"{self.ind_F_signif:.5}",
                        ">=",
                        f"{func.QuantilePearson(1 - self.trust, 1):.5}"))
        addIn(formRow3V("Значимість коефіцієнта Юла Q",
                        f"{abs(self.ind_Q_signif):.5}",
                        "<=",
                        f"{func.QuantileNorm(1 - self.trust / 2):.5}"))
        addIn(formRow3V("Значимість коефіцієнта Юла Y",
                        f"{abs(self.ind_Y_signif):.5}",
                        "<=",
                        f"{func.QuantileNorm(1 - self.trust / 2):.5}"))
        addIn(
            formRow3V("Значимість коефіцієнта Спірмена",
                      f"{abs(self.teta_c_signif):.5}",
                      "<=",
                      f"{func.QuantileTStudent(1 - self.trust / 2, nu):.5}"))
        addIn(
            formRow3V("Значим рангового коеф Кендалла",
                      f"{abs(self.teta_k_signif):.5}",
                      "<=",
                      f"{func.QuantileNorm(1 - self.trust / 2):.5}"))
        addIn(
            formRow3V("Значимість коефіцієнта Пірсона",
                      f"{self.C_Pearson_signif:.5}",
                      "<=",
                      f"{func.QuantilePearson(1 - self.trust, 10):.5}"))
        addIn(
            formRow3V("Значимість коеф Кендалла",
                      f"{abs(self.teta_b_signif):.5}",
                      "<=",
                      f"{func.QuantileNorm(1 - self.trust / 2):.5}"))

        if hasattr(self, "line_a"):
            addIn("")
            addIn("Параметри лінійної регресії: a + bx")
            addIn("-" * 16)
            addIn(
                formRow3V("Параметр a",
                          f"{self.line_a - self.det_line_a:.5}",
                          f"{self.line_a:.5}",
                          f"{self.line_a + self.det_line_a:.5}"))
            addIn(
                formRow3V("Параметр b",
                          f"{self.line_b - self.det_line_b:.5}",
                          f"{self.line_b:.5}",
                          f"{self.line_b + self.det_line_b:.5}"))

            addIn("")

            f = func.QuantileFisher(1 - self.trust, N - 1, N - 3)
            addIn(
                formRow3V("Адекватність відтвореної моделі регресії",
                          f"{self.line_f_signif:.5}",
                          "<=",
                          f"{f:.5}"))

        if hasattr(self, "parab_a"):
            t = func.QuantileTStudent(1 - self.trust / 2, N - 3)
            addIn("")
            addIn("Параметри параболічної регресії: a + bx + cx^2")
            addIn("-" * 16)

            addIn(
                formRow3V("Параметр a",
                          f"{self.parab_a - self.det_parab_a:.5}",
                          f"{self.parab_a:.5}",
                          f"{self.parab_a + self.det_parab_a:.5}"))
            addIn(
                formRow3V("Параметр b",
                          f"{self.parab_b - self.det_parab_b:.5}",
                          f"{self.parab_b:.5}",
                          f"{self.parab_b + self.det_parab_b:.5}"))
            addIn(
                formRow3V("Параметр c",
                          f"{self.parab_c - self.det_parab_c:.5}",
                          f"{self.parab_c:.5}",
                          f"{self.parab_c + self.det_parab_c:.5}"))

            addIn("")
            addIn(formRow3V("Значущість параметра a",
                            f"{self.parab_a_t:.5}", "<=",
                            f"{t:.5}"))
            addIn(formRow3V("Значущість параметра b",
                            f"{self.parab_b_t:.5}", "<=",
                            f"{t:.5}"))
            addIn(formRow3V("Значущість параметра c",
                            f"{self.parab_c_t:.5}", "<=",
                            f"{t:.5}"))

        if hasattr(self, "kvaz_a"):
            t = func.QuantileTStudent(1 - self.trust / 2, N - 3)
            addIn("")
            addIn("Параметри квазілійнійної регресії: a * exp(bx)")
            addIn("-" * 16)

            addIn(
                formRow3V("Параметр a",
                          f"{self.kvaz_a - self.det_kvaz_a:.5}",
                          f"{self.kvaz_a:.5}",
                          f"{self.kvaz_a + self.det_kvaz_a:.5}"))
            addIn(
                formRow3V("Параметр b",
                          f"{self.kvaz_b - self.det_kvaz_b:.5}",
                          f"{self.kvaz_b:.5}",
                          f"{self.kvaz_b + self.det_kvaz_b:.5}"))

        return "\n".join(info_protocol)
