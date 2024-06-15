import math
import numpy as np
import Datanalysis.functions as func
from Datanalysis.SamplingData import SamplingData
from Datanalysis.DoubleSampleRegression import DoubleSampleRegression
from Datanalysis.SamplesTools import timer, toCalcRankSeries, calculate_m

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def printHistogram(hist: np.ndarray, N):
    print(np.array([[hist[i, j] / N for j in range(hist.shape[1])]
                    for i in range(hist.shape[0] - 1, -1, -1)]))


def binaryFind(ranking_array, v):
    left = 0
    right = len(ranking_array) - 1
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


class DoubleSampleData(DoubleSampleRegression):
    def __init__(self, x: SamplingData, y: SamplingData, trust: float = 0.05):
        if len(x.raw) != len(y.raw):
            raise Exception('X and Y are independet samples')
        self.x = x
        self.y = y
        self.trust = trust

    def __len__(self):
        return len(self.x.raw)

    def set_trust(self, trust: float):
        self.trust = trust

    @timer
    def toCalculateCharacteristic(self):
        self.x_ = (self.x.x_, self.y.x_)
        self.pearson_correlation_coefficient()
        self.coeficientOfCorrelation()
        try:
            self.rangeCorrelation()
        except ZeroDivisionError:
            print('Error in range correlation')
        self.coefficientsOfCombinationsOfTables()
        self.linearCorrelationParametrs()

    # pair correlation coef
    def pearson_correlation_coefficient(self):
        N = len(self)

        xy_ = (self.x.raw @ self.y.raw) / N

        if self.x.Sigma == 0 or self.y.Sigma == 0:
            self.r = np.nan
        else:
            self.r = N / (N - 1) * (xy_ - self.x.x_ * self.y.x_) / (
                self.x.Sigma * self.y.Sigma)
        if abs(self.r) > 1.0:
            raise Exception('Error in pearson correlation coefficient')

        self.r_signif = self.r * (N - 2) ** 0.5 / (
            1 - self.r ** 2) ** 0.5

        self.r_det_v = self.r + self.r * (1 - self.r ** 2) / (2 * N)
        self.det_r = func.QuantileNorm(1 - self.trust / 2
                                       ) * (1 - self.r ** 2) / (N - 1) ** 0.5
        return self.r

    def generateMas3Dot3(self, k):
        y = [[] for _ in range(k + 1)]
        N = len(self)
        dx = (self.x.max - self.x.min) / k
        xy = np.c_[self.x.raw, self.y.raw]
        xy = xy[xy[:, 0].argsort()]

        i = 1
        sub_y_0 = 0
        for j in range(N):
            if not (self.x.min + (i-1) * dx <= xy[j, 0]
                    <= self.x.min + i * dx):
                y[i] = xy[sub_y_0:j - 1, 1]
                i += 1
                sub_y_0 = j
        y.append(xy[sub_y_0:, 1])
        return y

    def coeficientOfCorrelation(self):
        N = len(self)
        k = N // 2
        y = self.generateMas3Dot3(k)
        def m(i): return len(y[i])
        def _y_(i): return sum(y[i]) / m(i) if m(i) != 0 else 0
        y_ = self.y.x_
        self.po_2 = 0.0
        po_2_second_div = 0.0
        for i in range(k):
            self.po_2 += m(i) * (_y_(i) - y_) ** 2
        for i in range(k):
            for j in range(m(i)):
                po_2_second_div += (y[i][j] - y_) ** 2
        self.po_2 /= po_2_second_div
        self.po_2 = min(self.po_2, 1.0)
        self.po_k = k

        if self.po_2 == 1.0:
            self.det_less_po = math.inf
            self.det_more_po = math.inf
            self.po_signif_t = math.inf
            self.po_signif_f = math.inf
            return
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
        self.ind_Q = (N[0][0] * N[1][1] -
                      N[0][1] * N[1][0]) / (
                      N[0][0] * N[1][1] +
                      N[0][1] * N[1][0])
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

    @timer
    def rangeCorrelation(self):
        r = self.get_rank_series()

        self.teta_c, self.teta_c_signif, self.det_teta_c = \
            self.spearmans_rank_correlation_coefficient(r)

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

        self.teta_k, self.teta_k_signif, self.det_teta_k = \
            self.kendalls_rank_coefficient(r)

    def get_rank_series(self):
        N = len(self)
        x_raw = self.x.raw
        y_raw = self.y.raw
        x = [[i, 0] for i in x_raw]
        y = [[i, 0] for i in y_raw]
        x.sort(key=lambda k: k[0])
        y.sort(key=lambda k: k[0])

        toCalcRankSeries(x)
        toCalcRankSeries(y)

        r = []
        for i in range(N):
            rxi = binaryFind(x, x_raw[i])
            ryi = binaryFind(y, y_raw[i])
            r.append((x[rxi][1], y[ryi][1]))
        return r

    def spearmans_rank_correlation_coefficient(self, r):
        N = len(self)

        def rx(i): return r[i][0]
        def ry(i): return r[i][1]
        def d(i): return rx(i) - ry(i)

        teta_c_mn = 0.0
        for i in range(N):
            teta_c_mn += d(i) ** 2
        teta_c = 1 - 6 / (N * (N ** 2 - 1)) * teta_c_mn

        if abs(teta_c) != 1.0:
            teta_c_signif = teta_c * (N - 2) ** 0.5 / (
                1 - teta_c ** 2) ** 0.5
        else:
            teta_c_signif = np.nan
            teta_c = np.nan

        sigma_teta_c = ((1 - teta_c ** 2) / (N - 2)) ** 0.5
        det_teta_c = func.QuantileTStudent(1 - self.trust / 2, N - 2
                                           ) * sigma_teta_c

        return teta_c, teta_c_signif, det_teta_c

    def kendalls_rank_coefficient(self, r):
        N = len(self)

        S = 0.0
        for i in range(N - 1):
            for j in range(i + 1, N):
                ryi = r[i][1]
                ryj = r[j][1]
                if ryi < ryj:
                    S += 1
                elif ryi > ryj:
                    S -= 1
        teta_k = 2 * S / (N * (N - 1))
        teta_k_signif = 3 * teta_k * (N * (N - 1)) ** 0.5 / (
            2 * (2 * N + 5)) ** 0.5
        sigma_teta_k = ((4 * N + 10) / (9 * (N ** 2 - N))) ** 0.5
        det_teta_k = func.QuantileNorm(1 - self.trust / 2) * sigma_teta_k
        return teta_k, teta_k_signif, det_teta_k

    def linearCorrelationParametrs(self):
        N = len(self)
        self.sigma_eps = self.y.Sigma * (
            (1 - self.r ** 2) * (N - 1) / (N - 2)) ** 0.5
        self.line_f_signif = self.sigma_eps ** 2 / self.y.Sigma ** 2

    def get_histogram_data(self, column_number: int = 0):
        if column_number <= 0:
            column_number = calculate_m(len(self.x.raw))
            xn_ = len(self.x._x)
            yn_ = len(self.y._x)
            column_number = min(column_number, xn_, yn_)
        self.probability_table = np.zeros((column_number, column_number),
                                          dtype=int)
        N = len(self)
        h_x = (self.x.max - self.x.min) / column_number
        h_y = (self.y.max - self.y.min) / column_number
        x = self.x.raw
        y = self.y.raw

        for i in range(N):
            c = math.floor((x[i] - self.x.min) / h_x)
            if c == column_number:
                c -= 1
            r = math.floor((y[i] - self.y.min) / h_y)
            if r == column_number:
                r -= 1
            self.probability_table[r, c] += 1
        # printHistogram(self.probability_table, N)
        return self.probability_table

    def remove(self, x, y, w, h):
        self.fastRemove(x, y, w, h)

        self.x.setSeries(self.x.raw)
        self.y.setSeries(self.y.raw)

    def fastRemove(self, x, y, w, h):
        N = len(self)
        del_ind = []
        raw_x = list(self.x.raw)
        raw_y = list(self.y.raw)
        for i in range(N):
            if (x <= self.x.raw[i] <= x + w) and\
               (y <= self.y.raw[i] <= y + h):
                del_ind.append(i)

        for i in del_ind[::-1]:
            raw_x.pop(i)
            raw_y.pop(i)
        self.x.raw = raw_x
        self.y.raw = raw_y

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

        if is_item_del:
            self.x.setSeries(self.x.raw)
            self.y.setSeries(self.y.raw)

        return is_item_del

    def toIndependet(self):
        phi = math.atan(2 * self.r * self.x.Sigma * self.y.Sigma /
                        (self.x.S - self.y.S)) / 2
        eps = self.x.raw
        eta = self.y.raw
        new_eps = eps * math.cos(phi) + eta * math.sin(phi)
        new_eta = -eps * math.sin(phi) + eta * math.cos(phi)
        self.x.setSeries(new_eps)
        self.y.setSeries(new_eta)
        self.toCalculateCharacteristic()

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

        quant_p = func.QuantilePearson(1 - trust / 2, k - 1)
        logger.debug(f"дисперсії Бартлета: {X} <= {quant_p}")
        return X <= quant_p

    def xiXiTest(self, hist_data) -> tuple[bool, str]:
        f = self.toCreateNormalFunc()
        N = len(self)
        column_number = len(hist_data)
        h_x = (self.x.max - self.x.min) / column_number
        h_y = (self.y.max - self.y.min) / column_number

        def p(i, j): return f(h_x * (2 * i + 1) / 2,
                              h_y * (2 * j + 1) / 2) * h_x * h_y

        x_2 = 0.0
        for i in range(column_number):
            for j in range(column_number):
                if p(i, j) != 0:
                    x_2 += (hist_data[i][j] / N - p(i, j)) ** 2 / p(i, j)

        quant = func.QuantilePearson(1 - self.trust, column_number ** 2 - 2)
        self.xixitest_x2 = x_2
        self.xixitest_quant = quant
        return x_2 <= quant
