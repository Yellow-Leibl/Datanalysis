import math

import numpy as np

import functions as func
from Datanalysis.SamplingData import (SamplingData, formRow3V, formRow4V,
                                      toMakeRange, calc_reproduction_dx)


def printHistogram(hist, N):
    hist = [[j / N for j in i] for i in hist]
    hist = np.array(hist[::-1])
    print(hist)


class DoubleSampleData(SamplingData):
    def __init__(self, x: SamplingData, y: SamplingData, trust: float = 0.05):
        if len(x.getRaw()) != len(y.getRaw()):
            raise Exception()
        self.x = x
        self.y = y

        self.trust = trust

        self.probability_table = []

    def __len__(self):
        return len(self.x.getRaw())

    def toCalculateCharacteristic(self):
        self.x_ = (self.x.x_, self.y.x_)

        N = len(self.x.getRaw())
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
        self.rangeCorrelation()
        self.coefficientsOfCombinationsOfTables()

    def coeficientOfCorrelation(self):
        N = len(self.x.getRaw())
        k = SamplingData.calculateM(N)
        dx = calc_reproduction_dx(self.x.min, self.x.max, k)
        def x(i): return self.x.min + (i - 0.5) * dx
        y = []
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
        N = len(self.x.getRaw())
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
        print(f"{self.teta_c_signif} <= "
              f"{func.QuantileTStudent(1 - self.trust, N - 2)}")

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

    def getProtocol(self) -> str:
        info = []
        info.append("-" * 44 + "ПРОТОКОЛ" + "-" * 44 + "\n")
        info.append(formRow4V('Характеристика',
                              'INF', 'Значення', 'SUP', 'SKV') + "\n")

        info.append(formRow4V("Сер арифметичне X",
                              f"{self.x.x_-self.x.det_x_:.5}",
                              f"{self.x.x_:.5}",
                              f"{self.x.x_+self.x.det_x_:.5}",
                              f"{self.x.det_x_:.5}"))

        info.append(formRow4V("Сер арифметичне Y",
                              f"{self.y.x_-self.y.det_x_:.5}",
                              f"{self.y.x_:.5}",
                              f"{self.y.x_+self.y.det_x_:.5}",
                              f"{self.y.det_x_:.5}"))

        info.append("")

        info.append(formRow4V("Коефіціент кореляції",
                              f"{self.r_det_v - self.det_r:.5}",
                              f"{self.r:.5}",
                              f"{self.r_det_v + self.det_r:.5}",
                              f"{self.det_r:.5}"))

        info.append(formRow4V("Коеф кореляційного відношення p",
                              f"{self.det_less_po:.5}",
                              f"{self.po_2:.5}",
                              f"{self.det_more_po:.5}", ""))

        info.append("")

        info.append(formRow4V("Ранговий коеф кореляції Спірмена",
                              f"{self.teta_c - self.det_teta_c:.5}",
                              f"{self.teta_c:.5}",
                              f"{self.teta_c + self.det_teta_c:.5}",
                              f"{self.det_teta_c:.5}"))

        info.append(formRow4V("Ранговий коефіцієнт Кендалла",
                              f"{self.teta_k - self.det_teta_k:.5}",
                              f"{self.teta_k:.5}",
                              f"{self.teta_k + self.det_teta_k:.5}",
                              f"{self.det_teta_k:.5}"))

        info.append(formRow4V("Коефіцієнт сполучень Пірсона",
                              "", f"{self.C_Pearson:.5}", "", ""))

        info.append(formRow4V("Міра звʼязку Кендалла",
                              f"{self.teta_b - self.det_teta_b:.5}",
                              f"{self.teta_b:.5}",
                              f"{self.teta_b + self.det_teta_b:.5}",
                              f"{self.det_teta_b:.5}"))

        info.append("")

        info.append(formRow3V("Індекс Фехнера", "",
                              f"{self.ind_F:.5}", ""))
        info.append(formRow3V("Індекс сполучень", "",
                              f"{self.ind_Fi:.5}", ""))

        info.append("")

        N = len(self.x)
        nu = N - 2
        alpha = 1 - self.trust
        info.append(
            formRow3V("Значимість коефіцієнта кореліції",
                      f"{self.r_signif:.5}",
                      "<=",
                      f"{func.QuantileTStudent(alpha, nu):.5}"))
        info.append(
            formRow3V("Значимість коефіцієнта p, t-test",
                      f"{abs(self.po_signif_t):.5}",
                      "<=",
                      f"{func.QuantileTStudent(1 - self.trust, nu):.5}"))

        f_po = func.QuantileFisher(1 - self.trust,
                                   self.po_k - 1,
                                   N - self.po_k)
        info.append(
            formRow3V("Значимість коефіцієнта p, f-test",
                      f"{self.po_signif_f:.5}",
                      "<=",
                      f"{f_po:.5}"))
        info.append(formRow3V("Значимість коефіцієнта Фі",
                              f"{self.ind_F_signif:.5}",
                              ">=",
                              f"{func.QuantilePearson(1 - self.trust, 1):.5}"))
        info.append(formRow3V("Значимість коефіцієнта Юла Q",
                              f"{abs(self.ind_Q_signif):.5}",
                              "<=",
                              f"{func.QuantileNorm(1 - self.trust / 2):.5}"))
        info.append(formRow3V("Значимість коефіцієнта Юла Y",
                              f"{abs(self.ind_Y_signif):.5}",
                              "<=",
                              f"{func.QuantileNorm(1 - self.trust / 2):.5}"))
        info.append(
            formRow3V("Значимість коефіцієнта Спірмена",
                      f"{abs(self.teta_c_signif):.5}",
                      "<=",
                      f"{func.QuantileTStudent(1 - self.trust / 2, nu):.5}"))
        info.append(
            formRow3V("Значим рангового коеф Кендалла",
                      f"{abs(self.teta_k_signif):.5}",
                      "<=",
                      f"{func.QuantileNorm(1 - self.trust / 2):.5}"))
        info.append(
            formRow3V("Значимість коефіцієнта Пірсона",
                      f"{self.C_Pearson_signif:.5}",
                      "<=",
                      f"{func.QuantilePearson(1 - self.trust, 10):.5}"))
        info.append(
            formRow3V("Значимість коеф Кендалла",
                      f"{abs(self.teta_b_signif):.5}",
                      "<=",
                      f"{func.QuantileNorm(1 - self.trust / 2):.5}"))

        return "\n".join(info)

    def setTrust(self, trust: float):
        self.trust = trust

    def get_histogram_data(self, column_number: int = 0) -> list:
        if column_number <= 0:
            column_number = SamplingData.calculateM(len(self.x))

        self.probability_table = [[0 for x in range(column_number)]
                                  for y in range(column_number)]

        N = len(self.x.getRaw())
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
        N = len(self.x.getRaw())
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

        N = len(self.x.getRaw())
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

    def toCreateLineFunc(self):
        if not (self.x.isNormal() and self.y.isNormal() and
                self.identDispersionBarlet([self.x, self.y], self.trust) and
                self.x.critetionAbbe() and self.y.critetionAbbe()):
            print("The initial conditions arent correct for linear regression")
            return

        b = self.r * self.y.Sigma / self.x.Sigma
        a = self.y.x_ - b * self.x.x_

        def f(x): return a + b * x
        return f

    def xiXiTest(self, hist_data: list) -> bool:
        f = self.toCreateNormalFunc()
        x_2 = 0
        N = len(self.x.getRaw())
        column_number = len(hist_data)
        h_x = (self.x.max - self.x.min) / column_number
        h_y = (self.y.max - self.y.min) / column_number

        def p(i, j): return f(h_x * (2 * i + 1) / 2,
                              h_y * (2 * j + 1) / 2) * h_x * h_y

        for i in range(column_number):
            for j in range(column_number):
                if p(i, j) != 0:
                    x_2 += (hist_data[i][j] / N - p(i, j)) ** 2 / p(i, j)

        print(f"x_2={x_2} <="
              f"{func.QuantilePearson(self.trust, column_number ** 2 - 2)}")
        return x_2 <= func.QuantilePearson(
            1 - self.trust, column_number ** 2 - 2)
