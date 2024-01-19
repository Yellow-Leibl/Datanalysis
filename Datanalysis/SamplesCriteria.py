import math
import numpy as np
import logging

from Datanalysis import SamplingData, DoubleSampleData
from Datanalysis.SamplesTools import toCalcRankSeries
from Datanalysis.functions import (
    QuantileNorm, QuantileTStudent, QuantilePearson, QuantileFisher,
    L)


def identAvrTtestDependent(x: SamplingData, y: SamplingData, trust=0.05):
    N = len(x)
    z = [x[i] - y[i] for i in range(N)]
    z_ = sum(z) / N
    S = sum([(zl - z_) ** 2 for zl in z]) / (N - 1)
    t = z_ * N ** 0.5 / S ** 0.5

    logging.debug("середні залежні: t <= t_nu: "
                  f"{abs(t)} <= {QuantileTStudent(1 - trust / 2, N - 2)}")
    return abs(t) <= QuantileTStudent(1 - trust / 2, N - 2)


def identAvrTtestIndependent(x: SamplingData, y: SamplingData, trust=0.05):
    N1 = len(x)
    N2 = len(y)
    if N1 + N2 <= 25:
        S = ((N1 - 1) * x.S + (N2 - 1) * y.S) / (N1 + N2 - 2) / (
            (N1 * N2) / (N1 + N2))
    else:
        S = x.S / N1 + y.S / N2

    t = (x.x_ - y.x_) / math.sqrt(S)

    quant_t = QuantileTStudent(1 - trust / 2, N1 + N2 - 2)
    logging.debug(f"середні незалежні: t <= t_nu: {abs(t)} <= {quant_t}")
    return abs(t) <= quant_t


def identDispersionFtest(x: SamplingData, y: SamplingData, trust=0.05):
    N1 = len(x)
    N2 = len(y)
    if x.S > y.S:
        f = x.S / y.S
    else:
        f = y.S / x.S

    quant_f = QuantileFisher(1 - trust / 2, N1 - 1, N2 - 1)
    logging.debug(f"дисперсії f-test: {f} <= {quant_f}")
    return f <= quant_f


def identDispersionBarlet(samples, trust=0.05) -> bool:
    k = len(samples)

    def N(i): return len(samples[i])
    def S(i): return samples[i].S

    S_2 = sum([(N(i) - 1) * S(i) for i in range(k)])\
        / sum([N(i) - 1 for i in range(k)])

    B = - sum([(N(i) - 1) * math.log(S(i) / S_2) for i in range(k)])
    C = 1 + 1 / (3 * (k - 1)) * (sum([1 / (N(i) - 1) for i in range(k)])
                                 - 1 / sum([(N(i) - 1) for i in range(k)]))
    X = B / C

    quant_pearson = QuantilePearson(1 - trust / 2, k - 1)
    logging.debug(f"дисперсії Бартлета: {X} <= {quant_pearson}")
    return X <= quant_pearson


def identAvrFtest(samples, trust=0.05) -> bool:
    k = len(samples)

    def N(i): return len(samples[i])
    def S(i): return samples[i].S
    def x_(i): return samples[i].x_

    N_g = sum([N(i) for i in range(k)])
    x_g = sum([N(i) * x_(i) for i in range(k)]) / N_g

    S_M = 1 / (k - 1) * sum([N(i) * (x_(i) - x_g) for i in range(k)])

    S_B = 1 / (N_g - k) * sum([(N(i) - 1) * S(i) for i in range(k)])

    F = S_M / S_B

    quant_f = QuantileFisher(1 - trust / 2, k - 1, N_g - k)
    logging.debug(f"однофакторно дисп аналіз, середні: {F} <= {quant_f}")
    return F <= quant_f


def critetionSmirnovKolmogorov(x: SamplingData, y: SamplingData, trust=0.05):
    _, F, _ = x.toCreateNormalFunc()
    _, G, _ = y.toCreateNormalFunc()

    min_xl = min(x.min, y.min)
    max_xl = max(x.max, y.max)
    xl = np.linspace(min_xl, max_xl, 500, dtype=float)
    z = abs(F(xl[0]) - G(xl[0]))
    for xi in xl:
        zi = abs(F(xi) - G(xi))
        if z < zi:
            z = zi
    N1 = len(x)
    N2 = len(y)
    N = min(N1, N2)

    func_l = L(N ** 0.5 * z, N)
    logging.debug(f"Колмогорова: 1 - L(z) = {1 - func_l} > {trust}")
    return 1 - func_l > trust


def critetionWilcoxon(x: SamplingData, y: SamplingData, trust=0.05):
    r: list = [[i, 0, 'x'] for i in x._x] + [[i, 0, 'y'] for i in y._x]
    r.sort()

    prev = r[0][0]
    r[0][1] = 1
    v = 1
    avr_r = 0.0
    avr_i = 0
    for v in range(1, len(r)):
        if prev == r[v][0]:
            avr_r += v
            avr_i += 1
        else:
            r[v][1] = v + 1
            if avr_r != 0:
                avr_r /= avr_i
                j = v - 1
                while r[j][1] != 0:
                    r[j][1] = avr_r
                    j -= 1
                avr_r = 0
                avr_i = 0
        prev = r[v][0]

    W = 0.0
    for vect in r:
        if vect[2] == 'x':
            W += vect[1]

    N1 = len(x)
    N2 = len(y)
    N = N1 + N2
    E_W = N1 * (N + 1) / 2
    D_W = N1 * N2 * (N + 1) / 12

    w = (W - E_W) / (D_W) ** 0.5

    quant_n = QuantileNorm(1 - trust / 2)
    logging.debug(f"w ={abs(w)} <= {quant_n}")
    return abs(w) <= quant_n


def critetionUtest(x: SamplingData, y: SamplingData, trust=0.05):
    N1 = len(x)
    N2 = len(y)
    N = N1 + N2
    U = 0
    for yj in y._x:
        for xi in x._x:
            if xi > yj:
                U += 1
    E_U = N1 * N2 / 2
    D_U = N1 * N2 * (N + 1) / 12

    u = (U - E_U) / (D_U) ** 0.5

    quant_n = QuantileNorm(1 - trust / 2)
    logging.debug(f"u ={abs(u)} <= {quant_n}")
    return abs(u) <= quant_n


def critetionDiffAvrRanges(x: SamplingData, y: SamplingData, trust=0.05
                           ) -> bool:
    N1 = len(x)
    N2 = len(y)
    N = N1 + N2
    r = [(i, 0, 'x') for i in x._x] + [(i, 0, 'y') for i in y._x]
    r.sort()

    rx = 0.0
    ry = 0.0
    for i in range(N):
        if r[i][2] == 'x':
            rx += r[i][1]
        if r[i][2] == 'y':
            ry += r[i][1]
    rx /= N1
    ry /= N2

    nu = (rx - ry) / (N * ((N + 1) / (12 * N1 * N2)) ** 0.5)

    quant_n = QuantileNorm(1 - trust / 2)
    logging.debug(f"nu ={abs(nu)} <= {quant_n}")
    return abs(nu) <= quant_n


# Критерій Крускала Уоліса
def critetionKruskalaUolisa(samples: list[SamplingData], trust=0.05) -> bool:
    k = len(samples)

    x: list[list] = []
    for i in range(k):
        x += [[j, 0, i] for j in samples[i].raw]
    x.sort()
    N_G = len(x)

    toCalcRankSeries(x)

    def N(i): return len(samples[i])
    W = [0 for i in range(k)]
    for li in x:
        W[li[2]] += li[1]

    for i in range(k):
        W[i] /= N(i)

    H = 0

    E_W = (N_G + 1) / 2
    def D_W(i): return (N_G + 1) * (N_G - N(i)) / (12 * N(i))
    for i in range(k):
        H += (W[i] - E_W) ** 2 / D_W(i) * (1 - N(i) / N_G)

    quant_p = QuantilePearson(1 - trust, k - 1)
    logging.debug(f"H ={H} <= {quant_p}")
    return H <= quant_p


# Критерій знаків
def critetionSign(x_sample: SamplingData, y_sample: SamplingData, trust=0.05):
    N = len(x_sample)
    x = x_sample.raw
    y = y_sample.raw
    S = 0
    for i in range(N):
        if x[i] - y[i] > 0:
            S += 1
    S = (2 * S - 1 - N) / N ** 0.5

    quant_n = QuantileNorm(1 - trust / 2)
    logging.debug(f"S ={abs(S)} <= {quant_n}")
    return abs(S) <= quant_n


# Критерій Кохрена
def critetionKohrena(samples: list[SamplingData], trust=0.05) -> bool:
    k = len(samples)
    N = len(samples[0].raw)
    def u(i): return sum([samples[j].raw[i] for j in range(k)])
    def T(j): return sum(samples[j].raw)
    T_ = sum([T(j) for j in range(k)]) / k
    Q = k * (k - 1) * sum([(T(j) - T_) ** 2 for j in range(k)]) / (
        k * sum([u(i) for i in range(N)]) -
        sum([u(i) ** 2 for i in range(N)]))

    quant_p = QuantilePearson(1 - trust, k - 1)
    logging.debug(f"Q ={Q} <= {quant_p}")
    return Q <= quant_p


class SamplesCriteria:
    def __init__(self) -> None:
        self.samples: list[SamplingData] = []
        self.trust = 0.05

    def ident2Samples(self, row1: int, row2: int, trust: float = 0.05) -> bool:
        x, y = self.samples[row1], self.samples[row2]
        if x.kolmogorov_test(x.toCreateNormalFunc()[1]) and\
           y.kolmogorov_test(y.toCreateNormalFunc()[1]):
            # normal
            if len(x) == len(y):
                return identDispersionBarlet([x, y], trust) \
                    and identAvrTtestDependent(x, y, trust)
            else:
                return identDispersionFtest(x, y, trust) \
                    and identAvrTtestIndependent(x, y, trust) \
                    and critetionWilcoxon(x, y, trust) \
                    and critetionUtest(x, y, trust) \
                    and critetionDiffAvrRanges(x, y, trust)
        else:
            # other
            return critetionSmirnovKolmogorov(x, y, trust) \
                and critetionWilcoxon(x, y, trust) \
                and critetionUtest(x, y, trust) \
                and critetionDiffAvrRanges(x, y, trust) \
                and critetionSign(x, y, trust)

    def identKSamples(self, samples: list[SamplingData], trust: float = 0.05):
        if len(samples[0]) == 2:
            return critetionKohrena(samples, trust)
        isNormal = True
        for i in samples:
            if i.kolmogorov_test(i.toCreateNormalFunc()[1]) is False:
                isNormal = False
                break
        if isNormal:  # normal
            return identDispersionBarlet(samples, trust) \
                and critetionKruskalaUolisa(samples, trust)
        else:
            return critetionKruskalaUolisa(samples, trust)

    def ident2ModelsLine(self, samples: list[SamplingData],
                         trust: float = 0.05):
        d_samples = [DoubleSampleData(samples[i * 2], samples[i * 2 + 1])
                     for i in range(len(samples) // 2)]
        [i.toCalculateCharacteristic() for i in d_samples]

        y1 = d_samples[0]
        y2 = d_samples[1]
        y1.toCreateLinearRegressionMethodTeila()
        y2.toCreateLinearRegressionMethodTeila()

        N1 = len(d_samples[0])
        N2 = len(d_samples[1])
        S1 = y1.sigma_eps ** 2
        S2 = y2.sigma_eps ** 2
        sigma_x1_2 = y1.x.Sigma ** 2
        sigma_x2_2 = y2.x.Sigma ** 2
        b1 = y1.line_b
        b2 = y2.line_b
        x1_ = y1.x.x_
        x2_ = y2.x.x_
        y1_ = y1.y.x_
        y2_ = y2.y.x_
        quant_t = QuantileTStudent(1 - trust / 2, N1 + N2 - 4)
        S = (((N1 - 2) * S1 + (N2 - 2) * S2) / (N1 + N2 - 4)) ** 0.5
        b0 = (y1_ - y2_) / (x1_ - x2_)

        if S1 > S2:
            f = S1 / S2
        else:
            f = S2 / S1

        if f <= QuantileFisher(1 - trust, N1 - 2, N2 - 2):
            t = (b1 - b2) / (
                S * (1 / ((N1 - 1) * sigma_x1_2) +
                     1 / ((N2 - 1) * sigma_x2_2)) ** 0.5)
            if abs(t) <= quant_t:
                b = ((N1 - 1) * sigma_x1_2 * b1 + (N2 - 1) * sigma_x2_2 * b2
                     ) / ((N1 - 1) * sigma_x1_2 ** 2 + (N2 - 1) * sigma_x2_2)

                S0 = S ** 2 * (
                    1 / ((N1 - 1) * S1 + (N2 - 1) * S2) +
                    1 / (x1_ - x2_) ** 2 * (1 / N1 + 1 / N2))
                t = (b - b0) / S0
                return abs(t) <= quant_t
        else:
            t = (b1 - b2) / (
                S * (S1 / (N1 * sigma_x1_2) + S2 / (N2 * sigma_x2_2)) ** 0.5)
            C0 = S1 / (N1 * sigma_x1_2) / (
                S1 / (N1 * sigma_x1_2) + S2 / (N2 * sigma_x2_2))
            nu = round((C0 ** 2 / (
                N1 - 2) + (1 - C0) ** 2 / (N2 - 2)) ** -1)
            if t <= QuantileTStudent(1 - trust / 2, nu):
                b = (b1 * N1 * sigma_x1_2 / S1 + b2 * N2 * sigma_x2_2 / S2) / (
                    N1 * sigma_x1_2 / S1 + N2 * sigma_x2_2 / S2)
                S10 = (N2 * S1 + N1 * S2) / (
                    N1 * N2 * (x1_ - x2_) ** 2) + (S1 * S2) / (
                        N1 * sigma_x1_2 * S2 + N2 * sigma_x2_2 * S1)
                u = (b - b0) / S10
                if u <= QuantileNorm(1 - trust / 2):
                    return None
        return False

    def corelationRelation(self, samples: list, trust: float = 0.05):
        r = []
        k = len(samples)
        for i in range(0, k - 1, 2):
            d = DoubleSampleData(samples[i], samples[i + 1])
            d.toCalculateCharacteristic()
            r.append(d.r)
        if len(samples) % 2 == 1:
            d = DoubleSampleData(samples[0], samples[-1])
            d.toCalculateCharacteristic()
            r.append(d.r)

        def z(i): return 1 / 2 * math.log((1 + r[i]) / (1 - r[i]))
        def N(i): return len(samples[i])

        x_2 = sum([(N(i) - 3) * z(i) ** 2 for i in range(k)]) - sum(
            [(N(i) - 3) * z(i) for i in range(k)]) ** 2 / sum(
                [N(i) - 3 for i in range(k)])

        quant_p = QuantilePearson(1 - trust, k - 1)
        logging.debug(f"Кореляційний зв'язок: {x_2} <= {quant_p}")
        return x_2 <= quant_p

    def identAvrIfIdentDC2Samples(self, samples1: list[SamplingData],
                                  samples2: list[SamplingData]):
        def x(i, j): return samples1[i].raw[j]
        def y(i, j): return samples2[i].raw[j]
        N1 = len(samples1[0].raw)
        N2 = len(samples2[0].raw)
        n = len(samples1)

        S0 = np.zeros((n, n))
        S1 = np.zeros((n, n))
        div1 = 1 / (N1 + N2 - 2)
        div2 = 1 / (N1 + N2)
        for i in range(n):
            for j in range(n):
                E_xx_E_yy = sum([x(i, k) * x(j, k) for k in range(N1)]) +\
                    sum([y(i, k) * y(j, k) for k in range(N2)])
                E_xi = sum([x(i, k) for k in range(N1)])
                E_yi = sum([y(i, k) for k in range(N2)])
                E_xj = sum([x(j, k) for k in range(N1)])
                E_yj = sum([y(j, k) for k in range(N2)])
                S0[i][j] = div1 * (E_xx_E_yy -
                                   div2 * (E_xi + E_yi) * (E_xj + E_yj))
                S1[i][j] = div1 * (E_xx_E_yy -
                                   1 / N1 * E_xi * E_xj -
                                   1 / N2 * (E_yi + E_yj))
        V = - (N1 + N2 - 2 - n / 2) * math.log(
            np.linalg.det(S1) / np.linalg.det(S0))
        X_2 = QuantilePearson(1 - self.trust, n)
        self.identAvrIfIdentDC_V = V
        self.identAvrIfIdentDC_X_2 = X_2
        return V <= X_2

    def identAvrIfIdentDCProtocol(self, res):
        crits = f"{self.identAvrIfIdentDC_V:.5} <= " \
            f"{self.identAvrIfIdentDC_X_2:.5}"
        if res:
            return "2 n-вимірних середніх при збіжності " \
                f"дисперсійно-коваріаційних матриць збігаються:\n{crits}"
        else:
            return "2 n-вимірних середніх при збіжності " \
                f"дисперсійно-коваріаційних матриць незбігаються:\n{crits}"

    def identAvrIfNotIdentDC(self, samples: list[list[SamplingData]]):
        k = len(samples)
        n = len(samples[0])
        def N(d: int): return len(samples[d][0])

        x__cache = [np.array([[s.x_] for s in samples[d]]) for d in range(k)]
        def x_(d: int): return x__cache[d]

        def X(d: int, i: int):
            return np.array([[s.raw[i]] for s in samples[d]])

        def S(d: int): return 1 / (N(d) - 1) * sum(
            [(X(d, i) - x_(d)) @ np.transpose(X(d, i) - x_(d))
             for i in range(N(d))])

        _x_ = np.linalg.inv(
            sum([N(d) * np.linalg.inv(S(d)) for d in range(k)])
        ) @ sum([N(d) * np.linalg.inv(S(d)) @ x_(d) for d in range(k)])
        V = sum([N(d) * np.transpose(x_(d) - _x_) @
                 np.linalg.inv(S(d)) @ (x_(d) - _x_) for d in range(k)])
        X_2 = QuantilePearson(1 - self.trust, n * (k - 1))
        V = V[0][0]
        self.identAvrIfNotIdentDC_V = V
        self.identAvrIfNotIdentDC_X_2 = X_2
        return V <= X_2

    def identAvrIfNotIdentDCProtocol(self, res):
        crits = f"{self.identAvrIfNotIdentDC_V:.5} <= " \
            f"{self.identAvrIfNotIdentDC_X_2:.5}"
        if res:
            return "k n-вимірних середніх при розбіжності " \
                f"дисперсійно-коваріаційних матриць збігаються:\n{crits}"
        else:
            return "k n-вимірних середніх при розбіжності " \
                f"дисперсійно-коваріаційних матриць незбігаються:\n{crits}"

    def identDC(self, samples: list[list[SamplingData]]):
        k = len(samples)
        n = len(samples[0])
        def N(d: int): return len(samples[d][0])

        x__cache = [np.array([[s.x_] for s in samples[d]]) for d in range(k)]
        def x_(d: int): return x__cache[d]

        def X(d: int, i: int):
            return np.array([[s.raw[i]] for s in samples[d]])

        def S(d: int): return 1 / (N(d) - 1) * sum(
            [(X(d, i) - x_(d)) @ np.transpose(X(d, i) - x_(d))
             for i in range(N(d))])

        N_g = sum([N(d) for d in range(k)])
        det_S_g = np.linalg.det(1 / (N_g - k) * sum([(N(d) - 1) * S(d)
                                                     for d in range(k)]))
        V = sum([(N(d) - 1) / 2 * math.log(det_S_g / np.linalg.det(S(d)))
                 for d in range(k)])
        X_2 = QuantilePearson(1 - self.trust, n * (n + 1) * (k - 1) // 2)
        self.identDC_V = V
        self.identDC_X_2 = X_2
        return V <= X_2

    def identDCProtocol(self, res):
        crits = f"{self.identDC_V:.5} <= {self.identDC_X_2:.5}"
        if res:
            return "Дисперсійно-коваріаційні матриці нормально " \
                f"розподілених випадкових величин збігаються:\n{crits}"
        else:
            return "Дисперсійно-коваріаційні матриці нормально " \
                f"розподілених випадкових величин незбігаються:\n{crits}"

    def homogeneityProtocol(self, samples: list[list[SamplingData]]):
        res = self.identDC(samples)
        protocol = self.identDCProtocol(res)
        if len(samples) == 2:
            res = self.identAvrIfIdentDC2Samples(samples[0], samples[1])
            protocol += "\n" + self.identAvrIfIdentDCProtocol(res)
        else:
            res = self.identAvrIfNotIdentDC(samples)
            protocol += "\n" + self.identAvrIfNotIdentDCProtocol(res)
        return protocol
