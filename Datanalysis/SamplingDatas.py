from Datanalysis.SamplingData import SamplingData, toMakeRange
from Datanalysis.DoubleSampleData import DoubleSampleData
import math
from functions import (
    QuantileNorm, QuantileTStudent, QuantilePearson, QuantileFisher,
    L)

SPLIT_CHAR = ' '


def splitAndRemoveEmpty(s: str) -> list:
    return list(filter(lambda x: x != '\n' and x != '',
                       s.split(SPLIT_CHAR)))


def readVectors(text: str) -> list:
    split_float_data = [[float(j) for j in splitAndRemoveEmpty(i)]
                        for i in text]
    return [[vector[i] for vector in split_float_data]
            for i in range(len(split_float_data[0]))]


class SamplingDatas:
    def __init__(self):
        self.samples = []

    def appendSample(self, s: SamplingData):
        self.samples.append(s)

    def append(self, not_ranked_series_str: str):
        vectors = readVectors(not_ranked_series_str)
        for i, v in enumerate(vectors):
            self.samples.append(SamplingData(vectors[i]))
            self[-1].toRanking()
            self[-1].toCalculateCharacteristic()

    def __len__(self) -> int:
        return len(self.samples)

    def pop(self, i: int) -> SamplingData:
        return self.samples.pop(i)

    def __getitem__(self, i: int) -> SamplingData:
        return self.samples[i]

    def getMaxDepth(self) -> int:
        if len(self.samples) == 0:
            return 0
        return max([len(i.x) for i in self.samples])

    def identAvrTtestDependent(self,
                               x: SamplingData,
                               y: SamplingData,
                               trust: float = 0.05):
        N = len(x)
        z = [x[i] - y[i] for i in range(N)]
        z_ = sum(z) / N
        S = sum([(zl - z_) ** 2 for zl in z]) / (N - 1)
        t = z_ * N ** 0.5 / S ** 0.5

        print("середні залежні: t <= t_nu: "
              f"{abs(t)} <= {QuantileTStudent(1 - trust / 2, N - 2)}")
        return abs(t) <= QuantileTStudent(1 - trust / 2, N - 2)

    def identAvrTtestIndependent(self,
                                 x: SamplingData,
                                 y: SamplingData,
                                 trust: float = 0.05):
        N1 = len(x)
        N2 = len(y)
        if N1 + N2 <= 25:
            S = ((N1 - 1) * x.S + (N2 - 1) * y.S) / (N1 + N2 - 2) / (
                (N1 * N2) / (N1 + N2))
        else:
            S = x.S / N1 + y.S / N2

        t = (x.x_ - y.x_) / math.sqrt(S)

        print("середні незалежні: t <= t_nu: "
              f"{abs(t)} <= "
              f"{QuantileTStudent(1 - trust / 2, N1 + N2 - 2)}")
        return abs(t) <= QuantileTStudent(1 - trust / 2, N1 + N2 - 2)

    def identDispersionFtest(self,
                             x: SamplingData,
                             y: SamplingData,
                             trust: float = 0.05):
        N1 = len(x)
        N2 = len(y)
        if x.S > y.S:
            f = x.S / y.S
        else:
            f = y.S / x.S

        print("дисперсії f-test: "
              f"{f} <= {QuantileFisher(1 - trust / 2, N1 - 1, N2 - 2)}")
        return f <= QuantileFisher(1 - trust / 2, N1 - 1, N2 - 2)

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
              f"{X} <= {QuantilePearson(1 - trust / 2, k - 1)}")
        return X <= QuantilePearson(1 - trust / 2, k - 1)

    def identAvrFtest(self, samples, trust: float = 0.05) -> bool:
        k = len(samples)

        def N(i): return len(samples[i])
        def S(i): return samples[i].S
        def x_(i): return samples[i].x_

        N_g = sum([N(i) for i in range(k)])
        x_g = sum([N(i) * x_(i) for i in range(k)]) / N_g

        S_M = 1 / (k - 1) * sum([N(i) * (x_(i) - x_g) for i in range(k)])

        S_B = 1 / (N_g - k) * sum([(N(i) - 1) * S(i) for i in range(k)])

        F = S_M / S_B

        print("однофакторно дисп аналіз, середні: "
              f"{F} <= {QuantileFisher(1 - trust / 2, k - 1, N_g - k)}")
        return F <= QuantileFisher(1 - trust / 2, k - 1, N_g - k)

    def critetionSmirnovKolmogorov(self,
                                   x: SamplingData,
                                   y: SamplingData,
                                   trust: float = 0.05):
        f, F, DF = x.toCreateNormalFunc()
        g, G, DG = y.toCreateNormalFunc()

        min_xl = min(x[0], y[0])
        max_xl = max(x[-1], y[-1])
        xl = min_xl
        dx = SamplingData.calc_reproduction_dx(min_xl, max_xl)
        xl += dx
        z = abs(F(xl) - G(xl))
        while xl <= max_xl:
            zi = abs(F(xl) - G(xl))
            if z < zi:
                z = zi
            xl += dx
        N1 = len(x)
        N2 = len(y)
        N = min(N1, N2)

        print(f"1 - L(z) = {1 - L(N ** 0.5 * z, N)} > {trust}")
        return 1 - L(N ** 0.5 * z, N) > trust

    def critetionWilcoxon(self,
                          x: SamplingData,
                          y: SamplingData,
                          trust: float = 0.05):
        r = [[i, 0, 'x'] for i in x.x] + [[i, 0, 'y'] for i in y.x]
        r.sort()

        prev = r[0][0]
        r[0][1] = 1
        i = 1
        avr_r = 0
        avr_i = 0
        for i in range(1, len(r)):
            if prev == r[i][0]:
                avr_r += i
                avr_i += 1
            else:
                r[i][1] = i + 1
                if avr_r != 0:
                    avr_r = avr_r / avr_i
                    j = i - 1
                    while r[j][1] != 0:
                        r[j][1] = avr_r
                        j -= 1
                    avr_r = 0
                    avr_i = 0
            prev = r[i][0]

        W = 0
        for i in r:
            if i[2] == 'x':
                W += i[1]

        N1 = len(x)
        N2 = len(y)
        N = N1 + N2
        E_W = N1 * (N + 1) / 2
        D_W = N1 * N2 * (N + 1) / 12

        w = (W - E_W) / (D_W) ** 0.5

        print("w ="
              f"{abs(w)} <= {QuantileNorm(1 - trust / 2)}")
        return abs(w) <= QuantileNorm(1 - trust / 2)

    def critetionUtest(self,
                       x: SamplingData,
                       y: SamplingData,
                       trust: float = 0.05):
        N1 = len(x)
        N2 = len(y)
        N = N1 + N2
        U = 0
        for yj in y:
            for xi in x:
                if xi > yj:
                    U += 1
        E_U = N1 * N2 / 2
        D_U = N1 * N2 * (N + 1) / 12

        u = (U - E_U) / (D_U) ** 0.5

        print("u ="
              f"{abs(u)} <= {QuantileNorm(1 - trust / 2)}")
        return abs(u) <= QuantileNorm(1 - trust / 2)

    def critetionDiffAvrRanges(self,
                               x: SamplingData,
                               y: SamplingData,
                               trust: float = 0.05) -> bool:
        N1 = len(x)
        N2 = len(y)
        N = N1 + N2
        r = [[i, 0, 'x'] for i in x.x] + [[i, 0, 'y'] for i in y.x]
        r.sort()

        rx = 0
        ry = 0
        for i in range(N):
            if r[i][2] == 'x':
                rx += r[i][1]
            if r[i][2] == 'y':
                ry += r[i][1]
        rx /= N1
        ry /= N2

        nu = (rx - ry) / (N * ((N + 1) / (12 * N1 * N2)) ** 0.5)

        print("nu="
              f"{abs(nu)} <= {QuantileNorm(1 - trust / 2)}")
        return abs(nu) <= QuantileNorm(1 - trust / 2)

    def critetionKruskalaUolisa(self, samples, trust: float = 0.05) -> bool:
        k = len(samples)

        x = []
        for i in range(k):
            x += [[j, 0, i] for j in samples[i]]
        x.sort()
        N_G = len(x)

        toMakeRange(x)

        def N(i): return len(samples[i])
        W = [0 for i in range(k)]
        for i in x:
            W[i[2]] += i[1]

        for i in range(k):
            W[i] /= N(i)

        H = 0

        E_W = (N_G + 1) / 2
        def D_W(i): return (N_G + 1) * (N_G - N(i)) / (12 * N(i))
        for i in range(k):
            H += (W[i] - E_W) ** 2 / D_W(i) * (1 - N(i) / N_G)

        print("H ="
              f"{H} <= {QuantilePearson(1 - trust, k - 1)}")
        return H <= QuantilePearson(1 - trust, k - 1)

    def critetionSign(self,
                      x: SamplingData,
                      y: SamplingData,
                      trust: float = 0.05):
        N = len(x)
        S = 0
        for i in range(N):
            if x[i] - y[i] > 0:
                S += 1
        S = (2 * S - 1 - N) / N ** 0.5

        print("S*="
              f"{S} <= {QuantileNorm(1 - trust)}")
        return S <= QuantileNorm(1 - trust)

    def critetionKohrena(self, samples, trust: float = 0.05) -> bool:
        k = len(samples)
        N = len(samples[0])
        def u(i): return sum([samples[j][i] for j in range(k)])
        def T(j): return sum(samples[j].x)
        T_ = sum([T(j) for j in range(k)]) / k
        Q = k * (k - 1) * sum([(T(j) - T_) ** 2 for j in range(k)]) / (
            k * sum([u(i) for i in range(N)]) -
            sum([u(i) ** 2 for i in range(N)]))

        print(f"Q ={Q} <= {QuantilePearson(1 - trust, k - 1)}")
        return Q <= QuantilePearson(1 - trust, k - 1)

    def ident2Samples(self, row1: int, row2: int, trust: float = 0.05) -> bool:
        x, y = self.samples[row1], self.samples[row2]
        if x.kolmogorovTest(x.toCreateNormalFunc()[1]) and\
           y.kolmogorovTest(y.toCreateNormalFunc()[1]):
            # normal
            if len(x) == len(y):
                return self.identDispersionBarlet([x, y], trust) \
                        and self.identAvrTtestDependent(x, y, trust)
            else:
                return self.identDispersionFtest(x, y, trust) \
                        and self.identAvrTtestIndependent(x, y, trust) \
                        and self.critetionWilcoxon(x, y, trust) \
                        and self.critetionUtest(x, y, trust) \
                        and self.critetionDiffAvrRanges(x, y, trust)
        else:
            # other
            return self.critetionSmirnovKolmogorov(x, y, trust) \
                    and self.critetionWilcoxon(x, y, trust) \
                    and self.critetionUtest(x, y, trust) \
                    and self.critetionDiffAvrRanges(x, y, trust) \
                    and self.critetionSign(x, y, trust)
        return False

# TODO: Add kohrena
    def identKSamples(self, samples: list, trust: float = 0.05):
        isNormal = True
        for i in samples:
            if i.kolmogorovTest(i.toCreateNormalFunc()[1]) is False:
                isNormal = False
                break
        if isNormal:  # normal
            return self.identDispersionBarlet(samples, trust) \
                   and self.critetionKruskalaUolisa(samples, trust)
        else:
            return self.critetionKruskalaUolisa(samples, trust)
        return False

    def corelationRelation(self, samples: list, trust: float = 0.05):
        r = []
        k = len(samples)
        for i in range(0, k - 1, 2):
            d = DoubleSampleData(samples[i], samples[i + 1])
            d.toRanking()
            d.toCalculateCharacteristic()
            r.append(d.r)
        if len(samples) % 2 == 1:
            d = DoubleSampleData(samples[0], samples[-1])
            d.toRanking()
            d.toCalculateCharacteristic()
            r.append(d.r)

        def z(i): return 1 / 2 * math.log((1 + r[i]) / (1 - r[i]))
        def N(i): return len(samples[i])

        x_2 = sum([(N(i) - 3) * z(i) ** 2 for i in range(k)]) - sum(
            [(N(i) - 3) * z(i) for i in range(k)]) ** 2 / sum(
                [N(i) - 3 for i in range(k)])
        print(f"{x_2} <= {QuantilePearson(1 - trust, k - 1)}")
        return x_2 <= QuantilePearson(1 - trust, k - 1)
