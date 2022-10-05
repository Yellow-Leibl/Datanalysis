import math
from time import time
from func import (
    QuantileNorm, QuantileTStudent, QuantilePearson, QuantileFisher,
    FNorm, fNorm, fNorm_d_m, fNorm_d_sigma,
    FUniform, fUniform,
    FArcsin, fArcsin,
    FWeibull, fWeibull, fWeibull_d_alpha, fWeibull_d_beta,
    FExp, fExp, fExp_d_lamda,
    L,
    DF1Parametr, DF2Parametr)

NM_SMBLS = 32
VL_SMBLS = 16

# reproduction tools

NUM_DOT_REPRODUCTION = 500

SPLIT_CHAR = ' '


def calculateDx(x_start: float,
                x_end: float, n=NUM_DOT_REPRODUCTION) -> float:
    while n > 1:
        if (x_end - x_start) / n > 0:
            break
        n -= 1
    return (x_end - x_start) / n


class SamplingData:
    def __init__(self, not_ranked_series_x: list):
        self.not_ranked_series_x = not_ranked_series_x
        self.x = not_ranked_series_x.copy()
        self.probabilityX = []

        self.trust = 0.05

        self.h = 0.0
        self.hist_list = []

        self.min = 0.0
        self.max = 0.0
        self.x_ = 0.0       # mathematical exception
        self.S = 0.0        # dispersion
        self.Sigma = 0.0    # standart deviation
        self.A = 0.0        # asymmetric_c
        self.E = 0.0        # excess_c
        self.c_E = 0.0      # contre_excess_c : X/-\
        self.W_ = 0.0       # pearson_c
        self.Wp = 0.0       # param_var_c
        self.inter_range = 0.0

        self.MED = 0.0
        self.MED_Walsh = 0.0
        self.MAD = 0.0

    def __len__(self) -> int:
        return len(self.x)

    @staticmethod  # number of classes
    def calculateM(n: int) -> int:
        if n < 100:
            m = math.floor(math.sqrt(n))
        else:
            m = math.floor(n ** (1 / 3))
        m -= 1 - m % 2
        return m

    def __getitem__(self, i: int) -> float:
        return self.x[i]

    def toRanking(self):
        self.x.sort()

        prev: float = self.x[0] - 1
        number_all_observ = 0
        number_of_deleted_items = 0
        self.probabilityX = []
        for i in range(len(self.x)):
            if prev == self.x[i - number_of_deleted_items]:
                self.x.pop(i - number_of_deleted_items)
                number_of_deleted_items += 1
            else:
                self.probabilityX.append(0)
            self.probabilityX[len(self.probabilityX) - 1] += 1
            prev = self.x[i - number_of_deleted_items]
            number_all_observ += 1

        for i in range(len(self.probabilityX)):
            self.probabilityX[i] /= number_all_observ

    def setTrust(self, trust):
        self.trust: float = trust

    def toCalculateCharacteristic(self):
        t1 = time()
        N = len(self.x)

        self.h = 0.0

        self.Sigma = 0.0  # stand_dev

        self.min = self.x[0]
        self.max = self.x[N - 1]

        self.x_ = 0.0
        for i in range(N):
            self.x_ += self.x[i] * self.probabilityX[i]

        k = N // 2
        if 2 * k == N:
            self.MED = (self.x[k] + self.x[k + 1]) / 2
        else:
            self.MED = self.x[k + 1]
        self.MAD = 1.483 * self.MED

        PERCENT_USICH_SER = self.trust
        self.x_a = 0.0
        k = int(PERCENT_USICH_SER * N)
        for i in range(k + 1, N - k):
            self.x_a += self.x[i]
        self.x_a /= N - 2 * k

        # xl = []
        # for i in range(N):
        #     for j in range(i, N - 1):
        #         xl.append(0.5 * (self.x[i] * self.x[j]))

        # k_walsh = len(xl) // 2
        # if 2 * k_walsh == len(xl):
        #     self.MED_Walsh = (xl[k] + xl[k + 1]) / 2
        # else:
        #     self.MED_Walsh = xl[k + 1]

        u2 = 0.0
        u3 = 0.0
        u4 = 0.0
        u5 = 0.0
        u6 = 0.0
        u8 = 0.0
        for i in range(N):
            u2 += (self.x[i] - self.x_) ** 2 * self.probabilityX[i]
            u3 += (self.x[i] - self.x_) ** 3 * self.probabilityX[i]
            u4 += (self.x[i] - self.x_) ** 4 * self.probabilityX[i]
            u5 += (self.x[i] - self.x_) ** 5 * self.probabilityX[i]
            u6 += (self.x[i] - self.x_) ** 6 * self.probabilityX[i]
            u8 += (self.x[i] - self.x_) ** 8 * self.probabilityX[i]

        # u2 -= self.x_ ** 2
        self.u2 = u2
        self.u3 = u3
        sigma_u2 = math.sqrt(u2)
        self.S = u2 * N / (N - 1)
        self.Sigma = math.sqrt(self.S)

        self.A = u3 * math.sqrt(N * (N - 1)) / ((N - 2) * sigma_u2 ** 3)
        self.E = ((N ** 2 - 1) / ((N - 2) * (N - 3))) * (
            (u4 / sigma_u2 ** 4 - 3) + 6 / (N + 1))

        self.c_E = 1.0 / math.sqrt(abs(self.E))

        if self.x_ != 0:
            self.W_ = self.Sigma / self.x_
        else:
            self.W_ = math.inf

        self.Wp = self.MAD / self.MED

        ip = 0.0
        self.quant = []
        p = 0.0
        self.step_quant = 0.025
        # 0.025     0.05    0.075   0.1     0.125
        # 0.15      0.175   0.2     0.225   0.25
        # 0.275     0.3     0.325   0.35    0.375
        # 0.4       0.425   0.45    0.475   0.5
        # 0.525     0.55    0.575   0.6     0.675
        # 0.7       0.725   0.75    0.775   0.8
        # 0.825     0.85    0.875   0.9     0.925
        # 0.95      0.975   1.000
        for i in range(N):
            p += self.probabilityX[i]
            while ip + self.step_quant < p:
                ip += self.step_quant
                self.quant.append(self.x[i])
        ind75 = math.floor(0.75 / self.step_quant) - 1
        ind25 = math.floor(0.25 / self.step_quant) - 1
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

        print(f"Characteristic calculation time = {time() - t1} sec")

    def setSeries(self, not_ranked_series_x: list):
        self.x = not_ranked_series_x.copy()
        self.toRanking()
        self.toCalculateCharacteristic()

    def removeAnomaly(self, minimum: float, maximum: float):
        num_deleted = 0
        for i in range(len(self.x)):
            if not (minimum <= self.x[i - num_deleted] <= maximum):
                self.x.pop(i - num_deleted)
                num_deleted += 1
        if num_deleted != 0:
            self.toRanking()
            self.toCalculateCharacteristic()

    def autoRemoveAnomaly(self) -> bool:
        N = len(self.x)
        is_item_del = False

        t1 = 2 + 0.2 * math.log10(0.04 * N)
        t2 = (19 * math.sqrt(self.E + 2) + 1) ** 0.5
        a = 0
        b = 0
        if self.A < -0.2:
            a = self.x_ - t2 * self.S
            b = self.x_ + t1 * self.S
        elif self.A > 0.2:
            a = self.x_ - t1 * self.S
            b = self.x_ + t2 * self.S
        else:
            a = self.x_ - t1 * self.S
            b = self.x_ + t1 * self.S

        for i in range(N // 2 + N % 2):
            if not (a <= self.x[i] <= b) and not (a <= self.x[-i - 1] <= b):
                if abs(a - self.x[i]) > abs(b - self.x[-i - 1]):
                    self.x.pop(i)
                else:
                    self.x.pop(-i - 1)
                is_item_del = True
                break
            elif self.x[i] <= a:
                self.x.pop(i)
                is_item_del = True
                break
            elif self.x[-i - 1] >= b:
                self.x.pop(-i - 1)
                is_item_del = True
                break

        if is_item_del:
            self.toRanking()
            self.toCalculateCharacteristic()
        return is_item_del

    def toCreateNormalFunc(self) -> tuple:
        N = len(self.x)
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

    def toCreateExponentialFunc(self) -> tuple:
        # MM
        N = len(self.x)
        lamd_a = 1 / self.x_

        def f(x): return fExp(x, lamd_a)

        def F(x): return FExp(x, lamd_a)

        def DF(x): return DF1Parametr(fExp_d_lamda(x, lamd_a),
                                      lamd_a ** 2 / N)
        return f, F, DF

    def toCreateWeibullFunc(self) -> tuple:
        # MHK
        N = len(self.x)
        a11 = N - 1
        a12 = a21 = 0.0
        a22 = 0.0
        b1 = 0.0
        b2 = 0.0
        emp_func = 0.0
        for i in range(N - 1):
            emp_func += self.probabilityX[i]
            a12 += math.log(self.x[i])
            a22 += math.log(self.x[i]) ** 2
            b1 += math.log(math.log(1 / (1 - emp_func)))
            b2 += math.log(math.log(1 / (1 - emp_func))) * math.log(self.x[i])
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
                    math.log(alpha) - beta * math.log(self.x[i])) ** 2

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
        N = len(self.x)
        a_ = math.sqrt(2 * self.u2)
        def f(x): return fArcsin(x, a_)

        def F(x): return FArcsin(x, a_)

        def DF(x): return DF1Parametr(-x / (math.pi * a_ *
                                            math.sqrt(a_ ** 2 - x ** 2)),
                                      a_ ** 4 / (8 * N))
        return f, F, DF

    def toGenerateReproduction(self, f, F, DF) -> list:
        x_gen = []

        def limit(x):
            return QuantileNorm(1 - self.trust / 2) * math.sqrt(DF(x))

        dx = calculateDx(self.x[0], self.x[-1])
        x = self.x[0]
        while x < self.x[-1]:
            y = f(x)
            if y is not None:
                x_gen.append((x, f(x)))
            x += dx

        if len(x_gen) > 0 and x_gen[-1][1] != self.x[-1]:
            y = f(self.x[-1])
            if y is not None:
                # (x, y)
                x_gen.append((self.x[-1], y))

        k = self.h
        for i in range(len(x_gen)):
            # (x, dest y, low limit y, func y, high limit y)
            x = x_gen[i][0]
            y_f = x_gen[i][1]
            y_F = F(x)
            dy = limit(x)
            x_gen[i] = (x, y_f * k, (y_F - dy), y_F, (y_F + dy))

        return x_gen

    def kolmogorovTest(self, func_reproduction) -> bool:
        N = len(self.x)

        D = 0.0
        emp_func = 0.0
        for i in range(N):
            emp_func += self.probabilityX[i]
            DN_plus = abs(emp_func - func_reproduction(self.x[i]))
            DN_minus = abs(emp_func - func_reproduction(self.x[i - 1]))
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
        alpha_zgodi = 0.0
        if N > 100:
            alpha_zgodi = 0.05
        else:
            alpha_zgodi = 0.3

        if Pz >= alpha_zgodi:
            return True
        else:
            return False

    def xiXiTest(self, func_reproduction) -> bool:  # Pearson test
        hist_num = []
        # TODO: hist_list - hist glist
        M = len(self.hist_list)
        N = len(self.x)
        Xi = 0.0
        xi = self.min
        j = 0
        for i in range(M):
            hist_num.append(0)
            while j < N and \
                    self.h * i <= self.x[j] - self.min <= self.h * (i + 1):
                hist_num[i] += 1
                j += 1
            xi += self.h
            ni_o = N * (func_reproduction(xi) - func_reproduction(xi - self.h))
            if ni_o == 0:
                return
            Xi += (hist_num[i] - ni_o) ** 2 / ni_o

        Xi2 = QuantilePearson(1 - self.trust, M - 1)
        if Xi < Xi2:
            return True
        else:
            return False

    def toTransform(self):
        if self.min < 0:
            for i in range(len(self.x)):
                self.x[i] -= self.min - 1
        for i in range(len(self.x)):
            self.x[i] = math.log10(self.x[i])
        self.toCalculateCharacteristic()

    def toStandardization(self):
        for i in range(len(self.x)):
            self.x[i] = (self.x[i] - self.x_) / self.Sigma
        self.toCalculateCharacteristic()

    def toSlide(self, value: float = 1):
        if self.min < 0:
            for i in range(len(self.x)):
                self.x[i] -= self.min - value
        self.toCalculateCharacteristic()

    def get_histogram_data(self, number_of_column: int):
        n = len(self.x)
        M: int = 0
        if number_of_column > 0:
            M = number_of_column
        elif number_of_column <= len(self.x):
            M = SamplingData.calculateM(n)
        self.h = (self.x[n - 1] - self.x[0]) / M
        self.hist_list = []
        j = 0
        begin_j = self.x[0]
        for i in range(M):
            self.hist_list.append(0)
            while j < n and begin_j + self.h * (i + 1) >= self.x[j]:
                self.hist_list[len(self.hist_list) - 1] += self.probabilityX[j]
                j += 1
        return self.hist_list

    def getProtocol(self) -> str:
        info = []
        info.append("-" * 44 + "ПРОТОКОЛ" + "-" * 44 + "\n")
        info.append(formatName('Характеристика') +
                    formatValue('INF') + formatValue('Значення') +
                    formatValue('SUP') + formatValue('SKV') + "\n")

        info.append(formatName("Сер арифметичне") +
                    formatValue(f"{self.x_-self.det_x_:.5}") +
                    formatValue(f"{self.x_:.5}") +
                    formatValue(f"{self.x_+self.det_x_:.5}") +
                    formatValue(f"{self.det_x_:.5}"))

        info.append(formatName("Дисперсія") +
                    formatValue(f"{self.S - self.det_S:.5}") +
                    formatValue(f"{self.S:.5}") +
                    formatValue(f"{self.S + self.det_S:.5}") +
                    formatValue(f"{self.det_S:.5}"))

        info.append(formatName("Сер квадратичне") +
                    formatValue(f"{self.Sigma - self.det_Sigma:.5}") +
                    formatValue(f"{self.Sigma:.5}") +
                    formatValue(f"{self.Sigma + self.det_Sigma:.5}") +
                    formatValue(f"{self.det_Sigma:.5}"))

        info.append(formatName("Коеф. асиметрії") +
                    formatValue(f"{self.A - self.det_A:.5}") +
                    formatValue(f"{self.A:.5}") +
                    formatValue(f"{self.A + self.det_A:.5}") +
                    formatValue(f"{self.det_A:.5}"))

        info.append(formatName("коеф. ексцесу") +
                    formatValue(f"{self.E - self.det_E:.5}") +
                    formatValue(f"{self.E:.5}") +
                    formatValue(f"{self.E + self.det_E:.5}") +
                    formatValue(f"{self.det_E:.5}"))

        info.append(formatName("коеф. контрексцесу") +
                    formatValue(f"{self.c_E - self.det_c_E:.5}") +
                    formatValue(f"{self.c_E:.5}") +
                    formatValue(f"{self.c_E + self.det_c_E:.5}") +
                    formatValue(f"{self.det_c_E:.5}"))

        info.append(formatName("коеф. варіації Пірсона") +
                    formatValue(f"{self.W_ - self.det_W_:.5}") +
                    formatValue(f"{self.W_:.5}") +
                    formatValue(f"{self.W_ + self.det_W_:.5}") +
                    formatValue(f"{self.det_W_:.5}"))

        info.append("")

        info.append(formatName("MED") + formatValue(f"{self.MED:.5}"))
        info.append(formatName("усіченне середнє") +
                    formatValue(f"{self.x_a:.5}"))
        info.append(formatName("MED Уолша") +
                    formatValue(f"{self.MED_Walsh:.5}"))
        info.append(formatName("MAD") + formatValue(f"{self.MAD:.5}"))
        info.append(formatName("непарам. коеф. варіацій") +
                    formatValue(f"{self.Wp:.5}"))
        info.append(formatName("коеф. інтер. розмаху") +
                    formatValue(f"{self.inter_range:.5}"))

        info.append("")

        info.append(formatName("мат спод.інтерв.передбачення") +
                    formatValue(f"{self.x_ - self.vanga_x_:.5}") +
                    formatValue(f"{self.x_ + self.vanga_x_:.5}"))

        info.append("")

        info.append("Квантилі\n" + "-" * VL_SMBLS * 2 + "\n" +
                    formatValue("Ймовірність") + formatValue("X"))
        for i in range(len(self.quant)):
            info.append(formatValue(f"{self.step_quant * (i + 1):.3}") +
                        formatValue(f"{self.quant[i]:.5}"))

        return "\n".join(info)

    def critetionAbbe(self) -> float:
        N = len(self.not_ranked_series_x)
        d2 = 1 / (N - 1) * sum([(self.not_ranked_series_x[i + 1] -
                                 self.not_ranked_series_x[i]) ** 2
                               for i in range(N - 1)])

        q = d2 / (2 * self.S)

        E_q = 1
        D_q = (N - 2) / (N ** 2 - 1)
        U = (q - E_q) / D_q ** 0.5
        P = FNorm(U)
        return P


def formatName(n: str) -> str:
    return n.ljust(NM_SMBLS)


def formatValue(v: str) -> str:
    return v.center(VL_SMBLS)


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
        dx = calculateDx(min_xl, max_xl)
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

        prev = x[0][0]
        x[0][1] = 1
        i = 1
        avr_r = 0
        avr_i = 0
        for i in range(1, N_G):
            if prev == x[i][0]:
                avr_r += i
                avr_i += 1
            else:
                x[i][1] = i + 1
                if avr_r != 0:
                    avr_r = avr_r / avr_i
                    j = i - 1
                    while x[j][1] != 0:
                        x[j][1] = avr_r
                        j -= 1
                    avr_r = 0
                    avr_i = 0
            prev = x[i][0]

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
