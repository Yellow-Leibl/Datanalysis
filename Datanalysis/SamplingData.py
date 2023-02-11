import math
from functions import (
    QuantileNorm, QuantileTStudent, QuantilePearson,
    FNorm, fNorm, fNorm_d_m, fNorm_d_sigma,
    FUniform, fUniform,
    FArcsin, fArcsin,
    FWeibull, fWeibull, fWeibull_d_alpha, fWeibull_d_beta,
    FExp, fExp, fExp_d_lamda,
    DF1Parametr, DF2Parametr)

NM_SMBLS = 32
VL_SMBLS = 16


def calc_reproduction_dx(x_start: float,
                         x_end: float,
                         n=500) -> float:
    while n > 1:
        if (x_end - x_start) / n > 0:
            break
        n -= 1
    return (x_end - x_start) / n


def toMakeRange(x):  # (xl, rx)
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
    return x


def MED(r):
    N = len(r)
    if N == 1:
        med = r[0]
    elif N == 2:
        med = (r[0] + r[1]) / 2
    else:
        k = N // 2
        if 2 * k == N:
            med = (r[k] + r[k + 1]) / 2
        else:
            med = r[k + 1]

    return med


class SamplingData:
    def __init__(self, not_ranked_series_x: list, trust: float = 0.05):
        self.raw_x = not_ranked_series_x.copy()
        self.x = not_ranked_series_x.copy()
        self.probabilityX = []
        self.countX = []

        self.trust = trust

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

    def __getitem__(self, i: int) -> float:
        return self.x[i]

    def copy(self):
        t = SamplingData(self.getRaw())
        if len(self.probabilityX) > 0:
            t.toRanking()
            t.toCalculateCharacteristic()
        return t

    def getRaw(self) -> list:
        return self.raw_x

    @staticmethod  # number of classes
    def calculateM(n: int) -> int:
        if n == 2:
            return 2
        elif n < 100:
            m = math.floor(math.sqrt(n))
        else:
            m = math.floor(n ** (1 / 3))
        m -= 1 - m % 2
        return m

    def toRanking(self):
        self.x.sort()

        prev: float = self.x[0] - 1
        number_all_observ = 0
        number_of_deleted_items = 0
        self.countX = []
        self.probabilityX = []
        for i in range(len(self.x)):
            if prev == self.x[i - number_of_deleted_items]:
                self.x.pop(i - number_of_deleted_items)
                number_of_deleted_items += 1
            else:
                self.countX.append(0)
            self.countX[len(self.countX) - 1] += 1
            prev = self.x[i - number_of_deleted_items]
            number_all_observ += 1

        for i in range(len(self.countX)):
            self.probabilityX.append(self.countX[i] / number_all_observ)

    def setTrust(self, trust):
        self.trust: float = trust

    def toCalculateCharacteristic(self):
        N = len(self.x)

        self.Sigma = 0.0  # stand_dev

        self.min = self.x[0]
        self.max = self.x[N - 1]

        self.MED = MED(self.x)
        self.MAD = 1.483 * self.MED

        PERCENT_USICH_SER = self.trust
        self.x_a = 0.0
        k = int(PERCENT_USICH_SER * N)
        for i in range(k + 1, N - k):
            self.x_a += self.x[i]
        self.x_a /= N - 2 * k

        xl = [0] * (N * (N - 1) // 2)
        ll = 0
        for i in range(N):
            for j in range(i, N - 1):
                xl[ll] = 0.5 * (self.x[i] * self.x[j])
                ll += 1

        self.MED_Walsh = MED(xl)

        self.x_ = 0.0
        for i in range(N):
            self.x_ += self.x[i] * self.probabilityX[i]

        nu2 = 0.0
        u2 = 0.0
        u3 = 0.0
        u4 = 0.0
        u5 = 0.0
        u6 = 0.0
        u8 = 0.0
        for i in range(N):
            nu2 += self.x[i] ** 2 * self.probabilityX[i]
            u2 += (self.x[i] - self.x_) ** 2 * self.probabilityX[i]
            u3 += (self.x[i] - self.x_) ** 3 * self.probabilityX[i]
            u4 += (self.x[i] - self.x_) ** 4 * self.probabilityX[i]
            u5 += (self.x[i] - self.x_) ** 5 * self.probabilityX[i]
            u6 += (self.x[i] - self.x_) ** 6 * self.probabilityX[i]
            u8 += (self.x[i] - self.x_) ** 8 * self.probabilityX[i]

        # u2 -= self.x_ ** 2
        self.S_slide = nu2 - self.x_ ** 2
        self.Sigma_slide = self.S_slide ** 0.5
        self.u2 = u2
        self.u3 = u3
        sigma_u2 = math.sqrt(u2)
        self.S = u2 * N / (N - 1)
        self.Sigma = math.sqrt(self.S)

        if N != 2:
            self.A = u3 * math.sqrt(N * (N - 1)) / ((N - 2) * sigma_u2 ** 3)
            self.E = ((N ** 2 - 1) / ((N - 2) * (N - 3))) * (
                (u4 / sigma_u2 ** 4 - 3) + 6 / (N + 1))

            self.c_E = 1.0 / math.sqrt(abs(self.E))
        else:
            self.A = math.inf
            self.E = math.inf
            self.c_E = math.inf

        if self.x_ != 0:
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
        for i in range(N):
            p += self.probabilityX[i]
            while ip + step_quant < p:
                ip += step_quant
                self.quant.append(self.x[i])
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

    def setSeries(self, not_ranked_series_x: list):
        self.x = not_ranked_series_x.copy()
        self.toRanking()
        self.toCalculateCharacteristic()

# edit sample
    def remove(self, minimum: float, maximum: float):
        new_raw_x = []
        for i in self.raw_x:
            if minimum <= i <= maximum:
                new_raw_x.append(i)
        if len(new_raw_x) != len(self.raw_x):
            self.setSeries(new_raw_x)

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
            left_x = self.x[i]
            right_x = self.x[-i - 1]
            if not (a <= left_x <= b) and not (a <= right_x <= b):
                if abs(a - left_x) > abs(b - right_x):
                    self.raw_x.remove(left_x)
                else:
                    self.raw_x.remove(right_x)
                is_item_del = True
                break
            elif self.x[i] <= a:
                self.raw_x.remove(left_x)
                is_item_del = True
                break
            elif self.x[-i - 1] >= b:
                self.raw_x.remove(right_x)
                is_item_del = True
                break

        if is_item_del:
            self.toRanking()
            self.toCalculateCharacteristic()
        return is_item_del

    def toLogarithmus10(self):
        if self.min < 0:
            self.toStandardization()
            self.toSlide(3)
        for i in range(len(self.raw_x)):
            self.raw_x[i] = math.log10(self.raw_x[i])
        self.setSeries(self.raw_x)

    def toExp(self):
        for i in range(len(self.raw_x)):
            self.raw_x[i] = math.exp(self.raw_x[i], 10)
        self.setSeries(self.raw_x)

    def toStandardization(self):
        for i in range(len(self.raw_x)):
            self.raw_x[i] = (self.raw_x[i] - self.x_) / self.Sigma
        self.setSeries(self.raw_x)

    def toSlide(self, value: float = 1):
        for i in range(len(self.raw_x)):
            self.raw_x[i] += value
        self.setSeries(self.raw_x)

    def toMultiply(self, value: float = 1):
        for i in range(len(self.raw_x)):
            self.raw_x[i] *= value
        self.setSeries(self.raw_x)

    def toBinarization(self, x_):
        for i in range(len(self.raw_x)):
            self.raw_x[i] = 1.0 if self.raw_x[i] > x_ else 0.0
        self.setSeries(self.raw_x)

    def toTransform(self, f_tr):
        for i in range(len(self.raw_x)):
            self.raw_x[i] = f_tr(self.raw_x[i])
        self.setSeries(self.raw_x)
# end edit

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

    def toGenerateReproduction(self, f) -> list:
        x_gen = []
        dx = calc_reproduction_dx(self.min, self.max)
        x = self.min
        while x < self.max:
            if f(x) is not None:
                x_gen.append(x)
            x += dx

        if len(x_gen) > 0 and x_gen[-1] != self.max:
            x = self.max
            if f(x) is not None:
                x_gen.append(x)

        return x_gen

    def toCreateTrustIntervals(self, f, F, DF, h):
        u = QuantileNorm(1 - self.trust / 2)

        def limit(x):
            return u * math.sqrt(DF(x))

        def hist_f(x): return f(x) * h
        def lw_limit_F(x): return F(x) - limit(x)
        def hg_limit_F(x): return F(x) + limit(x)

        return hist_f, lw_limit_F, F, hg_limit_F

    def isNormal(self):
        return self.kolmogorovTest(self.toCreateNormalFunc()[1])

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

    def xiXiTest(self,
                 func_reproduction,
                 hist_list: list) -> bool:  # Pearson test
        hist_num = []
        M = len(hist_list)
        h = abs(self.max - self.min) / M
        N = len(self.x)
        Xi = 0.0
        xi = self.min
        j = 0
        for i in range(M):
            hist_num.append(0)
            while j < N and \
                    h * i <= self.x[j] - self.min <= h * (i + 1):
                hist_num[i] += 1
                j += 1
            xi += h
            ni_o = N * (func_reproduction(xi) - func_reproduction(xi - h))
            if ni_o == 0:
                return
            Xi += (hist_num[i] - ni_o) ** 2 / ni_o

        Xi2 = QuantilePearson(1 - self.trust, M - 1)
        if Xi < Xi2:
            return True
        else:
            return False

    def get_histogram_data(self, number_of_column: int) -> list:
        n = len(self.x)
        M: int = 0
        if number_of_column > 0:
            M = number_of_column
        elif number_of_column <= len(self.x):
            M = SamplingData.calculateM(n)
        h = abs(self.max - self.min) / M
        hist_list = []
        j = 0
        begin_j = self[0]
        for i in range(M):
            hist_list.append(0)
            while j < n and begin_j + h * (i + 1) >= self[j]:
                hist_list[-1] += self.probabilityX[j]
                j += 1
        return hist_list

    def getProtocol(self) -> str:
        info_protocol = []

        def add(text): info_protocol.append(text)
        add("-" * 44 + "ПРОТОКОЛ" + "-" * 44 + "\n")
        add(formatName('Характеристика') +
            formatValue('INF') + formatValue('Значення') +
            formatValue('SUP') + formatValue('SKV') + "\n")

        add(formatName("Сер арифметичне") +
            formatValue(f"{self.x_-self.det_x_:.5}") +
            formatValue(f"{self.x_:.5}") +
            formatValue(f"{self.x_+self.det_x_:.5}") +
            formatValue(f"{self.det_x_:.5}"))

        add(formatName("Дисперсія") +
            formatValue(f"{self.S - self.det_S:.5}") +
            formatValue(f"{self.S:.5}") +
            formatValue(f"{self.S + self.det_S:.5}") +
            formatValue(f"{self.det_S:.5}"))

        add(formatName("Сер квадратичне") +
            formatValue(f"{self.Sigma - self.det_Sigma:.5}") +
            formatValue(f"{self.Sigma:.5}") +
            formatValue(f"{self.Sigma + self.det_Sigma:.5}") +
            formatValue(f"{self.det_Sigma:.5}"))

        add(formatName("Коеф. асиметрії") +
            formatValue(f"{self.A - self.det_A:.5}") +
            formatValue(f"{self.A:.5}") +
            formatValue(f"{self.A + self.det_A:.5}") +
            formatValue(f"{self.det_A:.5}"))

        add(formatName("коеф. ексцесу") +
            formatValue(f"{self.E - self.det_E:.5}") +
            formatValue(f"{self.E:.5}") +
            formatValue(f"{self.E + self.det_E:.5}") +
            formatValue(f"{self.det_E:.5}"))

        add(formatName("коеф. контрексцесу") +
            formatValue(f"{self.c_E - self.det_c_E:.5}") +
            formatValue(f"{self.c_E:.5}") +
            formatValue(f"{self.c_E + self.det_c_E:.5}") +
            formatValue(f"{self.det_c_E:.5}"))

        add(formatName("коеф. варіації Пірсона") +
            formatValue(f"{self.W_ - self.det_W_:.5}") +
            formatValue(f"{self.W_:.5}") +
            formatValue(f"{self.W_ + self.det_W_:.5}") +
            formatValue(f"{self.det_W_:.5}"))

        add("")

        add(formatName("MED") + formatValue(f"{self.MED:.5}"))
        add(formatName("усіченне середнє") +
            formatValue(f"{self.x_a:.5}"))
        add(formatName("MED Уолша") +
            formatValue(f"{self.MED_Walsh:.5}"))
        add(formatName("MAD") + formatValue(f"{self.MAD:.5}"))
        add(formatName("непарам. коеф. варіацій") +
            formatValue(f"{self.Wp:.5}"))
        add(formatName("коеф. інтер. розмаху") +
            formatValue(f"{self.inter_range:.5}"))

        add("")

        add(formatName("мат спод.інтерв.передбачення") +
            formatValue(f"{self.x_ - self.vanga_x_:.5}") +
            formatValue(f"{self.x_ + self.vanga_x_:.5}"))

        add("")

        add("Квантилі\n" + "-" * VL_SMBLS * 2 + "\n" +
            formatValue("Ймовірність") + formatValue("X"))
        for i in range(len(self.quant)):
            step_quant = 1 / len(self.quant)
            add(formatValue(f"{step_quant * (i + 1):.3}") +
                formatValue(f"{self.quant[i]:.5}"))

        return "\n".join(info_protocol)

    def critetionAbbe(self) -> float:
        N = len(self.raw_x)
        d2 = 1 / (N - 1) * sum([(self.raw_x[i + 1] -
                                 self.raw_x[i]) ** 2
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


def formatRow2Value(name: str, v1, v2) -> str:
    return formatName(name) + \
        formatValue(v1) + \
        formatValue(v2)


def formRow3V(name: str, v1, v2, v3) -> str:
    return formatRow2Value(name, v1, v2) + \
        formatValue(v3)


def formRow4V(name: str, v1, v2, v3, v4) -> str:
    return formRow3V(name, v1, v2, v3) + \
        formatValue(v4)
