from time import time
import numpy as np
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


def timer(function):
    def wrapper(*args):
        t = time()
        if len(args) == 0:
            function()
        else:
            function(*args)
        print(f"{function.__name__}={time() - t}sec")
    return wrapper


def calc_reproduction_dx(x_start: float,
                         x_end: float,
                         n=500) -> float:
    dx = x_end - x_start
    while n > 1:
        if dx / n > 0:
            break
        n -= 1
    return dx / n


def toCalcRankSeries(x):  # (xl, rx)
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
    def __init__(self, not_ranked_series_x: list[float], trust: float = 0.05):
        self.raw = np.array(not_ranked_series_x)
        self._x = not_ranked_series_x.copy()
        self.countX: list[int] = []
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
        return len(self._x)

    def __getitem__(self, i: int) -> float:
        return self._x[i]

    def copy(self):
        t = SamplingData(list(self.raw))
        if len(self.probabilityX) > 0:
            t.toRanking()
            t.toCalculateCharacteristic()
        return t

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
        self._x.sort()
        prev = self._x[0] - 1
        number_all_observ = 0
        number_of_deleted_items = 0
        self.countX = []
        for i in range(len(self._x)):
            if prev == self._x[i - number_of_deleted_items]:
                self._x.pop(i - number_of_deleted_items)
                number_of_deleted_items += 1
            else:
                self.countX.append(0)
            self.countX[-1] += 1
            prev = self._x[i - number_of_deleted_items]
            number_all_observ += 1
        self.probabilityX = [c / number_all_observ for c in self.countX]

    def setTrust(self, trust: float):
        self.trust = trust

    def toCalculateCharacteristic(self):
        N = len(self._x)

        self.Sigma = 0.0  # stand_dev

        self.min = self._x[0]
        self.max = self._x[-1]

        self.MED = MED(self._x)
        self.MAD = 1.483 * self.MED

        PERCENT_USICH_SER = self.trust
        k = int(PERCENT_USICH_SER * N)
        self.x_a = sum([self._x[i] for i in range(k + 1, N - k)]) / (N - 2 * k)

        xl = [0.0] * (N * (N - 1) // 2)
        ll = 0
        for i in range(N):
            for j in range(i, N - 1):
                xl[ll] = 0.5 * (self._x[i] * self._x[j])
                ll += 1

        self.MED_Walsh = MED(xl)

        self.x_ = sum(self.raw) / len(self.raw)

        nu2 = 0.0
        u2 = 0.0
        u3 = 0.0
        u4 = 0.0
        u5 = 0.0
        u6 = 0.0
        u8 = 0.0
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

        if N != 2:
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
        for i in range(N):
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

    def setSeries(self, not_ranked_series_x):
        SamplingData.__init__(self, not_ranked_series_x)
        self.toRanking()
        self.toCalculateCharacteristic()

# edit sample
    def remove(self, minimum: float, maximum: float):
        new_raw_x = [x for x in self.raw if minimum <= x <= maximum]
        if len(new_raw_x) != len(self.raw):
            self.setSeries(new_raw_x)

    def autoRemoveAnomaly(self) -> bool:
        N = len(self._x)
        is_item_del = False

        t1 = 2 + 0.2 * math.log10(0.04 * N)
        t2 = (19 * math.sqrt(self.E + 2) + 1) ** 0.5
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

        for i in range(N // 2 + N % 2):
            left_x = self._x[i]
            right_x = self._x[-i - 1]
            if not (a <= left_x <= b) and not (a <= right_x <= b):
                if abs(a - left_x) > abs(b - right_x):
                    raw_x.remove(left_x)
                else:
                    raw_x.remove(right_x)
                is_item_del = True
                break
            elif self._x[i] <= a:
                raw_x.remove(left_x)
                is_item_del = True
                break
            elif self._x[-i - 1] >= b:
                raw_x.remove(right_x)
                is_item_del = True
                break

        if is_item_del:
            self.setSeries(raw_x)
        return is_item_del

    def toLogarithmus10(self):
        self.setSeries([math.log10(x) for x in self.raw])

    def toExp(self):
        self.setSeries([math.exp(x) for x in self.raw])

    def toStandardization(self):
        self.setSeries([(x - self.x_) / self.Sigma for x in self.raw])

    def toSlide(self, value: float = 1):
        self.setSeries([x + value for x in self.raw])

    def toMultiply(self, value: float = 1):
        self.setSeries([x * value for x in self.raw])

    def toBinarization(self, x_):
        self.setSeries([1 if x > x_ else 0 for x in self.raw])

    def toTransform(self, f_tr):
        self.setSeries([f_tr(x) for x in self.raw])

    def toCentralization(self):
        self.toSlide(-self.x_)
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

    def toCreateExponentialFunc(self) -> tuple:
        # MM
        N = len(self._x)
        lamd_a = 1 / self.x_

        def f(x): return fExp(x, lamd_a)

        def F(x): return FExp(x, lamd_a)

        def DF(x): return DF1Parametr(fExp_d_lamda(x, lamd_a),
                                      lamd_a ** 2 / N)
        return f, F, DF

    def toCreateWeibullFunc(self) -> tuple:
        # MHK
        N = len(self._x)
        a11 = N - 1
        a12 = a21 = 0.0
        a22 = 0.0
        b1 = 0.0
        b2 = 0.0
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

    def isNormal(self) -> bool:
        return self.kolmogorovTest(self.toCreateNormalFunc()[1])

    def kolmogorovTest(self, func_reproduction) -> bool:
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
        alpha_zgodi = 0.0
        if N > 100:
            alpha_zgodi = 0.05
        else:
            alpha_zgodi = 0.3
        self.kolmogorov_pz = Pz
        self.kolmogorov_alpha_zgodi = alpha_zgodi
        return Pz >= alpha_zgodi

    def kolmogorovTestProtocol(self, res):
        crits = f"{self.kolmogorov_pz:.5} >= {self.kolmogorov_alpha_zgodi:.5}"
        if res:
            return "Відтворення адекватне за критерієм" \
                f" згоди Колмогорова: {crits}"
        else:
            return "Відтворення неадекватне за критерієм" \
                f" згоди Колмогорова: {crits}"

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

    def xiXiTestProtocol(self, res):
        crits = f"{self.kolmogorov_pz:.5} < {self.xixitest_quant:.5}"
        if res:
            return "Відтворення адекватне за критерієм" \
                f" Пірсона: {crits}"
        else:
            return "Відтворення неадекватне за критерієм" \
                f" Пірсона: {crits}"

    def get_histogram_data(self, column_number=0) -> list:
        n = len(self._x)
        if column_number <= 0:
            column_number = SamplingData.calculateM(n)
        h = (self.max - self.min) / column_number
        hist_list = [0.0] * column_number
        for i in range(n - 1):
            j = math.floor((self._x[i] - self.min) / h)
            hist_list[j] += self.probabilityX[i]
        hist_list[-1] += self.probabilityX[-1]
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
        N = len(self.raw)
        d2 = 1 / (N - 1) * sum([(self.raw[i + 1] -
                                 self.raw[i]) ** 2
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


def formRowNV(name: str, *args) -> str:
    row = formatName(name)
    for arg in args:
        if type(arg) is str:
            row += formatValue(arg)
        else:
            row += formatValue(f"{arg:.5}")
    return row
