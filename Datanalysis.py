import math

# alpha 0.05 for tStudent

# 1.69   1.96

# 52   204

T_STUDENTA_INF_95P = 1.96
T_STUDENTA_INF_90P = 1.64


KVANT_NORMAL_69P = 0.5
KVANT_NORMAL_75P = 0.7
KVANT_NORMAL_90P = 1.3
KVANT_NORMAL_95P = 1.7
KVANT_NORMAL_99P = 2.4


NUM_DOT_REPRODUCTION = 500


def calculateDx(x_start, x_end, n = NUM_DOT_REPRODUCTION):
    while n > 1:
        if (x_end - x_start) / n > 0:
            break
        n -= 1
    return (x_end - x_start) / n

def calculateM(n) -> int:
    if n < 100:
        m = math.floor(math.sqrt(n))
    else:
        m = math.floor(n ** (1 / 3))
    m -= 1 - m % 2
    return m

def fNorm(x, m = 0, sigma = 1):
    return 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(- (x - m) ** 2 / (2 * sigma ** 2))

def fUniform(a, b):
    return 1 / b - a

def fExp(x, lam):
    if x < 0:
        return None
    try:
        return lam * math.exp(-lam * x)
    except OverflowError:
        return 1

def fWeibull(x, alpha, beta):
    return beta / alpha * x ** (beta - 1) * math.exp(-x ** beta / alpha)

def fArcsin(x, a):
    if not -a < x < a:
        return None
    return 1 / (math.pi * math.sqrt(a ** 2 - x ** 2))

class DataAnalysis:
    def __init__(self, not_ranked_series_x: list):
        self.x = not_ranked_series_x.copy()
        self.probabilityX = []

        self.h = 0.0

        self.min = 0.0
        self.max = 0.0
        self.x_ = 0.0 # math except
        self.S = 0.0  # dispersion
        self.Sigma = 0.0  # stand_dev
        self.A = 0.0  # asymmetric_c
        self.E = 0.0  # excess_c
        self.c_E = 0.0  # contre_excess_c : X/-\
        self.W_ = 0.0  # pearson_c
        self.Wp = 0.0  # param_var_c
        self.inter_range = 0.0

        self.MED = 0.0
        self.MED_Walsh = 0.0
        self.MAD = 0.0

    def setSeries(self, not_ranked_series_x: list):
        self.x = not_ranked_series_x.copy()
        self.toRanking()
        self.toCalculateCharacteristic()

    def RemoveAnomaly(self, minimum: float, maximum: float):
        num_deleted = 0
        for i in range(len(self.x)):
            if not (minimum <= self.x[i - num_deleted] <= maximum):
                self.x.pop(i - num_deleted)
                num_deleted += 1
        if num_deleted != 0:
            self.toRanking()
            self.toCalculateCharacteristic()
    
    def AutoRemoveAnomaly(self):
        N = len(self.x)
        is_item_del = False

        t1 = 2 + 0.2 * math.log10(0.04 * N)
        t2 = (19 * (self.E + 2) ** 0.5 + 1) ** 0.5
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
            if not(a <= self.x[i] <= b) and not (a <= self.x[-i - 1] <= b):
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

    def toCalculateCharacteristic(self):
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

        PERCENT_USICH_SER = 0.05
        self.x_a = 0.0
        k = int(PERCENT_USICH_SER * N)
        for i in range(k + 1, N - k):
            self.x_a += self.x[i]
        self.x_a /= N - 2 * k

        xl = []
        for i in range(N):
            for j in range(i, N - 1):
                xl.append(0.5 * (self.x[i] * self.x[j]))

        k_walsh = len(xl) // 2
        if 2 * k_walsh == len(xl):
            self.MED_Walsh = (xl[k] + xl[k + 1]) / 2
        else:
            self.MED_Walsh = xl[k + 1]

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
        self.E = ((N ** 2 - 1) / ((N - 2) * (N - 3))) * ((u4 / sigma_u2 ** 4 - 3) + 6 / (N + 1))

        self.c_E = 1.0 / math.sqrt(abs(self.E))

        self.W_ = self.Sigma / self.x_

        self.Wp = self.MAD / self.MED

        ip = 0.0
        self.kvant = []
        p = 0.0
        self.step_kvant = 0.025
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
            while ip + self.step_kvant < p:
                ip += self.step_kvant
                self.kvant.append(self.x[i])
        self.inter_range = self.kvant[math.floor(0.75 / self.step_kvant) - 1] - self.kvant[math.floor(0.25 / self.step_kvant) - 1]

        self.det_x_ = self.Sigma / math.sqrt(N) * T_STUDENTA_INF_95P
        self.det_Sigma = self.Sigma / math.sqrt(2 * N) * T_STUDENTA_INF_95P
        self.det_S = 2 * self.S / (N - 1)

        B1 = u3 * u3 / (u2 ** 3)
        B2 = u4 / (u2 ** 2)
        B3 = u3 * u5 / (u2 ** 4)
        B4 = u6 / (u2 ** 3)
        B6 = u8 / (u2 ** 4)

        # det_A is negative
        self.det_A = math.sqrt(abs(1.0 / (4 * N) * (4 * B4 - 12 * B3 - 24 * B2 + 9 * B2 * B1 + 35 * B1 - 36))) * T_STUDENTA_INF_95P
        self.det_A = math.sqrt(6 * (N - 2) / ((N + 1) * (N + 3)))
        self.det_E = math.sqrt(1.0 / N * (B6 - 4 * B4 * B2 - 8 * B3 + 4 * B2 ** 3 - B2 ** 2 + 16 * B2 * B1 + 16 * B1)) * T_STUDENTA_INF_95P
        self.det_c_E = math.sqrt(abs(u4 / sigma_u2 ** 4) / (29 * N)) * math.pow(abs(u4 / sigma_u2 ** 4 - 1) ** 3, 0.25) * T_STUDENTA_INF_95P
        self.det_W_ = self.W_ * math.sqrt((1 + 2 * self.W_ ** 2) / (2 * N)) * T_STUDENTA_INF_95P
        self.vanga_x_ = self.Sigma * math.sqrt(1 + 1 / N) * T_STUDENTA_INF_95P
    
    def toGenerateReproduction(self, func, max_height_column):
        x_gen = []
        N = len(self.x)
        
        if func == 0:
            f = lambda x : fNorm(x, self.x_, self.Sigma) # 
        elif func == 1:
            f = lambda x : fUniform(self.x_ - math.sqrt(3 * self.u2), self.x_ + math.sqrt(3 * self.u2))
        elif func == 2:
            f = lambda x : fExp(x, 1 / self.x_) # MM
        elif func == 3:
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

            alpha = math.exp(- (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21))
            beta = (b2 * a11 - b1 * a21) / (a11 * a22 - a12 * a21)

            f = lambda x : fWeibull(x, alpha, beta) # MHK
        elif func == 4:
            f = lambda x : fArcsin(x, math.sqrt(2 * self.u2))
        else:
            return []
        
        DF = 0.0
        DF = KVANT_NORMAL_90P * math.sqrt(DF)
        
        max_y = f(self.x[0])
        dx = calculateDx(self.x[0], self.x[-1])
        x = self.x[0]
        while x < self.x[-1]:
            y = f(x)
            if y != None:
                x_gen.append((x, f(x)))
                if max_y == None or max_y < y:
                    max_y = y
            x += dx
        if x_gen[-1][1] != self.x[-1]:
            y = f(self.x[-1])
            if y != None:
                x_gen.append((self.x[-1], y))
                if max_y < y:
                    max_y = y

        k = max_height_column / max_y
        for i in range(len(x_gen)):
            x_gen[i] = (x_gen[i][0], x_gen[i][1] * k)
        
        return x_gen

    def toTransform(self):
        if self.min < 0:
            for i in range(len(self.x)):
                self.x[i] -= self.min - 10
        for i in range(len(self.x)):
            self.x[i] = math.log10(self.x[i])
        self.toCalculateCharacteristic()

    def toStandardization(self):
        for i in range(len(self.x)):
            self.x[i] = (self.x[i] - self.x_) / self.Sigma
        self.toCalculateCharacteristic()

    def get_histogram_data(self, number_of_column: int):
        n = len(self.x)
        M: int = 0
        if number_of_column > 0:
            M = number_of_column
        elif number_of_column <= len(self.x):
            M = calculateM(n)
        self.h = (self.x[n - 1] - self.x[0]) / M
        hist_list = []
        j = 0
        begin_j = self.x[0]
        for i in range(M):
            hist_list.append(0)
            while j < n and begin_j + self.h * (i + 1) >= self.x[j]:
                hist_list[len(hist_list) - 1] += self.probabilityX[j]
                j += 1
        return hist_list

    def getProtocol(self) -> str:
        NM_SMBLS = 32
        VL_SMBLS = 16
        info = []
        info.append("-------------------------ПРОТОКОЛ-------------------------\n")
        info.append(f"{'Характеристика'.ljust(NM_SMBLS)}" +\
        f"{'INF'.ljust(VL_SMBLS)}{'Значення'.ljust(VL_SMBLS)}" +\
        f"{'SUP'.ljust(VL_SMBLS)}{'SKV'.ljust(VL_SMBLS)}\n")
        
        info.append("мат сподівання".ljust(NM_SMBLS) +\
            f"{self.x_-self.det_x_:.5}".ljust(VL_SMBLS) +\
                f"{self.x_:.5}".ljust(VL_SMBLS) +\
                    f"{self.x_+self.det_x_:.5}".ljust(VL_SMBLS) +\
                        f"{self.det_x_:.5}".ljust(VL_SMBLS))
        

        info.append("Дисперсія".ljust(NM_SMBLS) +\
            f"{self.S - self.det_S:.5}".ljust(VL_SMBLS) +\
                f"{self.S:.5}".ljust(VL_SMBLS) +\
                    f"{self.S + self.det_S:.5}".ljust(VL_SMBLS) +\
                        f"{self.det_S:.5}".ljust(VL_SMBLS))

        info.append("Сер квадратичне".ljust(NM_SMBLS) +\
            f"{self.Sigma - self.det_Sigma:.5}".ljust(VL_SMBLS) +\
                f"{self.Sigma:.5}".ljust(VL_SMBLS) +\
                    f"{self.Sigma + self.det_Sigma:.5}".ljust(VL_SMBLS) +\
                        f"{self.det_Sigma:.5}".ljust(VL_SMBLS))

        info.append("Коеф. асиметрії".ljust(NM_SMBLS) +\
            f"{self.A - self.det_A:.5}".ljust(VL_SMBLS) +\
                f"{self.A:.5}".ljust(VL_SMBLS) +\
                    f"{self.A + self.det_A:.5}".ljust(VL_SMBLS) +\
                        f"{self.det_A:.5}".ljust(VL_SMBLS))

        info.append("коеф. ексцесу".ljust(NM_SMBLS) +\
            f"{self.E - self.det_E:.5}".ljust(VL_SMBLS) +\
                f"{self.E:.5}".ljust(VL_SMBLS) +\
                    f"{self.E + self.det_E:.5}".ljust(VL_SMBLS) +\
                        f"{self.det_E:.5}".ljust(VL_SMBLS))

        info.append(f"коеф. контрексцесу".ljust(NM_SMBLS) +\
            f"{self.c_E - self.det_c_E:.5}".ljust(VL_SMBLS) +\
                f"{self.c_E:.5}".ljust(VL_SMBLS) +\
                    f"{self.c_E + self.det_c_E:.5}".ljust(VL_SMBLS) +\
                        f"{self.det_c_E:.5}".ljust(VL_SMBLS))

        info.append(f"коеф. варіації Пірсона".ljust(NM_SMBLS) +\
            f"{self.W_ - self.det_W_:.5}".ljust(VL_SMBLS) +\
                f"{self.W_:.5}".ljust(VL_SMBLS) +\
                    f"{self.W_ + self.det_W_:.5}".ljust(VL_SMBLS) +\
                        f"{self.det_W_:.5}".ljust(VL_SMBLS))
        
        info.append("")

        info.append("MED".ljust(NM_SMBLS) + f"{self.MED:.5}")
        info.append("усіченне середнє".ljust(NM_SMBLS) + f"{self.x_a:.5}")
        info.append("MED Уолша".ljust(NM_SMBLS) + f"{self.MED_Walsh:.5}")
        info.append("MAD".ljust(NM_SMBLS) + f"{self.MAD:.5}")
        info.append("непарам. коеф. варіацій".ljust(NM_SMBLS) + f"{self.Wp:.5}")
        info.append("коеф. інтер. розмаху".ljust(NM_SMBLS) + f"{self.inter_range:.5}")
        
        info.append("")

        info.append("мат спод.інтерв.передбачення".ljust(NM_SMBLS) +\
            f"{self.x_-self.vanga_x_:.5}".ljust(VL_SMBLS) +\
                f"{self.x_+self.vanga_x_:.5}".ljust(VL_SMBLS))
        
        info.append("")

        info.append("Квантилі\n------------------\nЙмовірність\tX")
        for i in range(len(self.kvant)):
            info.append(f"{self.step_kvant * (i + 1):.3}\t{self.kvant[i]}")

        return "\n".join(info)
