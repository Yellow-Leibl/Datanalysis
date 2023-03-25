import numpy as np
import math

from Datanalysis.SamplingData import SamplingData, timer, formRowNV
from Datanalysis.SamplesCriteria import SamplesCriteria
from Datanalysis.DoubleSampleData import DoubleSampleData
import functions as func

SPLIT_CHAR = ' '


def splitAndRemoveEmpty(s: str) -> list:
    return list(filter(lambda x: x != '\n' and x != '',
                       s.split(SPLIT_CHAR)))


def readVectors(text: list[str]) -> list:
    def strToFloat(x: str): return float(x.replace(',', '.'))
    split_float_data = [[strToFloat(j) for j in splitAndRemoveEmpty(i)]
                        for i in text]
    return [[vector[i] for vector in split_float_data]
            for i in range(len(split_float_data[0]))]


class SamplingDatas(SamplesCriteria):
    def __init__(self, samples: list[SamplingData] = [], trust=0.05):
        super().__init__()
        if type(samples) == list:
            [self.appendSample(s) for s in samples]
        self.trust = trust

    def appendSample(self, s: SamplingData):
        self.samples.append(s)

    @timer
    def append(self, not_ranked_series_str: list[str]):
        vectors = readVectors(not_ranked_series_str)

        def rankAndCalc(s: SamplingData):
            s.toRanking()
            s.toCalculateCharacteristic()
        for v in vectors:
            s = SamplingData(v)
            rankAndCalc(s)
            self.samples.append(s)

    def __len__(self) -> int:
        return len(self.samples)

    def pop(self, i: int) -> SamplingData:
        return self.samples.pop(i)

    def __getitem__(self, i: int) -> SamplingData:
        return self.samples[i]

    def getMaxDepthRangeData(self) -> int:
        if len(self.samples) == 0:
            return 0
        return max([len(i._x) for i in self.samples])

    def getMaxDepthRawData(self) -> int:
        if len(self.samples) == 0:
            return 0
        return max([len(i.getRaw()) for i in self.samples])

    def setTrust(self, trust):
        self.trust = trust

    @timer
    def toCalculateCharacteristic(self):
        n = len(self.samples)
        self.DC = np.zeros((n, n))
        self.R = self.DC.copy()
        for i in range(n):
            self.DC[i][i] = self.samples[i].S
            self.R[i][i] = 1.0
        self.R_Kendala = self.R.copy()

        for i in range(n):
            for j in range(i + 1, n):
                d2 = DoubleSampleData(self.samples[i], self.samples[j])
                d2.pearsonCorrelationСoefficient()
                cor = self.samples[i].Sigma * self.samples[j].Sigma * d2.r
                self.DC[i][j] = self.DC[j][i] = cor
                self.R[i][j] = self.R[j][i] = d2.r
                d2.rangeCorrelation()
                self.R_Kendala[i][j] = d2.teta_k

    def coeficientOfCorrelation(self, i, j, cd):
        if len(cd) >= 1:
            d = cd[-1]
            c = cd[:-1]
            r_ij_c = self.coeficientOfCorrelation(i, j, c)
            r_id_c = self.coeficientOfCorrelation(i, d, c)
            r_jd_c = self.coeficientOfCorrelation(j, d, c)
            r_ij_cd = (r_ij_c - r_id_c * r_jd_c) / (
                (1 - r_id_c ** 2) * (1 - r_jd_c ** 2)) ** 0.5
            return r_ij_cd
        else:
            return self.R[i][j]

    def coeficientOfRangeCorrelation(self, i, j, cd):
        if len(cd) >= 1:
            d = cd[-1]
            c = cd[:-1]
            r_ij_c = self.coeficientOfCorrelation(i, j, c)
            r_id_c = self.coeficientOfCorrelation(i, d, c)
            r_jd_c = self.coeficientOfCorrelation(j, d, c)
            r_ij_cd = (r_ij_c - r_id_c * r_jd_c) / (
                (1 - r_id_c ** 2) * (1 - r_jd_c ** 2)) ** 0.5
            return r_ij_cd
        else:
            return self.R_Kendala[i][j]

    def coeficientOfCorrelationTTestAndIntervals(self, r_ij_c, N, w):
        signif_r_ij_c = r_ij_c * (N - w - 2) ** 0.5 / (1 - r_ij_c ** 2) ** 0.5
        t = func.QuantileTStudent(1 - self.trust / 2, N - w - 2)
        print(f"{signif_r_ij_c} <= {t}")
        u = func.QuantileNorm(self.trust / 2)
        v1 = 1 / 2 * math.log((1 + r_ij_c) / (1 - r_ij_c)) - u / (N - w - 3)
        v2 = 1 / 2 * math.log((1 + r_ij_c) / (1 - r_ij_c)) + u / (N - w - 3)
        self.det_less_r_ij_c = (math.exp(2 * v1) - 1) / (math.exp(2 * v1) + 1)
        self.det_more_r_ij_c = (math.exp(2 * v2) - 1) / (math.exp(2 * v2) + 1)

    def multipleCorrelationCoefficient(self, k):
        n = len(self.R)
        Rkk = [[self.R[i][j] for j in range(n) if j != k]
               for i in range(n) if i != k]
        r_k = (1 - np.linalg.det(self.R) / np.linalg.det(Rkk)) ** 0.5
        N = len(self.samples[0])
        signif_r_k = (N - n - 1) / n * r_k ** 2 / (1 - r_k ** 2)
        f = func.QuantileFisher(1 - self.trust, n, N - n - 1)
        return r_k, signif_r_k, f

    def toCreateLinearRegressionMNK(self, yi: int):
        self.line_R, self.line_R_f_test, self.line_R_f_quant = \
            self.multipleCorrelationCoefficient(yi)
        self.line_R *= self.line_R
        n = len(self) - 1
        x_ = np.array([s.x_ for s in self.samples])
        X_x = np.array([[x - x_[i] for x in s.getRaw()]
                        for i, s in enumerate(self.samples) if i != yi])
        Y_T = np.array(self.samples[yi].getRaw())
        y_ = self.samples[yi].x_
        A_ = np.linalg.inv(X_x @ np.transpose(X_x)) @ (X_x) @ Y_T
        self.line_A = A_
        a0 = y_ - sum([A_[k] * x_[k] for k in range(n)])
        def f(*x): return a0 + sum([A_[k] * x[k] for k in range(n)])
        self.lineAccuracyParameters(A_, yi)
        less_f, more_f = self.lineTrustIntervals(f)
        return less_f, f, more_f

    def lineAccuracyParameters(self, A, yi: int):
        n = len(self) - 1
        N = len(self.samples[0])
        Y = self.samples[yi].getRaw()
        X = np.array([self.samples[i].getRaw()
                      for i in range(len(self)) if i != yi])
        self.line_Y = Y
        self.line_X = X
        C = np.linalg.inv(X @ np.transpose(X))
        self.line_C = C
        E = Y - A @ X
        self.line_E = E
        S_2 = E @ np.transpose(E)
        sigma = (S_2 / (N - n)) ** 0.5
        t = func.QuantileTStudent(1 - self.trust / 2, N - n)
        self.line_det_A = [t * sigma * C[k][k] ** 0.5 for k in range(n)]
        self.line_A_t_test = [self.line_A[k] / (sigma * C[k][k] ** 0.5)
                              for k in range(n)]
        self.line_A_t_quant = func.QuantileTStudent(1 - self.trust / 2, N - n)

        sigma_x = [self.samples[i].Sigma for i in range(len(self)) if i != yi]
        sigma_y = self.samples[yi].Sigma
        self.line_stand_A = [abs(self.line_A[k] * sigma_x[k] / sigma_y)
                             for k in range(n)]

        def X_2(alpha): return func.QuantilePearson(alpha, N - n)
        alpha1 = (1 - (1 - self.trust)) / 2
        alpha2 = (1 + (1 - self.trust)) / 2
        self.det_less_line_Sigma = S_2 * (N - n) / X_2(alpha2)
        self.line_S = sigma ** 2
        self.det_more_line_Sigma = S_2 * (N - n) / X_2(alpha1)
        self.line_sigma_signif_f_test = (N - n) * S_2 / sigma ** 2
        self.line_sigma_signif_f_quant = func.QuantilePearson(1 - self.trust,
                                                              N - n)

    def lineTrustIntervals(self, f):
        n = len(self) - 1
        N = len(self.samples[0])
        t = func.QuantileTStudent(1 - self.trust / 2, N - n)
        X = self.line_X
        C = self.line_C
        det_trust = t * self.line_S ** 0.5 * (
            1 + np.linalg.det(np.transpose(X) @ C @ X)) ** 0.5

        def less_f(*X): return f(*X) - det_trust
        def more_f(*X): return f(*X) + det_trust
        return less_f, more_f

    def getProtocol(self) -> str:
        info_protocol = []

        def addForm(title, *args):
            info_protocol.append(formRowNV(title, *args))

        def addIn(text=""): info_protocol.append(text)

        addIn("-" * 44 + "ПРОТОКОЛ" + "-" * 44 + "\n")
        addForm('Характеристика', 'INF', 'Значення', 'SUP', 'SKV')
        addIn()

        for i, s in enumerate(self.samples):
            addForm(f"Сер арифметичне X{i}",
                    f"{s.x_-s.det_x_:.5}",
                    f"{s.x_:.5}",
                    f"{s.x_+s.det_x_:.5}",
                    f"{s.det_x_:.5}")

        addIn()
        addIn("Оцінка дисперсійно-коваріаційної матриці DC:")
        n_DC = len(self.DC)
        addForm("", *[f"X{i}" for i in range(n_DC)])
        addIn()
        for i in range(n_DC):
            addForm(f"X{i}", *[f"{self.DC[i][j]:.5}" for j in range(n_DC)])

        addIn()
        addIn("Оцінка кореляційної матриці R:")
        n_R = len(self.R)
        addForm("", *[f"X{i}" for i in range(n_R)])
        addIn()
        for i in range(n_R):
            addForm(f"X{i}", *[f"{self.R[i][j]:.5}" for j in range(n_R)])

        if hasattr(self, "line_A"):
            addIn()
            addIn("Параметри лінійної регресії: Y = AX")
            addIn("-" * 16)
            addForm("Коефіцієнт детермінації", "",
                    f"{self.line_R:.5}")
            addForm("Перевірка значущості регресії",
                    f"{self.line_R_f_test:.5}",
                    ">",
                    f"{self.line_R_f_quant:.5}")
            addIn()
            addForm("",
                    f"{self.det_less_line_Sigma:.5}",
                    f"{self.line_S ** 2:.5}",
                    f"{self.det_more_line_Sigma:.5}")
            addForm("σ 2  ˆ 2",
                    f"{self.line_sigma_signif_f_test:.5}",
                    "≤",
                    f"{self.line_sigma_signif_f_quant:.5}")
            for k, a in enumerate(self.line_A):
                addForm(f"Параметр a{k}",
                        f"{a - self.line_det_A[k]:.5}",
                        f"{a:.5}",
                        f"{a + self.line_det_A[k]:.5}",
                        f"{self.line_det_A[k]:.5}")
                addForm(f"T-Тест a{k}",
                        f"{self.line_A_t_test[k]:.5}",
                        "≤",
                        f"{self.line_A_t_quant:.5}")
                addForm(f"Стандартизований параметр a{k}", '',
                        f"{self.line_stand_A[k]:.5}")
                addIn()

        return "\n".join(info_protocol)
