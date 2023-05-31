import numpy as np
import math

from Datanalysis.SamplingData import SamplingData, timer, formRowNV
from Datanalysis.SamplesCriteria import SamplesCriteria
from Datanalysis.DoubleSampleData import DoubleSampleData
import functions as func


def readVectors(text: list[str]) -> list:
    if ',' in text[0]:
        def to_float(x: str): return float(x.replace(',', '.'))
    else:
        def to_float(x: str): return float(x)

    def splitAndRemoveEmpty(s: str) -> list:
        return list(filter(lambda x: not x.isspace(), s.split()))

    n = len(splitAndRemoveEmpty(text[0]))
    vectors = [[0.0] * len(text) for i in range(n)]
    for j, line in enumerate(text):
        for i, str_num in enumerate(splitAndRemoveEmpty(line)):
            vectors[i][j] = to_float(str_num)
    return vectors


class SamplingDatas(SamplesCriteria):
    def __init__(self, samples: list[SamplingData] = [], trust=0.05):
        super().__init__()
        if type(samples) == list:
            [self.appendSample(s) for s in samples]
        self.trust = trust

    def appendSample(self, s: SamplingData):
        self.samples.append(s)

    def appendSamples(self, samples):
        self.samples += samples

    @timer
    def append(self, not_ranked_series_str: list[str]):
        vectors = readVectors(not_ranked_series_str)

        def rankAndCalc(s: SamplingData):
            s.toRanking()
            s.toCalculateCharacteristic()
        for v in vectors:
            s = SamplingData(v)
            rankAndCalc(s)
            self.appendSample(s)

    def copy(self) -> 'SamplingDatas':
        samples = [s.copy() for s in self.samples]
        new_sample = SamplingDatas(samples, self.trust)
        if hasattr(self, 'DC'):
            new_sample.DC = self.DC
            new_sample.R = self.R
            new_sample.R_Kendala = self.R_Kendala
            new_sample.r_multi = self.r_multi
            new_sample.DC_eigenval = self.DC_eigenval
            new_sample.DC_eigenvects = self.DC_eigenvects
            new_sample.DC_eigenval_part = self.DC_eigenval_part
            new_sample.DC_eigenval_accum = self.DC_eigenval_accum
        return new_sample

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

        self.r_multi = [self.multipleCorrelationCoefficient(i)
                        for i in range(n)]

        self.DC_eigenval, self.DC_eigenvects = func.EigenvalueJacob(self.DC)
        eigen_sum = sum(self.DC_eigenval)
        self.DC_eigenval_part = [val / eigen_sum for val in self.DC_eigenval]
        self.DC_eigenval_accum = [0.0] * n
        self.DC_eigenval_accum[0] = self.DC_eigenval_part[0]
        for i in range(n):
            self.DC_eigenval_accum[i] = \
                self.DC_eigenval_accum[i - 1] + self.DC_eigenval_part[i]

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

    def coeficientOfCorrelationTTest(self, r_ij_c, w):
        N = len(self.samples[0].getRaw())
        signif_r_ij_c = r_ij_c * (N - w - 2) ** 0.5 / (1 - r_ij_c ** 2) ** 0.5
        t = func.QuantileTStudent(1 - self.trust / 2, N - w - 2)
        self.partial_r_signif_t_test = signif_r_ij_c
        self.partial_r_signif_t_quant = t
        return signif_r_ij_c <= t

    def coeficientOfCorrelationIntervals(self, r_ij_c, w):
        N = len(self.samples[0].getRaw())
        u = func.QuantileNorm(self.trust / 2)
        v1 = 1 / 2 * math.log((1 + r_ij_c) / (1 - r_ij_c)) - u / (N - w - 3)
        v2 = 1 / 2 * math.log((1 + r_ij_c) / (1 - r_ij_c)) + u / (N - w - 3)
        det_less_partial_r = (math.exp(2 * v2) - 1) / (math.exp(2 * v2) + 1)
        det_more_partial_r = (math.exp(2 * v1) - 1) / (math.exp(2 * v1) + 1)
        self.det_less_partial_r = det_less_partial_r
        self.det_more_partial_r = det_more_partial_r

    def partialCoeficientOfCorrelationProtocol(self, i, j, cd):
        info_protocol = []

        def addForm(title, *args):
            info_protocol.append(formRowNV(title, *args))
        addForm('Характеристика', 'INF', 'Значення', 'SUP', 'SKV')
        info_protocol.append("")

        self.partial_r = self.coeficientOfCorrelation(i, j, cd)
        self.coeficientOfCorrelationIntervals(self.partial_r, len(cd) + 2)
        self.coeficientOfCorrelationTTest(self.partial_r, len(cd) + 2)
        addForm("Частковий коефіцієнт кореляції",
                self.det_less_partial_r,
                self.partial_r,
                self.det_more_partial_r)
        addForm("Т-тест",
                self.partial_r_signif_t_test,
                "≤",
                self.partial_r_signif_t_quant)
        return "\n".join(info_protocol)

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

    def multipleCorrelationCoefficient(self, k):
        n = len(self.R)
        Rkk = [[self.R[i][j] for j in range(n) if j != k]
               for i in range(n) if i != k]
        r_k = (1 - abs(np.linalg.det(self.R) / np.linalg.det(Rkk))) ** 0.5
        N = len(self.samples[0].getRaw())
        signif_r_k = (N - n - 1) / n * r_k ** 2 / (1 - r_k ** 2)
        f = func.QuantileFisher(1 - self.trust, n, N - n - 1)
        return r_k, signif_r_k, f

    def toIndependet(self):
        N = self.getMaxDepthRawData()
        n = len(self)
        vects = self.DC_eigenvects
        new_x = []
        def raw(i): return self[i].getRaw()
        for k in range(n):
            self[k].toCentralization()
            new_x.append([sum([vects[v, k] * raw(v)[i] for v in range(n)])
                          for i in range(N)])
        [s.setSeries(new_x[i]) for i, s in enumerate(self.samples)]

    def toReturnFromIndependet(self, w=0):
        n = len(self)
        if w == 0:
            w == n
        N = self.getMaxDepthRawData()
        vects = self.DC_eigenvects
        part = self.DC_eigenval_part
        old_x = [[]] * n

        sorted_by_disp = sorted([[part[i], i] for i in range(n)],
                                key=lambda i: i[0], reverse=True)

        def raw(i): return self[sorted_by_disp[i][1]].getRaw()
        def vect(i, k): return vects[i, sorted_by_disp[k][1]]
        for k in range(n):
            old_x[k] = [sum([vect(k, v) * raw(v)[i] for v in range(w)])
                        for i in range(N)]
        [s.setSeries(old_x[i]) for i, s in enumerate(self.samples)]

    def principalComponentAnalysis(self, w):
        independet_sample = self.copy()
        independet_sample.toIndependet()
        newold_sample = independet_sample.copy()
        newold_sample.toReturnFromIndependet(w)
        return independet_sample, newold_sample

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
        self.line_A0 = a0
        def f(*x): return a0 + A_ @ np.array(x)
        self.lineAccuracyParameters(f, yi)
        less_f, more_f = self.lineTrustIntervals(f)
        return less_f, f, more_f

    def lineAccuracyParameters(self, f, yi: int):
        n = len(self) - 1
        N = len(self.samples[0].getRaw())
        Y = self.samples[yi].getRaw()
        X = np.array([self.samples[i].getRaw()
                      for i in range(len(self)) if i != yi])
        self.line_Y = Y
        self.line_X = X
        C = np.linalg.inv(X @ X.transpose())
        self.line_C = C
        E = Y - f(*X)
        self.line_E = E
        S_2 = E @ np.transpose(E)
        sigma = (S_2 / (N - n)) ** 0.5
        sigma_2 = (S_2 / (N - n))
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
        self.line_S = sigma_2
        self.line_S_slide = S_2
        self.det_more_line_Sigma = S_2 * (N - n) / X_2(alpha1)
        self.line_sigma_signif_f_test = (N - n) * S_2 / sigma_2
        self.line_sigma_signif_f_quant = func.QuantilePearson(1 - self.trust,
                                                              N - n)

    def lineTrustIntervals(self, f):
        n = len(self) - 1
        N = len(self.samples[0].getRaw())
        t = func.QuantileTStudent(1 - self.trust / 2, N - n)
        C = self.line_C

        def det_f(X: np.ndarray) -> np.ndarray:
            return t * self.line_S ** 0.5 * (
                1 + (X.transpose() @ C @ X).diagonal()) ** 0.5

        def less_f(*X): return f(*X) - det_f(np.array(X))
        def more_f(*X): return f(*X) + det_f(np.array(X))
        return less_f, more_f

    def getProtocol(self) -> str:
        inf_protocol = []
        def add_text(text=""): inf_protocol.append(text)
        def addForm(title, *args): inf_protocol.append(formRowNV(title, *args))

        add_text("-" * 44 + "ПРОТОКОЛ" + "-" * 44 + "\n")
        addForm('Характеристика', 'INF', 'Значення', 'SUP', 'SKV')
        add_text()

        for i, s in enumerate(self.samples):
            addForm(f"Сер Арифметичне X{i+1}",
                    s.x_ - s.det_x_,
                    s.x_,
                    s.x_ + s.det_x_,
                    s.det_x_)
            addForm(f"Сер квадратичне X{i+1}",
                    s.Sigma - s.det_Sigma,
                    s.Sigma,
                    s.Sigma + s.det_Sigma,
                    s.det_Sigma)

        add_text()
        add_text("Оцінка дисперсійно-коваріаційної матриці DC:")
        n = len(self.samples)
        addForm("", *[f"X{i+1}" for i in range(n)])
        add_text()
        for i in range(n):
            addForm(f"X{i+1}", *[self.DC[i][j] for j in range(n)])

        add_text()
        add_text("Оцінка кореляційної матриці R:")
        addForm("", *[f"X{i+1}" for i in range(n)])
        add_text()
        for i in range(n):
            addForm(f"X{i+1}", *[self.R[i][j] for j in range(n)])

        add_text()
        for i in range(n):
            addForm(f"Множинний коефіцієнт кореляції X{i+1}",
                    "", self.r_multi[i][0])
            addForm("Т-тест коефіцієнту",
                    self.r_multi[i][1],
                    "≥",
                    self.r_multi[i][2])

        add_text()
        add_text("Власні вектори")
        addForm("", *[f"F{i+1}" for i in range(n)])
        for i in range(n):
            sum_xk = sum([self.DC_eigenvects[i][j] ** 2 for j in range(n)])
            addForm(f"X{i+1}", *([self.DC_eigenvects[i][j] for j in range(n)] +
                                 [sum_xk]))
        add_text()
        addForm("Власні числа", *[self.DC_eigenval[i] for i in range(n)])
        addForm("Частка %", *[self.DC_eigenval_part[i] for i in range(n)])
        addForm("Накопичена", *[self.DC_eigenval_accum[i] for i in range(n)])

        add_text()
        if hasattr(self, "line_A"):
            add_text("Параметри лінійної регресії: Y = AX")
            add_text("-" * 16)
            addForm("Коефіцієнт детермінації", "",
                    self.line_R)
            addForm("Перевірка значущості регресії",
                    self.line_R_f_test,
                    ">",
                    self.line_R_f_quant)
            add_text()
            addForm("Стандартна похибка регресії",
                    self.det_less_line_Sigma,
                    self.line_S_slide,
                    self.det_more_line_Sigma)
            addForm("σ^2 = σˆ^2",
                    self.line_sigma_signif_f_test,
                    "≤",
                    self.line_sigma_signif_f_quant)
            add_text()
            addForm(f"Параметр a{0}", "", self.line_A0)
            add_text()
            for k, a in enumerate(self.line_A):
                addForm(f"Параметр a{k+1}",
                        a - self.line_det_A[k],
                        a,
                        a + self.line_det_A[k],
                        self.line_det_A[k])
                addForm(f"T-Тест a{k+1}",
                        self.line_A_t_test[k],
                        "≤",
                        self.line_A_t_quant)
                addForm(f"Стандартизований параметр a{k+1}", '',
                        self.line_stand_A[k])
                add_text()

        return "\n".join(inf_protocol)
