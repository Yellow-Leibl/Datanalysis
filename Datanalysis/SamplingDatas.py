import numpy as np
import math

from Datanalysis import (
    SamplingData, DoubleSampleData, PolynomialRegressionModel)
from Datanalysis.SamplesCriteria import SamplesCriteria
from Datanalysis.SamplesTools import timer, formRowNV, calculate_m
import Datanalysis.functions as func
from Datanalysis.cluster import KMeans
import Datanalysis.cluster.AgglomerativeClustering as ac
import Datanalysis.cluster.SignifCluster as sc
import Datanalysis.cluster.distances as DIST
import Datanalysis.cluster as clstr

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SamplingDatas(SamplesCriteria):
    def __init__(self, samples: list[SamplingData] = None, trust=0.05):
        super().__init__()
        self.append_samples(samples)
        self.trust = trust
        self.clusters = None

    def append_sample(self, s: SamplingData):
        self.samples.append(s)

    def append_samples(self, samples: list[SamplingData]):
        if samples is not None:
            self.samples += samples

    def remove_observations(self, i):
        for s in self.samples:
            s.remove_observations(i)

    def remove_range(self, i, a, b):
        s = self.samples[i].raw
        self.remove_observations(np.where(np.logical_and(a <= s, s <= b)))

    @timer
    def append_calculate(self, samples: list[SamplingData]):
        self.samples += samples
        for s in samples:
            s.toRanking()
            s.toCalculateCharacteristic()

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

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.samples[i]
        if isinstance(i, int):
            return self.samples[i]

    def __iter__(self):
        return iter(self.samples)

    def to_numpy(self):
        """
        x1 x2 ... xn\n
        x1 x2 ... xn\n
        ............\n
        x1 x2 ... xn
        """
        return np.array([s.raw for s in self.samples]).T

    def get_names(self):
        return [s.name for s in self.samples]

    def get_ticks(self):
        return [s.ticks for s in self.samples]

    def getMaxDepthRangeData(self) -> int:
        if len(self.samples) == 0:
            return 0
        return max([len(i._x) for i in self.samples])

    def get_max_len_raw(self) -> int:
        if len(self.samples) == 0:
            return 0
        return max([len(i.raw) for i in self.samples])

    def set_trust(self, trust):
        self.trust = trust

    def get_clusters(self):
        return self.clusters

    @timer
    def toCalculateCharacteristic(self):
        n = len(self.samples)
        self.DC = np.empty((n, n))
        self.R = np.empty((n, n))

        for i in range(n):
            self.DC[i][i] = self.samples[i].S
            self.R[i][i] = 1.0

        self.R_Kendala = None

        for i in range(n):
            for j in range(i + 1, n):
                d2 = DoubleSampleData(self.samples[i], self.samples[j])
                r = d2.pearson_correlation_coefficient()
                cov = self.samples[i].Sigma * self.samples[j].Sigma * r
                self.DC[i][j] = self.DC[j][i] = cov
                self.R[i][j] = self.R[j][i] = r

        self.r_multi = [self.multipleCorrelationCoefficient(i)
                        for i in range(n)]

        self.calculate_pca()
        self.calculate_exploratory_data_analysis()
        self.to_calc_cluster_signif()

    def to_calc_cluster_signif(self):
        if self[0].clusters is None:
            return
        clusters = self[0].clusters
        X = self.to_numpy()
        S = [X[cluster] for cluster in clusters]
        d = DIST.get_distance_metric_one_interface(self[0].metric, X)

        self.signif_weighted_sum = sc.signif_weighted_sum(S, d)
        self.signif_pair_sum = sc.signif_pair_sum(S, d)
        self.signif_general_dispersion = sc.signif_general_dispersion(S)
        self.signif_relation_functionals = sc.signif_relation_functionals(S, d)

    def calculate_pca(self):
        n = len(self.samples)
        self.DC_eigenval, self.DC_eigenvects = \
            func.EigenvalueJacob(self.DC)
        self.DC_eigenval, self.DC_eigenvects = func.sort_evects_n_evals(
                self.DC_eigenval, self.DC_eigenvects)
        self.DC_eigenval_part = self.DC_eigenval.copy() / sum(self.DC_eigenval)
        self.DC_eigenval_part *= 100
        self.DC_eigenval_accum = np.empty(n, dtype=float)
        self.DC_eigenval_accum[0] = self.DC_eigenval_part[0]
        for i in range(1, n):
            self.DC_eigenval_accum[i] = \
                self.DC_eigenval_accum[i - 1] + self.DC_eigenval_part[i]

    def calculate_exploratory_data_analysis(self):
        n = len(self)

        def max_corr_method(R: np.ndarray):
            red = R.copy()
            for k in range(n):
                max = np.nan
                for v in range(n):
                    if v != k and (np.isnan(max) or max < abs(red[k, v])):
                        max = abs(red[k, v])
                red[k, k] = max
            return red

        def triads_method(R: np.ndarray):
            if R.shape[0] <= 3:
                return
            red = R.copy()
            for k in range(n):
                i = (k + 1) % n
                j = (k + 2) % n
                for v in range(n):
                    if red[k, i] < red[k, v] and v != k:
                        i = v
                for v in range(n):
                    if red[k, j] < red[k, v] and v != k and v != i:
                        j = v

                red[k, k] = abs(red[k, i] * red[k, j] / red[i, j])
            return red

        def average_method(R: np.ndarray):
            red = R.copy()
            for k in range(n):
                red[k, k] = 0.0
                for v in range(n):
                    red[k, k] += abs(red[k, v])
                red[k, k] /= n - 1
            return red

        def center_method(R: np.ndarray):
            red = max_corr_method(R)
            sum_r = np.sum(abs(red))
            hk_2 = np.empty(n, dtype=float)
            for k in range(n):
                hk = 0.0
                for v in range(n):
                    hk += abs(red[k, v])
                hk_2[k] = (hk ** 2) / sum_r

            for k in range(n):
                red[k, k] = hk_2[k]
            return red

        def averoid_method(R: np.ndarray):
            red = R.copy()
            sum_r = abs(R).sum() - abs(R.diagonal()).sum()
            for k in range(n):
                red[k, k] = 0.0
                for v in range(n):
                    if k != v:
                        red[k, k] += abs(R[k, v])
                red[k, k] = n / (n - 1) * (red[k, k] ** 2) / (
                    sum_r - sum(red[k]) + red[k, k])
            return red

        def pca_method(R: np.ndarray, w: int):
            red = R.copy()
            for k in range(n):
                red[k, k] = 0.0
                for v in range(w):
                    red[k, k] += self.DC_eigenvects[k, v] ** 2
            return red

        eval_R, _ = func.EigenvalueJacob(self.R)
        minimum_w = len(eval_R[eval_R > 1])

        red_mats = [
            max_corr_method(self.R),
            average_method(self.R), center_method(self.R),
            pca_method(self.R, minimum_w)]
        if n > 3:
            red_mats += [triads_method(self.R), averoid_method(self.R)]

        eigens_red = [func.EigenvalueJacob(Red) for Red in red_mats]

        def calc_f(redu: np.ndarray, evec_redu: np.ndarray):
            r_rest = redu - evec_redu @ evec_redu.transpose()

            f = 0.0
            for v in range(n):
                for q in range(n):
                    if v != q:
                        f += r_rest[v, q] ** 2
            return f

        f_all = [calc_f(Rh, A) for Rh, (_, A) in zip(red_mats, eigens_red)]

        min_f = min(f_all)
        min_f_index = f_all.index(min_f)
        Red = red_mats[min_f_index]
        eval_redu, evec_redu = eigens_red[min_f_index]

        def calc_w(eval_redu: np.ndarray):
            maximum_w = len(eval_redu[eval_redu > np.average(eval_redu)])
            return max(minimum_w, maximum_w)
        w = calc_w(eval_redu)

        f_prev = min_f
        prev_a = evec_redu.copy()

        def calc_dif_a(prev_a, a): return np.sum((a - prev_a) ** 2)

        def hk_2_less_1(hk): return True in (hk <= 1)

        def calc_hk(evec_redu: np.ndarray, w: int):
            hk = np.zeros(n)
            for k in range(n):
                for v in range(w):
                    hk[k] += evec_redu[k, v] ** 2
            return hk

        def calc_redu(Red: np.ndarray, hk):
            for k in range(n):
                Red[k, k] = hk[k]

        eps = 0.00001

        while True:
            hk = calc_hk(evec_redu, w)
            calc_redu(Red, hk)
            eval_redu, evec_redu = func.EigenvalueJacob(Red)
            f = calc_f(Red, evec_redu)
            w = calc_w(eval_redu)
            if f < f_prev and \
                calc_dif_a(prev_a, evec_redu) > eps and \
                    hk_2_less_1(hk):
                continue
            break

        major_f_ind = [(i, e_val) for i, e_val in enumerate(eval_redu)]
        major_f_ind.sort(key=lambda i: i[1], reverse=True)
        self.fact_mat = evec_redu[:, [major_f_ind[i][0] for i in range(w)]]

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
        N = len(self.samples[0].raw)
        signif_r_ij_c = r_ij_c * (N - w - 2) ** 0.5 / (1 - r_ij_c ** 2) ** 0.5
        t = func.QuantileTStudent(1 - self.trust / 2, N - w - 2)
        self.partial_r_signif_t_test = signif_r_ij_c
        self.partial_r_signif_t_quant = t
        return signif_r_ij_c <= t

    def coeficientOfCorrelationIntervals(self, r_ij_c, w):
        N = len(self.samples[0].raw)
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

    def calculate_r_kendala(self):
        self.R_Kendala = self.R.copy()
        n = len(self.samples)
        for i in range(n):
            for j in range(i + 1, n):
                d2 = DoubleSampleData(self.samples[i], self.samples[j])
                r = d2.get_rank_series()
                teta_k, _, _ = d2.kendalls_rank_coefficient(r)
                self.R_Kendala[i][j] = teta_k

    def coeficientOfRangeCorrelation(self, i, j, cd):
        if self.R_Kendala is None:
            self.calculate_r_kendala()

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
        N = len(self.samples[0].raw)
        signif_r_k = (N - n - 1) / n * r_k ** 2 / (1 - r_k ** 2)
        f = func.QuantileFisher(1 - self.trust, n, N - n - 1)
        return r_k, signif_r_k, f

    def fast_remove(self, ranges):
        def in_space(p, ranges):
            k = len(ranges)
            for i in range(k):
                start = ranges[i][0]
                end = ranges[i][1]
                if not (start <= p[i] <= end):
                    return False
            return True

        N = self.get_max_len_raw()
        del_ind = []
        for i in range(N):
            p = [s.raw[i] for s in self.samples]
            if in_space(p, ranges):
                del_ind.append(i)

        for s in self.samples:
            s.raw = np.delete(s.raw, del_ind)
        return len(del_ind)

    def get_histogram_data(self, column_number=0):
        n = len(self)
        if column_number <= 0:
            column_number = calculate_m(self.getMaxDepthRangeData())
        self.probability_table = np.zeros(
            tuple(column_number for i in range(n)))
        N = self.get_max_len_raw()
        h = [(s.max - s.min) / column_number for s in self.samples]
        pos = [0] * n
        for i in range(N):
            for j, s in enumerate(self.samples):
                pos[j] = math.floor((s.raw[i] - s.min) / h[j])
                if pos[j] == column_number:
                    pos[j] -= 1
            self.probability_table[tuple(pos)] += 1
        # print(self.probability_table)
        return self.probability_table

    def autoRemoveAnomaly(self, hist_data: np.ndarray):
        hist_shape = hist_data.shape
        n_samples = len(hist_shape)
        h = [(s.max - s.min) / n for n, s in zip(hist_shape, self.samples)]

        def ranges(i):
            r = []
            for j, n in enumerate(hist_shape):
                s = self.samples[j]
                col_ind = (i // np.prod(
                    [hist_shape[m] for m in range(j + 1, n_samples)])) % n
                start = s.min + col_ind * h[j]
                end = s.min + (col_ind + 1) * h[j]
                if col_ind + 1 == n:
                    end = s.max
                r.append([start, end])
            return r

        N = self.get_max_len_raw()
        hist_data = hist_data.reshape((np.prod(hist_data.shape)))
        item_del_count = 0
        for i, ni in enumerate(hist_data):
            p = ni / N
            if p <= self.trust and ni != 0:
                item_del_count += self.fast_remove(ranges(i))
        hist_data = hist_data.reshape(hist_shape)

        if item_del_count > 0:
            logger.debug(f"Deleted observe {item_del_count}")
            for s in self.samples:
                s.setSeries(s.raw)

        return item_del_count

    def toIndependet(self):
        N = self.get_max_len_raw()
        n = len(self)
        vects = self.DC_eigenvects
        new_serieses = []
        for k in range(n):
            self[k].to_centralization()
            new_series = np.zeros(N, dtype=float)
            for i in range(N):
                for v in range(n):
                    new_series[i] += vects[v, k] * self[v].raw[i]
            new_serieses.append(new_series)
        for new_raw, s in zip(new_serieses, self.samples):
            s.setSeries(new_raw)
        return self

    def toReturnFromIndependet(self, w=0):
        n = len(self)
        if w == 0:
            w = n
        N = self.get_max_len_raw()
        vects = self.DC_eigenvects
        old_serieses = []

        for k in range(n):
            old_series = np.zeros(N, dtype=float)
            for i in range(N):
                for v in range(w):
                    old_series[i] += vects[k, v] * self[v].raw[i]
            old_serieses.append(old_series)
        for old_raw, s in zip(old_serieses, self.samples):
            s.setSeries(old_raw)
        return self

    def pca(self, w):
        v_x_ = [s.x_ for s in self.samples]
        independet_sample = self.copy().toIndependet()
        newold_sample = independet_sample.copy().toReturnFromIndependet(w)
        [s.to_slide(x_) for s, x_ in zip(newold_sample.samples, v_x_)]
        return independet_sample, newold_sample

    def toCreateLinearRegressionMNK(self, yi: int):
        self.line_R, self.line_R_f_test, self.line_R_f_quant = \
            self.multipleCorrelationCoefficient(yi)
        self.line_R *= self.line_R
        n = len(self) - 1
        x_ = np.array([s.x_ for s in self.samples])
        X_x = np.array([[x - x_[i] for x in s.raw]
                        for i, s in enumerate(self.samples) if i != yi])
        Y_T = np.array(self.samples[yi].raw)
        y_ = self.samples[yi].x_
        A_ = np.linalg.inv(X_x @ X_x.transpose()) @ (X_x) @ Y_T
        self.line_A = A_
        a0 = y_ - sum([A_[k] * x_[k] for k in range(n)])
        self.line_A0 = a0
        def f(*x): return a0 + A_ @ np.array(x)
        self.lineAccuracyParameters(f, yi)
        less_f, more_f = self.lineTrustIntervals(f)
        return less_f, f, more_f

    def lineAccuracyParameters(self, f, yi: int):
        n = len(self) - 1
        N = len(self.samples[0].raw)
        Y = self.samples[yi].raw
        X = np.array([self.samples[i].raw
                      for i in range(len(self)) if i != yi])
        C = np.linalg.inv(X @ X.transpose())
        self.line_C = C
        E = Y - f(*X)
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
        N = len(self.samples[0].raw)
        t = func.QuantileTStudent(1 - self.trust / 2, N - n)
        C = self.line_C

        def det_f(X: np.ndarray) -> np.ndarray:
            return t * self.line_S ** 0.5 * (
                1 + (X.transpose() @ C @ X).diagonal()) ** 0.5

        def less_f(*X): return f(*X) - det_f(np.array(X))
        def more_f(*X): return f(*X) + det_f(np.array(X))
        return less_f, more_f

    def toCreateLinearVariationPlane(self):
        n = len(self)
        m = n - 1
        _, newold_sample = self.pca(m)
        def x(k, i): return newold_sample[k].raw[i]
        N = len(newold_sample[0].raw)
        rand_ind = [np.random.randint(0, N-1)]
        while len(rand_ind) < n:
            rnd = np.random.randint(0, N-1)
            if rnd not in rand_ind:
                rand_ind.append(rnd)
        ind_p0 = rand_ind[0]
        mat = np.empty((n, m))
        for k in range(n):
            for i, ind_pi in enumerate(rand_ind[1:]):
                mat[k, i] = x(k, ind_pi) - x(k, ind_p0)

        def det_minor(i): return np.linalg.det(np.delete(mat, i, 0))

        # 0,  1,  2,  ... n-1, n
        # x1, x2, x3, ... xn, d
        line_var_par = np.empty((n + 1))
        line_var_par[n] = 0.0
        for i in range(n):
            det_from_minor_i = det_minor(i) * (-1) ** i
            line_var_par[i] = det_from_minor_i
            line_var_par[n] -= x(i, ind_p0) * det_from_minor_i

        self.line_var_par = -(line_var_par / line_var_par[m])
        self.line_var_par = np.delete(self.line_var_par, m)

        def line_var_f(*x):
            return sum([self.line_var_par[i] * x[i] for i in range(m)]
                       ) + self.line_var_par[m]

        return None, line_var_f, None

    def to_create_polynomial_regression(self, degree):
        model = PolynomialRegressionModel(degree)
        Y = self.samples[-1].raw
        X = np.array([s.raw for s in self.samples[:-1]])
        model.fit(X, Y)

        def f(*X):
            return model.predict(np.array(X))

        return None, f, None

    def agglomerative_clustering(self, n_clusters,
                                 metric='euclidean', linkage='average'):
        clusters, dend_items = ac.linkage(self.to_numpy(), n_clusters,
                                          linkage=linkage, metric=metric)
        [s.set_clusters(clusters, metric) for s in self.samples]
        return clusters, dend_items

    def k_means_clustering(self, n_clusters,
                           init="random", metric="euclidean"):
        kmeans = KMeans(n_clusters, init, metric)
        clusters = kmeans.fit(self.to_numpy())
        [s.set_clusters(clusters, metric) for s in self.samples]

    def split_on_clusters(self):
        clusters = self[0].clusters
        X = self.to_numpy()
        S = [X[cluster] for cluster in clusters]
        samples = []
        for sample, s in zip(self.samples, S):
            for x in s.T:
                samplingdata = SamplingData(x, sample.trust, True,
                                            sample.name, sample.ticks)
                samplingdata.toRanking()
                samplingdata.toCalculateCharacteristic()
                samples.append(samplingdata)
        return samples

    def nearest_neighbor_classification_scores(self,
                                               train_size=0.7,
                                               metric="euclidean"):
        nn = clstr.NeighborsClassifier(metric)
        accuracy, ppv, tpr, fpr, fnr =\
            self.get_scores_from_model(nn, train_size)
        self.acc_class = accuracy
        self.ppv_class = ppv
        self.tpr_class = tpr
        self.fpr_class = fpr
        self.fnr_class = fnr

    def mod_nearest_neighbor_classification_scores(self,
                                                   train_size=0.7,
                                                   metric="euclidean"):
        nnm = clstr.NeighborsModClassifier(metric)
        accuracy, ppv, tpr, fpr, fnr =\
            self.get_scores_from_model(nnm, train_size)
        self.acc_class = accuracy
        self.ppv_class = ppv
        self.tpr_class = tpr
        self.fpr_class = fpr
        self.fnr_class = fnr

    def k_nearest_neighbor_classification_scores(self,
                                                 train_size=0.7,
                                                 k_neighbors=3,
                                                 metric="euclidean"):
        knn = clstr.KNeighborsClassifier(k_neighbors, metric)
        accuracy, ppv, tpr, fpr, fnr =\
            self.get_scores_from_model(knn, train_size)
        self.acc_class = accuracy
        self.ppv_class = ppv
        self.tpr_class = tpr
        self.fpr_class = fpr
        self.fnr_class = fnr

    def get_scores_from_model(self, model, train_size):
        X = self.to_numpy()
        Y = self.get_tags_array()
        x_train, x_test, y_train, y_test = \
            self.test_train_split(X, Y, train_size)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        return self.scores(y_test, y_pred)

    def get_tags_array(self):
        clusters = self[0].clusters
        N = len(self[0].raw)
        Y = np.empty(N, dtype=int)
        for i, cluster in enumerate(clusters):
            Y[cluster] = i
        return Y

    def test_train_split(self, x, y, train_size=0.7):
        N = len(x)
        indexes = np.arange(N)
        np.random.shuffle(indexes)
        train_ind = indexes[:int(N * train_size)]
        test_ind = indexes[int(N * train_size):]
        return x[train_ind], x[test_ind], y[train_ind], y[test_ind]

    def scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        K = self.get_error_matrix(y_true, y_pred)
        n = K.shape[0]
        ppv = np.zeros(n)
        tpr = np.zeros(n)
        fpr = np.zeros(n)
        fnr = np.zeros(n)
        for j in range(n):
            if K[:, j].sum() == 0:
                ppv[j] = 0
            else:
                ppv[j] = K[j, j] / K[:, j].sum()
            tpr[j] = K[j, j] / K[j, :].sum()
            fpr[j] = 1 - tpr[j]
            fnr[j] = 1 - ppv[j]
        accuracy = K.diagonal().sum() / K.sum()
        return accuracy, ppv, tpr, fpr, fnr

    def logistic_regression_scores(self, train_size, train_speed, num_iter):
        lc = clstr.LogisticClassifier()
        X = self.to_numpy()
        Y = self.get_tags_array()
        x_train, x_test, y_train, y_test = \
            self.test_train_split(X, Y, train_size)
        lc.fit(x_train, y_train, train_speed, num_iter)
        fpr_arr, tpr_arr = self.roc_curve_data(x_test, y_test, lc)

        accuracy, ppv, tpr, fpr, fnr = self.scores(y_test, lc.predict(X))
        self.acc_class = accuracy
        self.ppv_class = ppv
        self.tpr_class = tpr
        self.fpr_class = fpr
        self.fnr_class = fnr

        return fpr_arr, tpr_arr

    def roc_curve_data(self, x, y_true: np.ndarray,
                       lc: clstr.LogisticClassifier):
        xrange = np.arange(1.0, -0.01, -0.01)
        fpr = np.empty(len(xrange))
        tpr = np.empty(len(xrange))
        for i, a in enumerate(xrange):
            y_pred = lc.predict(x, a)
            fpr_i, tpr_i = self.binary_fpr_tpr(y_true, y_pred)
            fpr[i] = fpr_i
            tpr[i] = tpr_i
        return fpr, tpr

    def binary_fpr_tpr(self, y_true: np.ndarray, y_pred: np.ndarray):
        K = self.get_error_matrix(y_true, y_pred)
        TP, FN = K[0, 0], K[0, 1]
        FP, TN = K[1, 0], K[1, 1]
        P, N = TP + FN, FP + TN
        TPR = TP / P
        FPR = FP / N
        return FPR, TPR

    def binary_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        K = self.get_error_matrix(y_true, y_pred)
        TP, FP = K[0, 0], K[0, 1]
        FN, TN = K[1, 0], K[1, 1]
        P, N = TP + FN, FP + TN
        P_ = TP + FP
        N_ = FN + TN
        sensitivity = TP / P
        FPR = FP / N
        FNR = FN / P
        specificity = TN / N
        precision = TP / P_
        NPV = TN / N_
        FDR = FP / P_
        FOR = FN / N_
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        char = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "PPV": precision,
            "NPV": NPV,
            "FPR": FPR,
            "FNR": FNR,
            "FDR": FDR,
            "FOR": FOR,
            "accuracy": accuracy
        }
        return char

    def get_error_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        n = y_true.max() - y_true.min() + 1
        K = np.zeros((n, n))
        for i, j in zip(y_true, y_pred):
            K[n - i - 1, n - j - 1] += 1
        return K
