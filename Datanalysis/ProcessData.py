from Datanalysis import SamplingData
import Datanalysis.functions as func
import Datanalysis.SamplesTools as stool
import numpy as np


class ProcessData:
    def __init__(self, data: SamplingData):
        self.data = data

    def set_trust(self, trust: float):
        self.data.set_trust(trust)

    def get_histogram_data(self, number_column):
        return self.data.get_histogram_data(number_column)

    def get_intensity_function(self, column_number):
        if column_number <= 0:
            column_number = stool.calculate_m(len(self.data.raw))
            column_number = min(column_number, len(self.data._x))
        teta = np.linspace(0.0, self.data.max, column_number)

        n = np.empty(column_number)
        for i in range(column_number-1):
            n[i] = len([x for x in self.data.raw if teta[i] < x <= teta[i+1]])

        def N(i):
            return sum([n[j] for j in range(i)])
        N_g = len(self.data.raw)

        alpha_arr = np.empty(column_number - 1)
        h = (teta[-1] - teta[0]) / (column_number - 1)
        for i in range(column_number - 1):
            alpha_arr[i] = n[i] / ((N_g - N(i)) * h)

        to_del = []

        for i in range(column_number - 2):
            alphai = alpha_arr[i]
            alphai1 = alpha_arr[i + 1]
            t = (alphai1 - alphai) / ((n[i] - 1) * alphai1 ** 2 +
                                      (n[i + 1] - 1) * alphai ** 2) ** 0.5 *\
                ((n[i] * n[i + 1] - 2) / (n[i] + n[i + 1])) ** 0.5
            v = n[i] + n[i + 1] - 2
            t_q = func.QuantileTStudent(1 - self.data.trust / 2, v)
            if abs(t) <= t_q:
                alpha_arr[i] = (alpha_arr[i] + alpha_arr[i+1]) / 2
                alpha_arr[i + 1] = alpha_arr[i]
                to_del.append(i)
                n[i + 1] = n[i] + n[i + 1]
                n[i] = 0
        alpha_arr = np.delete(alpha_arr, to_del)
        return alpha_arr

    def to_calculate_characteristics(self):
        self.is_stationary_process()
        self.intens_stat = 1 / self.data.x_

    def is_stationary_process(self):
        _, F, _ = self.data.to_create_exp_func()
        res = self.data.kolmogorov_test(F)
        self.kolmogorov_pz_exp = self.data.kolmogorov_pz
        self.kolmogorov_alpha = self.data.kolmogorov_alpha_zgodi
        return res
