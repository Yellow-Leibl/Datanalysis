from Sessions.SessionMode import SessionMode
from Datanalysis import SamplingDatas
from GUI import DialogWindow, SpinBox, ComboBox, DoubleSpinBox
import GUI.PlotWidget as cplt


class SessionModeND(SessionMode):
    def __init__(self, window):
        super().__init__(window)
        self.datas_displayed = None
        self.__supported_metrics = {
            "Евклідова": "euclidean", "Манхеттенська": "manhattan",
            "Чебишева": "chebyshev", "Мінковського": "minkowski",
            "Махаланобіса": "mahalanobis"
            }

    def create_plot_layout(self):
        self.plot_widget.create_nd_plot(len(self.window.sel_indexes))

    def create_n_samples(self) -> SamplingDatas:
        samples = [self.window.all_datas[i] for i in self.window.sel_indexes]
        return SamplingDatas(samples, self.window.feature_area.get_trust())

    def get_active_samples(self) -> SamplingDatas:
        return self.datas_displayed

    def auto_remove_anomalys(self) -> bool:
        act_sample = self.get_active_samples()
        hist_data = act_sample.get_histogram_data(
            self.window.feature_area.get_number_classes())
        deleted_items = act_sample.autoRemoveAnomaly(hist_data)
        self.window.showMessageBox("Видалення аномалій",
                                   f"Було видалено {deleted_items} аномалій")
        return deleted_items

    def to_independent(self):
        self.datas_displayed.toIndependent()

    def update_sample(self, number_column=0):
        self.datas_displayed = self.create_n_samples()
        self.datas_displayed.toCalculateCharacteristic()
        self.plot_widget.plotND(self.datas_displayed, number_column)
        self.drawReproductionSeriesND(self.datas_displayed)
        super().update_sample(number_column)

    def drawReproductionSeriesND(self, datas):
        f = self.toCreateReproductionFuncND(datas, self.selected_regr_num)
        if f is None:
            return
        self.plot_widget.plotDiagnosticDiagram(datas, *f)

    def toCreateReproductionFuncND(self, datas: SamplingDatas, func_num):
        if func_num == 9:
            return datas.toCreateLinearRegressionMNK(len(datas) - 1)
        elif func_num == 10:
            return datas.toCreateLinearVariationPlane()
        elif func_num == 11:
            degree = self.get_degree()
            return datas.to_create_polynomial_regression(degree)

    def get_degree(self):
        title = "Введіть степінь полінома"
        dialog_window = DialogWindow(
            form_args=[title, SpinBox(min_v=1, max_v=10)])
        ret = dialog_window.get_vals()
        return ret.get(title)

    def pca(self):
        n = len(self.datas_displayed)
        title = "За кількістю головних компонент"
        dialog_window = DialogWindow(
            form_args=[title, SpinBox(min_v=1, max_v=n)])
        ret = dialog_window.get_vals()
        w = ret.get(title)

        active_samples = self.get_active_samples()
        ind, retn = active_samples.pca(w)
        self.window.all_datas.append_samples(ind.samples)
        self.window.all_datas.append_samples(retn.samples)
        self.window.table.update_table()

    def kmeans(self):
        k, metric, init = self.get_kmeans_parameters()
        active_samples = self.get_active_samples()
        active_samples.k_means_clustering(k, init, metric)
        self.update_sample()

    def get_kmeans_parameters(self):
        title1 = "Введіть кількість кластерів"
        title2 = "Відстань між об'єктами"
        title3 = "Вибір центрів кластерів"
        init = {"Випадковий": "random",
                "Перші k точок": "first"}
        N = len(self.datas_displayed[0].raw)
        dialog_window = DialogWindow(
            form_args=[title1, SpinBox(min_v=2, max_v=N-1),
                       title2, ComboBox(self.__supported_metrics),
                       title3, ComboBox(init)])
        ret = dialog_window.get_vals()
        k = ret.get(title1)
        metric = ret.get(title2)
        init = ret.get(title3)
        return k, metric, init

    def agglomerative_clustering(self):
        k, metric, linkage = self.get_agglomerative_parameters()

        active_samples = self.get_active_samples()
        c, z = active_samples.agglomerative_clustering(k, metric, linkage)
        self.update_sample()
        self.show_dendogram_plot(c, z)

    def get_agglomerative_parameters(self):
        title1 = "Введіть кількість кластерів"
        title2 = "Відстань між об'єктами"
        title3 = "Відстань між кластерами"
        linkage = {"Найближчого сусіда": "nearest",
                   "Найвіддаленішого сусіда": "furthest",
                   "Зваженого середнього": "average",
                   "Незваженого середнього": "unweighted",
                   "Медіанного": "median",
                   "Центроїдного": "centroid",
                   "Уорда": "wards"
                   }
        N = len(self.datas_displayed[0].raw)
        dialog_window = DialogWindow(
            form_args=[title1, SpinBox(min_v=2, max_v=N-1),
                       title2, ComboBox(self.__supported_metrics),
                       title3, ComboBox(linkage)])
        ret = dialog_window.get_vals()
        k = ret.get(title1)
        metric = ret.get(title2)
        linkage = ret.get(title3)
        return k, metric, linkage

    def show_dendogram_plot(self, c, z):
        plot_widget = cplt.PlotDendrogramWidget()
        d = cplt.PlotDialogWindow(plot=plot_widget,
                                  size=(1333, 733))
        plot_widget.plot_observers(c, z)
        self.keep_additional_window_is_open(d)
        d.show()

    def remove_clusters(self):
        for s in self.datas_displayed.samples:
            s.remove_clusters()
        self.update_sample()

    def split_on_clusters(self):
        a_s = self.datas_displayed.split_on_clusters()
        self.window.all_datas.append_samples(a_s)
        self.window.table.update_table()

    def nearest_neighbor_classification(self):
        train_size, metric = self.get_nearest_neighbor_parameters()
        self.datas_displayed.nearest_neighbor_classification_scores(
            train_size, metric)
        self.write_protocol()

    def mod_nearest_neighbor_classification(self):
        train_size, metric = self.get_nearest_neighbor_parameters()
        self.datas_displayed.nearest_neighbor_classification_scores(
            train_size, metric)
        self.write_protocol()

    def get_nearest_neighbor_parameters(self):
        title1 = "Навчальна вибірка"
        title2 = "Відстань між об'єктами"
        dialog_window = DialogWindow(
            form_args=[title1, DoubleSpinBox(min_v=0, max_v=1,
                                             decimals=5, value=0.7),
                       title2, ComboBox(self.__supported_metrics)])
        ret = dialog_window.get_vals()
        train_size = ret.get(title1)
        metric = ret.get(title2)
        return train_size, metric

    def k_nearest_neighbor_classification(self):
        train_size, k, metric = self.get_k_nearest_neighbor_parameters()
        self.datas_displayed.k_nearest_neighbor_classification_scores(
            train_size, k, metric)
        self.write_protocol()

    def get_k_nearest_neighbor_parameters(self):
        title1 = "Навчальна вибірка"
        title2 = "Кількість сусідів"
        title3 = "Відстань між об'єктами"
        dialog_window = DialogWindow(
            form_args=[title1, DoubleSpinBox(min_v=0, max_v=1,
                                             decimals=5, value=0.7),
                       title2, ComboBox(self.__supported_metrics),
                       title3, SpinBox(min_v=1, max_v=100)])
        ret = dialog_window.get_vals()
        train_size = ret.get(title1)
        k = ret.get(title2)
        metric = ret.get(title3)
        return train_size, k, metric

    def logistic_regression(self):
        train_size, alpha, num_iter = self.get_logistic_regression_parameters()
        fpr, tpr = self.datas_displayed.logistic_regression_scores(
            train_size, alpha, num_iter)
        plot_widget = cplt.PlotRocCurveWidget()
        d = cplt.PlotDialogWindow(plot=plot_widget,
                                  size=(400, 400))
        plot_widget.plot_observers(fpr, tpr)
        self.keep_additional_window_is_open(d)
        d.show()
        self.write_protocol()

    def get_logistic_regression_parameters(self):
        title1 = "Навчальна вибірка"
        title2 = "Швидкість навчання"
        title3 = "Кількість ітерацій"
        dialog_window = DialogWindow(
            form_args=[title1, DoubleSpinBox(min_v=0, max_v=1,
                                             decimals=5, value=0.7),
                       title2, DoubleSpinBox(min_v=0, max_v=1,
                                             decimals=5, value=0.1),
                       title3, SpinBox(min_v=1, max_v=1000, value=100)])
        ret = dialog_window.get_vals()
        train_size = ret.get(title1)
        alpha = ret.get(title2)
        num_iter = ret.get(title3)
        return train_size, alpha, num_iter

    def keep_additional_window_is_open(self, widget):
        self.additional_plot_widget = widget
