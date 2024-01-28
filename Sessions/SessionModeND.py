from Sessions.SessionMode import SessionMode
from Datanalysis import SamplingDatas
from GUI import DialogWindow, SpinBox, ComboBox
from GUI.PlotWidget import PlotDendrogramWidget, PlotDialogWindow


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
        N = len(self.datas_displayed[0])
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

        plot_widget = PlotDendrogramWidget()
        d = PlotDialogWindow(plot=plot_widget,
                             size=(1333, 733))
        plot_widget.plot_observers(c, z)
        d.exec()

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

    def remove_clusters(self):
        for s in self.datas_displayed.samples:
            s.remove_clusters()
        self.update_sample()

    def split_on_clusters(self):
        a_s = self.datas_displayed.split_on_clusters()
        self.window.all_datas.append_samples(a_s)
        self.window.table.update_table()
