from GUI.WindowLayout import WindowLayout
import Datanalysis as da
import GUI.PlotWidget as cplt
from GUI import DialogWindow, SpinBox, ComboBox, DoubleSpinBox


class SessionMode:
    def __init__(self, window: WindowLayout):
        self.window = window
        self.plot_widget = window.plot_widget
        self.selected_regr_num = -1
        self.__supported_metrics = {
            "Евклідова": "euclidean", "Манхеттенська": "manhattan",
            "Чебишева": "chebyshev", "Мінковського": "minkowski",
            "Махаланобіса": "mahalanobis"
            }
        self.datas_displayed = self.create_active_sampling_datas()

    def create_active_sampling_datas(self):
        sampling = da.SamplingDatas(self.get_selected_samples())
        if len(sampling) > 1:
            sampling.toCalculateCharacteristic()
        return sampling

    def get_active_sampling_datas(self):
        return self.datas_displayed

    def get_all_datas(self) -> da.SamplingDatas:
        return self.window.all_datas

    def get_selected_indexes(self) -> list:
        return self.window.sel_indexes

    def get_selected_samples(self) -> list:
        return [self.window.all_datas[i] for i in self.get_selected_indexes()]

    def set_regression_number(self, number):
        self.selected_regr_num = number
        self.select_new_sample()

    def configure(self):
        self.create_plot_layout()

    def create_plot_layout(self):
        pass

    def get_active_samples(self):
        pass

    def auto_remove_anomalys(self) -> bool:
        pass

    def to_independent(self):
        pass

    def select_new_sample(self):
        self.update_sample(self.window.feature_area.get_number_classes())

    def update_sample(self, number_column=0):
        self.write_protocol()
        self.write_critetion()

    def write_protocol(self):
        act_sample = self.get_active_samples()
        protocol_text = da.ProtocolGenerator.getProtocol(act_sample)

        if not isinstance(act_sample, da.SamplingDatas):
            sampling_datas = self.get_active_sampling_datas()
            protocol = []
            if sampling_datas.classifier is not None:
                da.ProtocolGenerator.get_for_classification(sampling_datas,
                                                            protocol)
            protocol_text += "\n".join(protocol)

        self.window.protocol.setText(protocol_text)

    def write_critetion(self):
        self.window.criterion_protocol.setText("")

    def kmeans(self):
        k, metric, init = self.get_kmeans_parameters()
        active_samples = self.get_active_sampling_datas()
        active_samples.k_means_clustering(k, init, metric)
        self.update_sample()

    def get_kmeans_parameters(self):
        title1 = "Введіть кількість кластерів"
        title2 = "Відстань між об'єктами"
        title3 = "Вибір центрів кластерів"
        init = {"Випадковий": "random",
                "Перші k точок": "first"}
        N = len(self.get_active_sampling_datas()[0].raw)
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

        active_samples = self.get_active_sampling_datas()
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
        N = len(self.get_active_sampling_datas()[0].raw)
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
        for s in self.get_active_sampling_datas().samples:
            s.remove_clusters()
        self.update_sample()

    def split_on_clusters(self):
        a_s = self.get_active_sampling_datas().split_on_clusters()
        self.get_all_datas().append_samples(a_s)
        self.window.table.update_table()

    def nearest_neighbor_classification(self):
        train_size, metric = self.get_nearest_neighbor_parameters()
        act_sample = self.get_active_sampling_datas()
        act_sample.nearest_neighbor_classification_scores(
            train_size, metric)
        self.write_protocol()

    def mod_nearest_neighbor_classification(self):
        train_size, metric = self.get_nearest_neighbor_parameters()
        act_sample = self.get_active_sampling_datas()
        act_sample.nearest_neighbor_classification_scores(
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
        act_sample = self.get_active_sampling_datas()
        act_sample.k_nearest_neighbor_classification_scores(
            train_size, k, metric)
        self.write_protocol()

    def get_k_nearest_neighbor_parameters(self):
        title1 = "Навчальна вибірка"
        title2 = "Кількість сусідів"
        title3 = "Відстань між об'єктами"
        dialog_window = DialogWindow(
            form_args=[title1, DoubleSpinBox(min_v=0, max_v=1,
                                             decimals=5, value=0.7),
                       title2, SpinBox(min_v=1, max_v=100),
                       title3, ComboBox(self.__supported_metrics)])
        ret = dialog_window.get_vals()
        train_size = ret.get(title1)
        k = ret.get(title2)
        metric = ret.get(title3)
        return train_size, k, metric

    def logistic_regression(self):
        train_size, alpha, num_iter = self.get_logistic_regression_parameters()
        act_sample = self.get_active_sampling_datas()
        fpr, tpr = act_sample.logistic_regression_scores(
            train_size, alpha, num_iter)
        self.plot_roc_curve(fpr, tpr)
        self.update_sample()
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

    def plot_roc_curve(self, fpr, tpr):
        plot_widget = cplt.PlotRocCurveWidget()
        d = cplt.PlotDialogWindow(plot=plot_widget,
                                  size=(400, 400))
        plot_widget.plot_observers(fpr, tpr)
        self.keep_additional_window_is_open(d)
        d.show()

    def keep_additional_window_is_open(self, widget):
        self.additional_plot_widget = widget

    def discriminant_analysis(self):
        train_size = self.get_discriminant_analysis_parameters()
        act_sample = self.get_active_sampling_datas()
        act_sample.discriminant_analysis(train_size)
        self.update_sample()
        self.write_protocol()

    def get_discriminant_analysis_parameters(self):
        title1 = "Навчальна вибірка"
        dialog_window = DialogWindow(
            form_args=[title1, DoubleSpinBox(min_v=0, max_v=1,
                                             decimals=5, value=0.7)])
        ret = dialog_window.get_vals()
        train_size = ret.get(title1)
        return train_size
