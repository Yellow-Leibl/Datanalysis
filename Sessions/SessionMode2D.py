from Sessions.SessionMode import SessionMode
from Datanalysis import DoubleSampleData, ProtocolGenerator, SamplingDatas
from GUI import DialogWindow, SpinBox


class SessionMode2D(SessionMode):
    def __init__(self, window):
        super().__init__(window)
        self.d2 = None
        self.hist_data_2d = None

    def create_plot_layout(self):
        self.plot_widget.show_2d_plot()

    def create_d2_sample(self) -> DoubleSampleData:
        x, y = self.get_selected_samples()
        return DoubleSampleData(x, y, self.window.feature_area.get_trust())

    def get_active_samples(self) -> DoubleSampleData:
        return self.d2

    def configure(self):
        super().configure()
        self.d2 = self.create_d2_sample()
        self.window.feature_area.silent_change_number_classes(0)
        self.window.feature_area.set_maximum_column_number(len(self.d2.x._x))

    def auto_remove_anomalys(self) -> bool:
        act_sample = self.get_active_samples()
        hist_data = act_sample.get_histogram_data(
            self.window.feature_area.get_number_classes())
        deleted_items = act_sample.autoRemoveAnomaly(hist_data)
        self.window.showMessageBox("Видалення аномалій",
                                   f"Було видалено {deleted_items} аномалій")
        return deleted_items

    def to_independent(self):
        self.d2.toIndependent()

    def pca(self):
        sd = SamplingDatas([self.d2.x, self.d2.y])
        sd.toCalculateCharacteristic()
        ind, retn = sd.pca(1)
        self.get_all_datas().append_samples(ind.samples)
        self.get_all_datas().append_samples(retn.samples)
        self.window.table.update_table()

    def update_sample(self, number_column: int = 0):
        self.d2.set_trust(self.window.feature_area.get_trust())
        self.d2.toCalculateCharacteristic()
        self.hist_data_2d = self.d2.get_histogram_data(number_column)
        self.window.feature_area.silent_change_number_classes(
            len(self.hist_data_2d))
        self.plot_widget.plot_2d_with_details(self.d2, self.hist_data_2d)
        self.drawReproductionSeries2D(self.d2)
        super().update_sample(number_column)

    def drawReproductionSeries2D(self, d2):
        f = self.toCreateReproductionFunc2D(d2, self.selected_regr_num)
        if f is None:
            return
        self.plot_widget.plot2DReproduction(d2, *f)

    def toCreateReproductionFunc2D(self, d_d: DoubleSampleData, func_num):
        if func_num == 5:
            return d_d.toCreateLinearRegressionMNK()
        elif func_num == 6:
            return d_d.toCreateLinearRegressionMethodTeila()
        elif func_num == 7:
            return d_d.toCreateParabolicRegression()
        elif func_num == 8:
            return d_d.toCreateKvazi8()
        elif func_num == 11:
            degree = self.get_degree()
            return d_d.to_create_polynomial_regression(degree)

    def get_degree(self):
        title = "Введіть степінь полінома"
        dialog_window = DialogWindow(
            form_args=[title, SpinBox(min_v=1, max_v=10)])
        ret = dialog_window.get_vals()
        return ret.get(title)

    def write_critetion(self):
        # TODO: move this into ProtocolGenerator and print it in protocol
        text = ProtocolGenerator.xixi_test_2d(self.d2, self.hist_data_2d)
        self.window.criterion_protocol.setText(text)

    def remove_clusters(self):
        self.d2.x.remove_clusters()
        self.d2.y.remove_clusters()
        self.update_sample()
