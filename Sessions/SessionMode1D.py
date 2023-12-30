from Sessions.SessionMode import SessionMode
from Datanalysis import SamplingData, ProtocolGenerator


class SessionMode1D(SessionMode):
    def __init__(self, window):
        super().__init__(window)
        self.d1_regr_F = None

    def create_plot_layout(self):
        self.plot_widget.show_1d_plot()

    def get_active_samples(self) -> SamplingData:
        return self.window.all_datas[self.window.sel_indexes[0]]

    def configure(self):
        super().configure()
        d = self.get_active_samples()
        self.window.feature_area.silent_change_number_classes(0)
        self.window.feature_area.set_maximum_column_number(len(d._x))

    def auto_remove_anomalys(self) -> bool:
        return self.get_active_samples().auto_remove_anomalys()

    def remove_anomaly_with_range(self):
        minmax = self.window.feature_area.get_borders()
        self.get_active_samples().remove(minmax[0], minmax[1])
        self.update_sample()

    def update_sample(self, number_column: int = 0):
        d = self.get_active_samples()
        d.set_trust(self.window.feature_area.get_trust())
        self.window.feature_area.set_borders(d.min, d.max)
        hist_data = d.get_histogram_data(number_column)
        self.window.feature_area.silent_change_number_classes(len(hist_data))
        self.plot_widget.plot1D(d, hist_data)
        self.drawReproductionSeries1D()

    def drawReproductionSeries1D(self):
        d = self.get_active_samples()
        f = self.toCreateReproductionFunc(d, self.selected_regr_num)
        if f is None:
            return
        h = abs(d.max - d.min) / self.window.feature_area.get_number_classes()
        f = d.toCreateTrustIntervals(*(*f, h))
        self.d1_regr_F = f[2]
        self.plot_widget.plot1DReproduction(d, *f)

    def toCreateReproductionFunc(self, d: SamplingData, func_num):
        if func_num == 0:
            return d.toCreateNormalFunc()
        elif func_num == 1:
            return d.toCreateUniformFunc()
        elif func_num == 2:
            return d.toCreateExponentialFunc()
        elif func_num == 3:
            return d.toCreateWeibullFunc()
        elif func_num == 4:
            return d.toCreateArcsinFunc()

    def write_critetion(self):
        # TODO: move this into ProtocolGenerator and print it in protocol
        d = self.get_active_samples()
        if self.d1_regr_F is None:
            return
        self.window.criterion_protocol.setText(
            self.writeCritetion1DSample(d, self.d1_regr_F))

    def writeCritetion1DSample(self, d: SamplingData, F):
        n = self.window.feature_area.get_number_classes()
        hist_class = d.get_histogram_data(n)
        return '\n' + ProtocolGenerator.xixi_test_1d(d, hist_class, F)
