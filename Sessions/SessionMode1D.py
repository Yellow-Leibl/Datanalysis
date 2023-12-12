from Sessions.SessionMode import SessionMode
from Datanalysis import SamplingData, ProtocolGenerator


class SessionMode1D(SessionMode):
    def __init__(self, window):
        super().__init__(window)
        self.d1_regr_F = None

    def create_plot_layout(self):
        self.window.plot_widget.create1DPlot()

    def get_active_samples(self) -> SamplingData:
        return self.window.all_datas[self.window.sel_indexes[0]]

    def auto_remove_anomalys(self) -> bool:
        return self.get_active_samples().auto_remove_anomalys()

    def update_graphics(self, number_column: int = 0):
        d = self.get_active_samples()
        d.setTrust(self.window.feature_area.get_trust())
        self.window.feature_area.set_borders(d.min, d.max)
        hist_data = d.get_histogram_data(number_column)
        self.window.feature_area.silent_change_number_classes(len(hist_data))
        self.window.plot_widget.plot1D(d, hist_data)
        self.drawReproductionSeries1D()

    def drawReproductionSeries1D(self):
        d = self.get_active_samples()
        f = self.toCreateReproductionFunc(d, self.selected_regr_num)
        if f is None:
            return
        h = abs(d.max - d.min) / self.window.feature_area.get_number_classes()
        f = d.toCreateTrustIntervals(*(*f, h))
        self.d1_regr_F = f[2]
        self.window.plot_widget.plot1DReproduction(d, *f)

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

    def write_protocol(self):
        d = self.get_active_samples()
        self.window.protocol.setText(ProtocolGenerator.getProtocol(d))

    def write_critetion(self):
        d = self.get_active_samples()
        if self.d1_regr_F is None:
            return
        self.window.criterion_protocol.setText(
            self.writeCritetion1DSample(d, self.d1_regr_F))

    def writeCritetion1DSample(self, d: SamplingData, F):
        criterion_text = d.kolmogorovTestProtocol(d.kolmogorovTest(F))
        try:
            xi_test_result = d.xiXiTest(
                F, d.get_histogram_data(
                    self.window.feature_area.get_number_classes()))
        except ZeroDivisionError:
            xi_test_result = False
        criterion_text += '\n' + d.xiXiTestProtocol(xi_test_result)
        return criterion_text
