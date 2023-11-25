from Sessions.SessionMode import SessionMode
from Datanalysis.SamplingDatas import SamplingDatas
from Datanalysis.ProtocolGenerator import ProtocolGenerator


class SessionModeND(SessionMode):
    def __init__(self, window):
        super().__init__(window)
        self.datas_displayed = None

    def create_plot_layout(self):
        self.window.plot_widget.createNDPlot(len(self.window.sel_indexes))

    def get_active_samples(self) -> SamplingDatas:
        return self.datas_displayed

    def auto_remove_anomaly(self) -> bool:
        act_sample = self.get_active_samples()
        hist_data = act_sample.get_histogram_data(
            self.window.getNumberClasses())
        deleted_items = act_sample.autoRemoveAnomaly(hist_data)
        self.window.showMessageBox("Видалення аномалій",
                                   f"Було видалено {deleted_items} аномалій")
        return deleted_items

    def to_independent(self):
        self.datas_displayed.toIndependent()

    def update_graphics(self, number_column: int = 0):
        samples = [self.window.all_datas[i] for i in self.window.sel_indexes]
        self.datas_displayed = SamplingDatas(samples, self.window.getTrust())
        self.datas_displayed.toCalculateCharacteristic()
        self.window.plot_widget.plotND(self.datas_displayed, number_column)
        self.drawReproductionSeriesND(self.datas_displayed)

    def drawReproductionSeriesND(self, datas):
        f = self.toCreateReproductionFuncND(
            datas, self.window.selected_regr_num)
        if f is None:
            return
        self.window.plot_widget.plotDiagnosticDiagram(datas, *f)

    def toCreateReproductionFuncND(self, datas: SamplingDatas, func_num):
        if func_num == 11:
            return datas.toCreateLinearRegressionMNK(len(datas) - 1)
        elif func_num == 12:
            return datas.toCreateLinearVariationPlane()

    def write_protocol(self):
        self.window.protocol.setText(
            ProtocolGenerator.getProtocol(self.datas_displayed))

    def write_critetion(self):
        self.window.criterion_protocol.setText("")