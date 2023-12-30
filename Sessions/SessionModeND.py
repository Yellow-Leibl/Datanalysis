from Sessions.SessionMode import SessionMode
from Datanalysis import SamplingDatas
from GUI import DialogWindow, SpinBox


class SessionModeND(SessionMode):
    def __init__(self, window):
        super().__init__(window)
        self.datas_displayed = None

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

    def pca(self):
        n = len(self.datas_displayed)
        title = "За кількістю головних компонент"
        dialog_window = DialogWindow(
            form_args=[title, SpinBox(min_v=1, max_v=n)])
        ret = dialog_window.get_vals()
        w = ret.get(title)

        active_samples = self.get_active_samples()
        ind, retn = active_samples.principalComponentAnalysis(w)
        self.window.all_datas.append_samples(ind.samples)
        self.window.all_datas.append_samples(retn.samples)
        self.window.table.update_table()
