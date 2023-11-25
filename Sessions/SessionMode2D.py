from Sessions.SessionMode import SessionMode
from Datanalysis.DoubleSampleData import DoubleSampleData
from Datanalysis.ProtocolGenerator import ProtocolGenerator


class SessionMode2D(SessionMode):
    def __init__(self, window):
        super().__init__(window)
        self.d2 = None
        self.hist_data_2d = None

    def create_plot_layout(self):
        self.window.plot_widget.create2DPlot()

    def get_active_samples(self) -> DoubleSampleData:
        x = self.window.all_datas[self.window.sel_indexes[0]]
        y = self.window.all_datas[self.window.sel_indexes[1]]
        return DoubleSampleData(x, y, self.window.getTrust())

    def auto_remove_anomaly(self) -> bool:
        act_sample = self.get_active_samples()
        hist_data = act_sample.get_histogram_data(
            self.window.getNumberClasses())
        deleted_items = act_sample.autoRemoveAnomaly(hist_data)
        self.window.showMessageBox("Видалення аномалій",
                                   f"Було видалено {deleted_items} аномалій")
        return deleted_items

    def to_independent(self):
        self.d2.toIndependent()

    def update_graphics(self, number_column: int = 0):
        x = self.window.all_datas[self.window.sel_indexes[0]]
        y = self.window.all_datas[self.window.sel_indexes[1]]
        self.d2 = DoubleSampleData(x, y, self.window.getTrust())
        self.d2.toCalculateCharacteristic()
        self.hist_data_2d = self.d2.get_histogram_data(number_column)
        self.window.silentChangeNumberClasses(len(self.hist_data_2d))
        self.window.plot_widget.plot2D(self.d2, self.hist_data_2d)
        self.drawReproductionSeries2D(self.d2)

    def drawReproductionSeries2D(self, d2):
        f = self.toCreateReproductionFunc2D(d2, self.window.selected_regr_num)
        if f is None:
            return
        self.window.plot_widget.plot2DReproduction(d2, *f)

    def toCreateReproductionFunc2D(self, d_d: DoubleSampleData, func_num):
        if func_num == 6:
            return d_d.toCreateLinearRegressionMNK()
        elif func_num == 7:
            return d_d.toCreateLinearRegressionMethodTeila()
        elif func_num == 8:
            return d_d.toCreateParabolicRegression()
        elif func_num == 9:
            return d_d.toCreateKvazi8()

    def write_protocol(self):
        self.window.protocol.setText(
            ProtocolGenerator.getProtocol(self.d2))

    def write_critetion(self):
        isNormal = self.d2.xiXiTest(self.hist_data_2d)
        self.window.criterion_protocol.setText(
            self.d2.xiXiTestProtocol(isNormal))
