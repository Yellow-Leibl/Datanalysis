import sys
import os

from PyQt6.QtWidgets import QFileDialog, QTableWidgetItem, QApplication

from Datanalysis.SamplingDatas import SamplingDatas
from Datanalysis.DoubleSampleData import DoubleSampleData
from Datanalysis.SamplingData import SamplingData
from mainlayout import MainLayout
from GeneralConstants import dict_edit, dict_regression, Edit


class Window(MainLayout):
    def __init__(self, file: str, is_file_name: bool = True):
        super().__init__()  # layout here
        self.datas = SamplingDatas()
        self.number_sample: list[int] = []
        self.reproduction_1d_F = None
        self.d2_active: list[int] = []
        self.datas_active: list[int] = []
        self.datas_crits: list[list] = []
        self.regr_num = -1
        if is_file_name:
            self.openFile(file)
        else:
            all_file = file.split('\n')
            self.loadFromData(all_file)
        # temp
        self.autoSelect()

    def autoSelect(self):
        # self.openFile("data/self/3lines_500_2.txt")
        # self.openFile("data/self/3lines_500_3.txt")
        # self.number_sample = [2, 1, 3]
        self.createPlotLayout(len(self.number_sample))
        self.regr_num = 9
        self.sampleChanged()

    def openFile(self, file_name: str):
        if file_name == '':
            file_name, _ = QFileDialog().getOpenFileName(
                self, "Відкрити файл", os.getcwd(), "Bci файли (*)")
        try:
            with open(file_name, 'r') as file:
                self.loadFromData(file.readlines())
        except FileNotFoundError:
            print(f"\"{file_name}\" not found")

    def loadFromData(self, all_file: list[str]):
        self.datas.append(all_file)
        self.writeTable()

    def saveFileAct(self):
        file_name, _ = QFileDialog().getSaveFileName(
            self, "Зберегти файл", os.getcwd(), "Bci файли (*)")
        with open(file_name, 'w') as file:
            def safe_access(lst: list, i):
                return str(lst[i]) if len(lst) > i else ''
            file.write('\n'.join(
                [' '.join([safe_access(self.datas[j].getRaw(), i)
                           for j in range(len(self.datas))])
                 for i in range(self.datas.getMaxDepthRawData())]))

    def sampleChanged(self):
        self.updateGraphics(self.getNumberClasses())
        self.writeProtocol()
        self.writeCritetion()
        self.writeTable()

    def selectSampleOrReproduction(self):
        self.updateGraphics(self.getNumberClasses())
        self.writeProtocol()
        self.writeCritetion()

    def getActiveSamples(self):
        if self.is1d():
            return self.datas[self.number_sample[0]]
        elif self.is2d():
            return self.d2
        elif self.isNd():
            return self.datas_act

    def editSampleEvent(self):
        act = Edit(dict_edit[self.sender().text()])
        if act == Edit.TRANSFORM:
            [self.datas[i].toLogarithmus10() for i in self.number_sample]
        elif act == Edit.STANDARTIZATION:
            [self.datas[i].toStandardization() for i in self.number_sample]
        elif act == Edit.SLIDE:
            [self.datas[i].toSlide() for i in self.number_sample]
        elif act == Edit.DELETE_ANOMALY:
            if not self.autoRemoveAnomaly():
                return
        self.sampleChanged()

    def duplicateSample(self):
        sel = self.getSelectedRows()
        for i in sel:
            self.datas.appendSample(self.datas[i].copy())
        self.writeTable()

    def removeAnomaly(self):
        if self.is1d():
            minmax = self.getMinMax()
            self.datas[self.number_sample[0]].remove(minmax[0], minmax[1])
            self.sampleChanged()

    def autoRemoveAnomaly(self) -> bool:
        if self.is1d():
            return self.datas[self.number_sample[0]].autoRemoveAnomaly()
        elif self.is2d():
            hist_data = self.d2.get_histogram_data(self.getNumberClasses())
            return self.d2.autoRemoveAnomaly(hist_data)
        return False

    def drawSamples(self):
        sel = self.getSelectedRows()
        if 0 == len(sel) or sel == self.number_sample:
            return
        if self.datas_active == sel or \
           self.d2_active == sel:
            return
        self.createPlotLayout(len(sel))
        self.number_sample = sel
        self.d2_active = [-1, -1]
        self.datas_active = []
        self.regr_num = -1
        self.silentChangeNumberClasses(0)
        self.setMaximumColumnNumber(max(len(d._x) for d in self.datas.samples))
        self.selectSampleOrReproduction()

    def deleteSamples(self):
        sel = self.getSelectedRows()
        p = 0
        for i in sel:
            if i - p in self.number_sample:
                self.number_sample = []
            self.datas.pop(i - p)
            p += 1
        self.writeTable()

    def is1d(self) -> bool:
        return len(self.number_sample) == 1

    def is2d(self) -> bool:
        return len(self.number_sample) == 2

    def is3d(self) -> bool:
        return len(self.number_sample) == 3

    def isNd(self) -> bool:
        return len(self.number_sample) >= 2

    def setReproductionSeries(self):
        self.regr_num = dict_regression[self.sender().text()]
        self.selectSampleOrReproduction()

    def changeTrust(self, trust: float):
        s = self.getActiveSamples()
        if s is not None:
            s.setTrust(trust)
            self.sampleChanged()

    def writeTable(self):
        self.table.clear()
        self.table.setColumnCount(self.datas.getMaxDepthRangeData() + 1)
        self.table.setRowCount(len(self.datas))
        for s in range(len(self.datas)):
            d = self.datas[s]
            self.table.setItem(s, 0, QTableWidgetItem(f"N={len(d.getRaw())}"))
            for i, e in enumerate(d._x):
                self.table.setItem(s, i + 1, QTableWidgetItem(f"{e:.5}"))
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()

    def numberColumnChanged(self):
        self.selectSampleOrReproduction()

    def writeProtocol(self):
        s = self.getActiveSamples()
        if s is not None:
            self.protocol.setText(s.getProtocol())

    def writeCritetion(self):
        if self.is1d():
            if self.reproduction_1d_F is None:
                return
            d = self.datas[self.number_sample[0]]
            self.criterion_protocol.setText(self.writeCritetion1DSample(
                d, self.reproduction_1d_F))
        elif self.is2d():
            isNormal = self.d2.xiXiTest(self.hist_data_2d)
            self.criterion_protocol.setText(self.d2.xiXiTestProtocol(isNormal))
        elif self.isNd():
            self.criterion_protocol.setText("")

    def updateGraphics(self, number_column: int = 0):
        if self.is1d():
            d = self.datas[self.number_sample[0]]
            self.setMinMax(d.min, d.max)
            hist_data = d.get_histogram_data(number_column)
            self.silentChangeNumberClasses(len(hist_data))
            self.plot_widget.plot1D(d, hist_data)
            self.drawReproductionSeries1D()
        elif self.is2d():
            self.d2_active = self.number_sample
            x = self.datas[self.number_sample[0]]
            y = self.datas[self.number_sample[1]]
            self.d2 = DoubleSampleData(x, y)
            self.d2.toCalculateCharacteristic()
            self.hist_data_2d = self.d2.get_histogram_data(number_column)
            self.silentChangeNumberClasses(len(self.hist_data_2d))
            self.plot_widget.plot2D(self.d2, self.hist_data_2d)
            self.drawReproductionSeries2D()
        elif self.isNd():
            samples = [self.datas[i] for i in self.number_sample]
            self.datas_active = self.number_sample
            self.datas_act = SamplingDatas(samples)
            self.datas_act.toCalculateCharacteristic()
            self.plot_widget.plotND(self.datas_act, number_column)
            if self.regr_num == 9:
                self.plot_widget.plotDiagnosticDiagram(self.datas_act)

    def drawReproductionSeries1D(self):
        d = self.datas[self.number_sample[0]]
        f, lF, F, hF = self.toCreateReproductionFunc(d, self.regr_num)
        if f is None:
            return
        self.reproduction_1d_F = F
        self.plot_widget.plot1DReproduction(d, f, lF, F, hF)

    def toCreateReproductionFunc(self, d: SamplingData, func_num):
        if func_num == 0:
            f, F, DF = d.toCreateNormalFunc()
        elif func_num == 1:
            f, F, DF = d.toCreateUniformFunc()
        elif func_num == 2:
            f, F, DF = d.toCreateExponentialFunc()
        elif func_num == 3:
            f, F, DF = d.toCreateWeibullFunc()
        elif func_num == 4:
            f, F, DF = d.toCreateArcsinFunc()
        else:
            return None, None, None, None

        h = abs(d.max - d.min) / self.getNumberClasses()
        return d.toCreateTrustIntervals(f, F, DF, h)

    def writeCritetion1DSample(self, d: SamplingData, F):
        criterion_text = d.kolmogorovTestProtocol(d.kolmogorovTest(F))
        try:
            xi_test_result = d.xiXiTest(
                F, d.get_histogram_data(self.getNumberClasses()))
        except ZeroDivisionError:
            xi_test_result = False
        criterion_text += '\n' + d.xiXiTestProtocol(xi_test_result)
        return criterion_text

    def drawReproductionSeries2D(self):
        tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f = \
            self.toCreateReproductionFunc2D(self.d2, self.regr_num)
        self.plot_widget.plot2DReproduction(
            self.d2, tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f)

    def toCreateReproductionFunc2D(self, d_d: DoubleSampleData, func_num):
        if func_num == 5:
            tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f = \
                d_d.toCreateLinearRegressionMNK()
        elif func_num == 6:
            tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f = \
                d_d.toCreateLinearRegressionMethodTeila()
        elif func_num == 7:
            tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f = \
                d_d.toCreateParabolicRegression()
        elif func_num == 8:
            tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f = \
                d_d.toCreateKvazi8()
        else:
            return None, None, None, None, None, None, None
        return tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f

    def drawReproductionSeries3D(self):
        self.plot_widget.plot3DReproduction(self.datas_act)

    def linearModelsCrit(self, trust: float):
        sel = self.getSelectedRows()
        if len(sel) == 4:
            res = self.datas.ident2ModelsLine(
                [self.datas[i] for i in sel], trust)
            title = "Лінійна регресія"
            descr = "Моделі регресійних прямих\n" + \
                f"({sel[0]}, {sel[1]}) і ({sel[2]}, {sel[3]})"
            if type(res) == bool:
                if res:
                    self.showMessageBox(title, descr + " - ідентичні")
                else:
                    self.showMessageBox(title, descr + " - неідентичні")
            elif type(res) == str:
                self.showMessageBox(title, descr +
                                    " - мають випадкову різницю регресій")

    def homogeneityAndIndependence(self, trust: float):
        sel = self.getSelectedRows()
        if len(sel) == 1:
            P = self.datas[sel[0]].critetionAbbe()
            title = "Критерій Аббе"
            if P > trust:
                self.showMessageBox(title, f"{P:.5} > {trust}" +
                                    "\nСпостереження незалежні")
            else:
                self.showMessageBox(title, f"{P:.5} < {trust}" +
                                    "\nСпостереження залежні")
        elif len(sel) == 2:
            if self.datas.ident2Samples(sel[0], sel[1], trust):
                self.showMessageBox("Вибірки однорідні", "")
            else:
                self.showMessageBox("Вибірки неоднорідні", "")
        else:
            if self.datas.identKSamples([self.datas[i] for i in sel], trust):
                self.showMessageBox("Вибірки однорідні", "")
            else:
                self.showMessageBox("Вибірки неоднорідні", "")

    def homogeneityNSamples(self):
        title = "Перевірка однорідності сукупностей"
        sel = self.getSelectedRows()
        self.unselectTable()
        if len(sel) == 0:
            if len(self.datas_crits) < 2:
                return
            text = self.datas.homogeneityProtocol(
                [[self.datas[j] for j in i] for i in self.datas_crits])
            self.showMessageBox(title, text)
            self.datas_crits = []
        elif sel not in self.datas_crits:
            if len(self.datas_crits) != 0 and \
               len(self.datas_crits[0]) != len(sel):
                return self.showMessageBox(
                    "Помилка",
                    f"Потрібен {len(self.datas_crits[0])}-вимірний розподіл")
            norm_test = [self.datas[i].isNormal() for i in sel]
            if False in norm_test:
                return self.showMessageBox(
                    "Помилка", "Не є нормальним розподілом:" +
                    str([sel[i]+1 for i, res in enumerate(norm_test)
                         if not res]))
            self.datas_crits.append(sel)
            self.showMessageBox(
                title, "Вибрані вибірки:\n" +
                "\n".join([str([i+1 for i in r]) for r in self.datas_crits]))

    def partialCorrelation(self):
        sel = self.getSelectedRows()
        w = len(sel)
        if w > 2:
            datas = SamplingDatas([self.datas.samples[i] for i in sel])
            datas.toCalculateCharacteristic()
            text = datas.partialCoeficientOfCorrelationProtocol(
                sel[0], sel[1], sel[2:])
            self.criterion_protocol.setText(text)


def applicationLoadFromFile(file: str = ''):
    app = QApplication(sys.argv)
    widget = Window(file)
    widget.show()
    sys.exit(app.exec())


def applicationLoadFromStr(file: str = ''):
    app = QApplication(sys.argv)
    widget = Window(file, False)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # applicationLoadFromFile("data/self/3lines_500_1.txt")
    # applicationLoadFromFile("data/self/line.txt")
    # applicationLoadFromFile("data/self/parable_n5000.txt")
    applicationLoadFromFile("data/6har.dat")
    # applicationLoadFromFile("data/500/norm3n.txt")
