import sys
import os
import logging

from PyQt6.QtWidgets import QFileDialog, QTableWidgetItem, QApplication
from PyQt6 import QtGui

from Datanalysis.SamplingDatas import SamplingDatas
from Datanalysis.DoubleSampleData import DoubleSampleData
from Datanalysis.SamplingData import SamplingData
from Datanalysis.ProtocolGenerator import ProtocolGenerator
from mainlayout import MainLayout

logging.basicConfig(level=logging.INFO)


class Window(MainLayout):
    def __init__(self, file: str, is_file_name: bool = True):
        super().__init__()  # layout here
        self.all_datas = SamplingDatas()
        self.sel_indexes: list[int] = []
        self.d2_indexes: list[int] = []
        self.datas_displayed_indexes: list[int] = []
        self.datas_crits: list[list] = []
        self.selected_regr_num = -1
        self.d1_regr_F = None
        if is_file_name:
            self.openFile(file)
        else:
            all_file = file.split('\n')
            self.loadFromData(all_file)
        self.autoSelect()  # temp

    def openFile(self, file_name: str):
        if file_name == '':
            file_name, _ = QFileDialog().getOpenFileName(
                self, "Відкрити файл", os.getcwd(), "Bci файли (*)")
        try:
            with open(file_name, 'r') as file:
                self.loadFromData(file.readlines())
        except FileNotFoundError:
            logging.error(f"\"{file_name}\" not found")

    def loadFromData(self, all_file: list[str]):
        self.all_datas.append(all_file)
        self.writeTable()

    def saveFileAct(self):
        file_name, _ = QFileDialog().getSaveFileName(
            self, "Зберегти файл", os.getcwd(), "Bci файли (*)")
        with open(file_name, 'w') as file:
            def safe_access(lst: list, i):
                return str(lst[i]) if len(lst) > i else ''
            file.write('\n'.join(
                [' '.join([safe_access(self.all_datas[j].raw, i)
                           for j in range(len(self.all_datas))])
                 for i in range(self.all_datas.getMaxDepthRawData())]))

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
            return self.all_datas[self.sel_indexes[0]]
        elif self.is2d():
            return self.d2
        elif self.isNd():
            return self.datas_displayed

    def editSampleEvent(self):
        edit_num = self.index_in_menu(self.get_edit_menu(), self.sender())
        if edit_num == 0:
            [self.all_datas[i].toLogarithmus10() for i in self.sel_indexes]
        elif edit_num == 1:
            [self.all_datas[i].toStandardization() for i in self.sel_indexes]
        elif edit_num == 2:
            [self.all_datas[i].toCentralization() for i in self.sel_indexes]
        elif edit_num == 3:
            [self.all_datas[i].toSlide(1.0) for i in self.sel_indexes]
        elif edit_num == 4:
            if not self.autoRemoveAnomaly():
                return
        elif edit_num == 5:
            if self.is2d():
                self.d2.toIndependet()
            elif self.isNd():
                self.datas_displayed.toIndependet()
        self.sampleChanged()

    def duplicateSample(self):
        sel = self.getSelectedRows()
        for i in sel:
            self.all_datas.appendSample(self.all_datas[i].copy())
        self.writeTable()

    def removeAnomaly(self):
        if self.is1d():
            minmax = self.getMinMax()
            self.all_datas[self.sel_indexes[0]].remove(minmax[0], minmax[1])
            self.sampleChanged()

    def autoRemoveAnomaly(self) -> bool:
        if self.is1d():
            return self.all_datas[self.sel_indexes[0]].autoRemoveAnomaly()
        elif self.is2d() or self.isNd():
            act_sample = self.getActiveSamples()
            hist_data = act_sample.get_histogram_data(self.getNumberClasses())
            deleted_items = act_sample.autoRemoveAnomaly(hist_data)
            self.showMessageBox("Видалення аномалій",
                                f"Було видалено {deleted_items} аномалій")
            return deleted_items
        return False

    def drawSamples(self):
        sel = self.getSelectedRows()
        if len(sel) == 0 or sel == self.sel_indexes:
            return
        if self.datas_displayed_indexes == sel or \
           self.d2_indexes == sel:
            return
        self.sel_indexes = sel
        self.createPlotLayout()
        self.d2_indexes = [-1, -1]
        self.datas_displayed_indexes = []
        self.selected_regr_num = -1
        self.silentChangeNumberClasses(0)
        self.setMaximumColumnNumber(self.all_datas.getMaxDepthRangeData())
        self.selectSampleOrReproduction()
        self.coloredTable(self.sel_indexes)

    def createPlotLayout(self):
        if self.is1d():
            self.plot_widget.create1DPlot()
        elif self.is2d():
            self.plot_widget.create2DPlot()
        else:
            self.plot_widget.createNDPlot(len(self.sel_indexes))

    def deleteSamples(self):
        sel = self.getSelectedRows()
        for p, i in enumerate(sel):
            if i - p in self.sel_indexes:
                self.sel_indexes = []
            self.all_datas.pop(i - p)
            p += 1
        self.writeTable()

    def is1d(self) -> bool:
        return len(self.sel_indexes) == 1

    def is2d(self) -> bool:
        return len(self.sel_indexes) == 2

    def is3d(self) -> bool:
        return len(self.sel_indexes) == 3

    def isNd(self) -> bool:
        return len(self.sel_indexes) >= 2

    def setReproductionSeries(self):
        self.selected_regr_num = \
            self.index_in_menu(self.get_regr_menu(), self.sender())
        self.selectSampleOrReproduction()

    def changeTrust(self, trust: float):
        self.sampleChanged()

    def writeTable(self):
        self.table.clear()
        self.table.setColumnCount(self.all_datas.getMaxDepthRangeData() + 1)
        self.table.setRowCount(len(self.all_datas))
        for s in range(len(self.all_datas)):
            d = self.all_datas[s]
            self.table.setItem(s, 0, QTableWidgetItem(f"N={len(d.raw)}"))
            for i, e in enumerate(d._x):
                self.table.setItem(s, i + 1, QTableWidgetItem(f"{e:.5}"))
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        self.coloredTable(self.sel_indexes)

    def coloredTable(self, sel_indexes: list[int]):
        for i in range(self.table.rowCount()):
            color = QtGui.QColor()
            if i in sel_indexes:
                color = QtGui.QColor(30, 150, 0)
            for j in range(len(self.all_datas[i]._x) + 1):
                self.table.item(i, j).setBackground(color)

    def numberColumnChanged(self):
        self.selectSampleOrReproduction()

    def writeProtocol(self):
        s = self.getActiveSamples()
        if s is not None:
            self.protocol.setText(ProtocolGenerator.getProtocol(s))

    def writeCritetion(self):
        if self.is1d():
            if self.d1_regr_F is None:
                return
            d = self.all_datas[self.sel_indexes[0]]
            self.criterion_protocol.setText(self.writeCritetion1DSample(
                d, self.d1_regr_F))
        elif self.is2d():
            isNormal = self.d2.xiXiTest(self.hist_data_2d)
            self.criterion_protocol.setText(self.d2.xiXiTestProtocol(isNormal))
        elif self.isNd():
            self.criterion_protocol.setText("")

    def updateGraphics(self, number_column: int = 0):
        if self.is1d():
            d = self.all_datas[self.sel_indexes[0]]
            d.setTrust(self.getTrust())
            self.setMinMax(d.min, d.max)
            hist_data = d.get_histogram_data(number_column)
            self.silentChangeNumberClasses(len(hist_data))
            self.plot_widget.plot1D(d, hist_data)
            self.drawReproductionSeries1D()
        elif self.is2d():
            self.d2_indexes = self.sel_indexes
            x = self.all_datas[self.sel_indexes[0]]
            y = self.all_datas[self.sel_indexes[1]]
            self.d2 = DoubleSampleData(x, y, self.getTrust())
            self.d2.toCalculateCharacteristic()
            self.hist_data_2d = self.d2.get_histogram_data(number_column)
            self.silentChangeNumberClasses(len(self.hist_data_2d))
            self.plot_widget.plot2D(self.d2, self.hist_data_2d)
            self.drawReproductionSeries2D(self.d2)
        elif self.isNd():
            samples = [self.all_datas[i] for i in self.sel_indexes]
            self.datas_displayed_indexes = self.sel_indexes
            self.datas_displayed = SamplingDatas(samples, self.getTrust())
            self.datas_displayed.toCalculateCharacteristic()
            self.plot_widget.plotND(self.datas_displayed, number_column)
            self.drawReproductionSeriesND(self.datas_displayed)

    def drawReproductionSeries1D(self):
        d = self.all_datas[self.sel_indexes[0]]
        f = self.toCreateReproductionFunc(d, self.selected_regr_num)
        if f is None:
            return
        h = abs(d.max - d.min) / self.getNumberClasses()
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

    def writeCritetion1DSample(self, d: SamplingData, F):
        criterion_text = d.kolmogorovTestProtocol(d.kolmogorovTest(F))
        try:
            xi_test_result = d.xiXiTest(
                F, d.get_histogram_data(self.getNumberClasses()))
        except ZeroDivisionError:
            xi_test_result = False
        criterion_text += '\n' + d.xiXiTestProtocol(xi_test_result)
        return criterion_text

    def drawReproductionSeries2D(self, d2):
        f = self.toCreateReproductionFunc2D(d2, self.selected_regr_num)
        if f is None:
            return
        self.plot_widget.plot2DReproduction(d2, *f)

    def toCreateReproductionFunc2D(self, d_d: DoubleSampleData, func_num):
        if func_num == 6:
            return d_d.toCreateLinearRegressionMNK()
        elif func_num == 7:
            return d_d.toCreateLinearRegressionMethodTeila()
        elif func_num == 8:
            return d_d.toCreateParabolicRegression()
        elif func_num == 9:
            return d_d.toCreateKvazi8()

    def drawReproductionSeriesND(self, datas):
        f = self.toCreateReproductionFuncND(datas, self.selected_regr_num)
        if f is None:
            return
        self.plot_widget.plotDiagnosticDiagram(datas, *f)

    def toCreateReproductionFuncND(self, datas: SamplingDatas, func_num):
        if func_num == 11:
            return datas.toCreateLinearRegressionMNK(len(datas) - 1)
        elif func_num == 12:
            return datas.toCreateLinearVariationPlane()

    def linearModelsCrit(self, trust: float):
        sel = self.getSelectedRows()
        if len(sel) == 4:
            res = self.all_datas.ident2ModelsLine(
                [self.all_datas[i] for i in sel], trust)
            title = "Лінійна регресія"
            descr = "Моделі регресійних прямих\n" + \
                f"({sel[0]}, {sel[1]}) і ({sel[2]}, {sel[3]})"
            if type(res) is bool:
                if res:
                    self.showMessageBox(title, descr + " - ідентичні")
                else:
                    self.showMessageBox(title, descr + " - неідентичні")
            elif type(res) is str:
                self.showMessageBox(title, descr +
                                    " - мають випадкову різницю регресій")

    def homogeneityAndIndependence(self, trust: float):
        sel = self.getSelectedRows()
        if len(sel) == 1:
            P = self.all_datas[sel[0]].critetionAbbe()
            title = "Критерій Аббе"
            if P > trust:
                self.showMessageBox(title, f"{P:.5} > {trust}" +
                                    "\nСпостереження незалежні")
            else:
                self.showMessageBox(title, f"{P:.5} < {trust}" +
                                    "\nСпостереження залежні")
        elif len(sel) == 2:
            if self.all_datas.ident2Samples(sel[0], sel[1], trust):
                self.showMessageBox("Вибірки однорідні", "")
            else:
                self.showMessageBox("Вибірки неоднорідні", "")
        else:
            if self.all_datas.identKSamples([self.all_datas[i] for i in sel],
                                            trust):
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
            text = self.all_datas.homogeneityProtocol(
                [[self.all_datas[j] for j in i] for i in self.datas_crits])
            self.showMessageBox(title, text)
            self.datas_crits = []
        elif sel not in self.datas_crits:
            if len(self.datas_crits) != 0 and \
               len(self.datas_crits[0]) != len(sel):
                return self.showMessageBox(
                    "Помилка",
                    f"Потрібен {len(self.datas_crits[0])}-вимірний розподіл")
            norm_test = [self.all_datas[i].isNormal() for i in sel]
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
            datas = SamplingDatas([self.all_datas.samples[i] for i in sel])
            datas.toCalculateCharacteristic()
            text = datas.partialCoeficientOfCorrelationProtocol(
                sel[0], sel[1], sel[2:])
            self.criterion_protocol.setText(text)

    def PCA(self):
        w = self.pCA_number.value()
        ind, retn = self.datas_displayed.principalComponentAnalysis(w)
        self.all_datas.appendSamples(ind.samples)
        self.all_datas.appendSamples(retn.samples)
        self.writeTable()

    def autoSelect(self):
        # self.sel_indexes = range(len(self.all_datas))
        self.sel_indexes = range(3)
        self.createPlotLayout()
        self.sampleChanged()


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
    # applicationLoadFromFile("data/iris_fish.txt")
    applicationLoadFromFile("data/500/norm3n.txt")
    # applicationLoadFromFile("data/self/line.txt")
