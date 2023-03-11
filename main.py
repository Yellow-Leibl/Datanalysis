import sys
import os

from PyQt6.QtWidgets import QFileDialog, QTableWidgetItem, QApplication

from Datanalysis.SamplingDatas import SamplingDatas
from Datanalysis.DoubleSampleData import DoubleSampleData
from Datanalysis.SamplingData import SamplingData
from mainlayout import MainLayout
from GeneralConstants import (dict_edit, dict_reproduction,
                              dict_regression, Edit)


class Window(MainLayout):
    def __init__(self, file: str, is_file_name: bool = True):
        super().__init__()  # layout here
        self.datas = SamplingDatas()
        self.d2_cache = [-1, -1]

        # temp
        self.number_sample = [1, 2]
        self.createPlotLayout(len(self.number_sample))

        if is_file_name:
            self.openFile(file)
        else:
            all_file = file.split('\n')
            self.loadFromData(all_file)

    def openFile(self, file_name: str):
        if file_name == '':
            file_name = QFileDialog().getOpenFileName(self, "Відкрити файл",
                                                      os.getcwd(),
                                                      "Bci файли (*)")[0]

        try:
            with open(file_name, 'r') as file:
                self.loadFromData(file.readlines())
        except FileNotFoundError:
            print(f"\"{file_name}\" not found")

    def loadFromData(self, all_file: list[str]):
        self.datas.append(all_file)
        self.reprod_num = -1
        self.sampleChanged()

    def saveFileAct(self):
        file_name = QFileDialog().getSaveFileName(self, "Зберегти файл",
                                                  os.getcwd(),
                                                  "Bci файли (*)")[0]
        with open(file_name, 'w') as file:
            def safe_access(lst: list, i):
                return str(lst[i]) if len(lst) > i else ''

            file.write('\n'.join(
                [' '.join([safe_access(self.datas[j].getRaw(), i)
                           for j in range(len(self.datas))])
                 for i in range(self.datas.getMaxDepthRawData())]))

    def sampleChanged(self):
        self.updateGraphics(self.getNumberClasses())
        self.writeTable()
        self.writeProtocol()

    def editSampleEvent(self):
        act = Edit(dict_edit[self.sender().text()])
        d = self.datas[self.number_sample[0]]
        if act == Edit.TRANSFORM:
            d.toLogarithmus10()
        elif act == Edit.STANDARTIZATION:
            d.toStandardization()
        elif act == Edit.SLIDE:
            d.toSlide()
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
        if not (0 < len(sel) < 4):
            return
        self.createPlotLayout(len(sel))
        self.number_sample = sel
        self.d2_cache = None
        self.reprod_num = -1
        self.silentChangeNumberClasses(0)
        self.setMaximumColumnNumber(len(self.datas[sel[0]]._x))
        self.sampleChanged()

    def deleteSamples(self):
        sel = self.getSelectedRows()
        p = 0
        for i in sel:
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
        return len(self.number_sample) >= 3

    def setReproductionSeries(self):
        if self.is1d():
            self.reprod_num = dict_reproduction[self.sender().text()]
        elif self.is2d():
            self.reprod_num = dict_regression[self.sender().text()]
        self.updateGraphics(self.getNumberClasses())
        self.writeProtocol()

    def changeTrust(self, trust: float):
        if self.is1d():
            self.datas[self.number_sample[0]].setTrust(trust)
        elif self.is2d():
            self.d2.setTrust(trust)
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

    def numberColumnChanged(self, value: int):
        self.updateGraphics(value)
        self.writeProtocol()

    def writeProtocol(self):
        if self.is1d():
            self.protocol.setText(
                self.datas[self.number_sample[0]].getProtocol())
        elif self.is2d():
            self.protocol.setText(self.d2.getProtocol())
        else:
            pass

    def writeCritetion(self, text):
        self.criterion_protocol.setText(text)

    def updateGraphics(self, number_column: int = 0):
        if self.is1d():
            d = self.datas[self.number_sample[0]]
            self.setMinMax(d.min, d.max)
            hist_data = d.get_histogram_data(number_column)
            self.silentChangeNumberClasses(len(hist_data))
            self.plot_widget.plot1D(d, hist_data)
            self.drawReproductionSeries()
        elif self.is2d():
            if self.d2_cache != self.number_sample:
                self.d2_cache = self.number_sample
                x = self.datas[self.number_sample[0]]
                y = self.datas[self.number_sample[1]]
                self.d2 = DoubleSampleData(x, y)
                self.d2.toCalculateCharacteristic()
            hist_data = self.d2.get_histogram_data(number_column)
            self.silentChangeNumberClasses(len(hist_data))
            self.plot_widget.plot2D(self.d2, hist_data)
            self.drawReproductionSeries2D()
            isNormal, crits = self.d2.xiXiTest(hist_data)
            if isNormal:
                self.writeCritetion(
                    f"Відтворення двовимірного розподілу адекватне: {crits}")
            else:
                self.writeCritetion(
                    f"Відтворення двовимірного розподілу неадекватне: {crits}")
        elif self.is3d():
            self.plot_widget.plot3D(
                [self.datas[i] for i in self.number_sample])

    def drawReproductionSeries(self):
        d = self.datas[self.number_sample[0]]
        f, lF, F, hF = self.toCreateReproductionFunc(d, self.reprod_num)
        if f is None:
            return
        self.writeCritetion1DSample(d, F)
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
        criterion_text = '\n'
        if d.kolmogorovTest(F):
            criterion_text += "Відтворення адекватне за критерієм" + \
                " згоди Колмогорова\n"
        else:
            criterion_text += "Відтворення неадекватне за критерієм" + \
                " згоди Колмогорова\n"

        try:
            xi_test_result = d.xiXiTest(
                F, d.get_histogram_data(self.getNumberClasses()))
        except ZeroDivisionError:
            xi_test_result = False

        if xi_test_result:
            criterion_text += "Відтворення адекватне за критерієм Пірсона\n"
        else:
            criterion_text += "Відтворення неадекватне за критерієм Пірсона\n"
        self.writeCritetion(criterion_text)

    def drawReproductionSeries2D(self):
        tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f = \
            self.toCreateReproductionFunc2D(self.d2, self.reprod_num)
        self.plot_widget.plot2DReproduction(
            self.d2, tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f)

    def toCreateReproductionFunc2D(self, d_d: DoubleSampleData, func_num):
        if func_num == 0:
            tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f = \
                d_d.toCreateLinearRegressionMNK()
        elif func_num == 1:
            tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f = \
                d_d.toCreateLinearRegressionMethodTeila()
        elif func_num == 2:
            tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f = \
                d_d.toCreateParabolicRegression()
        elif func_num == 3:
            tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f = \
                d_d.toCreateKvazi8()
        else:
            return None, None, None, None, None, None, None
        return tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f

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
    # applicationLoadFromFile("data/self/parable.txt")
    applicationLoadFromFile("data/6har.dat")
