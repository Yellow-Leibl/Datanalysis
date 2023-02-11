import numpy as np
import sys
import os

from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QApplication

from Datanalysis.SamplingDatas import SamplingDatas
from Datanalysis.DoubleSampleData import DoubleSampleData
from mainlayout import MainLayout
from GeneralConstants import (dict_edit, dict_reproduction,
                              dict_regression, Edit)

import pyqtgraph as pg


class Window(MainLayout):
    def __init__(self, file: str, is_file_name: bool = True):
        super().__init__()  # layout here

        self.number_sample = [0, 1]
        self.createPlotLayout(len(self.number_sample))
        self.datas = SamplingDatas()
        self.d_d = None
        self.d_d_cache = [-1, -1]

        if file != "":
            if is_file_name:
                self.openFile(file)
            else:
                all_file = file.split('\n')
                self.loadFromData(all_file)

    def openFile(self, file: str):
        file_name = file
        if file == '':
            file_name = QFileDialog().getOpenFileName(self, "Відкрити файл",
                                                      os.getcwd(),
                                                      "Bci файли (*)")[0]

        try:
            with open(file_name, 'r') as file:
                all_file = [i for i in file]
                self.loadFromData(all_file)
        except FileNotFoundError:
            print(f"\"{file_name}\" not found")

    def loadFromData(self, all_file: str):
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
            hist_data = self.d_d.get_histogram_data(self.getNumberClasses())
            return self.d_d.autoRemoveAnomaly(hist_data)

    def drawSamples(self):
        sel = self.getSelectedRows()
        if not (0 < len(sel) < 3):
            return
        self.createPlotLayout(len(sel))
        self.number_sample = sel
        self.d_d_cache = None
        self.reprod_num = -1
        self.silentChangeNumberClasses(0)
        self.setMaximumColumnNumber(len(self.datas[sel[0]].x))
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

    def setReproductionSeries(self):
        if self.is1d():
            self.reprod_num = dict_reproduction[self.sender().text()]
        else:
            self.reprod_num = dict_regression[self.sender().text()]
        self.updateGraphics(self.getNumberClasses())
        self.writeProtocol()

    def changeTrust(self, trust: float):
        if self.is1d():
            self.datas[self.number_sample[0]].setTrust(trust)
        elif self.is2d():
            self.d_d.setTrust(trust)
        self.sampleChanged()

    def writeTable(self):
        self.table.clear()
        self.table.setColumnCount(self.datas.getMaxDepthRangeData())
        self.table.setRowCount(len(self.datas))
        for s in range(len(self.datas)):
            for i, e in enumerate(self.datas[s].x):
                self.table.setItem(s,
                                   i,
                                   QTableWidgetItem(f"{self.datas[s][i]:.5}"))

    def numberColumnChanged(self, value: int):
        self.updateGraphics(value)
        self.writeProtocol()

    def writeProtocol(self):
        if self.is1d():
            self.protocol.setText(
                self.datas[self.number_sample[0]].getProtocol())
        elif self.is2d():
            self.protocol.setText(self.d_d.getProtocol())

    def writeCritetion(self, text):
        self.criterion_protocol.setText(text)

    def updateGraphics(self, number_column: int = 0):
        if self.is1d():
            d = self.datas[self.number_sample[0]]
            self.setMinMax(d.min, d.max)
            hist_data = d.get_histogram_data(number_column)
            h = abs(d.max - d.min) / len(hist_data)
            self.silentChangeNumberClasses(len(hist_data))
            self.drawHistogram(hist_data, d.min, h)
            self.drawEmpFunc(hist_data, d.min, h)
            self.drawReproductionSeries()
        elif self.is2d():
            if self.d_d_cache != self.number_sample:
                self.d_d_cache = self.number_sample
                x = self.datas[self.number_sample[0]]
                y = self.datas[self.number_sample[1]]
                self.d_d = DoubleSampleData(x, y)
                self.d_d.toCalculateCharacteristic()
            hist_data = self.d_d.get_histogram_data(number_column)
            self.silentChangeNumberClasses(len(hist_data))

            self.drawHistogram2D(hist_data)
            self.drawReproductionSeries2D()
            isNormal, crits = self.d_d.xiXiTest(hist_data)
            if isNormal:
                self.writeCritetion(
                    f"Відтворення двовимірного розподілу адекватне: {crits}")
            else:
                self.writeCritetion(
                    f"Відтворення двовимірного розподілу неадекватне: {crits}")

    def drawHistogram2D(self, hist_data: list):
        x = self.d_d.x
        y = self.d_d.y
        if len(x) != len(y):
            return

        h = np.array(hist_data)
        histogram_image = pg.ImageItem()
        histogram_image.setImage(h)
        start_x = x.min
        start_y = y.min
        w = x.max - start_x
        h = y.max - start_y
        histogram_image.setRect(start_x, start_y, w, h)

        self.cor_plot.clear()
        self.cor_plot.addItem(histogram_image)
        # points
        self.cor_plot.plot(x.getRaw(), y.getRaw(),
                            symbolBrush=(255, 0, 0, 175),
                            symbolPen=(0, 0, 0, 200), symbolSize=7,
                            pen=None)

    def drawHistogram(self, hist_data: list,
                      x_min: float, h: float):
        x = []
        y = []
        y_max: float = hist_data[0]
        for p, i in enumerate(hist_data):
            if y_max < i:
                y_max = i
            x.append(x_min + p * h)
            x.append(x_min + p * h)
            x.append(x_min + (p + 1) * h)
            y.append(0)
            y.append(i)
            y.append(i)

        self.hist_plot.clear()
        self.hist_plot.plot(x, y, fillLevel=0, brush=(250, 220, 70, 150))

    def drawEmpFunc(self, hist_data: list,
                    x_min: float, h: float):
        x_class = []
        y_class = []
        col_height = 0.0
        for p, i in enumerate(hist_data):
            if col_height > 1:
                col_height = 1
            x_class.append(x_min + p * h)
            x_class.append(x_min + p * h)
            x_class.append(x_min + (p + 1) * h)
            y_class.append(col_height)
            y_class.append(col_height + i)
            y_class.append(col_height + i)
            col_height += i

        d = self.datas[self.number_sample[0]]
        x_stat = []
        y_stat = []
        sum_ser = 0.0
        for i in range(len(d.probabilityX)):
            sum_ser += d.probabilityX[i]
            x_stat.append(d.x[i])
            y_stat.append(sum_ser)

        self.emp_plot.clear()
        self.emp_plot.plot(x_class, y_class, pen=newPen((255, 0, 0), 2))
        self.emp_plot.plot(x_stat, y_stat, pen=newPen((0, 255, 0), 2))

    def toCreateReproductionFunc(self, d, func_num):
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

    def writeCritetion1DSample(self, d, F):
        criterion_text = '\n'
        if d.kolmogorovTest(F):
            criterion_text += "Відтворення адекватне за критерієм" + \
                " згоди Колмогорова\n"
        else:
            criterion_text += "Відтворення неадекватне за критерієм" + \
                " згоди Колмогорова\n"

        try:
            xi_test_result = d.xiXiTest(
                F,
                d.get_histogram_data(self.getNumberClasses()))
        except ZeroDivisionError:
            xi_test_result = False

        if xi_test_result:
            criterion_text += "Відтворення адекватне за критерієм Пірсона\n"
        else:
            criterion_text += "Відтворення неадекватне за критерієм Пірсона\n"
        self.writeCritetion(criterion_text)

    def drawReproductionSeries(self):
        x_gen = []
        d = self.datas[self.number_sample[0]]
        f, lF, F, hF = self.toCreateReproductionFunc(d, self.reprod_num)
        if f is None:
            return
        x_gen = d.toGenerateReproduction(f)

        if len(x_gen) == 0:
            self.writeCritetion('')
            return
        else:
            self.writeCritetion1DSample(d, F)

        y_hist = []
        y_low = []
        y_emp = []
        y_high = []
        for x in x_gen:
            y_hist.append(f(x))
            y_low.append(lF(x))
            y_emp.append(F(x))
            y_high.append(hF(x))

        self.hist_plot.plot(x_gen, y_hist, pen=newPen((0, 0, 255), 3))
        self.emp_plot.plot(x_gen, y_low, pen=newPen((0, 128, 128), 2))
        self.emp_plot.plot(x_gen, y_emp, pen=newPen((0, 255, 255), 2))
        self.emp_plot.plot(x_gen, y_high, pen=newPen((128, 0, 128), 2))

    def toCreateReproductionFunc2D(self, d_d, func_num):
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

    def drawReproductionSeries2D(self):
        x_gen = []
        tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f = \
            self.toCreateReproductionFunc2D(self.d_d, self.reprod_num)
        if f is not None:
            x_gen = self.d_d.toGenerateReproduction(f)

        if len(x_gen) == 0:
            self.writeCritetion('')
            return

        y_tl_lf = []
        y_tl_mf = []
        y_tr_lf = []
        y_tr_mf = []
        y_tr_f_lf = []
        y_tr_f_mf = []
        y = []
        for x in x_gen:
            y_tl_lf.append(tl_lf(x))
            y_tl_mf.append(tl_mf(x))
            y_tr_lf.append(tr_lf(x))
            y_tr_mf.append(tr_mf(x))
            y_tr_f_lf.append(tr_f_lf(x))
            y_tr_f_mf.append(tr_f_mf(x))
            y.append(f(x))

        self.cor_plot.plot(x_gen, y_tl_lf, pen=newPen((0, 128, 128), 3))
        self.cor_plot.plot(x_gen, y_tl_mf, pen=newPen((0, 128, 128), 3))
        self.cor_plot.plot(x_gen, y_tr_lf, pen=newPen((0, 128, 255), 3))
        self.cor_plot.plot(x_gen, y_tr_mf, pen=newPen((0, 128, 255), 3))
        self.cor_plot.plot(x_gen, y_tr_f_lf, pen=newPen((0, 255, 128), 3))
        self.cor_plot.plot(x_gen, y_tr_f_mf, pen=newPen((128, 255, 128), 3))
        self.cor_plot.plot(x_gen, y, pen=newPen((255, 0, 255), 3))

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


def newPen(color, width):
    return {'color': color, 'width': width}


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
    applicationLoadFromFile("data/self/parable.txt")
    # applicationLoadFromFile("data/self/norm5n.txt")
