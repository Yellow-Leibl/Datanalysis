import numpy as np
import sys
import os

from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QApplication

from Datanalysis.SamplingDatas import SamplingDatas
from Datanalysis.DoubleSampleData import DoubleSampleData

from mainlayout import MainLayout
from GeneralConstants import dict_edit, dict_repr, dict_regr, Edit

import pyqtgraph as pg


class Window(MainLayout):
    def __init__(self, file: str, is_file_name: bool = True):
        super().__init__()  # layout here

        self.number_sample = [0, 1]
        self.datas = SamplingDatas()
        self.d_d = None
        self.d_d_number = [-1, -1]

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
        if len(self.number_sample) == 1:
            minmax = self.getMinMax()
            self.datas[self.number_sample[0]].remove(minmax[0], minmax[1])
            self.sampleChanged()

    def autoRemoveAnomaly(self) -> bool:
        if len(self.number_sample) == 1:
            return self.datas[self.number_sample[0]].autoRemoveAnomaly()
        elif len(self.number_sample) == 2:
            hist_data = self.d_d.get_histogram_data(self.getNumberClasses())
            return self.d_d.autoRemoveAnomaly(hist_data)

    def drawSamples(self):
        sel = self.getSelectedRows()
        if len(sel) == 1:
            self.reprod_num = -1
            self.setSample(sel[0])
        elif len(sel) == 2:
            self.reprod_num = -1
            self.setSample2D(sel[0], sel[1])
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
            self.reprod_num = dict_repr[self.sender().text()]
        else:
            self.reprod_num = dict_regr[self.sender().text()]
        self.updateGraphics(self.getNumberClasses())

    def changeTrust(self, trust: float):
        if len(self.number_sample) == 1:
            self.datas[self.number_sample[0]].setTrust(trust)
        elif len(self.number_sample) == 2:
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
        if len(self.number_sample) == 1:
            self.protocol.setText(
                self.datas[self.number_sample[0]].getProtocol())
        else:
            self.protocol.setText(self.d_d.getProtocol())

    def writeCritetion(self, text):
        self.criterion_protocol.setText(text)

    def setSample(self, row: int):
        self.number_sample = [row]
        self.silentChangeNumberClasses(0)
        self.setMaximumColumnNumber(len(self.datas[row].x))

    def setSample2D(self, row1: int, row2: int):
        self.number_sample = [row1, row2]
        self.d_d = DoubleSampleData(self.datas[row1], self.datas[row2])
        self.d_d.toCalculateCharacteristic()
        self.silentChangeNumberClasses(0)
        self.setMaximumColumnNumber(len(self.datas[row1].x))

    def updateGraphics(self, number_column: int = 0):
        if len(self.number_sample) == 1:
            d = self.datas[self.number_sample[0]]
            self.setMinMax(d.min, d.max)
            hist_data = d.get_histogram_data(number_column)
            h = abs(d.max - d.min) / len(hist_data)
            self.silentChangeNumberClasses(len(hist_data))

            self.drawHistogram(hist_data, d.min, h)
            self.drawEmpFunc(hist_data, d.min, h)
            self.drawReproductionSeries()
        elif len(self.number_sample) == 2:
            if self.d_d_number[0] != self.number_sample[0] or\
               self.d_d_number[1] != self.number_sample[1]:
                self.d_d_number = self.number_sample
                x = self.datas[self.number_sample[0]]
                y = self.datas[self.number_sample[1]]
                self.d_d = DoubleSampleData(x, y)
                self.d_d.toCalculateCharacteristic()
            hist_data = self.d_d.get_histogram_data(number_column)
            self.silentChangeNumberClasses(len(hist_data))

            self.drawHistogram2D(hist_data)
            self.drawReproductionSeries2D()
            self.writeProtocol()
            isNormal, crits = self.d_d.xiXiTest(hist_data)
            if isNormal:
                self.writeCritetion(
                    f"Відтворення двовимірного розподілу адекватне: {crits}")
            else:
                self.writeCritetion(
                    f"Відтворення двовимірного розподілу неадекватне: {crits}")

    def drawHistogram2D(self, hist_data: list):
        x = self.d_d.x.raw_x
        y = self.d_d.y.raw_x
        if len(x) != len(y):
            return

        h = np.array(hist_data)
        histogram_image = pg.ImageItem()
        histogram_image.setImage(h)
        pg.setConfigOption('imageAxisOrder', 'row-major')
        start_x = min(x)
        start_y = min(y)
        w = max(x) - start_x
        h = max(y) - start_y
        histogram_image.setRect(start_x, start_y, w, h)

        self.hist_plot.clear()
        self.hist_plot.addItem(histogram_image)
        # points
        self.hist_plot.plot(x, y, symbolBrush=(255, 0, 0, 175),
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
            return None, None, None

        return f, F, DF

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

    def drawReproductionSeries(self):
        x_gen = []
        d = self.datas[self.number_sample[0]]
        try:
            f, F, DF = self.toCreateReproductionFunc(d, self.reprod_num)
            if f is not None:
                h = abs(d.max - d.min) / self.getNumberClasses()
                x_gen = d.toGenerateReproduction(f, F, DF, h)
        except ValueError:
            print("Value error")
        except OverflowError:
            print("Overflow error")

        if len(x_gen) == 0:
            self.writeCritetion('')
            return

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

        x = []
        y_hist = []
        y_low = []
        y_emp = []
        y_high = []
        for i in x_gen:
            x.append(i[0])
            y_hist.append(i[1])
            y_low.append(i[2])
            y_emp.append(i[3])
            y_high.append(i[4])

        self.hist_plot.plot(x, y_hist, pen=newPen((0, 0, 255), 3))
        self.emp_plot.plot(x, y_low, pen=newPen((0, 128, 128), 2))
        self.emp_plot.plot(x, y_emp, pen=newPen((0, 255, 255), 2))
        self.emp_plot.plot(x, y_high, pen=newPen((128, 0, 128), 2))

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

        self.hist_plot.plot(x_gen, y_tl_lf, pen=newPen((0, 128, 128), 3))
        self.hist_plot.plot(x_gen, y_tl_mf, pen=newPen((0, 128, 128), 3))
        self.hist_plot.plot(x_gen, y_tr_lf, pen=newPen((0, 128, 255), 3))
        self.hist_plot.plot(x_gen, y_tr_mf, pen=newPen((0, 128, 255), 3))
        self.hist_plot.plot(x_gen, y_tr_f_lf, pen=newPen((0, 255, 128), 3))
        self.hist_plot.plot(x_gen, y_tr_f_mf, pen=newPen((128, 255, 128), 3))
        self.hist_plot.plot(x_gen, y, pen=newPen((255, 0, 255), 3))

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
    applicationLoadFromFile("data/self/kvaz_8.txt")
    # applicationLoadFromFile("data/self/norm5n.txt")
