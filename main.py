import numpy as np
import sys
import os

from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QApplication

from Datanalysis.SamplingDatas import SamplingDatas
from Datanalysis.DoubleSampleData import DoubleSampleData

from mainlayout import MainLayout
from historystack import HistoryStask
from GeneralConstants import dict_edit, dict_repr, Edit

import pyqtgraph as pg


class Window(MainLayout):
    def __init__(self, file: str, is_file_name: bool = True):
        super().__init__()  # layout here

        self.number_sample = [0]
        self.datas = SamplingDatas()
        self.d_d = None

        if file != "":
            if is_file_name:
                self.openFile(file)
            else:
                all_file = file.split('\n')
                self.loadFromData(all_file)
        #  temp
        self.setSample2D(0, 1)
        self.sampleChanged()

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
        self.history_series = HistoryStask(len(self.datas))

        if len(self.number_sample) == 1:
            d = self.datas[self.number_sample[0]]
            self.silentChangeNumberClasses(0)
            self.spin_number_column.setMaximum(len(d.x))

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
                [' '.join([safe_access(self.datas[j].raw_x, i)
                           for j in range(len(self.datas))])
                 for i in range(self.datas.getMaxDepth())]))

    def sampleChanged(self):
        if len(self.number_sample) == 1:
            d = self.datas[self.number_sample[0]]
            self.setMinMax(d.min, d.max)

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
            self.datas[self.number_sample[0]].remove(
                self.spin_box_min_x.value(),
                self.spin_box_max_x.value())
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
            self.setSample(sel[0])
        elif len(sel) == 2:
            self.setSample2D(sel[0], sel[1])
        self.sampleChanged()

    def deleteSamples(self):
        sel = self.getSelectedRows()
        p = 0
        for i in sel:
            self.datas.pop(i - p)
            p += 1
        self.writeTable()

    def setReproductionSeries(self):
        self.reprod_num = dict_repr[self.sender().text()]
        self.updateGraphics(self.getNumberClasses())

    def changeTrust(self, trust: float):
        if len(self.number_sample) == 1:
            self.datas[self.number_sample[0]].setTrust(trust)
        elif len(self.number_sample) == 2:
            self.d_d.setTrust(trust)
        self.sampleChanged()

    def writeTable(self):
        self.table.clear()
        self.table.setColumnCount(self.datas.getMaxDepth())
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
        self.spin_number_column.setMaximum(len(self.datas[row].x))

    def setSample2D(self, row1: int, row2: int):
        self.number_sample = [row1, row2]
        self.d_d = DoubleSampleData(self.datas[row1], self.datas[row2])
        self.d_d.toCalculateCharacteristic()
        self.silentChangeNumberClasses(0)
        self.spin_number_column.setMaximum(len(self.datas[row1].x))

    def updateGraphics(self, number_column: int = 0):
        if len(self.number_sample) == 1:
            d = self.datas[self.number_sample[0]]
            hist_data = d.get_histogram_data(number_column)
            h = abs(d.max - d.min) / len(hist_data)
            self.silentChangeNumberClasses(len(hist_data))

            self.drawHistogram(hist_data, d.min, h)
            self.drawEmpFunc(hist_data, d.min, h)
            self.drawReproductionSeries()
        elif len(self.number_sample) == 2:
            hist_data = self.d_d.get_histogram_data(number_column)
            self.silentChangeNumberClasses(len(hist_data))

            self.drawHistogram2D(hist_data)
            if self.d_d.xiXiTest(hist_data):
                self.writeCritetion(
                    "Відтворення двовимірного розподілу адекватне")
            else:
                self.writeCritetion(
                    "Відтворення двовимірного розподілу неадекватне")

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

        f = self.d_d.toCreateLineFunc()
        x = [self.d_d.x.min, self.d_d.x.max]
        y = [f(i) for i in x]
        self.hist_plot.plot(x, y, pen=newPen((128, 0, 255), 3))

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

    def toCreateReproductionFunc(self, func_num):
        d = self.datas[self.number_sample[0]]
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

    def drawReproductionSeries(self):
        x_gen = []
        d = self.datas[self.number_sample[0]]
        try:
            f, F, DF = self.toCreateReproductionFunc(self.reprod_num)
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
                d.get_histogram_data(self.spin_number_column.value()))
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

    def critsSamples(self, trust: float):
        sel = self.getSelectedRows()
        print(f"{sel}")
        if len(sel) == 1:
            P = self.datas[sel[0]].critetionAbbe()
            if P > trust:
                self.showMessageBox("Критерій Аббе",
                                    f"{P:.5} > {trust}" +
                                    "\nСпостереження незалежні")
            else:
                self.showMessageBox("Критерій Аббе",
                                    f"{P:.5} < {trust}" +
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
    applicationLoadFromFile("data/self/line.txt")
    # applicationLoadFromFile("data/self/norm5n.txt")
