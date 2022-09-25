import sys
import os

from PyQt5.QtWidgets import (QFileDialog, QTableWidgetItem, QApplication)
from PyQt5.QtChart import QLineSeries, QAreaSeries
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPen, QColor

from mainlayout import MainLayout
from Datanalysis import SamplingDatas
from historystack import HistoryStask
from GeneralConstants import dict_edit, dict_repr, Edit


class Window(MainLayout):
    def __init__(self, file: str):
        super().__init__()  # layout here

        self.number_sample = 0
        self.datas = SamplingDatas()

        if file != "":
            self.openFile(file)

    def openFileAct(self):
        self.openFile('')

    def openFile(self, file: str):
        file_name = file

        if file == '':
            file_name = QFileDialog().getOpenFileName(self, "Відкрити файл",
                                                      os.getcwd(),
                                                      "Bci файли (*)")[0]

        try:
            with open(file_name, 'r') as file:
                all_file = [i for i in file]

                self.datas.append(all_file)
                self.history_series = HistoryStask(len(self.datas))
                self.d = self.datas[self.number_sample]

                self.spin_box_number_column.blockSignals(True)
                self.spin_box_number_column.setMaximum(len(self.d.x))
                self.spin_box_number_column.setValue(0)
                self.spin_box_number_column.blockSignals(False)

                self.reprod_num = -1

                self.changeXSeries()
        except FileNotFoundError:
            print(f"\"{file_name}\" not found")

    def saveFileAct(self):
        file_name = QFileDialog().getSaveFileName(self, "Зберегти файл",
                                                  os.getcwd(),
                                                  "Bci файли (*)")[0]
        with open(file_name, 'w') as file:
            file.write('\n'.join([str(i) for i in self.d.x]))

    def changeXSeries(self):
        self.spin_box_min_x.setMinimum(self.d.min)
        self.spin_box_min_x.setMaximum(self.d.max)
        self.spin_box_min_x.setValue(self.d.min)

        self.spin_box_max_x.setMinimum(self.d.min)
        self.spin_box_max_x.setMaximum(self.d.max)
        self.spin_box_max_x.setValue(self.d.max)

        self.updateGraphics(self.spin_box_number_column.value())
        self.writeTable()
        self.writeProtocol()

    def editEvent(self):
        act = Edit(dict_edit[self.sender().text()])

        if act != Edit.UNDO:
            self.history_series.push(self.number_sample, self.d.x.copy())
        else:
            if self.history_series.len(self.number_sample) == 0:
                return
            self.d.setSeries(self.history_series.pop(self.number_sample))

        if act == Edit.TRANSFORM:
            self.d.toTransform()
        elif act == Edit.STANDARTIZATION:
            self.d.toStandardization()
        elif act == Edit.SLIDE:
            self.d.toSlide()
        elif act == Edit.DELETE_ANOMALY:
            if not self.d.autoRemoveAnomaly():
                return
        elif act == Edit.DELETE_SAMPLES:
            self.deleteSamples()
        self.changeXSeries()

    def removeAnomaly(self):
        self.d.removeAnomaly(self.spin_box_min_x.value(),
                             self.spin_box_max_x.value())

    def deleteSamples(self):
        sel = self.getSelectedRows()
        p = 0
        for i in sel:
            self.datas.pop(i - p)
            p += 1

    def writeTable(self):
        self.table.clear()
        self.table.setColumnCount(self.datas.getMaxDepth())
        self.table.setRowCount(len(self.datas))
        for s in range(len(self.datas)):
            for i, e in enumerate(self.datas[s].x):
                self.table.setItem(s,
                                   i,
                                   QTableWidgetItem(f"{self.datas[s][i]:.5}"))

    def numberColumnChanged(self):
        if self.d is None:
            return
        self.updateGraphics(self.spin_box_number_column.value())
        self.writeProtocol()

    def writeProtocol(self):
        self.protocol.setText(self.d.getProtocol())

    def writeCritetion(self, text):
        self.criterion_protocol.setText(text)

    def updateGraphics(self, number_column: int = 0):
        hist_data = self.d.get_histogram_data(number_column)
        self.drawHistogram(hist_data, self.d.min, self.d.max, self.d.h)
        self.drawEmpFunc(hist_data, self.d.min, self.d.max, self.d.h)
        self.drawReproductionSeries()
        self.spin_box_number_column.blockSignals(True)
        self.spin_box_number_column.setValue(len(hist_data))
        self.spin_box_number_column.blockSignals(False)

# [0-2] - class hist, [3-5] - reproduction series
# set setting axis must before add series to chart!!!!
    def drawHistogram(self, hist_data: list,
                      x_min: float, x_max: float, h: float):
        self.hist_axisX.setRange(x_min, x_max)
        self.hist_axisX.setTickCount(len(hist_data) + 1)

        self.hist_chart.removeAllSeries()

        self.hist_series_top_line = QLineSeries()
        self.hist_series_top_line.setPen(QPen(QColor(255, 255, 255), 5))
        self.hist_series_under_line = QLineSeries()

        y_max: float = hist_data[0]
        for p, i in enumerate(hist_data):
            if y_max < i:
                y_max = i
            self.hist_series_top_line.append(x_min + p * h, 0)
            self.hist_series_top_line.append(x_min + p * h, i)
            self.hist_series_top_line.append(x_min + (p + 1) * h, i)

        self.hist_series_under_line.append(QPointF(x_min, 0))
        self.hist_series_under_line.append(
            QPointF(x_min + h * len(hist_data), 0))

        self.hist_series_area = QAreaSeries(self.hist_series_under_line,
                                            self.hist_series_top_line)

        self.hist_chart.addSeries(self.hist_series_top_line)
        self.hist_chart.addSeries(self.hist_series_under_line)
        self.hist_chart.addSeries(self.hist_series_area)

        self.hist_axisY.setMax(y_max)

    def drawEmpFunc(self, hist_data: list,
                    x_min: float, x_max: float, h: float):
        self.emp_axisX.setRange(x_min, x_max)
        self.emp_axisX.setTickCount(len(hist_data) + 1)

        self.emp_chart.removeAllSeries()

        self.emp_series_class = QLineSeries()
        self.emp_series_func = QLineSeries()

        col_height = 0.0
        for p, i in enumerate(hist_data):
            col_height += i
            self.emp_series_class.append(x_min + p * h, col_height - i)
            self.emp_series_class.append(x_min + p * h, col_height)
            self.emp_series_class.append(x_min + (p + 1) * h, col_height)

        sum_ser = 0.0
        for i in range(len(self.d.probabilityX)):
            sum_ser += self.d.probabilityX[i]
            self.emp_series_func.append(self.d.x[i], sum_ser)

        self.emp_chart.addSeries(self.emp_series_class)
        self.emp_chart.addSeries(self.emp_series_func)

        self.emp_axisY.setMin(0)

    def setReproductionSeries(self):
        self.reprod_num = dict_repr[self.sender().text()]
        self.updateGraphics(self.spin_box_number_column.value())

    def toCreateReproductionFunc(self, func_num):
        if func_num == 0:
            f, F, DF = self.d.toCreateNormalFunc()
        elif func_num == 1:
            f, F, DF = self.d.toCreateUniformFunc()
        elif func_num == 2:
            f, F, DF = self.d.toCreateExponentialFunc()
        elif func_num == 3:
            f, F, DF = self.d.toCreateWeibullFunc()
        elif func_num == 4:
            f, F, DF = self.d.toCreateArcsinFunc()
        else:
            return None, None, None

        return f, F, DF

    def drawReproductionSeries(self):
        self.hist_series_reproduction = QLineSeries()
        self.emp_series_reproduction = QLineSeries()
        self.emp_series_reproduction_low_limit = QLineSeries()
        self.emp_series_reproduction_high_limit = QLineSeries()

        x_gen = []
        try:
            f, F, DF = self.toCreateReproductionFunc(self.reprod_num)
            if f is not None:
                x_gen = self.d.toGenerateReproduction(f, F, DF)
        except ValueError:
            print("Value error")
        except OverflowError:
            print("Overflow error")

        if len(x_gen) == 0:
            self.writeCritetion('')
            return

        criterion_text = '\n'
        if self.d.kolmogorovTest(F):
            criterion_text += "Відтворення адекватне за критерієм" + \
                " згоди Колмогорова\n"
        else:
            criterion_text += "Відтворення неадекватне за критерієм" + \
                " згоди Колмогорова\n"

        try:
            xi_test_result = self.d.xiXiTest(F)
        except ZeroDivisionError:
            xi_test_result = False

        if xi_test_result:
            criterion_text += "Відтворення адекватне за критерієм Пірсона\n"
        else:
            criterion_text += "Відтворення неадекватне за критерієм Пірсона\n"
        self.writeCritetion(criterion_text)

        for i in x_gen:
            self.hist_series_reproduction.append(i[0], i[1])
            self.emp_series_reproduction_low_limit.append(i[0], i[2])
            self.emp_series_reproduction.append(i[0], i[3])
            self.emp_series_reproduction_high_limit.append(i[0], i[4])

        self.hist_chart.addSeries(self.hist_series_reproduction)
        self.hist_series_reproduction.attachAxis(self.hist_axisY)

        self.emp_chart.addSeries(self.emp_series_reproduction)
        self.emp_chart.addSeries(self.emp_series_reproduction_low_limit)
        self.emp_chart.addSeries(self.emp_series_reproduction_high_limit)

    def setSample(self, row):
        self.number_sample = row
        self.d = self.datas[self.number_sample]

        self.spin_box_number_column.blockSignals(True)
        self.spin_box_number_column.setMaximum(len(self.d.x))
        self.spin_box_number_column.setValue(0)
        self.spin_box_number_column.blockSignals(False)
        self.changeXSeries()

    def critsSamples(self):
        sel = self.getSelectedRows()
        print(f"{sel}")
        if len(sel) == 1:
            P = self.datas[sel[0]].critetionAbbe()
            if P > 0.1:
                self.showMessageBox("Критерій Аббе",
                                    f"{P:.5} > 0.1\nСпостереження незалежні")
            else:
                self.showMessageBox("Критерій Аббе",
                                    f"{P:.5} < 0.1\nСпостереження залежні")

        elif len(sel) == 2:
            if self.datas.ident2Samples(sel[0], sel[1]):
                self.showMessageBox("Вибірки ідентичні", "")
            else:
                self.showMessageBox("Вибірки неідентичні", "")
        else:
            if self.datas.identKSamples([self.datas[i] for i in sel]):
                self.showMessageBox("Вибірки ідентичні", "")
            else:
                self.showMessageBox("Вибірки неідентичні", "")


def application(file: str = ''):
    app = QApplication(sys.argv)
    widget = Window(file)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    application("out.txt")
    # application()
