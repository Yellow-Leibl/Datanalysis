import sys
import os

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QApplication
from PyQt5.QtChart import QLineSeries, QAreaSeries, QValueAxis
from PyQt5.QtGui import QPen, QColor

from Datanalysis import SamplingDatas
from historystack import HistoryStask
from mainlayout import MainLayout, dict_edit


class Window(MainLayout):
    def __init__(self, file: str):
        super().__init__()  # layout here

        self.number_sample = 0

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

                self.datas = SamplingDatas(all_file)
                self.history_series = HistoryStask(self.datas.dimension)
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

    def EditEvent(self):
        act = dict_edit[self.sender().text()]

        if act != 3:
            self.history_series.push(self.number_sample, self.d.x.copy())
        else:
            if self.history_series.len(self.number_sample) == 0:
                return
            self.d.setSeries(self.history_series.pop(self.number_sample))

        if act == 0:
            self.d.toTransform()
        elif act == 1:
            self.d.toStandardization()
        elif act == 2:
            self.d.toSlide()
        elif act == 4:
            if not self.d.AutoRemoveAnomaly():
                return
        self.changeXSeries()

    def removeAnomaly(self):
        self.d.RemoveAnomaly(self.spin_box_min_x.value(),
                             self.spin_box_max_x.value())

    def writeTable(self):
        self.table.clear()
        self.table.setColumnCount(self.datas.getMaxDepth())
        self.table.setRowCount(self.datas.dimension)
        for s in range(self.datas.dimension):
            for i, e in enumerate(self.datas[s].x):
                self.table.setItem(s,
                                   i,
                                   QTableWidgetItem(str(self.datas[s][i])))

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
            if col_height > 1:
                col_height = 1
            self.emp_series_class.append(x_min + p * h, col_height)
            self.emp_series_class.append(x_min + p * h, col_height + i)
            self.emp_series_class.append(x_min + (p + 1) * h, col_height + i)
            col_height += i

        sum_ser = 0.0
        for i in range(len(self.d.probabilityX)):
            sum_ser += self.d.probabilityX[i]
            self.emp_series_func.append(self.d.x[i], sum_ser)

        self.emp_chart.addSeries(self.emp_series_class)
        self.emp_chart.addSeries(self.emp_series_func)

        self.emp_axisY.setMin(0)

    def setReproductionSeries(self):
        reprod = self.sender()
        for i in range(len(self.vidt_menu.actions())):
            if reprod == self.vidt_menu.actions()[i]:
                self.reprod_num = i
                break
        self.updateGraphics(self.spin_box_number_column.value())

    def drawReproductionSeries(self):
        self.hist_series_reproduction = QLineSeries()
        self.emp_series_reproduction = QLineSeries()
        self.emp_series_reproduction_low_limit = QLineSeries()
        self.emp_series_reproduction_high_limit = QLineSeries()

        x_gen = []
        try:
            f, F, DF = self.d.toCreateReproductionFunc(self.reprod_num)
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
        if self.d.KolmogorovTest(F):
            criterion_text += "Відтворення адекватне за критерієм" + \
                " згоди Колмогорова\n"
        else:
            criterion_text += "Відтворення неадекватне за критерієм" + \
                " згоди Колмогорова\n"

        if self.d.XiXiTest(F):
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

        emp_axis_y = QValueAxis()
        self.emp_chart.addAxis(emp_axis_y, Qt.AlignmentFlag.AlignLeft)
        self.emp_series_reproduction.attachAxis(emp_axis_y)
        self.emp_series_reproduction_low_limit.attachAxis(emp_axis_y)
        self.emp_series_reproduction_high_limit.attachAxis(emp_axis_y)

        self.emp_chart.axes()[-1].setVisible(False)

    def setSample(self, row):
        self.number_sample = row
        self.d = self.datas[self.number_sample]

        self.spin_box_number_column.blockSignals(True)
        self.spin_box_number_column.setMaximum(len(self.d.x))
        self.spin_box_number_column.setValue(0)
        self.spin_box_number_column.blockSignals(False)
        self.changeXSeries()


def application(file: str = ''):
    app = QApplication(sys.argv)
    widget = Window(file)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    application("data/500/norm3n.txt")
    # application()
