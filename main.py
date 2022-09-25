import sys
import os

from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QApplication

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
        self.drawHistogram(hist_data, self.d.min, self.d.h)
        self.drawEmpFunc(hist_data, self.d.min, self.d.h)
        self.drawReproductionSeries()
        self.spin_box_number_column.blockSignals(True)
        self.spin_box_number_column.setValue(len(hist_data))
        self.spin_box_number_column.blockSignals(False)

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

        x_stat = []
        y_stat = []
        sum_ser = 0.0
        for i in range(len(self.d.probabilityX)):
            sum_ser += self.d.probabilityX[i]
            x_stat.append(self.d.x[i])
            y_stat.append(sum_ser)

        self.emp_plot.clear()
        self.emp_plot.plot(x_class, y_class, pen=(255, 0, 0))
        self.emp_plot.plot(x_stat, y_stat, pen=(0, 255, 0))

    def setReproductionSeries(self):
        reprod = self.sender()
        for i in range(len(self.vidt_menu.actions())):
            if reprod == self.vidt_menu.actions()[i]:
                self.reprod_num = i
                break
        self.updateGraphics(self.spin_box_number_column.value())

    def drawReproductionSeries(self):
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
        self.hist_plot.plot(x, y_hist, pen=(0, 0, 255))
        self.emp_plot.plot(x, y_low, pen=(0, 128, 128))
        self.emp_plot.plot(x, y_emp, pen=(0, 0, 255))
        self.emp_plot.plot(x, y_high, pen=(128, 0, 128))

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
