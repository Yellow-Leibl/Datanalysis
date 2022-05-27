import sys
import os
from typing import List
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *
from PyQt5.QtCore import *
from PyQt5 import QtCore
from PyQt5.QtGui import *
from Datanalysis import DataAnalysis
import math

class Window(QMainWindow):
    def __init__(self, file: str):
        super(Window, self).__init__()
        self.setGeometry(150, 150, 1200, 800)
        self.setWindowTitle("Первинний аналіз")

        self.d: DataAnalysis = DataAnalysis([])
        self.series_list = []

        open_file_shortcut = Qt.CTRL + Qt.Key_O

        # menu bar
        self.menuBar = QMenuBar()
        self.setMenuBar(self.menuBar)
        file_menu = QMenu("&Файл", self)
        self.menuBar.addMenu(file_menu)
        file_menu.addAction("&Відкрити", self.openFileAct)
        file_menu.actions()[0].setShortcut(open_file_shortcut)
        file_menu.addAction("&Зберегти", self.saveFileAct)
        file_menu.addAction("В&ийти", exit)
        file_menu.actions()[-1].setShortcut(Qt.CTRL + Qt.Key_Q)

        edit_menu = QMenu("&Редагувати", self)
        self.menuBar.addMenu(edit_menu)
        edit_menu.addAction("&Перетворити", self.transform)
        edit_menu.actions()[0].setShortcut(Qt.CTRL + Qt.Key_T)
        edit_menu.addAction("&Стандартизувати", self.standardization)
        edit_menu.actions()[1].setShortcut(Qt.CTRL + Qt.Key_S)
        edit_menu.addAction("&Зсунути", self.sliding)
        edit_menu.actions()[2].setShortcut(Qt.CTRL + Qt.Key_P)
        edit_menu.addAction("&Повернути", self.UndoChange)
        edit_menu.actions()[3].setShortcut(Qt.CTRL + Qt.Key_Z)
        edit_menu.addAction("&Видалити аномалії", self.autoRemoveAnomaly)
        edit_menu.actions()[4].setShortcut(Qt.CTRL + Qt.Key_D)
        self.reprod_num = -1

        self.vidt_menu = QMenu("&Відтворити", self)
        self.menuBar.addMenu(self.vidt_menu)
        self.vidt_menu.addAction("&Нормальний", self.setReproductionSeries)
        self.vidt_menu.actions()[0].setShortcut(Qt.CTRL + Qt.Key_N)
        self.vidt_menu.addAction("&Рівномірний", self.setReproductionSeries)
        self.vidt_menu.actions()[1].setShortcut(Qt.CTRL + Qt.Key_U)
        self.vidt_menu.addAction("&Експоненціальний", self.setReproductionSeries)
        self.vidt_menu.actions()[2].setShortcut(Qt.CTRL + Qt.Key_E)
        self.vidt_menu.addAction("&Вейбулла", self.setReproductionSeries)
        self.vidt_menu.actions()[3].setShortcut(Qt.CTRL + Qt.Key_W)
        self.vidt_menu.addAction("&Арксинус", self.setReproductionSeries)
        self.vidt_menu.actions()[4].setShortcut(Qt.CTRL + Qt.Key_A)
        self.vidt_menu.addSeparator()
        self.vidt_menu.addAction("&Очистити", self.setReproductionSeries)
        self.vidt_menu.actions()[-1].setShortcut(Qt.CTRL + Qt.Key_C)

        # chart
        self.hist_chart = QChart()
        self.emp_chart = QChart()
        hist_chart_view = QChartView(self.hist_chart)
        emp_chart_view = QChartView(self.emp_chart)
        self.hist_chart.setTitle("Гістограмна оцінка")
        self.emp_chart.setTitle("Емпірична функція розподілу")
        self.hist_chart.legend().setVisible(False)
        self.emp_chart.legend().setVisible(False)

        # axis
        self.hist_axisX = QValueAxis()
        self.hist_axisY = QValueAxis()
        self.emp_axisX = QValueAxis()
        self.emp_axisY = QValueAxis()
        self.hist_chart.addAxis(self.hist_axisX, Qt.AlignBottom)
        self.hist_chart.addAxis(self.hist_axisY, Qt.AlignLeft)
        self.emp_chart.addAxis(self.emp_axisX, Qt.AlignBottom)
        self.emp_chart.addAxis(self.emp_axisY, Qt.AlignLeft)
        self.hist_axisX.setTitleText("x")
        self.hist_axisY.setTitleText("P")
        self.emp_axisX.setTitleText("x")
        self.emp_axisY.setTitleText("P")
        self.emp_axisY.setTickCount(11)
        self.hist_axisY.setTickCount(11)
        
        # series
        self.hist_series_top_line = QLineSeries()
        self.hist_chart.addSeries(self.hist_series_top_line)

        self.hist_series_under_line = QLineSeries()
        self.hist_series_area = QAreaSeries()
        
        self.emp_series_class = QLineSeries()
        self.emp_chart.addSeries(self.emp_series_class)
        self.emp_series_func = QLineSeries()
        
        self.hist_series_reproduction = QLineSeries()
        self.emp_series_reproduction = QLineSeries()

        self.hist_series_top_line.attachAxis(self.hist_axisX)
        self.hist_series_top_line.attachAxis(self.hist_axisY)
        
        self.emp_series_class.attachAxis(self.emp_axisX)
        self.emp_series_class.attachAxis(self.emp_axisY)

        # spin boxes
        self.spin_box_number_column = QSpinBox()
        self.spin_box_number_column.setMinimum(0)
        self.spin_box_number_column.valueChanged.connect(self.numberColumnChanged)
        self.spin_box_min_x = QDoubleSpinBox()
        self.spin_box_min_x.setDecimals(5)
        self.spin_box_max_x = QDoubleSpinBox()
        self.spin_box_max_x.setDecimals(5)
        self.remove_anomaly = QPushButton("Видалити аномалії")
        self.remove_anomaly.clicked.connect(self.removeAnomaly)

        # QTable
        self.table = QTableWidget()

        # QTextEdit
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setFontFamily("Monospace")
        self.text_edit.setFontFamily("Andale Mono")

        # # template
        widget = QWidget()
        vbox = QVBoxLayout()
        # table and textEdit tab
        tab_text_info = QTabWidget()
        tab_text_info.addTab(self.text_edit, "Протокол")
        tab_text_info.addTab(self.table, "Ранжований ряд")
        # box with transform func
        widget_func = QWidget()
        vbox_func = QVBoxLayout()

        h_step_box = QHBoxLayout()
        h_step_box.addWidget(QLabel("h"), 1)
        h_step_box.addWidget(self.spin_box_number_column, 9)

        anomaly_box = QHBoxLayout()
        anomaly_box.addWidget(QLabel("min"), 1)
        anomaly_box.addWidget(self.spin_box_min_x, 3)
        anomaly_box.addWidget(QLabel("max"), 1)
        anomaly_box.addWidget(self.spin_box_max_x, 3)

        vbox_func.addLayout(h_step_box)
        vbox_func.addLayout(anomaly_box)
        vbox_func.addWidget(self.remove_anomaly)

        widget_func.setLayout(vbox_func)
        # tab and add. functionality
        info_text_box = QHBoxLayout()
        info_text_box.addWidget(tab_text_info, 3)
        info_text_box.addWidget(widget_func, 1)
        # 2 chart box
        graphics_box = QHBoxLayout()
        graphics_box.addWidget(hist_chart_view)
        graphics_box.addWidget(emp_chart_view)

        vbox.addLayout(graphics_box, 3)
        vbox.addLayout(info_text_box, 1)

        widget.setLayout(vbox)
        self.setCentralWidget(widget)

        if file != "":
            self.openFile(file)

    @QtCore.pyqtSlot()    
    def openFileAct(self):
        self.openFile('')

    def openFile(self, file: str):
        file_name = file
        
        if file == '':
            file_name = QFileDialog().getOpenFileName(None,"Відкрити файл",
                                                      os.getcwd(), "Bci файли (*)")[0]

        try:
            with open(file_name, 'r') as file:
                self.series_list.clear()
                self.series_list.append([float(i.split(' ')[0]) for i in file])
                self.d = DataAnalysis(self.series_list[0])
                self.d.toRanking()
                self.d.toCalculateCharacteristic()

                self.spin_box_number_column.blockSignals(True)
                self.spin_box_number_column.setMaximum(len(self.d.x))
                self.spin_box_number_column.setValue(0)
                self.spin_box_number_column.blockSignals(False)
                
                self.reprod_num = -1

                self.changeXSeries()
        except FileNotFoundError:
            print("\"", file_name, "\" not found")
    
    @QtCore.pyqtSlot()
    def saveFileAct(self):
        file_name = QFileDialog().getSaveFileName(self, "Зберегти файл",
                                                  os.getcwd(), "Bci файли (*)")[0]
        try:
            with open(file_name, 'w') as file:
                file.write('\n'.join([str(i) for i in self.d.x]))
        except:
            print("Error saving file")

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

    @QtCore.pyqtSlot()
    def standardization(self):
        self.series_list.append(self.d.x.copy())
        self.d.toStandardization()
        self.changeXSeries()

    @QtCore.pyqtSlot()
    def transform(self):
        self.series_list.append(self.d.x.copy())
        self.d.toTransform()
        self.changeXSeries()

    @QtCore.pyqtSlot()
    def sliding(self):
        self.series_list.append(self.d.x.copy())
        self.d.toSlide()
        self.changeXSeries()

    @QtCore.pyqtSlot()
    def UndoChange(self):
        if len(self.series_list) <= 1:
            return
        self.d.setSeries(self.series_list.pop())
        self.changeXSeries()

    @QtCore.pyqtSlot()
    def removeAnomaly(self):
        self.series_list.append(self.d.x.copy())
        self.d.RemoveAnomaly(self.spin_box_min_x.value(), self.spin_box_max_x.value())
        self.changeXSeries()

    @QtCore.pyqtSlot()
    def autoRemoveAnomaly(self):
        self.series_list.append(self.d.x.copy())
        if self.d.AutoRemoveAnomaly():
            self.changeXSeries()

    def writeTable(self):
        self.table.setColumnCount(len(self.d.x))
        self.table.setRowCount(1)
        for i in range(len(self.d.x)):
            item = QTableWidgetItem(f"{self.d.x[i]:.5f}")
            self.table.setItem(0, i, item)

    @QtCore.pyqtSlot()
    def numberColumnChanged(self):
        if self.d is None:
            return
        self.updateGraphics(self.spin_box_number_column.value())
        self.writeProtocol()

    def writeProtocol(self):
        self.text_edit.setText(self.d.getProtocol())

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
    def drawHistogram(self, hist_data: List[float], x_min: float, x_max: float, h: float):
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
        self.hist_series_under_line.append(QPointF(x_min + h * len(hist_data), 0))

        self.hist_series_area = QAreaSeries(self.hist_series_under_line, self.hist_series_top_line)

        self.hist_chart.addSeries(self.hist_series_top_line)
        self.hist_chart.addSeries(self.hist_series_under_line)
        self.hist_chart.addSeries(self.hist_series_area)

        self.hist_axisY.setMax(y_max)

    @QtCore.pyqtSlot()
    def setReproductionSeries(self):
        reprod = self.sender()
        for i in range(len(self.vidt_menu.actions())):
            if reprod == self.vidt_menu.actions()[i]:
                self.reprod_num = i
        self.updateGraphics(self.spin_box_number_column.value())
    
    def drawReproductionSeries(self):
        self.hist_series_reproduction = QLineSeries()
        self.emp_series_reproduction = QLineSeries()
        self.emp_series_reproduction_low_limit = QLineSeries()
        self.emp_series_reproduction_high_limit = QLineSeries()
        
        x_gen = []
        try:
            x_gen = self.d.toGenerateReproduction(self.reprod_num)
        except:
            print("Error reproduction")
        if len(x_gen) == 0:
            return

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

    def drawEmpFunc(self, hist_data: List[float], x_min: float, x_max: float, h: float):
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


def application(file: str = ''):
    app = QApplication(sys.argv)
    widget = Window(file)
    widget.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    application("data/500/norm.txt")
    # application()
