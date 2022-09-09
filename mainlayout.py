from PyQt5.QtWidgets import (QMainWindow, QWidget, QMenu, QMenuBar,
                             QSpinBox, QDoubleSpinBox, QPushButton,
                             QTableWidget, QTextEdit, QTabWidget,
                             QHBoxLayout, QVBoxLayout, QLabel)
from PyQt5.QtChart import (QChart, QChartView, QValueAxis,
                           QLineSeries, QAreaSeries)
from PyQt5.QtCore import Qt

import platform


dict_edit = {
    "&Перетворити": 0,
    "&Стандартизувати": 1,
    "&Зсунути": 2,
    "&Повернути": 3,
    "&Видалити аномалії": 4
}

dict_shortcut = {
    "&Перетворити": Qt.CTRL + Qt.Key_T,
    "&Стандартизувати": Qt.CTRL + Qt.Key_S,
    "&Зсунути": Qt.CTRL + Qt.Key_P,
    "&Повернути": Qt.CTRL + Qt.Key_Z,
    "&Видалити аномалії": Qt.CTRL + Qt.Key_D
}


class MainLayout(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setGeometry(150, 100, 1200, 800)
        self.setWindowTitle("Первинний аналіз")

        # File menu
        file_menu = QMenu("&Файл", self)
        file_menu.addAction("&Відкрити", self.openFileAct)
        file_menu.actions()[0].setShortcut(Qt.CTRL + Qt.Key_O)
        file_menu.addAction("&Зберегти", self.saveFileAct)
        file_menu.addAction("В&ийти", exit)
        file_menu.actions()[-1].setShortcut(Qt.CTRL + Qt.Key_Q)

        # Editing menu
        edit_menu = QMenu("&Редагувати", self)
        for k, v in dict_edit.items():
            edit_menu.addAction(k, self.EditEvent)
            edit_menu.actions()[v].setShortcut(dict_shortcut[k])
        self.reprod_num = -1

        # Reproduction menu
        self.vidt_menu = QMenu("&Відтворити", self)
        self.vidt_menu.addAction("&Нормальний", self.setReproductionSeries)
        self.vidt_menu.actions()[0].setShortcut(Qt.CTRL + Qt.Key_N)
        self.vidt_menu.addAction("&Рівномірний", self.setReproductionSeries)
        self.vidt_menu.actions()[1].setShortcut(Qt.CTRL + Qt.Key_U)
        self.vidt_menu.addAction("&Експоненціальний",
                                 self.setReproductionSeries)
        self.vidt_menu.actions()[2].setShortcut(Qt.CTRL + Qt.Key_E)
        self.vidt_menu.addAction("&Вейбулла", self.setReproductionSeries)
        self.vidt_menu.actions()[3].setShortcut(Qt.CTRL + Qt.Key_W)
        self.vidt_menu.addAction("&Арксинус", self.setReproductionSeries)
        self.vidt_menu.actions()[4].setShortcut(Qt.CTRL + Qt.Key_A)
        self.vidt_menu.addSeparator()
        self.vidt_menu.addAction("&Очистити", self.setReproductionSeries)
        self.vidt_menu.actions()[-1].setShortcut(Qt.CTRL + Qt.Key_C)

        # Menu bar
        self.menuBar = QMenuBar()
        self.setMenuBar(self.menuBar)
        self.menuBar.addMenu(file_menu)
        self.menuBar.addMenu(edit_menu)
        self.menuBar.addMenu(self.vidt_menu)

        # Histogram chart
        self.hist_chart = QChart()
        self.hist_chart.setTitle("Гістограмна оцінка")
        self.hist_chart.legend().setVisible(False)
        hist_chart_view = QChartView(self.hist_chart)

        # Empirical chart
        self.emp_chart = QChart()
        self.emp_chart.setTitle("Емпірична функція розподілу")
        self.emp_chart.legend().setVisible(False)
        emp_chart_view = QChartView(self.emp_chart)

        # Histogram axes
        self.hist_axisX = QValueAxis()
        self.hist_axisY = QValueAxis()
        self.hist_axisX.setTitleText("x")
        self.hist_axisY.setTitleText("P")
        self.hist_axisY.setTickCount(11)
        self.hist_chart.addAxis(self.hist_axisX, Qt.AlignBottom)
        self.hist_chart.addAxis(self.hist_axisY, Qt.AlignLeft)

        # Empirical axes
        self.emp_axisX = QValueAxis()
        self.emp_axisY = QValueAxis()
        self.emp_axisX.setTitleText("x")
        self.emp_axisY.setTitleText("P")
        self.emp_axisY.setTickCount(11)
        self.emp_chart.addAxis(self.emp_axisX, Qt.AlignBottom)
        self.emp_chart.addAxis(self.emp_axisY, Qt.AlignLeft)

        # Histogram series
        self.hist_series_top_line = QLineSeries()
        self.hist_series_under_line = QLineSeries()
        self.hist_series_area = QAreaSeries()
        self.hist_series_reproduction = QLineSeries()

        # Empirical series
        self.emp_series_class = QLineSeries()
        self.emp_chart.addSeries(self.emp_series_class)
        self.emp_series_func = QLineSeries()
        self.emp_series_reproduction = QLineSeries()

        self.emp_series_class.attachAxis(self.emp_axisX)
        self.emp_series_class.attachAxis(self.emp_axisY)

        # spin boxes
        self.spin_box_number_column = QSpinBox()
        self.spin_box_number_column.setMinimum(0)
        self.spin_box_number_column.valueChanged.connect(
            self.numberColumnChanged)
        self.spin_box_min_x = QDoubleSpinBox()
        self.spin_box_min_x.setDecimals(5)
        self.spin_box_max_x = QDoubleSpinBox()
        self.spin_box_max_x.setDecimals(5)
        self.remove_anomaly = QPushButton("Видалити аномалії")
        self.remove_anomaly.clicked.connect(self.removeAnomaly)

        # Samples table
        self.table = QTableWidget()
        self.table.cellDoubleClicked.connect(
            lambda: self.setSample(self.table.currentRow()))

        # Protocol
        self.protocol = QTextEdit()
        self.protocol.setReadOnly(True)
        self.protocol.setFontFamily(MonoFontForSpecificOS())

        # Kriterii
        self.criterion_protocol = QTextEdit()
        self.criterion_protocol.setReadOnly(True)
        self.criterion_protocol.setFontFamily(MonoFontForSpecificOS())

        # # Layout
        widget = QWidget()
        main_vbox = QVBoxLayout()

        # table and textEdit tab
        tab_text_info = QTabWidget()
        tab_text_info.addTab(self.protocol, "Протокол")
        tab_text_info.addTab(self.criterion_protocol, "Критерії")
        tab_text_info.addTab(self.table, "Ранжований ряд")

        # box with transform func
        widget_func = QWidget()
        vbox_func = QVBoxLayout()

        # number of classes
        h_step_box = QHBoxLayout()
        h_step_box.addWidget(QLabel("h"), 1)
        h_step_box.addWidget(self.spin_box_number_column, 9)

        # borders
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

        main_vbox.addLayout(graphics_box, 3)
        main_vbox.addLayout(info_text_box, 1)

        widget.setLayout(main_vbox)
        self.setCentralWidget(widget)


def MonoFontForSpecificOS():
    name = platform.system()
    if name == 'Darwin':
        return "Andale Mono"
    else:
        return "Monospace"
