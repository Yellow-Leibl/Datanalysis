from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QMenu, QMenuBar,
    QSpinBox, QDoubleSpinBox, QPushButton,
    QTableWidget, QAbstractItemView,
    QTextEdit, QTabWidget,
    QHBoxLayout, QVBoxLayout, QFormLayout,
    QLabel, QMessageBox)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
# import pyqtgraph.opengl as _3d

import platform

from GeneralConstants import (
    dict_edit, dict_edit_shortcut, dict_repr, dict_repr_shortcut,
    dict_crit, dict_crit_shortcut, dict_regr, dict_regr_shortcut,
    Edit, Critetion)


class MainLayout(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setGeometry(150, 100, 1333, 733)
        self.setWindowTitle("Аналіз даних")

        # PyQtGraph global configuration
        pg.setConfigOption('imageAxisOrder', 'row-major')

        # 2 chart box
        self.layout_widget = pg.GraphicsLayoutWidget()
        self.createPlotLayout(2)

        # spin boxes
        self.__spin_number_column = QSpinBox()
        self.__spin_number_column.setMinimum(0)
        self.__spin_number_column.valueChanged.connect(
            lambda: self.numberColumnChanged(
                self.__spin_number_column.value()))

        self.__spin_box_level = QDoubleSpinBox()
        self.__spin_box_level.setDecimals(5)
        self.__spin_box_level.setMinimum(0)
        self.__spin_box_level.setMaximum(1)
        self.__spin_box_level.setValue(0.05)
        self.__spin_box_level.valueChanged.connect(
            lambda: self.changeTrust(self.__spin_box_level.value()))

        self.__spin_box_min_x = QDoubleSpinBox()
        self.__spin_box_min_x.setDecimals(5)
        self.__spin_box_max_x = QDoubleSpinBox()
        self.__spin_box_max_x.setDecimals(5)
        self.__remove_anomaly = QPushButton("Видалити аномалії")
        self.__remove_anomaly.clicked.connect(self.removeAnomaly)

        # Samples table
        self.table = QTableWidget()
        self.table.cellDoubleClicked.connect(
            lambda: self.drawSamples())
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # self.table.setSelectionMode(QAbstractItemView.MultiSelection)

        # Protocol
        self.protocol = QTextEdit()
        self.protocol.setReadOnly(True)
        self.protocol.setFontFamily(MonoFontForSpecificOS())

        # Kriterii
        self.criterion_protocol = QTextEdit()
        self.criterion_protocol.setReadOnly(True)
        self.criterion_protocol.setFontFamily(MonoFontForSpecificOS())

        # table and textEdit tab
        tab_text_info = QTabWidget()
        tab_text_info.addTab(self.protocol, "Протокол")
        tab_text_info.addTab(self.criterion_protocol, "Критерії")
        tab_text_info.addTab(self.table, "Ранжований ряд")

        # grid with transform func
        widget_func = QVBoxLayout()
        form_widget = QFormLayout()
        form_widget.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        widget_func.addLayout(form_widget)

        form_widget.addRow("Кількість класів:",
                           self.__spin_number_column)
        form_widget.addRow("Рівень значущості:",
                           self.__spin_box_level)

        self.setMenuBar(self.createMenuBar1d())

        # borders
        borders = QHBoxLayout()
        borders.addWidget(QLabel("min"))
        borders.addWidget(self.__spin_box_min_x)
        borders.addWidget(QLabel("max"))
        borders.addWidget(self.__spin_box_max_x)
        widget_func.addLayout(borders)
        widget_func.addWidget(self.__remove_anomaly)

        # tab and add. functionality
        info_text_box = QHBoxLayout()
        info_text_box.addWidget(tab_text_info, 3)
        info_text_box.addLayout(widget_func, 1)

        main_vbox = QVBoxLayout()
        main_vbox.addWidget(self.layout_widget, 3)
        main_vbox.addLayout(info_text_box, 1)

        widget = QWidget()
        widget.setLayout(main_vbox)
        self.setCentralWidget(widget)

    def createPlotLayout(self, n: int):
        graphics = pg.GraphicsLayout()
        if n == 1:
            self.hist_plot = graphics.addPlot(
                title="Гістограмна оцінка",
                labels={"left": "P", "bottom": "x"})
            self.emp_plot = graphics.addPlot(
                title="Емпірична функція розподілу",
                labels={"left": "P", "bottom": "x"})
        elif n == 2:
            self.hist_plot = graphics.addPlot()
        elif n == 3:
            pass
        self.layout_widget.setCentralWidget(graphics)

    def createMenuBar1d(self):
        # File menu
        file_menu = QMenu("&Файл", self)
        file_menu.addAction("&Відкрити", lambda: self.openFile(''))
        file_menu.actions()[0].setShortcut(Qt.CTRL + Qt.Key_O)
        file_menu.addAction("&Зберегти", self.saveFileAct)
        file_menu.addAction("В&ийти", exit)
        file_menu.actions()[-1].setShortcut(Qt.CTRL + Qt.Key_Q)

        # Editing menu
        edit_menu = QMenu("&Редагувати", self)
        for k, v in dict_edit.items():
            if v == Edit.DRAW_SAMPLES.value:
                edit_menu.addAction(k, self.drawSamples)
            elif v == Edit.DELETE_SAMPLES.value:
                edit_menu.addAction(k, self.deleteSamples)
            elif v == Edit.DUPLICATE.value:
                edit_menu.addAction(k, self.duplicateSample)
            else:
                edit_menu.addAction(k, self.editSampleEvent)
            edit_menu.actions()[-1].setShortcut(dict_edit_shortcut[k])
            if k == "&Видалити аномалії":
                edit_menu.addSeparator()
        self.reprod_num = -1

        # Reproduction menu
        vidt_menu = QMenu("&Відтворити", self)
        for k, v in dict_repr.items():
            if k == "&Очистити":
                vidt_menu.addSeparator()
            vidt_menu.addAction(k, self.setReproductionSeries)
            vidt_menu.actions()[-1].setShortcut(dict_repr_shortcut[k])

        # Regression menu
        regr_menu = QMenu("&Регресія", self)
        for k, v in dict_regr.items():
            if k == "&Очистити":
                regr_menu.addSeparator()
            regr_menu.addAction(k, self.setReproductionSeries)
            regr_menu.actions()[-1].setShortcut(dict_regr_shortcut[k])

        # Critetion menu
        crit_menu = QMenu("&Критерії", self)
        for k, v in dict_crit.items():
            if v == Critetion.HOMOGENEITY_INDEPENDENCE:
                crit_menu.addAction(k, lambda: self.homogeneityAndIndependence(
                    self.__spin_box_level.value()))
            if v == Critetion.LINEAR_REGRESSION_MODELS:
                crit_menu.addAction(k, lambda: self.linearModelsCrit(
                    self.__spin_box_level.value()))
            crit_menu.actions()[-1].setShortcut(dict_crit_shortcut[k])
        menuBar = QMenuBar()
        menuBar.addMenu(file_menu)
        menuBar.addMenu(edit_menu)
        menuBar.addMenu(vidt_menu)
        menuBar.addMenu(regr_menu)
        menuBar.addMenu(crit_menu)
        return menuBar

    def getMinMax(self):
        return (self.__spin_box_min_x.value(),
                self.__spin_box_max_x.value())

    def getSelectedRows(self) -> list:
        ranges = self.table.selectedRanges()
        sel_rows = []
        for r in ranges:
            for i in range(r.topRow(), r.bottomRow() + 1):
                sel_rows.append(i)
        return sel_rows

    def showMessageBox(self, text: str, informative_text: str):
        box = QMessageBox()
        box.setText(text)
        box.setInformativeText(informative_text)
        box.exec()

    def getNumberClasses(self) -> int:
        return self.__spin_number_column.value()

    def silentChangeNumberClasses(self, n: int) -> bool:
        self.__spin_number_column.blockSignals(True)
        self.__spin_number_column.setValue(n)
        self.__spin_number_column.blockSignals(False)

    def setMaximumColumnNumber(self, n: int):
        self.__spin_number_column.setMaximum(n)

    def setMinMax(self, min_x, max_x):
        self.__spin_box_min_x.setMinimum(min_x)
        self.__spin_box_min_x.setMaximum(max_x)
        self.__spin_box_min_x.setValue(min_x)

        self.__spin_box_max_x.setMinimum(min_x)
        self.__spin_box_max_x.setMaximum(max_x)
        self.__spin_box_max_x.setValue(max_x)


def MonoFontForSpecificOS():
    name = platform.system()
    if name == 'Darwin':
        return "Andale Mono"
    else:
        return "Monospace"


if __name__ == "__main__":
    import pyqtgraph.examples
    pyqtgraph.examples.run()
