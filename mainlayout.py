from PyQt6.QtWidgets import (
    QMainWindow, QWidget,
    QSpinBox, QDoubleSpinBox, QPushButton,
    QTableWidget, QAbstractItemView,
    QTextEdit, QTabWidget,
    QHBoxLayout, QVBoxLayout, QFormLayout,
    QLabel, QMessageBox, QMenu, QSplitter)
from PyQt6.QtCore import Qt
from PlotWidget import PlotWidget

import platform

from GeneralConstants import (
    dict_edit, dict_edit_shortcut, dict_crit, dict_crit_shortcut,
    dict_regression, dict_regr_shortcut,
    dict_file_shortcut, Edit, Critetion, dict_view_shortcut)


def addAction(menu: QMenu, title, action, shortcut_dict):
    menu.addAction(title, action)
    menu.actions()[-1].setShortcut(shortcut_dict[title])


class MainLayout(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setGeometry(150, 100, 1333, 733)
        self.setWindowTitle("Аналіз даних")

        # 2 chart box
        self.plot_widget = PlotWidget()
        self.plot_widget.create2DPlot()

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
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        # self.table.setSelectionMode(QAbstractItemView.SelectionBehavior.MultiSelection)

        # Protocol
        self.protocol = QTextEdit()
        self.protocol.setReadOnly(True)
        self.protocol.setFontFamily(MonoFontForSpecificOS())

        # Kriterii
        self.criterion_protocol = QTextEdit()
        self.criterion_protocol.setReadOnly(True)
        self.criterion_protocol.setFontFamily(MonoFontForSpecificOS())

        # table and textEdit tab
        self.tab_info = QTabWidget()
        self.tab_info.addTab(self.protocol, "Протокол")
        self.tab_info.addTab(self.criterion_protocol, "Критерії")
        self.tab_info.addTab(self.table, "Ранжований ряд")

        # grid with transform func
        widget_func = QVBoxLayout()
        form_widget = QFormLayout()
        form_widget.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form_widget.addRow("Кількість класів:",
                           self.__spin_number_column)
        form_widget.addRow("Рівень значущості:",
                           self.__spin_box_level)
        widget_func.addLayout(form_widget)

        self.createMenuBar()

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
        info_text_box.addWidget(self.tab_info, 3)
        info_text_box.addLayout(widget_func, 1)
        info_wid = QWidget()
        info_wid.setLayout(info_text_box)

        main_vbox = QSplitter(Qt.Orientation.Vertical)
        main_vbox.addWidget(self.plot_widget)
        main_vbox.addWidget(info_wid)

        self.setCentralWidget(main_vbox)

    def createPlotLayout(self, n: int):
        if n == 1:
            self.plot_widget.create1DPlot()
        elif n == 2:
            self.plot_widget.create2DPlot()
        elif n == 3:
            self.plot_widget.create3DPlot()

    def createMenuBar(self):
        # File menu
        menuBar = self.menuBar()
        file_menu = menuBar.addMenu("&Файл")
        addAction(file_menu, "&Відкрити", lambda: self.openFile(''),
                  dict_file_shortcut)
        addAction(file_menu, "&Зберегти", self.saveFileAct,
                  dict_file_shortcut)
        addAction(file_menu, "В&ийти", self.saveFileAct,
                  dict_file_shortcut)

        # Editing menu
        edit_menu = menuBar.addMenu("&Редагувати")
        for k, v in dict_edit.items():
            if v == Edit.DRAW_SAMPLES.value:
                addAction(edit_menu, k, self.drawSamples, dict_edit_shortcut)
            elif v == Edit.DELETE_SAMPLES.value:
                addAction(edit_menu, k, self.deleteSamples, dict_edit_shortcut)
            elif v == Edit.DUPLICATE.value:
                addAction(edit_menu, k, self.duplicateSample,
                          dict_edit_shortcut)
            else:
                addAction(edit_menu, k, self.editSampleEvent,
                          dict_edit_shortcut)
            if k == "&Видалити аномалії":
                edit_menu.addSeparator()
        self.reprod_num = -1

        view_menu = menuBar.addMenu("&Вигляд")
        for k, v in dict_view_shortcut.items():
            if k == "&Наступна вкладка":
                addAction(view_menu, k,
                          lambda: self.tab_info.setCurrentIndex(
                              (self.tab_info.currentIndex() + 1)
                              % len(self.tab_info.tabBar())),
                          dict_view_shortcut)

        # Regression menu
        regr_menu = menuBar.addMenu("&Регресія")
        for k, v in dict_regression.items():
            if v == 5 or v == 9 or v == 10:
                regr_menu.addSeparator()
            addAction(regr_menu, k, self.setReproductionSeries,
                      dict_regr_shortcut)

        # Critetion menu
        crit_menu = menuBar.addMenu("&Критерії")
        for k, v in dict_crit.items():
            if v == Critetion.HOMOGENEITY_INDEPENDENCE:
                crit_menu.addAction(k, lambda: self.homogeneityAndIndependence(
                    self.__spin_box_level.value()))
            if v == Critetion.LINEAR_REGRESSION_MODELS:
                crit_menu.addAction(k, lambda: self.linearModelsCrit(
                    self.__spin_box_level.value()))
            crit_menu.actions()[-1].setShortcut(dict_crit_shortcut[k])

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

    def silentChangeNumberClasses(self, n: int):
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
