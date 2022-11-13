from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QMenu, QMenuBar,
    QSpinBox, QDoubleSpinBox, QPushButton,
    QTableWidget, QAbstractItemView,
    QTextEdit, QTabWidget,
    QHBoxLayout, QVBoxLayout, QFormLayout,
    QLabel, QMessageBox)
from PyQt5.QtCore import Qt
from pyqtgraph import PlotWidget
import pyqtgraph as pg
# import pyqtgraph.opengl as _3d

import platform

from GeneralConstants import (
    dict_edit, dict_edit_shortcut, dict_repr, dict_repr_shortcut,
    dict_crit, dict_crit_shortcut, Edit)


class MainLayout(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setGeometry(150, 100, 1333, 733)
        self.setWindowTitle("Первинний аналіз")

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

        # Critetion menu
        crit_menu = QMenu("&Критерії", self)
        for k, v in dict_crit.items():
            crit_menu.addAction(k,
                                lambda: self.critsSamples(
                                    spin_box_level.value()))
            crit_menu.actions()[-1].setShortcut(dict_crit_shortcut[k])

        # Menu bar
        menuBar = QMenuBar()
        menuBar.addMenu(file_menu)
        menuBar.addMenu(edit_menu)
        menuBar.addMenu(vidt_menu)
        menuBar.addMenu(crit_menu)
        self.setMenuBar(menuBar)

        # Histogram chart
        self.hist_plot: PlotWidget = pg.PlotWidget(
            title="Гістограмна оцінка",
            labels={"left": "P", "bottom": "x"})

        # Empirical chart
        self.emp_plot: PlotWidget = pg.PlotWidget(
            title="Емпірична функція розподілу",
            labels={"left": "P", "bottom": "x"})

        # spin boxes
        self.spin_number_column = QSpinBox()
        self.spin_number_column.setMinimum(0)
        self.spin_number_column.valueChanged.connect(
            lambda: self.numberColumnChanged(
                self.spin_number_column.value()))

        spin_box_level = QDoubleSpinBox()
        spin_box_level.setDecimals(5)
        spin_box_level.setMinimum(0)
        spin_box_level.setMaximum(1)
        spin_box_level.setValue(0.05)
        spin_box_level.valueChanged.connect(
            lambda: self.changeTrust(spin_box_level.value()))

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
                           self.spin_number_column)
        form_widget.addRow("Рівень значущості:",
                           spin_box_level)

        # borders
        borders = QHBoxLayout()
        borders.addWidget(QLabel("min"))
        borders.addWidget(self.spin_box_min_x)
        borders.addWidget(QLabel("max"))
        borders.addWidget(self.spin_box_max_x)
        widget_func.addLayout(borders)
        widget_func.addWidget(self.remove_anomaly)

        # tab and add. functionality
        info_text_box = QHBoxLayout()
        info_text_box.addWidget(tab_text_info, 3)
        info_text_box.addLayout(widget_func, 1)

        # 2 chart box
        graphics_box = QHBoxLayout()
        graphics_box.addWidget(self.hist_plot)
        graphics_box.addWidget(self.emp_plot)

        main_vbox = QVBoxLayout()
        main_vbox.addLayout(graphics_box, 3)
        main_vbox.addLayout(info_text_box, 1)

        widget = QWidget()
        widget.setLayout(main_vbox)
        self.setCentralWidget(widget)

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
        return self.spin_number_column.value()

    def silentChangeNumberClasses(self, n: int) -> bool:
        self.spin_number_column.blockSignals(True)
        self.spin_number_column.setValue(n)
        self.spin_number_column.blockSignals(False)

    def setMinMax(self, min_x, max_x):
        self.spin_box_min_x.setMinimum(min_x)
        self.spin_box_min_x.setMaximum(max_x)
        self.spin_box_min_x.setValue(min_x)

        self.spin_box_max_x.setMinimum(min_x)
        self.spin_box_max_x.setMaximum(max_x)
        self.spin_box_max_x.setValue(max_x)


def MonoFontForSpecificOS():
    name = platform.system()
    if name == 'Darwin':
        return "Andale Mono"
    else:
        return "Monospace"


if __name__ == "__main__":
    import pyqtgraph.examples
    pyqtgraph.examples.run()
