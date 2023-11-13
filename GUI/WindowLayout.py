from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QPushButton,
    QTextEdit, QTabWidget,
    QHBoxLayout, QVBoxLayout, QFormLayout,
    QLabel, QMessageBox, QSplitter)
from PyQt6.QtCore import Qt
from PyQt6 import QtGui
from GUI.PlotWidget import PlotWidget
from GUI.TableWidget import TableWidget
from GUI.MenuBar import fill_menu_bar
from GUI.ui_tools import (
    SpinBox, DoubleSpinBox, BoxWithObjects, MonoFontForSpecificOS)


class WindowLayout(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setGeometry(150, 100, 1333, 733)
        self.setWindowTitle("Аналіз даних")

        # 2 chart box
        self.plot_widget = PlotWidget()
        self.plot_widget.create2DPlot()

        # spin boxes
        self.__spin_number_column = SpinBox(
            val_changed_f=self.numberColumnChanged)

        self.__trust_value = DoubleSpinBox(
            lambda: self.changeTrust(self.__trust_value.value()),
            0.0, 1.0, 5, 0.05)

        self.pCA_number = SpinBox(min=2, max=99)

        self.__spin_box_min_x = DoubleSpinBox(decimals=5)
        self.__spin_box_max_x = DoubleSpinBox(decimals=5)
        self.__remove_anomaly = QPushButton("Видалити аномалії")
        self.__remove_anomaly.clicked.connect(self.removeAnomaly)

        # Samples table
        self.table = TableWidget(
            cell_double_clicked=lambda: self.drawSamples())

        # Protocol
        self.protocol = QTextEdit()
        self.protocol.setWordWrapMode(QtGui.QTextOption.WrapMode.NoWrap)
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
        form_widget = QFormLayout()
        form_widget.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form_widget.addRow("Кількість класів:",
                           self.__spin_number_column)
        form_widget.addRow("Рівень значущості:",
                           self.__trust_value)
        form_widget.addRow("", QWidget())
        form_widget.addRow("Кількість перших компонентів для МГК:",
                           self.pCA_number)

        # borders
        borders = BoxWithObjects(
            QHBoxLayout(),
            QLabel("min"),
            self.__spin_box_min_x,
            QLabel("max"),
            self.__spin_box_max_x)

        form_func = BoxWithObjects(
            QVBoxLayout(),
            form_widget,
            borders,
            self.__remove_anomaly)

        # tab and add. functionality
        widget_func = QWidget()
        widget_func.setLayout(form_func)
        info_wid = BoxWithObjects(QSplitter(Qt.Orientation.Horizontal),
                                  self.tab_info,
                                  widget_func)

        main_vbox = BoxWithObjects(QSplitter(Qt.Orientation.Vertical),
                                   self.plot_widget,
                                   info_wid)

        self.setCentralWidget(main_vbox)
        fill_menu_bar(self)

    def get_edit_menu(self):
        return self.menuBar().actions()[1].menu()

    def get_regr_menu(self):
        return self.menuBar().actions()[3].menu()

    def index_in_menu(self, menu, act):
        return menu.actions().index(act)

    def getMinMax(self):
        return (self.__spin_box_min_x.value(),
                self.__spin_box_max_x.value())

    def nextProtocolTab(self):
        curr_ind = self.tab_info.currentIndex()
        total_tabs = len(self.tab_info.tabBar())
        self.tab_info.setCurrentIndex((curr_ind + 1) % total_tabs)

    def showMessageBox(self, title: str, text: str):
        mes = QMessageBox(self)
        mes.setText(title)
        font = mes.font()
        font.setPointSize(16)
        mes.setFont(font)
        mes.setInformativeText(text)
        mes.setBaseSize(400, 300)
        mes.exec()

    def getNumberClasses(self) -> int:
        return self.__spin_number_column.value()

    def getTrust(self):
        return self.__trust_value.value()

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


#  Examples pyqtgraph
if __name__ == "__main__":
    import pyqtgraph.examples
    pyqtgraph.examples.run()
# matrix display
