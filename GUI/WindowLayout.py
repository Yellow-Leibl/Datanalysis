from PyQt6.QtWidgets import (
    QMainWindow, QWidget,
    QSpinBox, QDoubleSpinBox, QPushButton,
    QTextEdit, QTabWidget,
    QHBoxLayout, QVBoxLayout, QFormLayout, QBoxLayout,
    QLabel, QMessageBox, QMenu, QSplitter)
from PyQt6.QtGui import QKeySequence
from PyQt6.QtCore import Qt
from PyQt6 import QtGui
from GUI.PlotWidget import PlotWidget
from GUI.TableWidget import TableWidget


def keySequence(sequence: str):
    return QKeySequence(sequence)


def justKey(key: Qt.Key):
    return key


def BoxWithObjects(box, *args):
    addObjects(box, *args)
    return box


def addObjects(box: QBoxLayout, *args):
    for arg in args:
        typ = str(type(arg))
        if 'Layout' in typ:
            box.addLayout(arg)
        elif 'Widget' in typ:
            box.addWidget(arg)
        else:
            raise Exception()


class WindowLayout(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setGeometry(150, 100, 1333, 733)
        self.setWindowTitle("Аналіз даних")

        # 2 chart box
        self.plot_widget = PlotWidget()
        self.plot_widget.create2DPlot()

        # spin boxes
        def spinBox(box, val_changed_f=None,
                    min=0, max=1, decimals=None, val=0):
            box.setMinimum(min)
            box.setMaximum(max)
            if decimals is not None:
                box.setDecimals(decimals)
            box.setValue(val)
            if val_changed_f is not None:
                box.valueChanged.connect(val_changed_f)
            return box

        self.__spin_number_column = spinBox(
            QSpinBox(), self.numberColumnChanged)

        self.__trust_value = spinBox(
            QDoubleSpinBox(),
            lambda: self.changeTrust(self.__trust_value.value()),
            min=0.0, max=1.0, decimals=5, val=0.05)

        self.pCA_number = spinBox(QSpinBox(), min=2, max=99)

        self.__spin_box_min_x = spinBox(QDoubleSpinBox(), decimals=5)
        self.__spin_box_max_x = spinBox(QDoubleSpinBox(), decimals=5)
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
        self.createMenuBar()

    def createMenuBar(self):
        def addAction(menu: QMenu, title, action, shortcut):
            act = menu.addAction(title, action)
            if act is not None:
                act.setShortcut(keySequence(shortcut))
        # File menu
        menuBar = self.menuBar()
        file_menu = menuBar.addMenu("Файл")
        addAction(file_menu, "Відкрити", lambda: self.openFile(''),
                  "Ctrl+O")
        addAction(file_menu, "Зберегти", self.saveFileAct,
                  "Ctrl+S")
        addAction(file_menu, "Вийти", self.saveFileAct,
                  "Ctrl+Q")

        # Editing menu
        edit_menu = menuBar.addMenu("Редагувати")
        addAction(edit_menu, "Перетворити", self.editSampleEvent, "Ctrl+T")
        addAction(edit_menu, "Стандартизувати", self.editSampleEvent,
                  "Ctrl+W")
        addAction(edit_menu, "Центрувати", self.editSampleEvent,
                  "Ctrl+Shift+W")
        addAction(edit_menu, "Зсунути", self.editSampleEvent, "Ctrl+P")
        addAction(edit_menu, "Видалити аномалії", self.editSampleEvent,
                  "Ctrl+A")
        addAction(edit_menu, "До незалежних величин", self.editSampleEvent,
                  "Ctrl+Y")
        edit_menu.addSeparator()
        addAction(edit_menu, "Клонувати", self.duplicateSample, "Ctrl+C")
        addAction(edit_menu, "Зобразити розподіл", self.drawSamples, "Ctrl+D")
        addAction(edit_menu, "Видалити спостереження",
                  self.delete_observations, "Ctrl+Backspace")

        view_menu = menuBar.addMenu("Вигляд")
        addAction(view_menu, "Наступна вкладка", self.nextProtocolTab,
                  "Alt+Tab")

        self.reprod_num = -1

        # Regression menu
        regr_menu = menuBar.addMenu("&Регресія")
        addAction(regr_menu, "&Нормальний", self.setReproductionSeries,
                  "Ctrl+Alt+N")
        addAction(regr_menu, "&Рівномірний", self.setReproductionSeries,
                  "Ctrl+Alt+U")
        addAction(regr_menu, "&Експоненціальний", self.setReproductionSeries,
                  "Ctrl+Alt+E")
        addAction(regr_menu, "&Вейбулла", self.setReproductionSeries,
                  "Ctrl+Alt+W")
        addAction(regr_menu, "&Арксинус", self.setReproductionSeries,
                  "Ctrl+Alt+A")
        regr_menu.addSeparator()
        addAction(regr_menu, "Лінійна &МНК", self.setReproductionSeries,
                  "Ctrl+Alt+M")
        addAction(regr_menu, "Лінійна Метод &Тейла",
                  self.setReproductionSeries, "Ctrl+Alt+T")
        addAction(regr_menu, "&Парабола", self.setReproductionSeries,
                  "Ctrl+Alt+P")
        addAction(regr_menu, "Квазілінійна y = a * exp(b * x)",
                  self.setReproductionSeries, "Ctrl+Alt+K")
        regr_menu.addSeparator()
        addAction(regr_menu, "Лінійна МНК", self.setReproductionSeries,
                  "Ctrl+Alt+V")
        addAction(regr_menu, "Лінійне різноманіття",
                  self.setReproductionSeries, "Ctrl+Alt+R")
        regr_menu.addSeparator()
        addAction(regr_menu, "&Очистити", self.setReproductionSeries,
                  "Ctrl+Alt+C")

        # Critetion menu
        crit_menu = menuBar.addMenu("&Критерії")
        def setTrust(func): return lambda: func(self.__trust_value.value())
        addAction(crit_menu, "Перевірка однорідності/незалежності",
                  setTrust(self.homogeneityAndIndependence), "Ctrl+Shift+I")
        addAction(crit_menu, "Порівняння лінійних регресій",
                  setTrust(self.linearModelsCrit), "Ctrl+Shift+E")
        addAction(crit_menu, "Перевірка однорідності сукупностей (Нормальні)",
                  self.homogeneityNSamples, "Ctrl+Shift+N")
        addAction(crit_menu, "Частковий коефіцієнт кореляції",
                  self.partialCorrelation, "Ctrl+Shift+P")

        anal_menu = menuBar.addMenu("К&омпонентний та факторний аналіз")
        addAction(anal_menu, "Метод головних компонент",
                  self.PCA, "Ctrl+Shift+M")

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


def MonoFontForSpecificOS():
    import platform
    name = platform.system()
    if name == 'Darwin':
        return "Andale Mono"
    else:
        return "Monospace"


#  Examples pyqtgraph
if __name__ == "__main__":
    import pyqtgraph.examples
    pyqtgraph.examples.run()
# matrix display
