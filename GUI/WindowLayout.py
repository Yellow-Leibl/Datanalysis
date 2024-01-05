if __name__ == "__main__":
    import pyqtgraph.examples
    pyqtgraph.examples.run()

import os
import logging

from PyQt6.QtWidgets import (QMessageBox, QSplitter,
                             QFileDialog)
from PyQt6.QtCore import Qt
from PyQt6 import QtGui, QtWidgets
from GUI import (PlotWidget, TableWidget, fill_menu_bar,
                 TextEdit, TabWidget,
                 FeatureArea, BoxWithObjects)

logging.basicConfig(level=logging.INFO)


class WindowLayout(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(150, 100, 1333, 733)
        self.setWindowTitle("Аналіз даних")

        self.plot_widget = PlotWidget()

        self.table = TableWidget(cell_double_clicked=self.draw_samples)

        self.protocol = TextEdit(read_only=True, mono_font=True)
        self.protocol.setWordWrapMode(QtGui.QTextOption.WrapMode.NoWrap)

        self.criterion_protocol = TextEdit(read_only=True, mono_font=True)
        self.criterion_protocol.setWordWrapMode(
            QtGui.QTextOption.WrapMode.NoWrap)

        self.tab_info = TabWidget(self.protocol, "Протокол",
                                  self.criterion_protocol, "Критерії",
                                  self.table, "Ранжований ряд")

        self.feature_area = FeatureArea(self)
        info_wid = BoxWithObjects(QSplitter(Qt.Orientation.Horizontal),
                                  self.tab_info,
                                  self.feature_area)
        info_wid.setStretchFactor(0, 3)
        info_wid.setStretchFactor(1, 1)

        main_vbox = BoxWithObjects(QSplitter(Qt.Orientation.Vertical),
                                   self.plot_widget,
                                   info_wid)

        self.setCentralWidget(main_vbox)
        fill_menu_bar(self)

    def nextProtocolTab(self):
        curr_ind = self.tab_info.currentIndex()
        total_tabs = len(self.tab_info.tabBar())
        self.tab_info.setCurrentIndex((curr_ind + 1) % total_tabs)

    def showMessageBox(self, title: str, text: str):
        if title == "" and text == "":
            return
        mes = QMessageBox(self)
        mes.setText(title)
        font = mes.font()
        font.setPointSize(16)
        mes.setFont(font)
        mes.setInformativeText(text)
        mes.setBaseSize(400, 300)
        mes.exec()

    def open_file_act(self):
        filename = self.open_file_dialog()
        self.open_file(filename)

    def open_file_dialog(self) -> str:
        file_name, _ = QFileDialog().getOpenFileName(
            self, "Відкрити файл", os.getcwd(), "Bci файли (*)")
        return file_name

    def save_file_act(self):
        file_name, _ = QFileDialog().getSaveFileName(
            self, "Зберегти файл", os.getcwd(), "Bci файли (*)")
        with open(file_name, 'w') as file:
            def safe_access(lst: list, i):
                return str(lst[i]) if len(lst) > i else ''
            file.write('\n'.join(
                [' '.join([safe_access(self.all_datas[j].raw, i)
                           for j in range(len(self.all_datas))])
                 for i in range(self.all_datas.get_max_len_raw())]))
