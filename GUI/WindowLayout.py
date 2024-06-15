if __name__ == "__main__":
    import pyqtgraph.examples
    pyqtgraph.examples.run()

import os
import logging

from PyQt6.QtWidgets import (QMessageBox, QSplitter,
                             QFileDialog)
from PyQt6.QtCore import Qt
from PyQt6 import QtWidgets
from GUI import (TableWidget, fill_menu_bar,
                 TextEdit, TabWidget,
                 FeatureArea, BoxWithObjects)
import GUI.PlotWidget as cplt

logging.basicConfig(level=logging.INFO)


class WindowLayout(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(150, 100, 1333, 733)
        self.setWindowTitle("Аналіз даних")

        self.plot_widget = cplt.PlotWidget()
        self.additional_plot_widget = None

        self.table = TableWidget(cell_double_clicked=self.draw_samples)

        self.protocol = TextEdit(read_only=True, mono_font=True, nowrap=True)

        self.criterion_protocol = TextEdit(read_only=True, mono_font=True,
                                           nowrap=True)

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
        filenames = self.open_file_dialog()
        if len(filenames) == 0:
            return
        self.open_files(filenames)

    def open_file_dialog(self) -> str:
        file_name, _ = QFileDialog().getOpenFileNames(
            self, "Відкрити файл", os.getcwd(), "Bci файли (*)")
        return file_name

    def open_clusters_files_act(self):
        filenames = self.open_file_dialog()
        if len(filenames) == 0:
            return
        self.open_clusters_files(filenames)

    def save_file_act(self):
        filename = self.save_file_dialog()
        if filename == '':
            return
        self.save_file(filename)

    def save_file_as_obj_act(self):
        filename = self.save_file_dialog()
        if filename == '':
            return
        self.save_file_as_obj(filename)

    def save_file_dialog(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Зберегти файл", os.getcwd(), "Bci файли (*)")
        return file_name
