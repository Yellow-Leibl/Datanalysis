from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QAbstractItemView
from PyQt6 import QtGui
from Datanalysis import SamplingDatas, SamplingData
import numpy as np


class TableWidget(QTableWidget):
    def __init__(self, parent=None, cell_double_clicked=None):
        super().__init__(parent)
        self.__info_cells_count = 2
        self.selected_indexes = None

        self.__selected_row_color = QtGui.QColor(30, 150, 0)
        self.__max_min_color = QtGui.QColor(215, 0, 97)

        if cell_double_clicked is not None:
            self.cellDoubleClicked.connect(cell_double_clicked)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

    def set_datas(self, datas: SamplingDatas):
        self.datas = datas
        self.update_table()

    def get_val_item(self, row, col):
        return self.item(row, col + self.__info_cells_count)

    def create_cell(self, text):
        item = QTableWidgetItem(text)
        return item

    def update_table(self):
        self.clear()
        self.setColumnCount(self.datas.get_max_len_raw()
                            + self.__info_cells_count)
        self.setRowCount(len(self.datas))
        for s in range(len(self.datas)):
            d = self.datas[s]
            description = self.get_description(d)
            self.setItem(s, 0, self.create_cell(description))
            self.setItem(s, 1, self.create_cell(
                self.format_sample_description(d)))
            for i, val in enumerate(d.raw):
                self.setItem(s, i + self.__info_cells_count,
                             self.create_cell(self.format_cell_value(val)))
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.colorize_selections(self.selected_indexes)
        self.colorize_max_min_value()

    def get_description(self, d: SamplingData):
        return f"{d.name}: {d.discrt_res}"

    def format_sample_description(self, d: SamplingData):
        if np.issubdtype(type(d.min), np.integer):
            return f"N={len(d.raw)}, [{d.min}; {d.max}]"
        else:
            return f"N={len(d.raw)}, [{d.min:.5}; {d.max:.5}]"

    def format_cell_value(self, val):
        if np.issubdtype(type(val), np.integer):
            return f"{val}"
        else:
            return f"{val:.5}"

    def select_rows(self, indexes: list[int]):
        self.clearSelection()
        self.selected_indexes = indexes
        self.colorize_selections(indexes)
        self.colorize_max_min_value()

    def colorize_selections(self, sel_indexes: list[int]):
        if sel_indexes is None:
            return
        for i in range(self.rowCount()):
            color = QtGui.QColor()
            if i in sel_indexes:
                color = self.__selected_row_color
            for j in range(len(self.datas[i].raw) + self.__info_cells_count):
                self.item(i, j).setBackground(color)

    def colorize_max_min_value(self):
        for i in range(self.rowCount()):
            d = self.datas[i]
            for j in range(len(self.datas[i].raw)):
                if d.raw[j] == d.max or d.raw[j] == d.min:
                    self.get_val_item(i, j).setBackground(self.__max_min_color)

    def get_observations_to_remove(self):
        items = self.get_active_items()
        obsers_to_remove = [[] for _ in range(len(self.datas))]
        for item in items:
            if item[1] >= self.__info_cells_count:
                obser_col = item[1] - self.__info_cells_count
                obsers_to_remove[item[0]].append(obser_col)
        return obsers_to_remove

    def get_active_items(self):
        return [(index.row(), index.column())
                for index in self.selectedIndexes()]

    def get_active_rows(self) -> list:
        ranges = self.selectedRanges()
        sel_rows = []
        for r in ranges:
            sel_rows += [*range(r.topRow(), r.bottomRow() + 1)]
        return sel_rows
