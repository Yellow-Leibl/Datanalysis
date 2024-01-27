from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QAbstractItemView
from PyQt6 import QtGui
from Datanalysis import SamplingDatas, SamplingData
import numpy as np
from Datanalysis.SamplesTools import timer


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

    def set_text(self, row, col, text):
        item = QTableWidgetItem(text)
        self.setItem(row, col, item)

    @timer
    def update_table(self):
        self.clear()
        self.setColumnCount(self.datas.get_max_len_raw()
                            + self.__info_cells_count)
        self.setRowCount(len(self.datas))
        for i in range(len(self.datas)):
            d = self.datas[i]
            self.fill_row(i, d)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.colorize_selections(self.selected_indexes)
        self.colorize_max_min_value()

    def select_rows(self, indexes: list[int]):
        self.clearSelection()
        self.selected_indexes = indexes
        self.colorize_selections(indexes)
        self.colorize_max_min_value()

    @timer
    def fill_row(self, i, d: SamplingData):
        self.set_text(i, 0, self.get_description(d))
        self.set_text(i, 1, self.format_sample_description(d))
        if self.is_int(d.raw[0]) or d.ticks is not None:
            self.fill_unformat_row(i, d)
        else:
            self.fill_double_row(i, d)

    def is_int(self, val):
        return np.issubdtype(type(val), np.integer)

    def fill_double_row(self, i, d: SamplingData):
        for j, val in enumerate(d.raw):
            self.set_text(i, j + self.__info_cells_count, f"{val:.5}")

    def fill_unformat_row(self, i, d: SamplingData):
        for j, val in enumerate(d.raw):
            if d.ticks is not None:
                val = d.ticks[int(val)]
            self.set_text(i, j + self.__info_cells_count, f"{val}")

    def get_description(self, d: SamplingData):
        if d.ticks is None:
            return d.name
        if len(d.ticks) > 40:
            ticks_str = f"{str(np.array(d.ticks[:25]))[:-1]} ..."
        else:
            ticks_str = f"{np.array(d.ticks)}"
        return f"{d.name}: {ticks_str}"

    def format_sample_description(self, d: SamplingData):
        if np.issubdtype(type(d.min), np.integer):
            return f"N={len(d.raw)}, [{d.min}; {d.max}]"
        else:
            return f"N={len(d.raw)}, [{d.min:.5}; {d.max:.5}]"

    def colorize_selections(self, sel_indexes: list[int]):
        if sel_indexes is None:
            return
        for i in range(self.rowCount()):
            color = QtGui.QColor()
            if i in sel_indexes:
                color = self.__selected_row_color
            self.item(i, 0).setBackground(color)
            self.item(i, 1).setBackground(color)

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
