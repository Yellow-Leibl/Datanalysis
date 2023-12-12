import sys
import logging
import numpy as np

from GUI.WindowLayout import WindowLayout, QApplication

from Sessions import (
    SessionMode, SessionMode1D,
    SessionMode2D, SessionModeND, SessionModeTimeSeries)

from Datanalysis import SamplingDatas, ReaderDatas

logging.basicConfig(level=logging.INFO)


class Window(WindowLayout):
    def __init__(self,
                 file: str,
                 is_file_name: bool = True,
                 auto_select: list = None):
        super().__init__()
        self.session: SessionMode = None
        self.all_datas = SamplingDatas()
        self.table.set_datas(self.all_datas)
        self.reader = ReaderDatas()
        self.sel_indexes: list[int] = []
        self.datas_crits: list[list] = []
        self.open_file(file, is_file_name)
        if auto_select is not None:
            self.auto_select(auto_select)

    def open_file(self, file, is_file_name: bool = True):
        if is_file_name:
            all_vectors = self.reader.read_from_file(file)
        else:
            all_vectors = self.reader.read_from_text(file.split('\n'))
        self.loadFromData(all_vectors)

    def loadFromData(self, vectors):
        self.all_datas.append(vectors)
        self.table.update_table()

    def active_sample_changed(self, selected_new_samples: bool = False):
        self.session.update_graphics(self.feature_area.get_number_classes())
        self.session.write_protocol()
        self.session.write_critetion()
        if selected_new_samples:
            self.table.select_rows(self.sel_indexes)
        else:
            self.table.update_table()

    def change_sample_type_mode(self):
        if type(self.session) is SessionMode1D:
            self.session = SessionModeTimeSeries(self)
        elif type(self.session) is SessionModeTimeSeries:
            self.session = SessionMode1D(self)
        else:
            return
        self.configure_session()

    def editSampleEvent(self):
        edit_num = self.index_in_menu(self.get_edit_menu(), self.sender())
        if edit_num == 0:
            [self.all_datas[i].toLogarithmus10() for i in self.sel_indexes]
        elif edit_num == 1:
            [self.all_datas[i].toStandardization() for i in self.sel_indexes]
        elif edit_num == 2:
            [self.all_datas[i].toCentralization() for i in self.sel_indexes]
        elif edit_num == 3:
            [self.all_datas[i].toSlide(1.0) for i in self.sel_indexes]
        elif edit_num == 4:
            self.session.auto_remove_anomalys()
        elif edit_num == 5:
            self.session.to_independent()
        self.active_sample_changed()

    def duplicateSample(self):
        sel = self.table.get_active_rows()
        for i in sel:
            self.all_datas.append_sample(self.all_datas[i].copy())
        self.table.update_table()

    def removeAnomaly(self):
        if self.is1d():
            minmax = self.feature_area.get_borders()
            self.all_datas[self.sel_indexes[0]].remove(minmax[0], minmax[1])
            self.active_sample_changed()

    def draw_samples(self):
        sel = self.table.get_active_rows()
        if len(sel) == 0 or sel == self.sel_indexes:
            return
        self.sel_indexes = sel
        self.init_session()
        self.configure_session()

    def remove_trend(self):
        if type(self.session) is SessionModeTimeSeries:
            self.session.remove_trend()
            self.active_sample_changed()

    def init_session(self):
        if self.is1d():
            if type(self.session) is not SessionModeTimeSeries:
                self.session = SessionMode1D(self)
        elif self.is2d():
            self.session = SessionMode2D(self)
        elif self.isNd():
            self.session = SessionModeND(self)

    def configure_session(self):
        self.session.create_plot_layout()
        self.feature_area.silent_change_number_classes(0)
        self.feature_area.set_maximum_column_number(
            self.all_datas.getMaxDepthRangeData())
        self.active_sample_changed(selected_new_samples=True)
        self.table.select_rows(self.sel_indexes)

    def is1d(self) -> bool:
        return len(self.sel_indexes) == 1

    def is2d(self) -> bool:
        return len(self.sel_indexes) == 2

    def is3d(self) -> bool:
        return len(self.sel_indexes) == 3

    def isNd(self) -> bool:
        return len(self.sel_indexes) >= 2

    def delete_observations(self):
        all_obsers = self.table.get_observations_to_remove()
        sample_changed = False
        update_table = sum([len(obsers) for obsers in all_obsers]) > 0
        for i, obsers in list(enumerate(all_obsers))[::-1]:
            if len(obsers) == 0:
                continue
            if len(obsers) >= len(self.all_datas[i].raw):
                self.all_datas.pop(i)
                continue
            self.all_datas[i].remove_observations(obsers)
            if i in self.sel_indexes:
                sample_changed = True
        if sample_changed:
            self.active_sample_changed()
        elif update_table:
            self.table.update_table()

    def setReproductionSeries(self):
        regr_num = self.index_in_menu(self.get_regr_menu(), self.sender())
        self.session.set_regression_number(regr_num)
        self.active_sample_changed(selected_new_samples=True)

    def smooth_series(self):
        if type(self.session) is SessionModeTimeSeries:
            smth_num = self.index_in_menu(self.get_smth_menu(), self.sender())
            self.session.smooth_time_series(smth_num)
            self.active_sample_changed(selected_new_samples=True)

    def change_trust(self):
        self.active_sample_changed()

    def numberColumnChanged(self):
        self.active_sample_changed(selected_new_samples=True)

    def linearModelsCrit(self, trust: float):
        sel = self.table.get_active_rows()
        if len(sel) == 4:
            res = self.all_datas.ident2ModelsLine(
                [self.all_datas[i] for i in sel], trust)
            title = "Лінійна регресія"
            descr = "Моделі регресійних прямих\n" + \
                f"({sel[0]}, {sel[1]}) і ({sel[2]}, {sel[3]})"
            if res is None:
                self.showMessageBox(title, descr +
                                    " - мають випадкову різницю регресій")
            else:
                res_str = "- ідентичні" if res else "- неідентичні"
                self.showMessageBox(title, descr + res_str)

    def homogeneityAndIndependence(self, trust: float):
        sel = self.table.get_active_rows()
        if len(sel) == 1:
            self.critetion_abbe(sel, trust)
        elif len(sel) == 2:
            self.are_independent_2_samples(sel, trust)
        elif len(sel) > 2:
            self.are_independent_k_samples(sel, trust)

    def critetion_abbe(self, sel, trust: float):
        P = self.all_datas[sel[0]].critetionAbbe()
        if P > trust:
            descr = f"{P:.5} > {trust}\nСпостереження незалежні"
        else:
            descr = f"{P:.5} < {trust}\nСпостереження залежні"
        self.showMessageBox("Критерій Аббе", descr)

    def are_independent_2_samples(self, sel, trust: float):
        if self.all_datas.ident2Samples(sel[0], sel[1], trust):
            title = "Вибірки однорідні"
        else:
            title = "Вибірки неоднорідні"
        self.showMessageBox(title, "")

    def are_independent_k_samples(self, sel, trust: float):
        if self.all_datas.identKSamples([self.all_datas[i] for i in sel],
                                        trust):
            title = "Вибірки однорідні"
        else:
            title = "Вибірки неоднорідні"
        self.showMessageBox(title, "")

    def homogeneityNSamples(self):
        sel = self.table.get_active_rows()
        self.table.clearSelection()
        if len(sel) == 0:
            self.confirm_homogeneity()
        else:
            self.add_data_to_homogeneity(sel)

    def confirm_homogeneity(self):
        if len(self.datas_crits) < 2:
            return
        text = self.all_datas.homogeneityProtocol(
            [[self.all_datas[j] for j in i] for i in self.datas_crits])
        self.showMessageBox("Перевірка однорідності сукупностей", text)
        self.datas_crits = []

    def add_data_to_homogeneity(self, sel):
        if sel in self.datas_crits:
            self.showMessageBox("Помилка", "Розподіл вже вибраний")
            return
        n = len(self.datas_crits[0])
        if len(self.datas_crits) != 0 and n != len(sel):
            self.showMessageBox("Помилка", f"Потрібен {n}-вимірний розподіл")
            return

        failed_test = self.get_failed_norm_test_indexes(sel)
        if len(failed_test) != 0:
            self.showMessageBox("Помилка",
                                f"Не є нормальним розподілом: {failed_test+1}")
            return

        self.datas_crits.append(sel)
        self.showMessageBox(
            "Перевірка однорідності сукупностей",
            f"Вибрані вибірки:\n{np.array(self.datas_crits)+1}")

    def get_failed_norm_test_indexes(self, sel):
        norm_test = [self.all_datas[i].isNormal() for i in sel]
        return np.array([i for i, res in enumerate(norm_test) if not res])

    def partialCorrelation(self):
        sel = self.table.get_active_rows()
        w = len(sel)
        if w > 2:
            datas = SamplingDatas([self.all_datas.samples[i] for i in sel])
            datas.toCalculateCharacteristic()
            text = datas.partialCoeficientOfCorrelationProtocol(
                sel[0], sel[1], sel[2:])
            self.criterion_protocol.setText(text)

    def PCA(self):
        if type(self.session) is SessionModeND:
            self.session.pca(self.pCA_number.value())

    def auto_select(self, sel_index):
        self.sel_indexes = sel_index
        # self.init_session()
        self.session = SessionModeTimeSeries(self)
        self.configure_session()


def applicationLoadFromFile(file: str = ''):
    launch_app(file, is_file_name=True)


def applicationLoadFromStr(file: str = ''):
    launch_app(file, is_file_name=False)


def demo_mode_show():
    file = "data/self/norm18n.txt"
    launch_app(file, is_file_name=True, auto_select=range(18))


# def demo_mode_time_series_show():
#     file = "data/self/3_normal_2700.txt"
#     launch_app(file, is_file_name=True, auto_select=range(3))


def demo_mode_time_series_show():
    file = "data/self/time_series_500_3n.txt"
    launch_app(file, is_file_name=True, auto_select=range(1))


def demo_mode_course_work():
    file = "data/self/Life Expectancy Data.csv"
    launch_app(file, is_file_name=True, auto_select=range(1))


def launch_app(file, is_file_name: bool, auto_select=None):
    app = QApplication(sys.argv)
    widget = Window(file, is_file_name=is_file_name, auto_select=auto_select)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    demo_mode_time_series_show()
