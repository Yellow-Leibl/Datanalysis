import sys
import logging
import numpy as np

from GUI.WindowLayout import WindowLayout, QtWidgets

from Sessions import (
    SessionMode, SessionMode1D,
    SessionMode2D, SessionModeND, SessionModeTimeSeries)

from Datanalysis import SamplingDatas, ReaderDatas

logging.basicConfig(level=logging.INFO)


class Window(WindowLayout):
    def __init__(self,
                 file: str,
                 is_file_name=True,
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

    def open_file(self, file: str, is_file_name=True):
        if file == '':
            return
        if is_file_name:
            all_vectors = self.reader.read_from_file(file)
        else:
            all_vectors = self.reader.read_from_text(file.split('\n'))
        self.load_from_data(all_vectors)

    def load_from_data(self, vectors):
        self.all_datas.append(vectors)
        self.table.update_table()

    def select_new_sample(self):
        self.session.select_new_sample()
        self.table.select_rows(self.sel_indexes)

    def update_sample(self):
        self.session.select_new_sample()
        self.table.update_table()

    def edit_sample_event(self, edit_num):
        if edit_num == 0:
            [self.all_datas[i].toLogarithmus10() for i in self.sel_indexes]
        elif edit_num == 1:
            [self.all_datas[i].to_standardization() for i in self.sel_indexes]
        elif edit_num == 2:
            [self.all_datas[i].toCentralization() for i in self.sel_indexes]
        elif edit_num == 3:
            [self.all_datas[i].toSlide(1.0) for i in self.sel_indexes]
        elif edit_num == 4:
            self.session.auto_remove_anomalys()
        elif edit_num == 5:
            self.session.to_independent()
        self.update_sample()

    def duplicate_sample(self):
        sel = self.table.get_active_rows()
        for i in sel:
            self.all_datas.append_sample(self.all_datas[i].copy())
        self.table.update_table()

    def remove_anomaly_with_range(self):
        if type(self.session) is SessionMode1D:
            self.session.remove_anomaly_with_range()

    def draw_samples(self):
        sel = self.table.get_active_rows()
        if len(sel) == 0 or sel == self.sel_indexes:
            return
        self.sel_indexes = sel
        self.make_session()
        self.configure_session()

    def make_session(self):
        if len(self.sel_indexes) == 1:
            if type(self.session) is not SessionModeTimeSeries:
                self.session = SessionMode1D(self)
        elif len(self.sel_indexes) == 2:
            self.session = SessionMode2D(self)
        elif len(self.sel_indexes) >= 2:
            self.session = SessionModeND(self)

    def change_sample_type_mode(self):
        if type(self.session) is SessionMode1D:
            self.session = SessionModeTimeSeries(self)
        elif type(self.session) is SessionModeTimeSeries:
            self.session = SessionMode1D(self)
        else:
            return
        self.configure_session()

    def remove_trend(self):
        if type(self.session) is SessionModeTimeSeries:
            self.session.remove_trend()
            self.update_sample()

    def configure_session(self):
        self.session.configure()
        self.select_new_sample()

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
            self.update_sample()
        elif update_table:
            self.table.update_table()

    def clear_plot(self):
        self.set_reproduction_series(-1)
        self.smooth_series(-1)

    def set_reproduction_series(self, regr_num):
        self.session.set_regression_number(regr_num)

    def smooth_series(self, smth_num):
        if type(self.session) is SessionModeTimeSeries:
            self.session.smooth_time_series(smth_num)
            self.session.update_sample()

    def ssa(self, ssa_num):
        if type(self.session) is SessionModeTimeSeries:
            if ssa_num == 0:
                self.session.ssa_visualize_components()
            elif ssa_num == 1:
                self.session.ssa_reconstruction()
            elif ssa_num == 2:
                self.session.test_forecast_ssa()
            elif ssa_num == 3:
                self.session.forecast_ssa()
            self.session.update_sample()

    def linear_models_crit(self, trust: float):
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

    def homogeneity_and_independence(self, trust: float):
        sel = self.table.get_active_rows()
        if len(sel) == 1:
            self.critetion_abbe(sel, trust)
        elif len(sel) == 2:
            self.are_independent_2_samples(sel, trust)
        elif len(sel) > 2:
            self.are_independent_k_samples(sel, trust)

    def critetion_abbe(self, sel, trust: float):
        P = self.all_datas[sel[0]].critetion_abbe()
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

    def homogeneity_n_samples(self):
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
        norm_test = [self.all_datas[i].is_normal() for i in sel]
        return np.array([i for i, res in enumerate(norm_test) if not res])

    def partial_correlation(self):
        sel = self.table.get_active_rows()
        w = len(sel)
        if w > 2:
            datas = SamplingDatas([self.all_datas.samples[i] for i in sel])
            datas.toCalculateCharacteristic()
            text = datas.partialCoeficientOfCorrelationProtocol(
                sel[0], sel[1], sel[2:])
            self.criterion_protocol.setText(text)

    def pca(self):
        if type(self.session) is SessionModeND:
            self.session.pca()

    def auto_select(self, sel_index):
        self.sel_indexes = sel_index
        self.make_session()
        self.configure_session()


def applicationLoadFromFile(file: str = ''):
    launch_app(file, is_file_name=True)


def applicationLoadFromStr(file: str = '', auto_select=None):
    launch_app(file, is_file_name=False, auto_select=auto_select)


def demo_mode_show():
    file = "data/self/norm18n.txt"
    launch_app(file, is_file_name=True, auto_select=range(18))


def demo_mode_time_series_show():
    file = "data/self/3_normal_2700.txt"
    launch_app(file, is_file_name=True, auto_select=range(3))


def demo_mode_time_series_show_2():
    file = "data/self/kaggle_example_200.txt"
    launch_app(file, is_file_name=True, auto_select=range(1))


def demo_mode_time_series_show_1():
    file = "data/self/time_series_500_3n.txt"
    launch_app(file, is_file_name=True, auto_select=range(1))


def demo_mode_course_work():
    # file = "data/course/student-mat.csv"
    file = "data/course/CO2 Emissions_Canada.csv"
    launch_app(file, is_file_name=True, auto_select=range(12))


def launch_app(file, is_file_name: bool, auto_select=None):
    app = QtWidgets.QApplication(sys.argv)
    widget = Window(file, is_file_name=is_file_name, auto_select=auto_select)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # demo_mode_time_series_show_2()
    demo_mode_course_work()
