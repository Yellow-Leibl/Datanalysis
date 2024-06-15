import sys
import logging
import numpy as np

from GUI.WindowLayout import WindowLayout, QtWidgets
from GUI import DialogWindow, DoubleSpinBox, ComboBox

import Sessions as ses

from Datanalysis import SamplingDatas, IODatas, SamplingData
from Datanalysis.SamplesTools import timer

logging.basicConfig(level=logging.INFO)


class Window(WindowLayout):
    def __init__(self,
                 file: str,
                 is_file_name=True,
                 auto_select: list = None):
        super().__init__()
        self.session: ses.SessionMode = None
        self.all_datas = SamplingDatas()
        self.table.set_datas(self.all_datas)
        self.io_datas = IODatas()
        self.sel_indexes: list[int] = []
        self.datas_crits: list[list] = []
        self.open_files([file], is_file_name)
        if auto_select is not None:
            self.auto_select(auto_select)

    def open_files(self, files: list[str] | str, is_file_names=True):
        if is_file_names:
            samples = []
            for file in files:
                samples += self.io_datas.read_from_file(file)
        else:
            samples = self.io_datas.read_from_text(files.split('\n'))
        self.load_from_data(samples)

    def open_clusters_files(self, files: list[str]):
        samples = []
        for file in files:
            samples.append(self.io_datas.read_from_file(file))

        n = len(samples[0])
        n_arr = [len(sample[0].raw) for sample in samples]
        clusters = []
        prev_n = 0
        for ni in n_arr:
            clusters.append(np.arange(prev_n, prev_n + ni))
            prev_n += ni

        merged_samples = []
        for i in range(n):
            samples_i_raw = [sample[i].raw for sample in samples]
            raw = np.concatenate(samples_i_raw)
            s = SamplingData(raw, name=samples[0][i].name)
            s.set_clusters(clusters, 'euclidean')
            merged_samples.append(s)

        self.load_from_data(merged_samples)

    def save_file(self, filename):
        self.io_datas.save_to_csv(filename,
                                  self.all_datas.get_names(),
                                  self.all_datas.get_ticks(),
                                  self.all_datas.to_numpy())

    def save_file_as_obj(self, filename):
        self.io_datas.save_project(filename, self.all_datas.samples)

    def load_from_data(self, sampling_datas):
        if not hasattr(sampling_datas[0], "calculated"):
            self.all_datas.append_calculate(sampling_datas)
        else:
            self.all_datas.append_samples(sampling_datas)
        self.table.update_table()

    @timer
    def select_new_sample(self):
        self.session.select_new_sample()
        self.table.select_rows(self.sel_indexes)

    def update_sample(self):
        self.session.select_new_sample()
        self.table.update_table()

    def edit_sample_event(self, edit_num):
        if edit_num == 0:
            [self.all_datas[i].to_log10() for i in self.sel_indexes]
        elif edit_num == 1:
            [self.all_datas[i].to_standardization() for i in self.sel_indexes]
        elif edit_num == 2:
            [self.all_datas[i].to_centralization() for i in self.sel_indexes]
        elif edit_num == 3:
            slide = self.get_slide()
            [self.all_datas[i].to_slide(slide) for i in self.sel_indexes]
        elif edit_num == 4:
            self.session.auto_remove_anomalys()
        elif edit_num == 5:
            self.session.to_independent()
        elif edit_num == 6:
            i, a, b = self.get_remove_ranges()
            if i == -1:
                return
            self.all_datas.remove_range(i, a, b)
        self.update_sample()

    def get_slide(self):
        return self.get_double_number("Введіть зсув", 0)

    def get_double_number(self, title: str, default_val=None):
        dialog_window = DialogWindow(
            form_args=[title, DoubleSpinBox(min_v=0, max_v=100, value=0)])
        ret = dialog_window.get_vals()
        return ret.get(title, default_val)

    def get_remove_ranges(self):
        title1 = "Виберіть вибірку"
        title2 = "Введіть початок діапазону"
        title3 = "Введіть кінець діапазону"
        items_combobox = {}
        for i in range(len(self.all_datas)):
            items_combobox[self.all_datas[i].name] = i
        dialog_window = DialogWindow(form_args=[
            title1, ComboBox(items_combobox),
            title2, DoubleSpinBox(min_v=-1.79e+308, max_v=1.79e+308, value=0),
            title3, DoubleSpinBox(min_v=-1.79e+308, max_v=1.79e+308, value=0)])
        ret = dialog_window.get_vals()
        i = ret.get(title1, -1)
        a = ret.get(title2)
        b = ret.get(title3)
        return i, a, b

    def duplicate_sample(self):
        sel = self.table.get_active_rows()
        for i in sel:
            self.all_datas.append_sample(self.all_datas[i].copy())
        self.table.update_table()

    def draw_samples(self):
        sel = self.table.get_active_rows()
        if len(sel) == 0 or sel == self.sel_indexes:
            return
        self.sel_indexes = sel
        self.make_session()
        self.configure_session()

    def make_session(self):
        if len(self.sel_indexes) == 1:
            if (type(self.session) is not ses.SessionModeTimeSeries and
                    type(self.session) is not ses.SessionModeTMO):
                self.session = ses.SessionMode1D(self)
        elif len(self.sel_indexes) == 2:
            self.session = ses.SessionMode2D(self)
        elif len(self.sel_indexes) >= 2:
            self.session = ses.SessionModeND(self)

    def change_sample_type_mode(self, mode: int):
        if mode == 0:
            self.session = ses.SessionMode1D(self)
        elif mode == 1:
            self.session = ses.SessionModeTimeSeries(self)
        elif mode == 2:
            self.session = ses.SessionModeTMO(self)
        self.configure_session()

    def remove_trend(self):
        if type(self.session) is ses.SessionModeTimeSeries:
            self.session.remove_trend()
            self.update_sample()

    def rename_sample(self):
        sel = self.table.get_active_rows()
        if len(sel) != 1:
            return
        d = self.all_datas[sel[0]]
        new_name = self.get_name_dialog(d.name)
        if new_name != "":
            self.all_datas[sel[0]].name = new_name
            if sel[0] in self.sel_indexes:
                self.update_sample()
            else:
                self.table.update_table()

    def get_name_dialog(self, old_name):
        title = "Введіть нове ім'я"
        dialog_window = DialogWindow(
            form_args=[title, QtWidgets.QLineEdit(old_name)],
            size=(600, 100))
        ret = dialog_window.get_vals()
        return ret.get(title, "")

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
        if type(self.session) is ses.SessionModeTimeSeries:
            self.session.smooth_time_series(smth_num)
            self.session.update_sample()

    def ssa(self, ssa_num):
        if type(self.session) is ses.SessionModeTimeSeries:
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
            self.datas_crits = []
            return
        text = self.all_datas.homogeneityProtocol(
            [[self.all_datas[j] for j in i] for i in self.datas_crits])
        self.showMessageBox("Перевірка однорідності сукупностей", text)
        self.datas_crits = []

    def add_data_to_homogeneity(self, sel):
        if sel in self.datas_crits:
            self.showMessageBox("Помилка", "Розподіл вже вибраний")
            return

        if len(self.datas_crits) != 0 and len(self.datas_crits[0]) != len(sel):
            n = len(self.datas_crits[0])
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
        return np.array([i for i, res in zip(sel, norm_test) if not res])

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
        if type(self.session) is ses.SessionModeND or \
                type(self.session) is ses.SessionMode2D:
            self.session.pca()

    def kmeans(self):
        self.session.kmeans()

    def agglomerative_clustering(self):
        self.session.agglomerative_clustering()

    def remove_clusters(self):
        self.session.remove_clusters()

    def split_on_clusters(self):
        self.session.split_on_clusters()

    def merge_as_clusters(self):
        self.session.merge_as_clusters()

    def nearest_neighbor_classification(self):
        self.session.nearest_neighbor_classification()

    def mod_nearest_neighbor_classification(self):
        self.session.mod_nearest_neighbor_classification()

    def k_nearest_neighbor_classification(self):
        self.session.k_nearest_neighbor_classification()

    def logistic_regression(self):
        self.session.logistic_regression()

    def discriminant_analysis(self):
        self.session.discriminant_analysis()

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


def demo_mode_course_work_0():
    file = "data/course/CO2 Emissions_Canada.csv"
    launch_app(file, is_file_name=True, auto_select=range(12))


def demo_mode_course_work_1():
    file = "data/course/CO2_without_mpg.csv"
    launch_app(file, is_file_name=True, auto_select=range(11))


def demo_mode_course_work_2():
    file = "data/course/2_CO2_fuel_type_zx.csv"
    launch_app(file, is_file_name=True, auto_select=range(11))


def demo_mode_course_work_3():
    file = "data/course/3_CO2_zx_removed_an.csv"
    launch_app(file, is_file_name=True, auto_select=range(11))


def demo_mode_course_work_4():
    file = "data/course/4_CO2_zx_log.csv"
    launch_app(file, is_file_name=True, auto_select=range(11))


def demo_mode_course_work_5():
    file = "data/course/5_CO2_zx_rm_tail.csv"
    launch_app(file, is_file_name=True, auto_select=range(11))


def demo_mode_classification():
    file = "data/iris_fish.txt"
    launch_app(file, is_file_name=True, auto_select=[2, 3])


def demo_mode_tmo():
    file = "data/500/exp.txt"
    launch_app(file, is_file_name=True, auto_select=range(1))


def demo_mode_dyplom():
    file = "/Users/user/Desktop/source/Dyplom/drone_videos/combined_hand_proj.sdatas"
    launch_app(file, is_file_name=True, auto_select=range(1))


def demo_mode_dyplom1():
    file = "/Users/user/Desktop/source/Dyplom/drone_videos/rgb_proj.sdatas"
    launch_app(file, is_file_name=True, auto_select=range(1))


def launch_app(file, is_file_name: bool, auto_select=None):
    app = QtWidgets.QApplication(sys.argv)
    widget = Window(file, is_file_name=is_file_name, auto_select=auto_select)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    demo_mode_dyplom()
