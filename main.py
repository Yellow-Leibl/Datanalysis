import sys
import logging

from PyQt6.QtWidgets import QApplication
from GUI.WindowLayout import WindowLayout

from Sessions.SessionMode import SessionMode
from Sessions.SessionMode1D import SessionMode1D
from Sessions.SessionMode2D import SessionMode2D
from Sessions.SessionModeND import SessionModeND

from Datanalysis.SamplingDatas import SamplingDatas

logging.basicConfig(level=logging.INFO)


class Window(WindowLayout):
    def __init__(self,
                 file: str,
                 is_file_name: bool = True,
                 demo_mode: bool = False):
        super().__init__()
        self.session: SessionMode = None
        self.all_datas = SamplingDatas()
        self.sel_indexes: list[int] = []
        self.datas_crits: list[list] = []
        self.selected_regr_num = -1
        if is_file_name:
            all_file = self.open_file(file)
        else:
            all_file = file.split('\n')
        self.loadFromData(all_file)
        if demo_mode:
            self.autoSelect()

    def loadFromData(self, all_file: list[str]):
        self.all_datas.append(all_file)
        self.table.update_table(self.all_datas)

    def active_sample_changed(self, selected_new_samples: bool = False):
        self.session.update_graphics(self.getNumberClasses())
        self.session.write_protocol()
        self.session.write_critetion()
        if selected_new_samples:
            self.table.select_rows(self.sel_indexes)
        else:
            self.table.update_table(self.all_datas)

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
            if not self.session.auto_remove_anomaly():
                return
        elif edit_num == 5:
            self.session.to_independent()
        self.active_sample_changed()

    def duplicateSample(self):
        sel = self.table.get_active_rows()
        for i in sel:
            self.all_datas.append_sample(self.all_datas[i].copy())
        self.table.update_table(self.all_datas)

    def removeAnomaly(self):
        if self.is1d():
            minmax = self.getMinMax()
            self.all_datas[self.sel_indexes[0]].remove(minmax[0], minmax[1])
            self.active_sample_changed()

    def drawSamples(self):
        sel = self.table.get_active_rows()
        if len(sel) == 0 or sel == self.sel_indexes:
            return
        self.sel_indexes = sel

        if self.is1d():
            self.session = SessionMode1D(self)
        elif self.is2d():
            self.session = SessionMode2D(self)
        elif self.isNd():
            self.session = SessionModeND(self)

        self.session.create_plot_layout()
        self.selected_regr_num = -1
        self.silentChangeNumberClasses(0)
        self.setMaximumColumnNumber(self.all_datas.getMaxDepthRangeData())
        self.active_sample_changed(selected_new_samples=True)
        self.table.select_rows(self.sel_indexes)

    def delete_observations(self):
        all_obsers = self.table.get_observations_to_remove()
        sample_changed = False
        update_table = sum([len(obsers) for obsers in all_obsers]) > 0
        for i, obsers in list(enumerate(all_obsers))[::-1]:
            if len(obsers) == 0:
                continue
            if len(obsers) == len(self.all_datas[i].raw):
                self.all_datas.pop(i)
                continue
            for obser in obsers:
                self.all_datas[i].remove_observation(obser)
            if i in self.sel_indexes:
                sample_changed = True
        if sample_changed:
            self.active_sample_changed()
        elif update_table:
            self.table.update_table(self.all_datas)

    def is1d(self) -> bool:
        return len(self.sel_indexes) == 1

    def is2d(self) -> bool:
        return len(self.sel_indexes) == 2

    def is3d(self) -> bool:
        return len(self.sel_indexes) == 3

    def isNd(self) -> bool:
        return len(self.sel_indexes) >= 2

    def setReproductionSeries(self):
        self.selected_regr_num = \
            self.index_in_menu(self.get_regr_menu(), self.sender())
        self.active_sample_changed(selected_new_samples=True)

    def changeTrust(self, trust: float):
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
            if type(res) is bool:
                if res:
                    self.showMessageBox(title, descr + " - ідентичні")
                else:
                    self.showMessageBox(title, descr + " - неідентичні")
            elif type(res) is str:
                self.showMessageBox(title, descr +
                                    " - мають випадкову різницю регресій")

    def homogeneityAndIndependence(self, trust: float):
        sel = self.table.get_active_rows()
        if len(sel) == 1:
            P = self.all_datas[sel[0]].critetionAbbe()
            title = "Критерій Аббе"
            if P > trust:
                self.showMessageBox(title, f"{P:.5} > {trust}" +
                                    "\nСпостереження незалежні")
            else:
                self.showMessageBox(title, f"{P:.5} < {trust}" +
                                    "\nСпостереження залежні")
        elif len(sel) == 2:
            if self.all_datas.ident2Samples(sel[0], sel[1], trust):
                self.showMessageBox("Вибірки однорідні", "")
            else:
                self.showMessageBox("Вибірки неоднорідні", "")
        elif len(sel) > 2:
            if self.all_datas.identKSamples([self.all_datas[i] for i in sel],
                                            trust):
                self.showMessageBox("Вибірки однорідні", "")
            else:
                self.showMessageBox("Вибірки неоднорідні", "")

    def homogeneityNSamples(self):
        title = "Перевірка однорідності сукупностей"
        sel = self.table.get_active_rows()
        self.table.clearSelection()
        if len(sel) == 0:
            if len(self.datas_crits) < 2:
                return
            text = self.all_datas.homogeneityProtocol(
                [[self.all_datas[j] for j in i] for i in self.datas_crits])
            self.showMessageBox(title, text)
            self.datas_crits = []
        elif sel not in self.datas_crits:
            if len(self.datas_crits) != 0 and \
               len(self.datas_crits[0]) != len(sel):
                return self.showMessageBox(
                    "Помилка",
                    f"Потрібен {len(self.datas_crits[0])}-вимірний розподіл")
            norm_test = [self.all_datas[i].isNormal() for i in sel]
            if False in norm_test:
                return self.showMessageBox(
                    "Помилка", "Не є нормальним розподілом:" +
                    str([sel[i]+1 for i, res in enumerate(norm_test)
                         if not res]))
            self.datas_crits.append(sel)
            self.showMessageBox(
                title, "Вибрані вибірки:\n" +
                "\n".join([str([i+1 for i in r]) for r in self.datas_crits]))

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
        w = self.pCA_number.value()
        active_samples = self.session.get_active_samples()
        ind, retn = active_samples.principalComponentAnalysis(w)
        self.all_datas.append_samples(ind.samples)
        self.all_datas.append_samples(retn.samples)
        self.table.update_table(self.all_datas)

    def autoSelect(self):
        self.sel_indexes = range(3)
        self.session = SessionModeND(self)
        self.session.create_plot_layout()
        self.active_sample_changed()


def applicationLoadFromFile(file: str = ''):
    app = QApplication(sys.argv)
    widget = Window(file)
    widget.show()
    sys.exit(app.exec())


def applicationLoadFromStr(file: str = ''):
    app = QApplication(sys.argv)
    widget = Window(file, False)
    widget.show()
    sys.exit(app.exec())


def demo_mode_show():
    file = "data/500/norm3n.txt"
    app = QApplication(sys.argv)
    widget = Window(file, demo_mode=True)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    demo_mode_show()
