from GUI.WindowLayout import WindowLayout
from Datanalysis import SamplingDatas, SamplingData, DoubleSampleData


class SessionMode:
    def __init__(self, window: WindowLayout):
        self.window = window
        self.selected_regr_num = -1

    def set_regression_number(self, number):
        self.selected_regr_num = number

    def create_plot_layout(self):
        pass

    def get_active_samples(
            self) -> SamplingDatas | DoubleSampleData | SamplingData:
        pass

    def auto_remove_anomalys(self) -> bool:
        pass

    def to_independent(self):
        pass

    def update_graphics(self, number_column):
        pass

    def write_protocol(self):
        pass

    def write_critetion(self):
        pass
