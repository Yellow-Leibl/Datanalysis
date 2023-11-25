from GUI.WindowLayout import WindowLayout
from Datanalysis.SamplingDatas import SamplingDatas
from Datanalysis.DoubleSampleData import DoubleSampleData
from Datanalysis.SamplingData import SamplingData


class SessionMode:
    def __init__(self, window: WindowLayout):
        self.window = window

    def create_plot_layout(self):
        pass

    def get_active_samples(
            self) -> SamplingDatas | DoubleSampleData | SamplingData:
        pass

    def auto_remove_anomaly(self) -> bool:
        pass

    def to_independent(self):
        pass

    def update_graphics(self, number_column):
        pass

    def write_protocol(self):
        pass

    def write_critetion(self):
        pass
