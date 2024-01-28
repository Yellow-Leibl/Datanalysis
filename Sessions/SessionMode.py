from GUI.WindowLayout import WindowLayout
from Datanalysis import ProtocolGenerator


class SessionMode:
    def __init__(self, window: WindowLayout):
        self.window = window
        self.plot_widget = window.plot_widget
        self.selected_regr_num = -1

    def set_regression_number(self, number):
        self.selected_regr_num = number
        self.select_new_sample()

    def configure(self):
        self.create_plot_layout()

    def create_plot_layout(self):
        pass

    def get_active_samples(self):
        pass

    def auto_remove_anomalys(self) -> bool:
        pass

    def to_independent(self):
        pass

    def select_new_sample(self):
        self.update_sample(self.window.feature_area.get_number_classes())

    def update_sample(self, number_column=0):
        self.write_protocol()
        self.write_critetion()

    def write_protocol(self):
        act_sample = self.get_active_samples()
        protocol_text = ProtocolGenerator.getProtocol(act_sample)
        self.window.protocol.setText(protocol_text)

    def write_critetion(self):
        self.window.criterion_protocol.setText("")
