from Sessions.SessionMode import SessionMode
from Datanalysis import ProcessData


class SessionModeTMO(SessionMode):
    def __init__(self, window):
        super().__init__(window)
        self.process_data = None

    def create_plot_layout(self):
        self.plot_widget.show_1d_plot()

    def create_proccess_data(self):
        data = self.window.all_datas[self.window.sel_indexes[0]]
        return ProcessData(data)

    def get_active_samples(self):
        return self.process_data

    def configure(self):
        super().configure()
        self.process_data = self.create_proccess_data()
        _x = self.process_data.data._x
        self.window.feature_area.silent_change_number_classes(0)
        self.window.feature_area.set_maximum_column_number(len(_x))

    def update_sample(self, number_column: int = 0):
        self.process_data = self.create_proccess_data()
        self.process_data.to_calculate_characteristics()
        self.process_data.set_trust(self.window.feature_area.get_trust())
        hist_data = self.process_data.get_histogram_data(number_column)
        inten_data = self.process_data.get_intensity_function(number_column)
        self.window.feature_area.silent_change_number_classes(len(hist_data))
        self.plot_widget.plot1DHist(self.process_data.data, hist_data)
        self.plot_widget.plot_emp_hist(self.process_data.data, inten_data)
        super().update_sample(number_column)
