from Sessions.SessionMode import SessionMode
from Datanalysis import SamplingData, ProtocolGenerator, TimeSeriesData


class SessionModeTimeSeries(SessionMode):
    def __init__(self, window):
        super().__init__(window)
        self.time_series = None
        self.showed_smooth = None
        self.x_trend = None
        self.showed_index = -1

    def init_new_session(self):
        self.showed_smooth = None
        self.x_trend = None
        self.save_selected_index()

    def create_plot_layout(self):
        self.window.plot_widget.create_time_series_plot()

    def get_active_samples(self) -> SamplingData:
        return self.window.all_datas[self.window.sel_indexes[0]]

    def is_changed_sample(self):
        return self.showed_index != self.window.sel_indexes[0]

    def save_selected_index(self):
        self.showed_index = self.window.sel_indexes[0]

    def auto_remove_anomalys(self) -> bool:
        return self.time_series.auto_remove_anomalys()

    def update_graphics(self, _: int = 0):
        if self.is_changed_sample():
            self.init_new_session()

        d = self.get_active_samples()
        self.time_series = TimeSeriesData(d)
        self.window.feature_area.set_borders(d.min, d.max)
        self.window.plot_widget.plot_time_series(self.time_series)
        if self.showed_smooth is not None:
            self.window.plot_widget.plot_time_series_smooth(self.showed_smooth)
        if self.x_trend is not None:
            self.window.plot_widget.plot_time_series_trend(self.x_trend)

    def remove_all_added_lines(self):
        self.showed_smooth = None
        self.x_trend = None

    def smooth_time_series(self, num):
        if num == 0:
            self.showed_smooth = self.time_series.moving_average_method(k=7)
        elif num == 1:
            self.showed_smooth = self.time_series.median_method()
        elif num == 2:
            self.showed_smooth = self.time_series.sma_method()
        elif num == 3:
            self.showed_smooth = self.time_series.wma_method()
        elif num == 4:
            self.showed_smooth = self.time_series.ema_method()
        elif num == 5:
            self.showed_smooth = self.time_series.dma_method()
        elif num == 6:
            self.showed_smooth = self.time_series.tma_method()
        elif num == 8:
            self.remove_all_added_lines()
        elif num == 10 and self.showed_smooth is not None:
            self.time_series.set_series(self.showed_smooth)
            self.remove_all_added_lines()
            self.window.table.update_table()

    def write_protocol(self):
        self.window.protocol.setText(
            ProtocolGenerator.getProtocol(self.time_series))

    def remove_trend(self):
        if self.x_trend is None:
            self.x_trend = self.time_series.remove_poly_trend()
        else:
            self.time_series.set_series(self.time_series.x - self.x_trend)
            self.remove_all_added_lines()
            self.update_graphics()

    def ssa(self):
        self.showed_smooth = self.time_series.ssa_method()
