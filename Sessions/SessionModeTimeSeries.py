from Sessions.SessionMode import SessionMode
from Datanalysis import SamplingData, TimeSeriesData
from GUI.DialogWindow import DialogWindow
from GUI import SpinBox


class SessionModeTimeSeries(SessionMode):
    def __init__(self, window):
        super().__init__(window)
        self.init_new_session()

    def init_new_session(self):
        self.showed_index = -1
        self.time_series = None
        self.showed_smooth = None
        self.x_trend = None
        self.showed_components = None
        self.save_selected_index()

    def remove_all_added_lines(self):
        self.showed_smooth = None
        self.x_trend = None
        self.showed_components = None

    def create_plot_layout(self):
        self.plot_widget.create_time_series_plot()

    def create_time_series(self):
        data = self.window.all_datas[self.window.sel_indexes[0]]
        return TimeSeriesData(data)

    def get_active_samples(self) -> SamplingData:
        return self.time_series

    def is_changed_sample(self):
        return self.showed_index != self.window.sel_indexes[0]

    def save_selected_index(self):
        self.showed_index = self.window.sel_indexes[0]

    def auto_remove_anomalys(self) -> bool:
        return self.time_series.auto_remove_anomalys()

    def update_sample(self, _: int = 0):
        if self.is_changed_sample():
            self.init_new_session()

        self.time_series = self.create_time_series()
        self.plot_widget.plot_time_series(self.time_series)
        if self.showed_smooth is not None:
            self.plot_widget.plot_time_series_smooth(self.showed_smooth)
        if self.x_trend is not None:
            self.plot_widget.plot_time_series_trend(self.x_trend)
        if self.showed_components is not None:
            self.plot_widget.plot_time_series_components(
                self.showed_components)

    def smooth_time_series(self, num):
        if num == 0:
            self.showed_smooth = self.time_series.moving_average_method(k=7)
        elif num == 1:
            self.showed_smooth = self.time_series.median_method()
        elif num == 2:
            n = self.get_n_for_smooth()
            self.showed_smooth = self.time_series.sma_method(n)
        elif num == 3:
            n = self.get_n_for_smooth()
            self.showed_smooth = self.time_series.wma_method(n)
        elif num == 4:
            n = self.get_n_for_smooth()
            self.showed_smooth = self.time_series.ema_method(n)
        elif num == 5:
            n = self.get_n_for_smooth()
            self.showed_smooth = self.time_series.dma_method(n)
        elif num == 6:
            n = self.get_n_for_smooth()
            self.showed_smooth = self.time_series.tma_method(n)
        elif num == 8:
            self.remove_all_added_lines()
        elif num == 10 and self.showed_smooth is not None:
            self.time_series.set_series(self.showed_smooth)
            self.remove_all_added_lines()
            self.window.table.update_table()

    def get_n_for_smooth(self):
        N = len(self.time_series.x)
        title = "Введіть кількість точок для згладжування"
        dialog_window = DialogWindow(
            form_args=[title, SpinBox(min_v=1, max_v=N//2)])
        ret = dialog_window.get_vals()
        return ret.get(title)

    def remove_trend(self):
        if self.x_trend is None:
            self.x_trend = self.time_series.remove_poly_trend()
        else:
            self.time_series.set_series(self.time_series.x - self.x_trend)
            self.remove_all_added_lines()
            self.update_sample()

    def ssa_visualize_components(self):
        M = self.get_parameters_for_ssa_components()
        components = self.time_series.components_ssa_method(M)
        self.showed_components = components

    def get_parameters_for_ssa_components(self):
        title1 = "Введіть розмір гусені:"
        return self.get_parameters_for_args(title1)

    def ssa_reconstruction(self):
        M, n_components = self.get_parameters_for_ssa_reconstruction()
        p = self.time_series.reconstruction_ssa_method(M, n_components)
        self.showed_smooth = p

    def get_parameters_for_ssa_reconstruction(self):
        title1 = "Введіть розмір гусені:"
        title2 = "Введіть кількість компонент:"
        return self.get_parameters_for_args(title1, title2)

    def test_forecast_ssa(self):
        M, n_components, cnt_forecast = self.get_parameters_for_forecast_ssa()
        self.showed_smooth = self.time_series.test_forecast_ssa_method(
            M, n_components, cnt_forecast)

    def forecast_ssa(self):
        M, n_components, cnt_forecast = self.get_parameters_for_forecast_ssa()
        self.showed_smooth = self.time_series.forecast_ssa_method(
            self.time_series.x, M, n_components, cnt_forecast)

    def get_parameters_for_forecast_ssa(self):
        title1 = "Введіть розмір гусені:"
        title2 = "Введіть кількість компонент:"
        title3 = "Введіть кількість точок для прогнозування:"
        return self.get_parameters_for_args(title1, title2, title3)

    def get_parameters_for_args(self, *args):
        N = len(self.time_series.x)
        form_list = []
        for arg in args:
            form_list.append(arg)
            form_list.append(SpinBox(min_v=1, max_v=N - 1))
        dialog_window = DialogWindow(form_args=form_list)
        ret = dialog_window.get_vals()
        return_list = [ret[arg] for arg in args]
        if len(return_list) == 1:
            return return_list[0]
        return return_list
