import GUI.ui_tools as gui
from GUI.ui_tools import QtGui
from PyQt6 import QtCore
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from Datanalysis import (
    SamplingData, DoubleSampleData, SamplingDatas, TimeSeriesData)
from Datanalysis.SamplesTools import calculate_m
import numpy as np
import matplotlib.pyplot as plt
import math

pg.setConfigOption('imageAxisOrder', 'row-major')
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


def off_warning_for_pyqtgraph(widget: pg.GraphicsLayoutWidget):
    widget.viewport().setAttribute(
        QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)


class PlotWidget(gui.QtWidgets.QStackedWidget):
    def __init__(self) -> None:
        super().__init__()
        self.cache_graphics = [False] * 7

        self.__scatter_plot = PlotScatterWidget()
        self.__heatmap_plot = PlotHeatMapWidget()
        self.__parallel_plot = PlotParallelWidget()
        self.__3d_plot = Plot3dWidget()
        self.__buble_plot = PlotBubleWidget()
        self.__glyph_plot = PlotGlyphWidget()

        #  1D & 2D & Time Series
        __2d_widget = pg.GraphicsLayoutWidget()
        self.__2d_layout = __2d_widget.ci
        #  Diagnostic diagram
        __E_widget = pg.GraphicsLayoutWidget()
        self.__diagnostic_plot = __E_widget.ci.addPlot(
            title="Похибка регресії",
            labels={"left": "ε", "bottom": "Y"})
        #  N canvas
        self.__nd_widget = gui.TabWidget(
            __E_widget, "Діагностична діаграма",
            self.__scatter_plot, "Діаграма розкиду",
            self.__heatmap_plot, "Теплова карта",
            self.__parallel_plot, "Паралельні координати",
            self.__3d_plot, "3-вимірний простір",
            self.__buble_plot, "Бульбашкова діаграма",
            self.__glyph_plot, "Гліф діаграма")

        self.__nd_widget.setCurrentIndex(1)
        self.__nd_widget.currentChanged.connect(self.updateCharts)

        self.addWidget(__2d_widget)
        self.addWidget(self.__nd_widget)

        off_warning_for_pyqtgraph(__2d_widget)
        off_warning_for_pyqtgraph(__E_widget)

    def set_enabled_3d(self):
        self.__nd_widget.setTabEnabled(4, True)
        self.__nd_widget.setTabEnabled(5, True)
        self.__nd_widget.setTabEnabled(6, True)

    def set_disabled_3d(self):
        self.__nd_widget.setTabEnabled(4, False)
        self.__nd_widget.setTabEnabled(5, False)
        self.__nd_widget.setTabEnabled(6, False)

    def getCurrentTabIndex(self):
        return self.__nd_widget.currentIndex()

    def plotND(self, dn: SamplingDatas, col=0):
        self.__diagnostic_plot.clear()
        self.column_count = col
        self.datas = dn
        self.cache_graphics = [False] * 7
        self.updateCharts()

    def updateCharts(self):
        index = self.getCurrentTabIndex()
        if self.cache_graphics[index]:
            return
        self.cache_graphics[index] = True
        if index == 1:
            self.plotScatterDiagram(self.datas.samples, self.column_count)
        if index == 2:
            self.__heatmap_plot.plot_observers(self.datas)
        if index == 3:
            self.__parallel_plot.plot_observers(self.datas.to_numpy().T,
                                                self.datas.get_names())
        if index == 4:
            self.plot3D(self.datas)
        if index == 5:
            self.__buble_plot.plot_observers(self.datas.samples)
        if index == 6:
            self.__glyph_plot.plot_observers(self.datas.samples,
                                             self.column_count)

    #
    #  Creating plot
    #

    def create_time_series_plot(self):
        self.setCurrentIndex(0)
        self.__2d_layout.clear()
        self.time_val_plot = self.__2d_layout.addPlot(
            title="Часовий ряд",
            labels={"left": "X(t)", "bottom": "t"},
            row=0, col=0)
        self.time_auto_cor_plot = self.__2d_layout.addPlot(
            title="Автокореляційна функція",
            labels={"left": "r(𝜏)", "bottom": "𝜏"},
            row=0, col=1)

    def show_1d_plot(self):
        self.setCurrentIndex(0)
        self.create_1d_plot()

    def create_1d_plot(self):
        self.__2d_layout.clear()
        self.hist_plot = self.__2d_layout.addPlot(
            title="Гістограмна оцінка",
            labels={"left": "P"})
        self.emp_plot = self.__2d_layout.addPlot(
            title="Емпірична функція розподілу",
            labels={"left": "P"})

    def show_2d_plot(self):
        self.setCurrentIndex(0)
        self.create_2d_plot()

    def create_2d_plot(self):
        self.__2d_layout.clear()
        self.corr_plot = self.__2d_layout.addPlot(
            title="Корреляційне поле")

    def create_nd_plot(self, n: int):
        if n == 3:
            self.set_enabled_3d()
        else:
            self.set_disabled_3d()
        self.__scatter_plot.create_scatter_plot(n)
        self.setCurrentIndex(1)

    #
    #  Plotting
    #

    def plot_time_series(self, d: TimeSeriesData):
        N = len(d.x)
        t = np.arange(N)
        self.time_val_plot.clear()
        self.time_val_plot.plot(t, d.x,
                                pen=newPen(unique_colors[-1], 3),
                                symbol='o', symbolSize=5,)

        teta = t[1:]
        cor_val = [d.auto_cor_f(ti) for ti in teta]
        self.time_auto_cor_plot.clear()
        self.time_auto_cor_plot.plot(teta, cor_val,
                                     pen=newPen((0, 0, 0), 1))

    def plot_time_series_smooth(self, x):
        N = len(x)
        t = np.arange(N)
        self.time_val_plot.plot(t, x,
                                pen=newPen((255, 0, 0), 1))

    def plot_time_series_trend(self, x):
        N = len(x)
        t = np.arange(N)
        self.time_val_plot.plot(t, x,
                                pen=newPen((0, 0, 255), 2))

    def plot_time_series_components(self, components: np.ndarray):
        N = components.shape[1]
        t = np.arange(N)
        for i in range(components.shape[0]):
            self.time_val_plot.plot(t, components[i],
                                    pen=newPen(unique_colors[-i], 3))

    def plot1D(self, d: SamplingData, hist_data):
        self.plot1DHist(d, hist_data)
        self.plot1DEmp(d, hist_data)

    def plot1DHist(self, d: SamplingData, hist_data: np.ndarray,
                   hist_plot: pg.PlotItem = None):
        h = abs(d.max - d.min) / len(hist_data)
        x = np.empty(len(hist_data) * 3, dtype=float)
        y = np.empty(len(hist_data) * 3, dtype=float)
        y_max: float = hist_data[0]
        for p, i in enumerate(hist_data):
            if y_max < i:
                y_max = i
            x[p*3] = x[p*3+1] = d.min + p * h
            x[p*3+2] = d.min + (p + 1) * h
            y[p*3] = 0
            y[p*3+1] = y[p*3+2] = i

        if hist_plot is None:
            hist_plot = self.hist_plot
        hist_plot.clear()
        hist_plot.getAxis("bottom").setLabel(text=d.name)
        hist_plot.plot(x, y, fillLevel=0,
                       brush=(30, 120, 180),
                       pen=newPen((0, 0, 0), 1))

    def plot_emp_hist(self, d: SamplingData, hist_data: np.ndarray):
        x_class = np.empty((len(hist_data), 3), dtype=float)
        y_class = np.empty((len(hist_data), 3), dtype=float)
        cum_hist_data = np.cumsum(hist_data)
        y_class[:, 0] = cum_hist_data
        y_class[:, 1] = cum_hist_data
        y_class[:, 2] = np.nan

        x = np.linspace(d.min, d.max, len(hist_data) + 1, dtype=float)
        x_class[:, 0] = x[:-1]
        x_class[:, 1] = x[1:]
        x_class[:, 2] = np.nan

        # flatten: always returns a copy
        # ravel: returns a view of the original array whenever possible
        x_class = x_class.ravel()
        y_class = y_class.ravel()

        self.emp_plot.clear()
        self.emp_plot.getAxis("bottom").setLabel(text=d.name)
        self.emp_plot.plot(x_class, y_class,
                           pen=newPen((255, 0, 0), 2))

    def plot1DEmp(self, d: SamplingData, hist_data: np.ndarray):
        self.plot_emp_hist(d, hist_data)

        y_stat = np.cumsum(d.probabilityX)
        self.emp_plot.plot(d._x, y_stat,
                           pen=newPen((0, 255, 0), 2))

    def plot1DReproduction(self, d: SamplingData, f, lF, F, hF):
        if f is None:
            return
        x_gen = np.linspace(d.min, d.max, 500, dtype=float)
        y_hist = np.empty(len(x_gen), dtype=float)
        y_low = np.empty(len(x_gen), dtype=float)
        y_emp = np.empty(len(x_gen), dtype=float)
        y_high = np.empty(len(x_gen), dtype=float)
        for i, x in enumerate(x_gen):
            y_hist[i] = f(x)
            y_low[i] = lF(x)
            y_emp[i] = F(x)
            y_high[i] = hF(x)

        self.hist_plot.plot(x_gen, y_hist, pen=newPen((255, 0, 0), 3))
        self.emp_plot.plot(x_gen, y_low, pen=newPen((0, 128, 128), 2))
        self.emp_plot.plot(x_gen, y_emp, pen=newPen((0, 255, 255), 2))
        self.emp_plot.plot(x_gen, y_high, pen=newPen((128, 0, 128), 2))

    def plot_2d_with_details(self, d2: DoubleSampleData, hist_data):
        self.corr_plot.clear()
        self.corr_plot.setLabel("left", d2.y.name)
        self.corr_plot.setLabel("bottom", d2.x.name)
        self.plot_2d_histogram(d2, hist_data, self.corr_plot)
        self.plot_2d(d2, self.corr_plot)

    def plot_2d_histogram(self,
                          d2: DoubleSampleData,
                          hist_data,
                          corr_plot: pg.PlotItem):
        x = d2.x
        y = d2.y

        histogram_image = pg.ImageItem(hist_data)
        width = x.max - x.min
        height = y.max - y.min
        histogram_image.setRect(x.min, y.min, width, height)

        corr_plot.addItem(histogram_image)

    def plot_2d(self, d2: DoubleSampleData, corr_plot: pg.PlotItem):
        x = d2.x
        y = d2.y

        if x.clusters is not None:
            for i, cluster in enumerate(x.clusters):
                corr_plot.plot(x.raw[cluster], y.raw[cluster],
                               symbolBrush=unique_colors[i],
                               symbolPen=(0, 0, 0, 200), symbolSize=6,
                               pen=None)
        else:
            corr_plot.plot(x.raw, y.raw,
                           symbolBrush=(30, 120, 240),
                           symbolPen=(0, 0, 0, 200), symbolSize=6,
                           pen=None)

    def plot2DReproduction(self, d2: DoubleSampleData,
                           tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f):
        x = np.linspace(d2.x.min, d2.x.max, 500, dtype=float)
        self.corr_plot.plot(x, tl_lf(x), pen=newPen((0, 128, 128), 3))
        self.corr_plot.plot(x, tl_mf(x), pen=newPen((0, 128, 128), 3))
        self.corr_plot.plot(x, tr_lf(x), pen=newPen((255, 0, 255), 3))
        self.corr_plot.plot(x, tr_mf(x), pen=newPen((255, 0, 255), 3))
        self.corr_plot.plot(x, tr_f_lf(x), pen=newPen((0, 255, 128), 3))
        self.corr_plot.plot(x, tr_f_mf(x), pen=newPen((0, 255, 128), 3))
        self.corr_plot.plot(x, f(x), pen=newPen((255, 0, 0), 3))

    def plot3D(self, d3: SamplingDatas):
        X1 = d3[0].raw
        X2 = d3[1].raw
        X3 = d3[2].raw
        names = [d3[i].name for i in range(3)]
        self.__3d_plot.plot_observers(X1, X2, X3, names)

    def plotDiagnosticDiagram(self, dn: SamplingDatas, tr_l_f, f, tr_m_f):
        Y = dn[-1].raw
        X = [d.raw for d in dn[:-1]]
        E = Y - f(*X)
        self.__diagnostic_plot.plot(Y, E,
                                    symbolBrush=(30, 120, 180),
                                    symbolPen=(0, 0, 0, 200),
                                    symbolSize=7, pen=None)
        if len(dn) == 3:
            self.plot3D(dn)
            self.plot3DReproduction(dn, tr_l_f, f, tr_m_f)

    def plot3DReproduction(self, d3: SamplingDatas, tr_l_f, f, tr_m_f):
        x1 = d3[0]
        x2 = d3[1]
        self.__3d_plot.plot_regression(x1.min, x1.max, x2.min, x2.max,
                                       tr_l_f, f, tr_m_f)

    def plotScatterDiagram(self, dn: list[SamplingData], col):
        self.__scatter_plot.set_labels_for_scatter_plot(dn)
        plots = self.__scatter_plot.get_plots()
        n = len(dn)
        diag_i = 0
        slide_cells = 0
        for i in range(n):
            self.plot1DHist(dn[i], dn[i].get_histogram_data(col),
                            plots[diag_i])
            diag_i += n - i
            slide_cells += i
            for j in range(diag_i - n + i + 1, diag_i):
                d2 = DoubleSampleData(dn[i], dn[(j + slide_cells) % n])
                plots[j].clear()
                self.plot_2d_histogram(d2, d2.get_histogram_data(col),
                                       plots[j])
                self.plot_2d(d2, plots[j])


def newPen(color, width):
    return {'color': color, 'width': width}


class Plot3dWidget(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure()
        super().__init__(self.figure)
        self.__ax: plt.Axes = self.figure.add_subplot(projection='3d')

    def clear_plot(self):
        self.__ax.clear()

    def __update_plot(self):
        self.figure.canvas.draw()

    def plot_observers(self, X1, X2, X3, names):
        self.clear_plot()

        self.__ax.set(xlabel=names[0], ylabel=names[1], zlabel=names[2])
        self.__ax.scatter(X1, X2, X3)
        self.__update_plot()

    def plot_regression(self, x1_min, x1_max, x2_min, x2_max,
                        tr_l_f, f, tr_m_f):
        x1_lin = np.linspace(x1_min, x1_max, 50)
        x2_lin = np.linspace(x2_min, x2_max, 50)
        X1, X2 = np.meshgrid(x1_lin, x2_lin)

        X3 = self.__make_plane_mesh(X1, X2, f)
        self.__ax.plot_surface(X1, X2, X3, alpha=0.65, color='red')

        if tr_l_f is not None:
            X3_l = self.__make_plane_mesh(X1, X2, tr_l_f)
            self.__ax.plot_surface(
                X1, X2, X3_l, alpha=0.25, color='purple')
        if tr_m_f is not None:
            X3_m = self.__make_plane_mesh(X1, X2, tr_m_f)
            self.__ax.plot_surface(
                X1, X2, X3_m, alpha=0.25, color='purple')

        self.__update_plot()

    def __make_plane_mesh(self, X1, X2, f):
        x3 = np.array(f(np.ravel(X1), np.ravel(X2)))
        X3 = x3.reshape(X1.shape)
        return X3


class PlotParallelWidget(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        off_warning_for_pyqtgraph(self)
        self.__ax: pg.PlotItem = self.ci.addPlot()

    def set_labels(self, names):
        xdict = dict(enumerate(names))
        ax: pg.AxisItem = self.__ax.getAxis("bottom")
        ax.setTicks([xdict.items()])

        self.__ax.setLabel("left", "Умовні величини")

    def plot_observers(self, X: np.ndarray, names):
        self.__ax.clear()
        self.set_labels(names)

        n, N = X.shape

        def tr2v(x: np.ndarray):
            dx = x.max() - x.min()
            if dx == 0:
                return 0.5
            return (x - x.min()) / dx

        x_indexes = list(range(n))
        x_data = (x_indexes + [np.nan]) * N

        y_data = np.empty((N, n + 1), dtype=float)
        for j in range(n):
            y_data[:, j] = tr2v(X[j])
        y_data[:, n] = np.nan
        y_data = y_data.reshape((n + 1) * N)

        self.__ax.plot(x_data, y_data, pen=newPen((0, 0, 255), 1))


class PlotHeatMapWidget(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        off_warning_for_pyqtgraph(self)
        self.__heatmap_layout = self.ci

    def plot_observers(self, dn: SamplingDatas):
        heatmap_plot = self.__create_plot(dn)

        histogram_image = pg.ImageItem()
        values_image = np.array([s.raw for s in dn.samples])
        for i, row in enumerate(values_image):
            values_image[i] = (row - dn[i].min) / (dn[i].max - dn[i].min)
        histogram_image.setImage(values_image.transpose())
        n = len(dn)
        N = len(dn[0].raw)
        histogram_image.setRect(-0.5, -0.5, n, N)
        heatmap_plot.addItem(histogram_image)

    def __create_plot(self, dn: SamplingDatas) -> pg.PlotItem:
        n = len(dn)
        cols = [f"X{i+1}" for i in range(n)]
        colsdict = dict(enumerate(cols))
        colsaxis = pg.AxisItem(orientation='bottom')
        colsaxis.setTicks([colsdict.items()])

        self.__heatmap_layout.clear()
        heatmap_plot = self.__heatmap_layout.addPlot(
            axisItems={'bottom': colsaxis})
        heatmap_plot.getViewBox().invertY(True)
        heatmap_plot.getViewBox().setDefaultPadding(0.0)

        N = len(dn[0].raw)
        t = dict((i, f"{i+1}") for i in range(N)).items()
        heatmap_plot.getAxis("left").setTicks((t, []))

        return heatmap_plot


class PlotBubleWidget(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        off_warning_for_pyqtgraph(self)
        self.__buble_plot = self.ci.addPlot(
            labels={"left": "Y", "bottom": "X"})

    def plot_observers(self, dn: list[SamplingData]):
        x_raw = dn[0].raw
        y_raw = dn[1].raw
        z_raw = dn[2].raw
        sz = dn[2]
        def f_norm(z): return (z - sz.min) / (sz.max - sz.min)
        z_norm = [f_norm(z) * 25 + 2 for z in z_raw]

        self.__buble_plot.clear()
        self.__buble_plot.plot(x_raw, y_raw,
                               symbolSize=z_norm, alphaHint=0.6, pen=None)


class PlotGlyphWidget(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        off_warning_for_pyqtgraph(self)
        self.__glyph_plot = self.ci.addPlot(
            labels={"left": "Y", "bottom": "X"})
        colorMap = pg.colormap.get("CET-D1")
        self.__glyph_bar = pg.ColorBarItem(colorMap=colorMap)

    def plot_observers(self, dn: list[SamplingData], col=0):
        x_raw = dn[0].raw
        y_raw = dn[1].raw
        z_raw = dn[2].raw
        x = dn[0]
        y = dn[1]
        if col == 0:
            col = calculate_m(len(x_raw))

        glyph_data_sum = np.zeros((col, col))
        glyph_data_count = np.zeros((col, col), dtype=np.uint8)
        h_x = (x.max - x.min) / col
        h_y = (y.max - y.min) / col
        for i, z in enumerate(z_raw):
            c = math.floor((x_raw[i] - x.min) / h_x)
            if c == col:
                c -= 1
            r = math.floor((y_raw[i] - y.min) / h_y)
            if r == col:
                r -= 1
            glyph_data_count[r, c] += 1
            glyph_data_sum[r, c] += z

        glyph_data = glyph_data_sum
        for i in range(col):
            for j in range(col):
                if glyph_data_count[i, j] != 0:
                    glyph_data[i, j] /= glyph_data_count[i, j]

        glyph_image = pg.ImageItem(glyph_data)
        width = x.max - x.min
        height = y.max - y.min
        glyph_image.setRect(x.min, y.min, width, height)
        self.__glyph_plot.clear()
        self.__glyph_plot.addItem(glyph_image)
        self.__glyph_bar.values = glyph_image.quickMinMax()
        self.__glyph_bar.setImageItem(glyph_image, insert_in=self.__glyph_plot)

        self.__glyph_plot.plot(x_raw, y_raw,
                               symbolBrush=(30, 120, 180),
                               symbolPen=(0, 0, 0, 200), symbolSize=5,
                               pen=None)


class PlotScatterWidget(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        off_warning_for_pyqtgraph(self)
        self.__scatter_diagram_layout = self.ci
        self.__scatter_diagram_plots: list[pg.PlotItem] = []

    def get_plots(self):
        return self.__scatter_diagram_plots

    def create_scatter_plot(self, n):
        self.__scatter_diagram_plots: list[pg.PlotItem] = []
        self.__scatter_diagram_layout.clear()
        #
        # 111 ... ... ...
        # 000 111 ... ...
        # 000 000 111 ...
        # 000 000 000 111
        #
        # 111 000 000 000 111 000 000 111 000 111
        #
        for i in range(n):
            for j in range(i, n):
                plot_item = self.__scatter_diagram_layout.addPlot(
                    row=j, col=i)
                plot_item.getViewBox().setDefaultPadding(0.0)
                self.__scatter_diagram_plots.append(plot_item)

    def set_labels_for_scatter_plot(self, dn: list[SamplingData]):
        n = len(dn)
        k = 0
        font = QtGui.QFont()
        font.setPixelSize(7)

        def set_axis_style(plot: pg.PlotItem, axis):
            plot.showAxis(axis)
            plot.getAxis(axis).setStyle(tickFont=font)

        for i in range(n):
            for j in range(i, n):
                plot_item = self.__scatter_diagram_plots[k]
                k += 1
                plot_item.hideAxis("left")
                plot_item.hideAxis("bottom")
                labels = {}
                if i == 0:
                    labels["left"] = dn[j].name
                    set_axis_style(plot_item, "left")
                if j == n - 1:
                    labels["bottom"] = dn[i].name
                    set_axis_style(plot_item, "bottom")
                plot_item.setLabels(**labels)


class PlotDendrogramWidget(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        off_warning_for_pyqtgraph(self)
        self.__dendrogram_plot = self.ci.addPlot(
            labels={"left": "Відстань", "bottom": "Кластери"})
        self.__dendrogram_plot.getViewBox().setDefaultPadding(0.0)

    def plot_observers(self, C, Z):
        self.__dendrogram_plot.clear()

        self.set_ticks(C)

        indexes_arr = []
        for c in C:
            indexes_arr += c
        cluster2index = dict([(ind, i) for i, ind in enumerate(indexes_arr)])

        def find_prev_cluster(i):
            d1 = 0
            d2 = 0
            for j in range(i - 1, -1, -1):
                if Z[i][0] in Z[j][:3] and d1 == 0:
                    d1 = Z[j][2]
                if Z[i][1] in Z[j][:3] and d2 == 0:
                    d2 = Z[j][2]
            return d1, d2

        for i in range(len(Z)):
            z = Z[i]
            x1 = cluster2index[z[0]]
            x2 = cluster2index[z[1]]
            y = z[2]
            y1, y2 = find_prev_cluster(i)
            cluster2index[z[0]] = (x1 + x2) / 2
            self.__dendrogram_plot.plot(
                [x1, x1, x2, x2], [y1, y, y, y2],
                pen=newPen((30, 120, 240), 3))

    def set_ticks(self, clusters):
        ticks = []
        for cluster in clusters:
            ticks += [f"{i}" for i in cluster]
        xdict = dict(enumerate(ticks))
        self.__dendrogram_plot.getAxis("bottom").setTicks([xdict.items()])


class PlotDialogWindow(gui.QtWidgets.QWidget):
    def __init__(self, **args):
        super().__init__()
        title = args.get("title", "")
        self.setWindowTitle(title)
        size = args.get("size", (0, 0))
        self.resize(size[0], size[1])

        plot_widget = args.get("plot")

        vbox = gui.QtWidgets.QVBoxLayout()
        vbox.addWidget(plot_widget)
        self.setLayout(vbox)


class PlotRocCurveWidget(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        off_warning_for_pyqtgraph(self)
        self.__roc_curve_plot = self.ci.addPlot(
            labels={"left": "True Positive Rate",
                    "bottom": "False Positive Rate"})

    def plot_observers(self, fpr: np.ndarray, tpr: np.ndarray):
        self.__roc_curve_plot.clear()
        self.__roc_curve_plot.plot([0, 1], [0, 1], pen=pg.mkPen(
            'r', width=3, style=QtCore.Qt.PenStyle.DashLine))
        self.__roc_curve_plot.plot(fpr, tpr, pen=newPen((0, 0, 255), 3))


unique_colors = [
    "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941",
    "#006FA6", "#A30059", "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC",
    "#B79762", "#004D43", "#8FB0FF", "#997D87", "#5A0007", "#809693",
    "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9",
    "#B903AA", "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299",
    "#300018", "#0AA6D8", "#013349", "#00846F", "#372101", "#FFB500",
    "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68",
    "#7A87A1", "#788D66", "#885578", "#FAD09F", "#FF8A9A", "#D157A0",
    "#BEC459", "#456648", "#0086ED", "#886F4C",

    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9",
    "#FF913F", "#938A81", "#575329", "#00FECF", "#B05B6F", "#8CD0FF",
    "#3B9700", "#04F757", "#C8A1A1", "#1E6E00", "#7900D7", "#A77500",
    "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0",
    "#3A2465", "#922329", "#5B4534", "#FDE8DC", "#404E55", "#0089A3",
    "#CB7E98", "#A4E804", "#324E72", "#6A3A4C", "#83AB58", "#001C1E",
    "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
    "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4",
    "#1E0200", "#5B4E51", "#C895C5", "#320033", "#FF6832", "#66E1D3",
    "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"
]
