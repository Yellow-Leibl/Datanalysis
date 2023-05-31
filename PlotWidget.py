from PyQt6.QtWidgets import QStackedWidget, QTabWidget
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from Datanalysis.SamplingData import SamplingData
from Datanalysis.DoubleSampleData import DoubleSampleData
from Datanalysis.SamplingDatas import SamplingDatas
import numpy as np
import math

pg.setConfigOption('imageAxisOrder', 'row-major')
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class PlotWidget(QStackedWidget):
    def __init__(self) -> None:
        super().__init__()
        self.cache_graphics = [False] * 7
        #  1D & 2D
        __2d_widget = pg.GraphicsLayoutWidget()
        self.__2d_layout = __2d_widget.ci
        #  3D
        __3d_figure = Figure()
        self.update_3d = lambda: __3d_figure.canvas.draw()
        __3d_widget = FigureCanvasQTAgg(__3d_figure)
        self.__3d_plot: plt.Axes = __3d_figure.add_subplot(projection='3d')
        #  Diagnostic diagram
        __E_widget = pg.GraphicsLayoutWidget()
        self.__diagnostic_plot = __E_widget.ci.addPlot(
            title="Похибка регресії",
            labels={"left": "ε", "bottom": "Y"})
        #  Scatter diagram
        __scatter_widget = pg.GraphicsLayoutWidget()
        self.__scatter_diagram_layout = __scatter_widget.ci
        #  Parallel coordinates
        __parallel_widget = pg.GraphicsLayoutWidget()
        self.__parallel_layout = __parallel_widget.ci
        #  Heat map
        __heatmap_widget = pg.GraphicsLayoutWidget()
        self.__heatmap_layout = __heatmap_widget.ci
        #  Buble diagram
        __buble_widget = pg.GraphicsLayoutWidget()
        self.__buble_plot = __buble_widget.ci.addPlot(
            labels={"left": "Y", "bottom": "X"})
        #  Glyph diagram
        __glyph_widget = pg.GraphicsLayoutWidget()
        self.__glyph_plot = __glyph_widget.ci.addPlot(
            labels={"left": "Y", "bottom": "X"})
        colorMap = pg.colormap.get("CET-D1")
        self.__glyph_bar = pg.ColorBarItem(colorMap=colorMap)
        #  N canvas
        self.__nd_widget = QTabWidget()
        self.__nd_widget.addTab(__parallel_widget, "Паралельні координати")
        self.__nd_widget.addTab(__scatter_widget, "Діаграма розкиду")
        self.__nd_widget.addTab(__heatmap_widget, "Теплова карта")
        self.__nd_widget.addTab(__E_widget, "Діагностична діаграма")
        self.__nd_widget.addTab(__3d_widget, "3-вимірний простір")
        self.__nd_widget.addTab(__buble_widget, "Бульбашкова діаграма")
        self.__nd_widget.addTab(__glyph_widget, "Гліф діаграма")
        self.__nd_widget.currentChanged.connect(self.updateCharts)
        #  General canvas
        self.addWidget(__2d_widget)
        self.addWidget(self.__nd_widget)

    def setEnabled3d(self):
        self.__nd_widget.setTabEnabled(4, True)
        self.__nd_widget.setTabEnabled(5, True)
        self.__nd_widget.setTabEnabled(6, True)

    def setDisabled3d(self):
        self.__nd_widget.setTabEnabled(4, False)
        self.__nd_widget.setTabEnabled(5, False)
        self.__nd_widget.setTabEnabled(6, False)

    def getCurrentTabIndex(self):
        return self.__nd_widget.currentIndex()
    #
    #  Creating plot
    #

    def create1DPlot(self):
        self.setCurrentIndex(0)
        self.__2d_layout.clear()
        self.hist_plot = self.__2d_layout.addPlot(
            title="Гістограмна оцінка",
            labels={"left": "P", "bottom": "x"})
        self.emp_plot = self.__2d_layout.addPlot(
            title="Емпірична функція розподілу",
            labels={"left": "P", "bottom": "x"})

    def create2DPlot(self):
        self.setCurrentIndex(0)
        self.__2d_layout.clear()
        self.corr_plot = self.__2d_layout.addPlot(
            title="Корреляційне поле",
            labels={"left": "Y", "bottom": "X"})

    def createNDPlot(self, n: int):
        if n == 3:
            self.setEnabled3d()
        else:
            self.setDisabled3d()
        self.createScatterPlot(n)
        self.createParallelPlot(n)
        self.createHeatmapPlot(n)
        self.setCurrentIndex(1)

    def createScatterPlot(self, n):
        self.__scatter_diagram_plots = []
        self.__scatter_diagram_layout.clear()
        # *
        # 111 ... ... ...
        # 000 111 ... ...
        # 000 000 111 ...
        # 000 000 000 111
        #
        # 111 000 000 000 111 000 000 111 000 111
        # *
        for i in range(n):
            for j in range(i, n):
                left = ""
                bottom = ""
                if i == 0:
                    left = f"X{j+1}"
                if j == n - 1:
                    bottom = f"X{i+1}"
                plot_item = self.__scatter_diagram_layout.addPlot(
                    row=j, col=i, labels={"left": left, "bottom": bottom})
                plot_item.getViewBox().setDefaultPadding(0.0)
                self.__scatter_diagram_plots.append(plot_item)

    def createParallelPlot(self, n):
        x = [f"X{i+1}" for i in range(n)]
        xdict = dict(enumerate(x))
        self.__parallel_x_data = list(xdict.keys())
        stringaxis = pg.AxisItem(orientation='bottom')
        stringaxis.setTicks([xdict.items()])

        self.__parallel_layout.clear()
        self.__parallel_plot = self.__parallel_layout.addPlot(
            labels={"left": "Умовні величини"},
            axisItems={'bottom': stringaxis})

    def createHeatmapPlot(self, n):
        cols = [f"X{i+1}" for i in range(n)]
        colsdict = dict(enumerate(cols))
        # self.__heatmap_x_data = list(colsdict.keys())
        colsaxis = pg.AxisItem(orientation='bottom')
        colsaxis.setTicks([colsdict.items()])

        self.__heatmap_layout.clear()
        self.__heatmap_plot = self.__heatmap_layout.addPlot(
            axisItems={'bottom': colsaxis})
        self.__heatmap_plot.getViewBox().invertY(True)
        self.__heatmap_plot.getViewBox().setDefaultPadding(0.0)

    #
    #  Plotting
    #

    def plot1D(self, d: SamplingData, hist_data: list):
        self.plot1DHist(d, hist_data)
        self.plot1DEmp(d, hist_data)

    def plot1DHist(self, d: SamplingData, hist_data: list,
                   hist_plot: pg.PlotItem = None):
        h = abs(d.max - d.min) / len(hist_data)
        x = []
        y = []
        y_max: float = hist_data[0]
        for p, i in enumerate(hist_data):
            if y_max < i:
                y_max = i
            x.append(d.min + p * h)
            x.append(d.min + p * h)
            x.append(d.min + (p + 1) * h)
            y.append(0)
            y.append(i)
            y.append(i)

        if hist_plot is None:
            hist_plot = self.hist_plot
        hist_plot.clear()
        hist_plot.plot(x, y, fillLevel=0,
                       brush=(30, 120, 180),
                       pen=newPen((0, 0, 0), 1))

    def plot1DEmp(self, d: SamplingData, hist_data: list):
        h = abs(d.max - d.min) / len(hist_data)
        x_class = []
        y_class = []
        col_height = 0.0
        for p, i in enumerate(hist_data):
            if col_height > 1:
                col_height = 1
            x_class.append(d.min + p * h)
            x_class.append(d.min + p * h)
            x_class.append(d.min + (p + 1) * h)
            y_class.append(col_height)
            y_class.append(col_height + i)
            y_class.append(col_height + i)
            col_height += i

        x_stat = []
        y_stat = []
        sum_ser = 0.0
        for i in range(len(d.probabilityX)):
            sum_ser += d.probabilityX[i]
            x_stat.append(d._x[i])
            y_stat.append(sum_ser)

        self.emp_plot.clear()
        self.emp_plot.plot(x_class, y_class,
                           pen=newPen((255, 0, 0), 2))
        self.emp_plot.plot(x_stat, y_stat,
                           pen=newPen((0, 255, 0), 2))

    def plot1DReproduction(self, d: SamplingData, f, lF, F, hF):
        if f is None:
            return
        x_gen = d.toGenerateReproduction(f)
        y_hist = []
        y_low = []
        y_emp = []
        y_high = []
        for x in x_gen:
            y_hist.append(f(x))
            y_low.append(lF(x))
            y_emp.append(F(x))
            y_high.append(hF(x))

        self.hist_plot.plot(x_gen, y_hist, pen=newPen((255, 0, 0), 3))
        self.emp_plot.plot(x_gen, y_low, pen=newPen((0, 128, 128), 2))
        self.emp_plot.plot(x_gen, y_emp, pen=newPen((0, 255, 255), 2))
        self.emp_plot.plot(x_gen, y_high, pen=newPen((128, 0, 128), 2))

    def plot2D(self, d2: DoubleSampleData, hist_data, corr_plot=None):
        x = d2.x
        y = d2.y
        if len(x.getRaw()) != len(y.getRaw()):
            return

        histogram_image = pg.ImageItem(hist_data)
        width = x.max - x.min
        height = y.max - y.min
        histogram_image.setRect(x.min, y.min, width, height)

        if corr_plot is None:
            corr_plot = self.corr_plot
        corr_plot.clear()
        corr_plot.addItem(histogram_image)
        corr_plot.plot(x.getRaw(), y.getRaw(),
                       symbolBrush=(30, 120, 180),
                       symbolPen=(0, 0, 0, 200), symbolSize=6,
                       pen=None)

    def plot2DReproduction(self, d2: DoubleSampleData,
                           tl_lf, tl_mf, tr_lf, tr_mf, tr_f_lf, tr_f_mf, f):
        if f is None:
            return
        x_gen = d2.toGenerateReproduction(f)
        y_tl_lf, y_tl_mf = [], []
        y_tr_lf, y_tr_mf = [], []
        y_tr_f_lf, y_tr_f_mf = [], []
        y = []
        for x in x_gen:
            y_tl_lf.append(tl_lf(x))
            y_tl_mf.append(tl_mf(x))
            y_tr_lf.append(tr_lf(x))
            y_tr_mf.append(tr_mf(x))
            y_tr_f_lf.append(tr_f_lf(x))
            y_tr_f_mf.append(tr_f_mf(x))
            y.append(f(x))

        self.corr_plot.plot(x_gen, y_tl_lf, pen=newPen((0, 128, 128), 3))
        self.corr_plot.plot(x_gen, y_tl_mf, pen=newPen((0, 128, 128), 3))
        self.corr_plot.plot(x_gen, y_tr_lf, pen=newPen((255, 0, 255), 3))
        self.corr_plot.plot(x_gen, y_tr_mf, pen=newPen((255, 0, 255), 3))
        self.corr_plot.plot(x_gen, y_tr_f_lf, pen=newPen((0, 255, 128), 3))
        self.corr_plot.plot(x_gen, y_tr_f_mf, pen=newPen((0, 255, 128), 3))
        self.corr_plot.plot(x_gen, y, pen=newPen((255, 0, 0), 3))

    def plot3D(self, d3: list[SamplingData]):
        x1_raw = d3[0].getRaw()
        x2_raw = d3[1].getRaw()
        x3_raw = d3[2].getRaw()
        self.__3d_plot.clear()
        self.__3d_plot.set(xlabel="$X1$", ylabel="$X2$", zlabel="$X3$")
        self.__3d_plot.scatter(x1_raw, x2_raw, x3_raw)
        self.update_3d()

    def plotDiagnosticDiagram(self, dn: SamplingDatas):
        tr_l_f, f, tr_m_f = dn.toCreateLinearRegressionMNK(len(dn) - 1)
        self.__diagnostic_plot.plot(dn.line_Y, dn.line_E,
                                    symbolBrush=(30, 120, 180),
                                    symbolPen=(0, 0, 0, 200), symbolSize=7,
                                    pen=None)
        if len(dn) == 3:
            self.plot3DReproduction(dn, tr_l_f, f, tr_m_f)

    def plot3DReproduction(self, d3: SamplingDatas, tr_l_f, f, tr_m_f):
        x1 = d3[0]
        x2 = d3[1]
        x1_min = x1.min
        x1_max = x1.max
        x1_plane = [x1_min, x1_max, x1_min, x1_max]
        x2_min = x2.min
        x2_max = x2.max
        x2_plane = [x2_min, x2_max, x2_min, x2_max]
        X1, X2 = np.meshgrid(x1_plane, x2_plane)
        zs_f = np.array(f(np.ravel(X1), np.ravel(X2)))
        zs_tr_l_f = np.array(tr_l_f(np.ravel(X1), np.ravel(X2)))
        zs_tr_m_f = np.array(tr_m_f(np.ravel(X1), np.ravel(X2)))
        X3 = zs_f.reshape(X1.shape)
        X3_l = zs_tr_l_f.reshape(X1.shape)
        X3_m = zs_tr_m_f.reshape(X1.shape)
        self.__3d_plot.plot_surface(X1, X2, X3, alpha=0.1, color='red')
        self.__3d_plot.plot_surface(X1, X2, X3_l, alpha=0.05, color='purple')
        self.__3d_plot.plot_surface(X1, X2, X3_m, alpha=0.05, color='purple')
        self.update_3d()

    def plotScatterDiagram(self, dn: list[SamplingData], col):
        n = len(dn)
        diag_i = 0
        slide_cells = 0
        for i in range(n):
            self.plot1DHist(dn[i], dn[i].get_histogram_data(col),
                            self.__scatter_diagram_plots[diag_i])
            diag_i += n - i
            slide_cells += i
            for j in range(diag_i - n + i + 1, diag_i):
                d2 = DoubleSampleData(dn[i], dn[(j + slide_cells) % n])
                self.plot2D(d2, d2.get_histogram_data(col),
                            self.__scatter_diagram_plots[j])

    def plotParallelCoordinates(self, dn: list[SamplingData]):
        n = len(dn)
        N = len(dn[0].getRaw())

        def tr2v(d: SamplingData, i):
            return (d.getRaw()[i] - d.min) / (
                d.max - d.min)
        self.__parallel_plot.clear()
        x_data = []
        y_data = []
        for i in range(N):
            x_data += self.__parallel_x_data
            y_data += [tr2v(dn[j], i) for j in range(n)]
            x_data += self.__parallel_x_data[-2::-1]
            y_data += [tr2v(dn[j], i) for j in range(n - 2, -1, -1)]

        self.__parallel_plot.plot(x_data, y_data,
                                  pen=newPen((0, 0, 255), 1))

    def plotHeatMap(self, dn: SamplingDatas):
        n = len(dn)
        N = len(dn[0].getRaw())
        histogram_image = pg.ImageItem()
        values_image = np.array([s.getRaw() for s in dn.samples])
        for i, row in enumerate(values_image):
            values_image[i] = (row - dn[i].min) / (dn[i].max - dn[i].min)
        histogram_image.setImage(values_image.transpose())
        histogram_image.setRect(-0.5, -0.5, n, N)
        self.__heatmap_plot.addItem(histogram_image)
        t = dict((i, f"{i+1}") for i in range(N)).items()
        self.__heatmap_plot.getAxis("left").setTicks((t, []))

    def plotBubleDiagram(self, dn: list[SamplingData]):
        x_raw = dn[0].getRaw()
        y_raw = dn[1].getRaw()
        z_raw = dn[2].getRaw()
        sz = dn[2]
        def f_norm(z): return (z - sz.min) / (sz.max - sz.min)
        z_norm = [f_norm(z) * 25 + 2 for z in z_raw]

        self.__buble_plot.clear()
        self.__buble_plot.plot(x_raw, y_raw,
                               symbolSize=z_norm, alphaHint=0.6, pen=None)

    def plotGlyphDiagram(self, dn: list[SamplingData], col):
        x_raw = dn[0].getRaw()
        y_raw = dn[1].getRaw()
        z_raw = dn[2].getRaw()
        x = dn[0]
        y = dn[1]
        if col == 0:
            col = SamplingData.calculateM(len(x_raw))

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
        if index == 0:
            self.plotParallelCoordinates(self.datas.samples)
        if index == 1:
            self.plotScatterDiagram(self.datas.samples, self.column_count)
        if index == 2:
            self.plotHeatMap(self.datas)
        if index == 4:
            self.plot3D(self.datas.samples)
        if index == 5:
            self.plotBubleDiagram(self.datas.samples)
        if index == 6:
            self.plotGlyphDiagram(self.datas.samples, self.column_count)


def newPen(color, width):
    return {'color': color, 'width': width}


#  Example matplotlib 3d
if __name__ == "__main__":
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    ax = plt.figure().add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.025)
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5,
                    rstride=8, cstride=8, alpha=0.3)
    ax.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')
    ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
           xlabel='X', ylabel='Y', zlabel='Z')
    plt.show()
