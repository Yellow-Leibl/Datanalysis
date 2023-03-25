from PyQt6.QtWidgets import QStackedWidget
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from Datanalysis.SamplingData import SamplingData
from Datanalysis.DoubleSampleData import DoubleSampleData
from Datanalysis.SamplingDatas import SamplingDatas
import numpy as np

pg.setConfigOption('imageAxisOrder', 'row-major')
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class PlotWidget(QStackedWidget):
    def __init__(self) -> None:
        super().__init__()
        #  2D
        __2d_widget = pg.GraphicsLayoutWidget()
        self.__2d_layout = __2d_widget.ci
        #  3D
        self.__3d_figure = Figure()
        __3d_widget = FigureCanvasQTAgg(self.__3d_figure)
        self.__axes = self.__3d_figure.add_subplot(projection='3d')

        self.addWidget(__2d_widget)
        self.addWidget(__3d_widget)

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

    def create3DPlot(self):
        self.setCurrentIndex(1)

    def plot1D(self, d: SamplingData, hist_data: list):
        self.plot1DHist(d, hist_data)
        self.plot1DEmp(d, hist_data)

    def plot1DHist(self, d: SamplingData, hist_data: list):
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

        self.hist_plot.clear()
        self.hist_plot.plot(x, y, fillLevel=0,
                            brush=(30, 120, 180))

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

    def plot2D(self, d2: DoubleSampleData, hist_data):
        x = d2.x
        y = d2.y
        if len(x.getRaw()) != len(y.getRaw()):
            return

        histogram_image = pg.ImageItem()
        histogram_image.setImage(hist_data)
        width = x.max - x.min
        height = y.max - y.min
        histogram_image.setRect(x.min, y.min, width, height)

        self.corr_plot.clear()
        self.corr_plot.addItem(histogram_image)
        self.corr_plot.plot(x.getRaw(), y.getRaw(),
                            symbolBrush=(30, 120, 180),
                            symbolPen=(0, 0, 0, 200), symbolSize=7,
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
        x0_raw = d3[0].getRaw()
        x1_raw = d3[1].getRaw()
        x2_raw = d3[2].getRaw()
        self.__axes.clear()
        self.__axes.scatter(x0_raw, x1_raw, x2_raw)

        self.__axes.set_xlabel("$X0$")
        self.__axes.set_ylabel("$X1$")
        self.__axes.set_zlabel("$X2$")
        self.__3d_figure.canvas.draw()

    def plot3DReproduction(self, d3: SamplingDatas):
        x0 = d3[0]
        x1 = d3[1]
        x0_min = x0.min
        x0_max = x0.max
        x0_plane = [x0_min, x0_max, x0_min, x0_max]
        x1_min = x1.min
        x1_max = x1.max
        x1_plane = [x1_min, x1_max, x1_min, x1_max]
        tr_l_f, f, tr_m_f = d3.toCreateLinearRegressionMNK(2)
        X0, X1 = np.meshgrid(x0_plane, x1_plane)
        zs_f = np.array(f(np.ravel(X0), np.ravel(X1)))
        zs_tr_l_f = np.array(tr_l_f(np.ravel(X0), np.ravel(X1)))
        zs_tr_m_f = np.array(tr_m_f(np.ravel(X0), np.ravel(X1)))
        X2 = zs_f.reshape(X0.shape)
        X2_l = zs_tr_l_f.reshape(X0.shape)
        X2_m = zs_tr_m_f.reshape(X0.shape)
        self.__axes.plot_surface(X0, X1, X2, alpha=0.1, color='red')
        self.__axes.plot_surface(X0, X1, X2_l, alpha=0.0, color='purple')
        self.__axes.plot_surface(X0, X1, X2_m, alpha=0.0, color='purple')

        self.__3d_figure.canvas.draw()

    def plotND(self, dn: SamplingDatas):
        pass


def newPen(color, width):
    return {'color': color, 'width': width}


if __name__ == "__main__":
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt

    ax = plt.figure().add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.025)

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5,
                    rstride=8, cstride=8, alpha=0.3)

    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit
    # on the 'walls' of the graph.
    ax.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

    ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
           xlabel='X', ylabel='Y', zlabel='Z')

    plt.show()
