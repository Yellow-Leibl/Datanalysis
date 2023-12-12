from PyQt6 import QtWidgets
from PyQt6.QtGui import QKeySequence


def fill_menu_bar(self: QtWidgets.QMainWindow):
    def addAction(menu: QtWidgets.QMenu, title, action, shortcut):
        act = menu.addAction(title, action)
        if act is not None:
            act.setShortcut(QKeySequence(shortcut))

    menuBar = self.menuBar()
    file_menu = menuBar.addMenu("Файл")

    addAction(file_menu, "Відкрити", self.open_file_act, "Ctrl+O")
    addAction(file_menu, "Зберегти", self.save_file_act, "Ctrl+S")
    addAction(file_menu, "Вийти", exit, "Ctrl+Q")

    edit_menu = menuBar.addMenu("Редагувати")
    addAction(edit_menu, "Перетворити", self.editSampleEvent, "Ctrl+T")
    addAction(edit_menu, "Стандартизувати", self.editSampleEvent, "Ctrl+W")
    addAction(edit_menu, "Центрувати", self.editSampleEvent, "Ctrl+Shift+W")
    addAction(edit_menu, "Зсунути", self.editSampleEvent, "Ctrl+P")
    addAction(edit_menu, "Видалити аномалії", self.editSampleEvent, "Ctrl+A")
    addAction(edit_menu, "До незалежних величин", self.editSampleEvent,
              "Ctrl+Y")
    edit_menu.addSeparator()
    addAction(edit_menu, "Клонувати", self.duplicateSample, "Ctrl+C")
    addAction(edit_menu, "Зобразити розподіл", self.draw_samples, "Ctrl+D")
    addAction(edit_menu, "Видалити спостереження", self.delete_observations,
              "Ctrl+Backspace")
    edit_menu.addSeparator()
    addAction(edit_menu, "Вилучення тренду", self.remove_trend, "Ctrl+R")

    view_menu = menuBar.addMenu("Вигляд")
    addAction(view_menu, "Наступна вкладка", self.nextProtocolTab, "Alt+Tab")
    addAction(view_menu, "Часовий ряд/Часовий зріз",
              self.change_sample_type_mode, "Alt+T")

    self.reprod_num = -1

    regr_menu = menuBar.addMenu("&Регресія")
    addAction(regr_menu, "&Нормальний", self.setReproductionSeries,
              "Ctrl+Alt+N")
    addAction(regr_menu, "&Рівномірний", self.setReproductionSeries,
              "Ctrl+Alt+U")
    addAction(regr_menu, "&Експоненціальний", self.setReproductionSeries,
              "Ctrl+Alt+E")
    addAction(regr_menu, "&Вейбулла", self.setReproductionSeries,
              "Ctrl+Alt+W")
    addAction(regr_menu, "&Арксинус", self.setReproductionSeries,
              "Ctrl+Alt+A")
    regr_menu.addSeparator()
    addAction(regr_menu, "Лінійна &МНК", self.setReproductionSeries,
              "Ctrl+Alt+M")
    addAction(regr_menu, "Лінійна Метод &Тейла", self.setReproductionSeries,
              "Ctrl+Alt+T")
    addAction(regr_menu, "&Парабола", self.setReproductionSeries,
              "Ctrl+Alt+P")
    addAction(regr_menu, "Квазілінійна y = a * exp(b * x)",
              self.setReproductionSeries, "Ctrl+Alt+K")
    regr_menu.addSeparator()
    addAction(regr_menu, "Лінійна МНК", self.setReproductionSeries,
              "Ctrl+Alt+V")
    addAction(regr_menu, "Лінійне різноманіття",
              self.setReproductionSeries, "Ctrl+Alt+R")
    regr_menu.addSeparator()
    addAction(regr_menu, "&Очистити", self.setReproductionSeries,
              "Ctrl+Alt+C")

    smth_menu = menuBar.addMenu("Згладжування")
    addAction(smth_menu, "Ковзне середнє", self.smooth_series, "Ctrl+Alt+M")
    addAction(smth_menu, "Медіанне", self.smooth_series, "Ctrl+Alt+N")
    addAction(smth_menu, "Просте ковзне середнє  SMA",
              self.smooth_series, "Ctrl+Alt+S")
    addAction(smth_menu, "Зважене ковзне середнє WMA",
              self.smooth_series, "Ctrl+Alt+W")
    addAction(smth_menu, "Експоненціальне ковзне середнє EMA",
              self.smooth_series, "Ctrl+Alt+E")
    addAction(smth_menu, "Подвійне експоненціальне ковзне середнє DMA",
              self.smooth_series, "Ctrl+Alt+R")
    addAction(smth_menu, "Потрійне експоненціальне ковзне середнє TMA",
              self.smooth_series, "Ctrl+Alt+T")
    smth_menu.addSeparator()
    addAction(smth_menu, "Очистити", self.smooth_series, "Ctrl+Alt+C")
    smth_menu.addSeparator()
    addAction(smth_menu, "Підтвердити згладжування", self.smooth_series,
              "Ctrl+Alt+A")

    crit_menu = menuBar.addMenu("&Критерії")
    def setTrust(func): return lambda: func(self.feature_area.get_trust())
    addAction(crit_menu, "Перевірка однорідності/незалежності",
              setTrust(self.homogeneityAndIndependence), "Ctrl+Shift+I")
    addAction(crit_menu, "Порівняння лінійних регресій",
              setTrust(self.linearModelsCrit), "Ctrl+Shift+E")
    addAction(crit_menu, "Перевірка однорідності сукупностей (Нормальні)",
              self.homogeneityNSamples, "Ctrl+Shift+N")
    addAction(crit_menu, "Частковий коефіцієнт кореляції",
              self.partialCorrelation, "Ctrl+Shift+P")

    anal_menu = menuBar.addMenu("Компонентний аналіз")
    addAction(anal_menu, "Метод головних компонент", self.PCA, "Ctrl+Shift+M")
