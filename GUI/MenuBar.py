from PyQt6 import QtWidgets
from PyQt6.QtGui import QKeySequence


def fill_menu_bar(self: QtWidgets.QMainWindow):
    def addAction(menu: QtWidgets.QMenu, title, action, shortcut):
        act = menu.addAction(title, action)
        if act is not None:
            act.setShortcut(QKeySequence(shortcut))

    def addSection(menu: QtWidgets.QMenu, title):
        sec = menu.addAction(title)
        sec.setDisabled(True)

    menuBar = self.menuBar()
    file_menu = menuBar.addMenu("Файл")

    addAction(file_menu, "Відкрити", self.open_file_act, "Ctrl+O")
    addAction(file_menu, "Зберегти", self.save_file_act, "Ctrl+S")
    addAction(file_menu, "Вийти", exit, "Ctrl+Q")

    edit_menu = menuBar.addMenu("Редагувати")
    addAction(edit_menu, "Логарифмувати",
              lambda: self.edit_sample_event(0), "Ctrl+T")
    addAction(edit_menu, "Стандартизувати",
              lambda: self.edit_sample_event(1), "Ctrl+W")
    addAction(edit_menu, "Центрувати",
              lambda: self.edit_sample_event(2), "Ctrl+Shift+W")
    addAction(edit_menu, "Зсунути",
              lambda: self.edit_sample_event(3), "Ctrl+P")
    addAction(edit_menu, "Видалити аномалії",
              lambda: self.edit_sample_event(4), "Ctrl+A")
    addAction(edit_menu, "До незалежних величин",
              lambda: self.edit_sample_event(5), "Ctrl+Y")
    addAction(edit_menu, "Видалити проміжок",
              lambda: self.edit_sample_event(6), "")
    edit_menu.addSeparator()
    addAction(edit_menu, "Клонувати", self.duplicate_sample, "Ctrl+C")
    addAction(edit_menu, "Зобразити розподіл", self.draw_samples, "Ctrl+D")
    addAction(edit_menu, "Видалити спостереження", self.delete_observations,
              "Ctrl+Backspace")
    addAction(edit_menu, "Перейменувати", self.rename_sample, "Ctrl+R")
    edit_menu.addSeparator()
    addAction(edit_menu, "Вилучення тренду", self.remove_trend, "Ctrl+T")

    view_menu = menuBar.addMenu("Вигляд")
    addAction(view_menu, "Наступна вкладка", self.nextProtocolTab, "Alt+Tab")
    addAction(view_menu, "Часовий ряд/Часовий зріз",
              self.change_sample_type_mode, "Alt+T")
    addAction(view_menu, "Очистити графік", self.clear_plot, "Ctrl+Alt+C")

    self.reprod_num = -1

    regr_menu = menuBar.addMenu("&Регресія")
    addAction(regr_menu, "&Нормальний",
              lambda: self.set_reproduction_series(0), "Ctrl+Alt+Q")
    addAction(regr_menu, "&Рівномірний",
              lambda: self.set_reproduction_series(1), "Ctrl+Alt+W")
    addAction(regr_menu, "&Експоненціальний",
              lambda: self.set_reproduction_series(2), "Ctrl+Alt+E")
    addAction(regr_menu, "&Вейбулла",
              lambda: self.set_reproduction_series(3), "Ctrl+Alt+R")
    addAction(regr_menu, "&Арксинус",
              lambda: self.set_reproduction_series(4), "Ctrl+Alt+T")
    regr_menu.addSeparator()
    addAction(regr_menu, "Лінійна МНК (2D)",
              lambda: self.set_reproduction_series(5), "Ctrl+Alt+Y")
    addAction(regr_menu, "Лінійна Метод Тейла",
              lambda: self.set_reproduction_series(6), "Ctrl+Alt+U")
    addAction(regr_menu, "Парабола",
              lambda: self.set_reproduction_series(7), "Ctrl+Alt+I")
    addAction(regr_menu, "Квазілінійна y = a * exp(b * x)",
              lambda: self.set_reproduction_series(8), "Ctrl+Alt+O")
    regr_menu.addSeparator()
    addAction(regr_menu, "Лінійна МНК",
              lambda: self.set_reproduction_series(9), "Ctrl+Alt+P")
    addAction(regr_menu, "Лінійне різноманіття",
              lambda: self.set_reproduction_series(10), "Ctrl+Alt+[")
    regr_menu.addSeparator()
    addAction(regr_menu, "Поліноміальна регресія",
              lambda: self.set_reproduction_series(11), "Ctrl+Alt+]")

    time_menu = menuBar.addMenu("Часові ряди")

    addSection(time_menu, "Згладжування")
    addAction(time_menu, "Ковзне середнє",
              lambda: self.smooth_series(0), "Ctrl+Alt+S")
    addAction(time_menu, "Медіанне",
              lambda: self.smooth_series(1), "Ctrl+Alt+F")
    addAction(time_menu, "Просте ковзне середнє  SMA",
              lambda: self.smooth_series(2), "Ctrl+Alt+G")
    addAction(time_menu, "Зважене ковзне середнє WMA",
              lambda: self.smooth_series(3), "Ctrl+Alt+J")
    addAction(time_menu, "Експоненціальне ковзне середнє EMA",
              lambda: self.smooth_series(4), "Ctrl+Alt+K")
    addAction(time_menu, "Подвійне експоненціальне ковзне середнє DMA",
              lambda: self.smooth_series(5), "Ctrl+Alt+L")
    addAction(time_menu, "Потрійне експоненціальне ковзне середнє TMA",
              lambda: self.smooth_series(6), "Ctrl+Alt+;")
    time_menu.addSeparator()
    addAction(time_menu, "Підтвердити згладжування",
              lambda: self.smooth_series(-2), "Ctrl+Alt+A")
    time_menu.addSeparator()

    addSection(time_menu, "Метод Гусені SSA")
    addAction(time_menu, "Візуалізувати головні компоненти",
              lambda: self.ssa(0), "Ctrl+Alt+Z")
    addAction(time_menu, "Реконструкція",
              lambda: self.ssa(1), "Ctrl+Alt+X")
    addAction(time_menu, "Прогнозування на наявних даних",
              lambda: self.ssa(2), "Ctrl+Alt+V")
    addAction(time_menu, "Прогнозування майбутніх даних",
              lambda: self.ssa(3), "Ctrl+Alt+B")

    crit_menu = menuBar.addMenu("&Критерії")
    def set_trust(func): return lambda: func(self.feature_area.get_trust())
    addAction(crit_menu, "Перевірка однорідності/незалежності",
              set_trust(self.homogeneity_and_independence), "Ctrl+Shift+I")
    addAction(crit_menu, "Порівняння лінійних регресій",
              set_trust(self.linear_models_crit), "Ctrl+Shift+E")
    addAction(crit_menu, "Перевірка однорідності сукупностей (Нормальні)",
              self.homogeneity_n_samples, "Ctrl+Shift+N")
    addAction(crit_menu, "Частковий коефіцієнт кореляції",
              self.partial_correlation, "Ctrl+Shift+P")

    anal_menu = menuBar.addMenu("Компонентний аналіз")
    addAction(anal_menu, "Метод головних компонент", self.pca, "Ctrl+Shift+M")

    clust_menu = menuBar.addMenu("Кластерний аналіз")
    addAction(clust_menu, "Кластеризація методом k-середніх",
              self.kmeans, "")
    addAction(clust_menu, "Кластеризація аглоративним методом",
              self.agglomerative_clustering, "")

    class_menu = menuBar.addMenu("Класифікація")
