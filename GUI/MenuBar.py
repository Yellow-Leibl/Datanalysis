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
    addAction(edit_menu, "Перетворити", self.edit_sample_event, "Ctrl+T")
    addAction(edit_menu, "Стандартизувати", self.edit_sample_event, "Ctrl+W")
    addAction(edit_menu, "Центрувати", self.edit_sample_event, "Ctrl+Shift+W")
    addAction(edit_menu, "Зсунути", self.edit_sample_event, "Ctrl+P")
    addAction(edit_menu, "Видалити аномалії", self.edit_sample_event, "Ctrl+A")
    addAction(edit_menu, "До незалежних величин", self.edit_sample_event,
              "Ctrl+Y")
    edit_menu.addSeparator()
    addAction(edit_menu, "Клонувати", self.duplicate_sample, "Ctrl+C")
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
    addAction(regr_menu, "Лінійна &МНК",
              lambda: self.set_reproduction_series(5), "Ctrl+Alt+Y")
    addAction(regr_menu, "Лінійна Метод &Тейла",
              lambda: self.set_reproduction_series(6), "Ctrl+Alt+U")
    addAction(regr_menu, "&Парабола",
              lambda: self.set_reproduction_series(7), "Ctrl+Alt+I")
    addAction(regr_menu, "Квазілінійна y = a * exp(b * x)",
              lambda: self.set_reproduction_series(8), "Ctrl+Alt+O")
    regr_menu.addSeparator()
    addAction(regr_menu, "Лінійна МНК",
              lambda: self.set_reproduction_series(9), "Ctrl+Alt+P")
    addAction(regr_menu, "Лінійне різноманіття",
              lambda: self.set_reproduction_series(10), "Ctrl+Alt+[")
    regr_menu.addSeparator()
    addAction(regr_menu, "&Очистити",
              lambda: self.set_reproduction_series(-1), "Ctrl+Alt+C")

    smth_menu = menuBar.addMenu("Згладжування")
    addAction(smth_menu, "Ковзне середнє",
              self.smooth_series, "Ctrl+Alt+S")
    addAction(smth_menu, "Медіанне",
              self.smooth_series, "Ctrl+Alt+F")
    addAction(smth_menu, "Просте ковзне середнє  SMA",
              self.smooth_series, "Ctrl+Alt+G")
    addAction(smth_menu, "Зважене ковзне середнє WMA",
              self.smooth_series, "Ctrl+Alt+J")
    addAction(smth_menu, "Експоненціальне ковзне середнє EMA",
              self.smooth_series, "Ctrl+Alt+K")
    addAction(smth_menu, "Подвійне експоненціальне ковзне середнє DMA",
              self.smooth_series, "Ctrl+Alt+L")
    addAction(smth_menu, "Потрійне експоненціальне ковзне середнє TMA",
              self.smooth_series, "Ctrl+Alt+;")
    smth_menu.addSeparator()
    addAction(smth_menu, "Очистити", self.smooth_series, "Ctrl+Alt+C")
    smth_menu.addSeparator()
    addAction(smth_menu, "Підтвердити згладжування",
              self.smooth_series, "Ctrl+Alt+A")

    ssa_menu = menuBar.addMenu("Метод Гусені SSA")
    addAction(ssa_menu, "Візуалізувати головні компоненти",
              lambda: self.ssa(0), "Ctrl+Alt+Z")
    addAction(ssa_menu, "Реконструкція",
              lambda: self.ssa(1), "Ctrl+Alt+X")
    addAction(ssa_menu, "Прогнозування на наявних даних",
              lambda: self.ssa(2), "Ctrl+Alt+V")
    addAction(ssa_menu, "Прогнозування майбутніх даних",
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
