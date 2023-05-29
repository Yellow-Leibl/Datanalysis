from PyQt6.QtGui import QKeySequence
from PyQt6.QtCore import Qt
from enum import IntFlag, auto


class Critetion(IntFlag):
    HOMOGENEITY_INDEPENDENCE = auto(),
    LINEAR_REGRESSION_MODELS = auto(),
    HOMOGENEITY_N_SAMPLES = auto(),
    PARTIAL_CORRELATION = auto(),


dict_crit = {
    "Перевірка однорідності/незалежності":  Critetion.HOMOGENEITY_INDEPENDENCE,
    "Порівняння лінійних регресій":         Critetion.LINEAR_REGRESSION_MODELS,
    "Перевірка однорідності сукупностей "
    "(Нормальні)":                          Critetion.HOMOGENEITY_N_SAMPLES,
    "Частковий коефіцієнт кореляції":       Critetion.PARTIAL_CORRELATION,
}

dict_crit_shortcut = {
    "Перевірка однорідності/незалежності":              QKeySequence("Ctrl+I"),
    "Порівняння лінійних регресій":                     QKeySequence("Ctrl+E"),
    "Перевірка однорідності сукупностей (Нормальні)":   QKeySequence("Ctrl+N"),
    "Частковий коефіцієнт кореляції":                   QKeySequence("Ctrl+P"),
}


class Edit(IntFlag):
    TRANSFORM = auto(),
    STANDARTIZATION = auto(),
    SLIDE = auto(),
    DUPLICATE = auto(),
    DELETE_ANOMALY = auto(),
    DRAW_SAMPLES = auto(),
    DELETE_SAMPLES = auto(),


dict_edit = {
    "&Клонувати":           Edit.DUPLICATE.value,
    "&Перетворити":         Edit.TRANSFORM.value,
    "&Стандартизувати":     Edit.STANDARTIZATION.value,
    "&Зсунути":             Edit.SLIDE.value,
    "&Видалити аномалії":   Edit.DELETE_ANOMALY.value,
    "Зо&бразити розподіл":  Edit.DRAW_SAMPLES.value,
    "Видалити &розподіл":   Edit.DELETE_SAMPLES.value,
}
dict_edit_shortcut = {
    "&Перетворити":         QKeySequence("Ctrl+T"),
    "&Стандартизувати":     QKeySequence("Ctrl+W"),
    "&Зсунути":             QKeySequence("Ctrl+P"),
    "&Клонувати":           QKeySequence("Ctrl+C"),
    "&Видалити аномалії":   QKeySequence("Ctrl+A"),
    "Зо&бразити розподіл":  Qt.Key.Key_D,
    "Видалити &розподіл":   Qt.Key.Key_Backspace
}

dict_regression = {
    # 1D
    "&Нормальний":          0,
    "&Рівномірний":         1,
    "&Експоненціальний":    2,
    "&Вейбулла":            3,
    "&Арксинус":            4,
    # 2D
    "Лінійна &МНК":         5,
    "Лінійна Метод &Тейла": 6,
    "&Парабола":            7,
    "Квазілінійна: y = a * exp(b * x)":   8,
    # 3D
    "Лінійна МНК":         9,

    "&Очистити":            10,
}
dict_regr_shortcut = {
    # 1D
    "&Нормальний":          QKeySequence("Ctrl+Alt+N"),
    "&Рівномірний":         QKeySequence("Ctrl+Alt+U"),
    "&Експоненціальний":    QKeySequence("Ctrl+Alt+E"),
    "&Вейбулла":            QKeySequence("Ctrl+Alt+W"),
    "&Арксинус":            QKeySequence("Ctrl+Alt+A"),
    # 2D
    "Лінійна &МНК":         QKeySequence("Ctrl+Alt+M"),
    "Лінійна Метод &Тейла": QKeySequence("Ctrl+Alt+T"),
    "&Парабола":            QKeySequence("Ctrl+Alt+P"),
    "Квазілінійна: y = a * exp(b * x)": QKeySequence("Ctrl+Alt+K"),
    # 3D
    "Лінійна МНК":          QKeySequence("Ctrl+Alt+V"),
    "&Очистити":            QKeySequence("Ctrl+Alt+C"),
}


dict_file_shortcut = {
    "&Відкрити":            QKeySequence("Ctrl+O"),
    "&Зберегти":            QKeySequence("Ctrl+S"),
    "В&ийти":               QKeySequence("Ctrl+Q"),
}

dict_view_shortcut = {
    "&Наступна вкладка":    QKeySequence("Alt+Tab"),
}
