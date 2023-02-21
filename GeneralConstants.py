from PyQt6.QtGui import QKeySequence
from PyQt6.QtCore import Qt
from enum import Enum


class Critetion(Enum):
    HOMOGENEITY_INDEPENDENCE = 0,
    LINEAR_REGRESSION_MODELS = 1


dict_crit = {
    "Перевірка однорідності/незалежності":  Critetion.HOMOGENEITY_INDEPENDENCE,
    "Порівняння лінійних регресій":         Critetion.LINEAR_REGRESSION_MODELS
}

dict_crit_shortcut = {
    "Перевірка однорідності/незалежності":  QKeySequence("Ctrl+I"),
    "Порівняння лінійних регресій":         QKeySequence("Ctrl+E"),
}


class Edit(Enum):
    TRANSFORM = 0
    STANDARTIZATION = 1
    SLIDE = 2
    DUPLICATE = 3
    DELETE_ANOMALY = 4
    DRAW_SAMPLES = 5
    DELETE_SAMPLES = 6


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
    "Видалити &розподіл":   Qt.Key.Key_Delete
}

dict_reproduction = {
    "&Нормальний":          0,
    "&Рівномірний":         1,
    "&Експоненціальний":    2,
    "&Вейбулла":            3,
    "&Арксинус":            4,
    "&Очистити":            5,
}
dict_repr_shortcut = {
    "&Нормальний":          QKeySequence("Ctrl+Alt+N"),
    "&Рівномірний":         QKeySequence("Ctrl+Alt+U"),
    "&Експоненціальний":    QKeySequence("Ctrl+Alt+E"),
    "&Вейбулла":            QKeySequence("Ctrl+Alt+W"),
    "&Арксинус":            QKeySequence("Ctrl+Alt+A"),
    "&Очистити":            QKeySequence("Ctrl+Alt+C"),
}

dict_regression = {
    "Лінійна &МНК":         0,
    "Лінійна Метод &Тейла": 1,
    "&Парабола":            2,
    "Квазілінійна: y = a * exp(b * x)":   3,
    "&Очистити":            4,
}

dict_regr_shortcut = {
    "Лінійна &МНК":         QKeySequence("Ctrl+Alt+M"),
    "Лінійна Метод &Тейла": QKeySequence("Ctrl+Alt+T"),
    "&Парабола":            QKeySequence("Ctrl+Alt+P"),
    "Квазілінійна: y = a * exp(b * x)": QKeySequence("Ctrl+Alt+K"),
    "&Очистити":            QKeySequence("Ctrl+Alt+C"),
}


dict_file = {
    "&Відкрити":            0,
    "&Зберегти":            1,
    "В&ийти":               2
}


dict_file_shortcut = {
    "&Відкрити":            QKeySequence("Ctrl+O"),
    "&Зберегти":            QKeySequence("Ctrl+S"),
    "В&ийти":               QKeySequence("Ctrl+Q"),
}
