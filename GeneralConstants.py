from PyQt5.QtCore import Qt
from enum import Enum


class Critetion(Enum):
    HOMOGENEITY_INDEPENDENCE = 0,
    LINEAR_REGRESSION_MODELS = 1


dict_crit = {
    "Перевірка однорідності/незалежності":  Critetion.HOMOGENEITY_INDEPENDENCE,
    "Порівняння лінійних регресій":         Critetion.LINEAR_REGRESSION_MODELS
}

dict_crit_shortcut = {
    "Перевірка однорідності/незалежності":  Qt.CTRL + Qt.Key_I,
    "Порівняння лінійних регресій":         Qt.CTRL + Qt.Key_E
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
    "&Перетворити":         Qt.CTRL + Qt.Key_T,
    "&Стандартизувати":     Qt.CTRL + Qt.Key_S,
    "&Зсунути":             Qt.CTRL + Qt.Key_P,
    "&Клонувати":           Qt.CTRL + Qt.Key_Z,
    "&Видалити аномалії":   Qt.CTRL + Qt.Key_A,
    "Зо&бразити розподіл":  Qt.Key_D,
    "Видалити &розподіл":   Qt.CTRL + Qt.Key_D
}

dict_repr = {
    "&Нормальний":          0,
    "&Рівномірний":         1,
    "&Експоненціальний":    2,
    "&Вейбулла":            3,
    "&Арксинус":            4,
    "&Очистити":            5,
}
dict_repr_shortcut = {
    "&Нормальний":          Qt.CTRL + Qt.ALT + Qt.Key_N,
    "&Рівномірний":         Qt.CTRL + Qt.ALT + Qt.Key_U,
    "&Експоненціальний":    Qt.CTRL + Qt.ALT + Qt.Key_E,
    "&Вейбулла":            Qt.CTRL + Qt.ALT + Qt.Key_W,
    "&Арксинус":            Qt.CTRL + Qt.ALT + Qt.Key_A,
    "&Очистити":            Qt.CTRL + Qt.ALT + Qt.Key_C,
}

dict_regr = {
    "Лінійна &МНК":         0,
    "Лінійна Метод &Тейла": 1,
    "&Парабола":            2,
    "y = a * exp(b * x)":   3,
    "&Очистити":            4,
}

dict_regr_shortcut = {
    "Лінійна &МНК":         Qt.CTRL + Qt.ALT + Qt.Key_M,
    "Лінійна Метод &Тейла": Qt.CTRL + Qt.ALT + Qt.Key_T,
    "&Парабола":            Qt.CTRL + Qt.ALT + Qt.Key_P,
    "y = a * exp(b * x)":   Qt.CTRL + Qt.ALT + Qt.Key_K,
    "&Очистити":            Qt.CTRL + Qt.ALT + Qt.Key_C,
}
