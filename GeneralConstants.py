from PyQt5.QtCore import Qt
from enum import Enum

dict_crit = {
    "Перевірка однорідності/незалежності":   0
}

dict_crit_shortcut = {
    "Перевірка однорідності/незалежності":   Qt.CTRL + Qt.Key_I
}


class Edit(Enum):
    TRANSFORM = 0
    STANDARTIZATION = 1
    SLIDE = 2
    UNDO = 3
    DELETE_ANOMALY = 4
    DELETE_SAMPLES = 5


dict_edit = {
    "&Перетворити":         Edit.TRANSFORM.value,
    "&Стандартизувати":     Edit.STANDARTIZATION.value,
    "&Зсунути":             Edit.SLIDE.value,
    "&Повернути":           Edit.UNDO.value,
    "&Видалити аномалії":   Edit.DELETE_ANOMALY.value,
    "Видалити &розподіл":   Edit.DELETE_SAMPLES
}
dict_edit_shortcut = {
    "&Перетворити":         Qt.CTRL + Qt.Key_T,
    "&Стандартизувати":     Qt.CTRL + Qt.Key_S,
    "&Зсунути":             Qt.CTRL + Qt.Key_P,
    "&Повернути":           Qt.CTRL + Qt.Key_Z,
    "&Видалити аномалії":   Qt.CTRL + Qt.Key_D,
    "Видалити &розподіл":   Qt.CTRL + Qt.Key_R
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
    "&Нормальний":          Qt.CTRL + Qt.Key_N,
    "&Рівномірний":         Qt.CTRL + Qt.Key_U,
    "&Експоненціальний":    Qt.CTRL + Qt.Key_E,
    "&Вейбулла":            Qt.CTRL + Qt.Key_W,
    "&Арксинус":            Qt.CTRL + Qt.Key_A,
    "&Очистити":            Qt.CTRL + Qt.Key_C,
}
