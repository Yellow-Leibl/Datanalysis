from PyQt6.QtWidgets import (QWidget, QLabel, QPushButton, QStackedWidget)
from GUI.ui_tools import (FormLayout, SpinBox, WidgetWithLayout,
                          DoubleSpinBox, HBoxLayout, VBoxLayout)


class FeatureArea(QStackedWidget):
    def __init__(self, parent):
        super().__init__()
        self.addWidget(self.create_1d_layout(parent))

    def create_1d_layout(self, parent) -> QWidget:
        self.__spin_number_column = SpinBox(
            val_changed_f=parent.update_sample)

        self.__trust_value = DoubleSpinBox(
            parent.update_sample, 0.0, 1.0, 5, 0.05)

        self.pca_number = SpinBox(min_v=2, max_v=99)

        self.__spin_box_min_x = DoubleSpinBox(decimals=5)
        self.__spin_box_max_x = DoubleSpinBox(decimals=5)
        self.__remove_anomaly = QPushButton("Видалити аномалії")
        self.__remove_anomaly.clicked.connect(parent.remove_anomaly_with_range)

        form_widget = FormLayout(
            "Кількість класів:", self.__spin_number_column,
            "Рівень значущості:", self.__trust_value,
            "", QWidget(),
            "Кількість перших компонентів для МГК:", self.pca_number)

        self.borders = HBoxLayout(QLabel("min"),
                                  self.__spin_box_min_x,
                                  QLabel("max"),
                                  self.__spin_box_max_x)

        form_func = VBoxLayout(form_widget,
                               self.borders,
                               self.__remove_anomaly)

        return WidgetWithLayout(form_func)

    def set_borders(self, min_x, max_x):
        self.__spin_box_min_x.setMinimum(min_x)
        self.__spin_box_min_x.setMaximum(max_x)
        self.__spin_box_min_x.setValue(min_x)

        self.__spin_box_max_x.setMinimum(min_x)
        self.__spin_box_max_x.setMaximum(max_x)
        self.__spin_box_max_x.setValue(max_x)

    def get_borders(self):
        return (self.__spin_box_min_x.value(),
                self.__spin_box_max_x.value())

    def silent_change_number_classes(self, n: int):
        self.__spin_number_column.blockSignals(True)
        self.__spin_number_column.setValue(n)
        self.__spin_number_column.blockSignals(False)

    def set_maximum_column_number(self, n: int):
        self.__spin_number_column.setMaximum(n)

    def get_number_classes(self):
        return self.__spin_number_column.value()

    def get_trust(self):
        return self.__trust_value.value()

    # TODO: switch visible layout by mode (QStackedWidget)
    def switch_to_1d_mode(self):
        pass

    def switch_to_2d_mode(self):
        pass

    def switch_to_nd_mode(self):
        pass

    def switch_to_time_series_mode(self):
        pass
