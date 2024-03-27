from GUI.ui_tools import (FormLayout, SpinBox, WidgetWithLayout,
                          DoubleSpinBox, VBoxLayout, QtWidgets)


class FeatureArea(QtWidgets.QStackedWidget):
    def __init__(self, parent):
        super().__init__()
        self.addWidget(self.create_1d_layout(parent))

    def create_1d_layout(self, parent) -> QtWidgets.QWidget:
        self.__spin_number_column = SpinBox(
            val_changed_f=parent.update_sample)

        self.__trust_value = DoubleSpinBox(
            parent.update_sample, 0.0, 1.0, 5, 0.05)

        form_widget = FormLayout(
            "Кількість класів:", self.__spin_number_column,
            "Рівень значущості:", self.__trust_value)

        form_func = VBoxLayout(form_widget)

        return WidgetWithLayout(form_func)

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
