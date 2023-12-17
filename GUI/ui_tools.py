from PyQt6 import QtWidgets


def BoxWithObjects(box, *args):
    addObjects(box, *args)
    return box


def addObjects(box: QtWidgets.QBoxLayout, *args):
    for arg in args:
        bases = str(arg.__class__.__bases__)
        type_name = str(arg.__class__.__name__)
        if 'Layout' in bases or 'Layout' in type_name:
            box.addLayout(arg)
        elif 'Widget' in bases or 'Widget' in type_name:
            box.addWidget(arg)
        else:
            raise Exception()


class WidgetWithLayout(QtWidgets.QWidget):
    def __init__(self, layout: QtWidgets.QLayout) -> None:
        super().__init__()
        self.setLayout(layout)


class VBoxLayout(QtWidgets.QVBoxLayout):
    def __init__(self, *args) -> None:
        super().__init__()
        addObjects(self, *args)


class HBoxLayout(QtWidgets.QHBoxLayout):
    def __init__(self, *args) -> None:
        super().__init__()
        addObjects(self, *args)


class SpinBox(QtWidgets.QSpinBox):
    def __init__(self, val_changed_f=None,
                 min_v=0, max_v=1, val=0):
        super().__init__()
        self.setMinimum(min_v)
        self.setMaximum(max_v)
        self.setValue(val)
        if val_changed_f is not None:
            self.valueChanged.connect(val_changed_f)


class DoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self, val_changed_f=None,
                 min_v=0, max_v=1, decimals=None, val=0):
        super().__init__()
        self.setMinimum(min_v)
        self.setMaximum(max_v)
        if decimals is not None:
            self.setDecimals(decimals)
        self.setValue(val)
        if val_changed_f is not None:
            self.valueChanged.connect(val_changed_f)


class TextEdit(QtWidgets.QTextEdit):
    def __init__(self, text="", read_only=False, mono_font=False) -> None:
        super().__init__(text)
        self.setReadOnly(read_only)
        if mono_font:
            self.setFontFamily(get_mono_font_any_os())


class FormLayout(QtWidgets.QFormLayout):
    def __init__(self, *args) -> None:
        super().__init__()
        self.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        if len(args) % 2 != 0:
            raise Exception("Wrong number of arguments")
        for i in range(0, len(args), 2):
            if args[i] == "":
                self.addRow(args[i + 1])
            else:
                self.addRow(args[i], args[i + 1])


class TabWidget(QtWidgets.QTabWidget):
    def __init__(self, *args) -> None:
        super().__init__()
        if len(args) % 2 != 0:
            return
        for i in range(0, len(args), 2):
            self.addTab(args[i], args[i + 1])


def get_mono_font_any_os():
    import platform
    name = platform.system()
    if name == 'Darwin':
        return "Andale Mono"
    else:
        return "Monospace"
