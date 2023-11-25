from PyQt6 import QtWidgets


def MonoFontForSpecificOS():
    import platform
    name = platform.system()
    if name == 'Darwin':
        return "Andale Mono"
    else:
        return "Monospace"


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


class SpinBox(QtWidgets.QSpinBox):
    def __init__(self, val_changed_f=None,
                 min=0, max=1, val=0):
        super().__init__()
        self.setMinimum(min)
        self.setMaximum(max)
        self.setValue(val)
        if val_changed_f is not None:
            self.valueChanged.connect(val_changed_f)


class DoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self, val_changed_f=None,
                 min=0, max=1, decimals=None, val=0):
        super().__init__()
        self.setMinimum(min)
        self.setMaximum(max)
        if decimals is not None:
            self.setDecimals(decimals)
        self.setValue(val)
        if val_changed_f is not None:
            self.valueChanged.connect(val_changed_f)
