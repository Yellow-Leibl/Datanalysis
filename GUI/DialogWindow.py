from GUI.ui_tools import QtWidgets, FormLayout


class DialogWindow(QtWidgets.QDialog):
    def __init__(self, **args):
        super().__init__()
        title = args.get("title", "")
        self.setWindowTitle(title)
        size = args.get("size", (0, 0))
        self.resize(size[0], size[1])

        self.setModal(True)

        self.form_list = args.get("form_args", [])

        but_ok = QtWidgets.QPushButton("OK")
        but_ok.clicked.connect(self.ok_action)
        self.form_list += ["", but_ok]

        self.form = FormLayout(*self.form_list)

        self.setLayout(self.form)

        self.return_from_dialog = {}

    def ok_action(self):
        self.return_from_dialog = self.get_values()
        self.accept()

    def get_vals(self):
        self.exec()
        return self.return_from_dialog

    def get_values(self):
        vals = {}
        for i in range(0, len(self.form_list) - 2, 2):
            wid = self.form_list[i + 1]
            if isinstance(wid, QtWidgets.QComboBox):
                vals[self.form_list[i]] = wid.currentData()
            elif isinstance(wid, QtWidgets.QLineEdit):
                vals[self.form_list[i]] = wid.text()
            else:
                vals[self.form_list[i]] = self.form_list[i + 1].value()
        return vals
